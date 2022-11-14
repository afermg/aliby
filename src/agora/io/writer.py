import itertools
import logging
from collections.abc import Iterable
from time import perf_counter
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import yaml
from utils_find_1st import cmp_equal, find_1st

from agora.io.bridge import BridgeH5
from agora.io.utils import timed

#################### Dynamic version ##################################


def load_attributes(file: str, group="/"):
    '''
    Load the metadata from an h5 file and convert to a dictionary, including the "parameters" field which is stored as YAML.

    Parameters
    ----------
    file: str
        Name of the h5 file
    group: str, optional
        The group in the h5 file from which to read the data
    '''
    # load the metadata, stored as attributes, from the h5 file and return as a dictionary
    with h5py.File(file, "r") as f:
        meta = dict(f[group].attrs.items())
    if "parameters" in meta:
        # convert from yaml format into dict
        meta["parameters"] = yaml.safe_load(meta["parameters"])
    return meta


class DynamicWriter:
    '''
    A parent class for all writers
    '''
    # a dict giving a tuple of the maximum size, as a 2D tuple, and the type for each dataset
    data_types = {}
    # the group in the h5 file to write to
    group = ""
    # compression info
    compression = "gzip"
    compression_opts = 9

    def __init__(self, file: str):
        self.file = file
        # the metadata is stored as attributes in the h5 file
        self.metadata = load_attributes(file)

    def _append(self, data, key, hgroup):
        """
        Append data to existing dataset in the h5 file otherwise create a new one

        Parameters
        ----------
        data
            Data to be written, typically a numpy array
        key: str
            Name of dataset
        hgroup: str
            Destination group in the h5 file
        """
        try:
            n = len(data)
        except Exception as e:
            logging.debug(
                "DynamicWriter: Attributes have no length: {}".format(e)
            )
            n = 1
        if key in hgroup:
            # append to existing dataset
            try:
                # FIXME This is broken by bugged mother-bud assignment
                dset = hgroup[key]
                dset.resize(dset.shape[0] + n, axis=0)
                dset[-n:] = data
            except Exception as e:
                logging.debug(
                    "DynamicWriter: Inconsistency between dataset shape and new empty data: {}".format(
                        e
                    )
                )
        else:
            # create new dataset
            # TODO Include sparsity check
            max_shape, dtype = self.datatypes[key]
            shape = (n,) + max_shape[1:]
            hgroup.create_dataset(
                key,
                shape=shape,
                maxshape=max_shape,
                dtype=dtype,
                compression=self.compression,
                compression_opts=self.compression_opts
                if self.compression is not None
                else None,
            )
            # write all data, signified by the empty tuple
            hgroup[key][()] = data


    def _overwrite(self, data, key, hgroup):
        """
        Delete and then replace existing dataset in h5 file

        Parameters
        ----------
        data
            Data to be written, typically a numpy array
        key: str
            Name of dataset
        hgroup: str
            Destination group in the h5 file
        """
        # We do not append to mother_assign; raise error if already saved
        data_shape = np.shape(data)
        max_shape, dtype = self.datatypes[key]
        # delete existing data
        if key in hgroup:
            del hgroup[key]
        # write new data
        hgroup.require_dataset(
            key, shape=data_shape, dtype=dtype, compression=self.compression,
        )
        # write all data, signified by the empty tuple
        hgroup[key][()] = data

    # def _check_key(self, key):
    #     if key not in self.datatypes:
    #         raise KeyError(f"No defined data type for key {key}")

    def write(self, data: dict, overwrite: list, meta={}):
        '''
        Write data and metadata to h5 file

         Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        meta: dict, optional
            Metadata to be written as attributes of the h5 file
        '''
        with h5py.File(self.file, "a") as store:
            # open group, creating if necessary
            hgroup = store.require_group(self.group)
            # write data
            for key, value in data.items():
                # only save data with a pre-defined data-type
                 if key not in self.datatypes:
                     raise KeyError(f"No defined data type for key {key}")
                 else:
                    try:
                        if key.startswith("attrs/"):
                            # metadata
                            key = key.split("/")[1]
                            hgroup.attrs[key] = value
                        elif key in overwrite:
                            # delete and replace existing dataset
                            self._overwrite(value, key, hgroup)
                        else:
                            # append or create new dataset
                            self._append(value, key, hgroup)
                    except Exception as e:
                        print(key, value)
                        raise (e)
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


##################### Special instances #####################
class TilerWriter(DynamicWriter):
    datatypes = {
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        "attrs/tile_size": ((1,), np.uint16),
        "attrs/max_size": ((1,), np.uint16),
    }
    group = "trap_info"

    def write(self, data, overwrite: list, tp: int, meta={}):
        """
        Custom function to avoid writing over any data that already exists at time point tp
        """
        skip = False
        # append to h5 file
        with h5py.File(self.file, "a") as store:
            # open group, creating if necessary
            hgroup = store.require_group(self.group)
            # find xy drift for each time point as proof that it has already been processed
            nprev = hgroup.get("drifts", None)
            if nprev and tp < nprev.shape[0]:
                # data already exists
                print(f"Tiler: Skipping timepoint {tp}")
                skip = True
        if not skip:
            super().write(data=data, overwrite=overwrite, meta=meta)

# Alan: why's this here?
tile_size = 117

# Alan: we use complex numbers because...
@timed()
def save_complex(array, dataset):
    # append array, an 1D array of complex numbers, onto dataset, a 2D array of real numbers
    n = len(array)
    if n > 0:
        dataset.resize(dataset.shape[0] + n, axis=0)
        dataset[-n:, 0] = array.real
        dataset[-n:, 1] = array.imag


@timed()
def load_complex(dataset):
    # convert 2D dataset into a 1D array of complex numbers
    array = dataset[:, 0] + 1j * dataset[:, 1]
    return array


class BabyWriter(DynamicWriter):
    compression = "gzip"
    max_ncells = 2e5  # Alan: Could just make this None
    max_tps = 1e3  # Could just make this None
    # the number of cells in a chunk for edge masks
    chunk_cells = 25
    default_tile_size = 117
    datatypes = {
        "centres": ((None, 2), np.uint16),
        "position": ((None,), np.uint16),
        "angles": ((None,), h5py.vlen_dtype(np.float32)),
        "radii": ((None,), h5py.vlen_dtype(np.float32)),
        "edgemasks": ((max_ncells, max_tps, tile_size, tile_size), bool),
        "ellipse_dims": ((None, 2), np.float32),
        "cell_label": ((None,), np.uint16),
        "trap": ((None,), np.uint16),
        "timepoint": ((None,), np.uint16),
        # "mother_assign": ((None,), h5py.vlen_dtype(np.uint16)),
        "mother_assign_dynamic": ((None,), np.uint16),
        "volumes": ((None,), np.float32),
    }
    group = "cell_info"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._traps_initialised = False

    def __init_trap_info(self):
        # requires traps to have been initialised
        trap_metadata = load_attributes(self.file, "trap_info")
        tile_size = trap_metadata.get("tile_size", self.default_tile_size)
        max_tps = self.metadata["time_settings/ntimepoints"][0]
        self.datatypes["edgemasks"] = (
            (self.max_ncells, max_tps, tile_size, tile_size),
            bool,
        )
        self._traps_initialised = True

    def __init_edgemasks(self, hgroup, edgemasks, current_indices, n_cells):
        # create the values dataset in the h5 file
        # holds the edge masks and has shape (n_tps, n_cells, tile_size, tile_size)
        key = "edgemasks"
        max_shape, dtype = self.datatypes[key]
        shape = (n_cells, 1) + max_shape[2:]
        chunks = (self.chunk_cells, 1) + max_shape[2:]
        val_dset = hgroup.create_dataset(
            "values",
            shape=shape,
            maxshape=max_shape,
            dtype=dtype,
            chunks=chunks,
            compression=self.compression,
        )
        val_dset[:, 0] = edgemasks
        # create index dataset in the h5 file:
        # the (trap, cell_id) description used to index into the values and has shape (n_cells, 2)
        ix_max_shape = (max_shape[0], 2)
        ix_shape = (0, 2)
        ix_dtype = np.uint16
        ix_dset = hgroup.create_dataset(
            "indices",
            shape=ix_shape,
            maxshape=ix_max_shape,
            dtype=ix_dtype,
            compression=self.compression,
        )
        save_complex(current_indices, ix_dset)

    def __append_edgemasks(self, hgroup, edgemasks, current_indices):
        val_dset = hgroup["values"]
        ix_dset = hgroup["indices"]
        existing_indices = load_complex(ix_dset)
        # check if there are any new labels
        available = np.in1d(current_indices, existing_indices)
        missing = current_indices[~available]
        all_indices = np.concatenate([existing_indices, missing])
        # resizing
        debug_t = perf_counter()  # for timing code for debugging
        n_tps = val_dset.shape[1] + 1
        n_add_cells = len(missing)
        # resize dataset for Time and Cells
        new_shape = (val_dset.shape[0] + n_add_cells, n_tps) + val_dset.shape[2:]
        val_dset.resize(new_shape)
        logging.debug(f"Timing:resizing:{perf_counter() - debug_t}")
        # write data
        cell_indices = np.where(np.in1d(all_indices, current_indices))[0]
        for ix, mask in zip(cell_indices, edgemasks):
            try:
                val_dset[ix, n_tps - 1] = mask
            except Exception as e:
                logging.debug(
                    "Exception: {}:{}, {}, {}".format(
                        e, ix, n_tps, val_dset.shape
                    )
                )
        # save the index values
        save_complex(missing, ix_dset)

    def write_edgemasks(self, data, keys, hgroup):
        # data has shape (trap_ids, cell_labels, edgemasks) and is a structured array
        if not self._traps_initialised:
            self.__init_trap_info()
        key = "edgemasks"
        val_key = "values"
        traps, cell_labels, edgemasks = data
        n_cells = len(cell_labels)
        hgroup = hgroup.require_group(key)
        # create complex indices with traps as real part and cell_labels as imaginary part
        current_indices = np.array(traps) + 1j * np.array(cell_labels)
        if val_key not in hgroup:
            self.__init_edgemasks(hgroup, edgemasks, current_indices, n_cells)
        else:
            self.__append_edgemasks(hgroup, edgemasks, current_indices)

    def write(self, data, overwrite: list, tp: int = None, meta={}):
        with h5py.File(self.file, "a") as store:
            hgroup = store.require_group(self.group)
            for key, value in data.items():
                if key not in self.datatypes:
                     raise KeyError(f"BabyWriter: No defined data type for key {key}")
                else:
                    try:
                        if key.startswith("attrs/"):
                            # metadata
                            key = key.split("/")[1]
                            hgroup.attrs[key] = value
                        elif key in overwrite:
                            # delete and replace existing dataset
                            self._overwrite(value, key, hgroup)
                        elif key == "edgemasks":
                            keys = ["trap", "cell_label", "edgemasks"]
                            value = [data[x] for x in keys]
                            edgemask_dset = hgroup.get(key + "/values", None)
                            if (
                                edgemask_dset
                                and tp < edgemask_dset[()].shape[1]
                            ):
                                # data already exists
                                print(f"BabyWriter: Skipping edgemasks in tp {tp}")
                            else:
                                self.write_edgemasks(value, keys, hgroup)
                        else:
                            # append or create new dataset
                            self._append(value, key, hgroup)
                    except Exception as e:
                        print(key, value)
                        raise (e)
        # write metadata
        for key, value in meta.items():
            hgroup.attrs[key] = value


class LinearBabyWriter(DynamicWriter):
    # TODO make this YAML
    compression = "gzip"
    datatypes = {
        "centres": ((None, 2), np.uint16),
        "position": ((None,), np.uint16),
        "angles": ((None,), h5py.vlen_dtype(np.float32)),
        "radii": ((None,), h5py.vlen_dtype(np.float32)),
        "edgemasks": ((None, tile_size, tile_size), bool),
        "ellipse_dims": ((None, 2), np.float32),
        "cell_label": ((None,), np.uint16),
        "trap": ((None,), np.uint16),
        "timepoint": ((None,), np.uint16),
        # "mother_assign": ((None,), h5py.vlen_dtype(np.uint16)),
        "mother_assign_dynamic": ((None,), np.uint16),
        "volumes": ((None,), np.float32),
    }
    group = "cell_info"

    def write(self, data, overwrite: list, tp=None, meta={}):
        # Data is a dictionary, if not, make it one
        # Overwrite data is a list

        with h5py.File(self.file, "a") as store:
            hgroup = store.require_group(self.group)
            available_tps = hgroup.get("timepoint", None)
            if not available_tps or tp not in np.unique(available_tps[()]):
                super().write(data, overwrite)
            else:
                print(f"BabyWriter: Skipping tp {tp}")

            for key, value in meta.items():
                hgroup.attrs[key] = value


class StateWriter(DynamicWriter):
    datatypes = {
        "max_lbl": ((None, 1), np.uint16),
        "tp_back": ((None, 1), np.uint16),
        "trap": ((None, 1), np.int16),
        "cell_lbls": ((None, 1), np.uint16),
        "prev_feats": ((None, None), np.float32),
        "lifetime": ((None, 2), np.uint16),
        "p_was_bud": ((None, 2), np.float32),
        "p_is_mother": ((None, 2), np.float32),
        "ba_cum": ((None, None), np.float32),
    }
    group = "last_state"
    compression = "gzip"

    @staticmethod
    def format_field(states: list, field: str):
        # Flatten a field in the states list to save as an hdf5 dataset
        fields = [pos_state[field] for pos_state in states]
        return fields

    @staticmethod
    def format_values_tpback(states: list, val_name: str):
        tp_back, trap, value = [
            [[] for _ in states[0][val_name]] for _ in range(3)
        ]

        lbl_tuples = [
            (tp_back, trap, cell_label)
            for trap, state in enumerate(states)
            for tp_back, value in enumerate(state[val_name])
            for cell_label in value
        ]
        if len(lbl_tuples):
            tp_back, trap, value = zip(*lbl_tuples)

        return tp_back, trap, value

    @staticmethod
    def format_values_traps(states: list, val_name: str):
        formatted = np.array(
            [
                (trap, clabel_val)
                for trap, state in enumerate(states)
                for clabel_val in state[val_name]
            ]
        )
        return formatted

    @staticmethod
    def pad_if_needed(array: np.ndarray, pad_size: int):
        padded = np.zeros((pad_size, pad_size)).astype(float)
        length = len(array)
        padded[:length, :length] = array

        return padded

    def format_states(self, states: list):
        formatted_state = {"max_lbl": [state["max_lbl"] for state in states]}
        tp_back, trap, cell_label = self.format_values_tpback(
            states, "cell_lbls"
        )
        _, _, prev_feats = self.format_values_tpback(states, "prev_feats")

        # Heterogeneous datasets
        formatted_state["tp_back"] = tp_back
        formatted_state["trap"] = trap
        formatted_state["cell_lbls"] = cell_label
        formatted_state["prev_feats"] = np.array(prev_feats)

        # One entry per cell label - tp_back independent
        for val_name in ("lifetime", "p_was_bud", "p_is_mother"):
            formatted_state[val_name] = self.format_values_traps(
                states, val_name
            )

        bacum_max = max([len(state["ba_cum"]) for state in states])

        formatted_state["ba_cum"] = np.array(
            [
                self.pad_if_needed(state["ba_cum"], bacum_max)
                for state in states
            ]
        )

        return formatted_state

    def write(self, data, overwrite: Iterable, tp: int = None):
        # formatted_data = self.format_states(data)
        # super().write(data=formatted_data, overwrite=overwrite)
        if len(data):
            last_tp = 0
            if tp is None:
                tp = 0

            try:
                with h5py.File(self.file, "r") as f:
                    gr = f.get(self.group, None)
                    if gr:
                        last_tp = gr.attrs.get("tp", 0)

                # print(f"{ self.file } - tp: {tp}, last_tp: {last_tp}")
                if tp == 0 or tp > last_tp:
                    # print(f"Writing timepoint {tp}")
                    formatted_data = self.format_states(data)
                    super().write(data=formatted_data, overwrite=overwrite)
                    with h5py.File(self.file, "a") as f:
                        # print(f"Writing tp {tp}")
                        f[self.group].attrs["tp"] = tp
                elif tp > 0 and tp <= last_tp:
                    print(f"BabyWriter: Skipping timepoint {tp}")
            except Exception as e:
                raise (e)
        else:
            print("Skipping overwriting empty state")


#################### Extraction version ###############################
class Writer(BridgeH5):
    """
    Class in charge of transforming data into compatible formats

    Decoupling interface from implementation!

    Parameters
    ----------
        filename: str Name of file to write into
        flag: str, default=None
            Flag to pass to the default file reader. If None the file remains closed.
        compression: str, default=None
            Compression method passed on to h5py writing functions (only used for
        dataframes and other array-like data.)
    """

    def __init__(self, filename, compression=None):
        super().__init__(filename, flag=None)

        if compression is None:
            self.compression = "gzip"

    def write(
        self,
        path: str,
        data: Iterable = None,
        meta: Dict = {},
        overwrite: str = None,
    ):
        """
        Parameters
        ----------
        path : str
            Path inside h5 file to write into.
        data : Iterable, default = None
        meta : Dict, default = {}

        """
        self.id_cache = {}
        with h5py.File(self.filename, "a") as f:
            if overwrite == "overwrite":  # TODO refactor overwriting
                if path in f:
                    del f[path]
            # elif overwrite == "accumulate":  # Add a number if needed
            #     if path in f:
            #         parent, name = path.rsplit("/", maxsplit=1)
            #         n = sum([x.startswith(name) for x in f[path]])
            #         path = path + str(n).zfill(3)
            # elif overwrite == "skip":
            #     if path in f:
            #         logging.debug("Skipping dataset {}".format(path))

            logging.debug(
                "{} {} to {} and {} metadata fields".format(
                    overwrite, type(data), path, len(meta)
                )
            )
            if data is not None:
                self.write_dset(f, path, data)
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, data=metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        if isinstance(data, pd.DataFrame):
            self.write_pd(f, path, data, compression=self.compression)
        elif isinstance(data, pd.MultiIndex):
            self.write_index(f, path, data)  # , compression=self.compression)
        elif isinstance(data, Dict) and np.all(
            [isinstance(x, pd.DataFrame) for x in data.values]
        ):
            for k, df in data.items():
                self.write_dset(f, path + f"/{k}", df)
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, attr: str, data: Iterable):
        obj = f.require_group(path)

        obj.attrs[attr] = data

    @staticmethod
    def write_arraylike(f: h5py.File, path: str, data: Iterable, **kwargs):
        if path in f:
            del f[path]

        narray = np.array(data)

        chunks = None
        if narray.any():
            chunks = (1, *narray.shape[1:])

        dset = f.create_dataset(
            path,
            shape=narray.shape,
            chunks=chunks,
            dtype="int",
            compression=kwargs.get("compression", None),
        )
        dset[()] = narray

    @staticmethod
    def write_index(f, path, pd_index, **kwargs):
        f.require_group(path)  # TODO check if we can remove this
        for i, name in enumerate(pd_index.names):
            ids = pd_index.get_level_values(i)
            id_path = path + "/" + name
            f.create_dataset(
                name=id_path,
                shape=(len(ids),),
                dtype="uint16",
                compression=kwargs.get("compression", None),
            )
            indices = f[id_path]
            indices[()] = ids

    def write_pd(self, f, path, df, **kwargs):
        values_path = (
            path + "values" if path.endswith("/") else path + "/values"
        )
        if path not in f:
            max_ncells = 2e5

            max_tps = 1e3
            f.create_dataset(
                name=values_path,
                shape=df.shape,
                # chunks=(min(df.shape[0], 1), df.shape[1]),
                # dtype=df.dtypes.iloc[0], This is making NaN in ints into negative vals
                dtype="float",
                maxshape=(max_ncells, max_tps),
                compression=kwargs.get("compression", None),
            )
            dset = f[values_path]
            dset[()] = df.values

            for name in df.index.names:
                indices_path = "/".join((path, name))
                f.create_dataset(
                    name=indices_path,
                    shape=(len(df),),
                    dtype="uint16",  # Assuming we'll always use int indices
                    chunks=True,
                    maxshape=(max_ncells,),
                )
                dset = f[indices_path]
                dset[()] = df.index.get_level_values(level=name).tolist()

            if (
                df.columns.dtype == np.int
                or df.columns.dtype == np.dtype("uint")
                or df.columns.name == "timepoint"
            ):
                tp_path = path + "/timepoint"
                f.create_dataset(
                    name=tp_path,
                    shape=(df.shape[1],),
                    maxshape=(max_tps,),
                    dtype="uint16",
                )
                tps = list(range(df.shape[1]))
                f[tp_path][tps] = tps
            else:
                f[path].attrs["columns"] = df.columns.tolist()
        else:
            dset = f[values_path]

            # Filter out repeated timepoints
            new_tps = set(df.columns)
            if path + "/timepoint" in f:
                new_tps = new_tps.difference(f[path + "/timepoint"][()])
            df = df[new_tps]

            if (
                not hasattr(self, "id_cache")
                or df.index.nlevels not in self.id_cache
            ):  # Use cache dict to store previously-obtained indices
                self.id_cache[df.index.nlevels] = {}
                existing_ids = self.get_existing_ids(
                    f, [path + "/" + x for x in df.index.names]
                )
                # Split indices in existing and additional
                new = df.index.tolist()
                if (
                    df.index.nlevels == 1
                ):  # Cover for cases with a single index
                    new = [(x,) for x in df.index.tolist()]
                (
                    found_multis,
                    self.id_cache[df.index.nlevels]["additional_multis"],
                ) = self.find_ids(
                    existing=existing_ids,
                    new=new,
                )
                found_indices = np.array(
                    locate_indices(existing_ids, found_multis)
                )

                # We must sort our indices for h5py indexing
                incremental_existing = np.argsort(found_indices)
                self.id_cache[df.index.nlevels][
                    "found_indices"
                ] = found_indices[incremental_existing]
                self.id_cache[df.index.nlevels]["found_multi"] = found_multis[
                    incremental_existing
                ]

            existing_values = df.loc[
                [
                    _tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels]["found_multi"]
                ]
            ].values
            new_values = df.loc[
                [
                    _tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels][
                        "additional_multis"
                    ]
                ]
            ].values
            ncells, ntps = f[values_path].shape

            # Add found cells
            dset.resize(dset.shape[1] + df.shape[1], axis=1)
            dset[:, ntps:] = np.nan

            # TODO refactor this indices sorting. Could be simpler
            found_indices_sorted = self.id_cache[df.index.nlevels][
                "found_indices"
            ]

            # Cover for case when all labels are new
            if found_indices_sorted.any():
                # h5py does not allow bidimensional indexing,
                # so we have to iterate over the columns
                for i, tp in enumerate(df.columns):
                    dset[found_indices_sorted, tp] = existing_values[:, i]
            # Add new cells
            n_newcells = len(
                self.id_cache[df.index.nlevels]["additional_multis"]
            )
            dset.resize(dset.shape[0] + n_newcells, axis=0)
            dset[ncells:, :] = np.nan

            for i, tp in enumerate(df.columns):
                dset[ncells:, tp] = new_values[:, i]

            # save indices
            for i, name in enumerate(df.index.names):
                tmp = path + "/" + name
                dset = f[tmp]
                n = dset.shape[0]
                dset.resize(n + n_newcells, axis=0)
                dset[n:] = (
                    self.id_cache[df.index.nlevels]["additional_multis"][:, i]
                    if len(
                        self.id_cache[df.index.nlevels][
                            "additional_multis"
                        ].shape
                    )
                    > 1
                    else self.id_cache[df.index.nlevels]["additional_multis"]
                )

            tmp = path + "/timepoint"
            dset = f[tmp]
            n = dset.shape[0]
            dset.resize(n + df.shape[1], axis=0)
            dset[n:] = df.columns.tolist()

    @staticmethod
    def get_existing_ids(f, paths):
        # Fetch indices and convert them to a (nentries, nlevels) ndarray
        return np.array([f[path][()] for path in paths]).T

    @staticmethod
    def find_ids(existing, new):
        # Compare two tuple sets and return the intersection and difference
        # (elements in the 'new' set not in 'existing')
        set_existing = set([tuple(*x) for x in zip(existing.tolist())])
        existing_cells = np.array(list(set_existing.intersection(new)))
        new_cells = np.array(list(set(new).difference(set_existing)))

        return (
            existing_cells,
            new_cells,
        )


# @staticmethod
def locate_indices(existing, new):
    if new.any():
        if new.shape[1] > 1:
            return [
                find_1st(
                    (existing[:, 0] == n[0]) & (existing[:, 1] == n[1]),
                    True,
                    cmp_equal,
                )
                for n in new
            ]
        else:
            return [
                find_1st(existing[:, 0] == n, True, cmp_equal) for n in new
            ]
    else:
        return []


def _tuple_or_int(x):
    # Convert tuple to int if it only contains one value
    if len(x) == 1:
        return x[0]
    else:
        return x
