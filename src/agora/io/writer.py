import logging
from collections.abc import Iterable
from pathlib import Path
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
    """
    Load the metadata from an h5 file and convert to a dictionary, including the "parameters" field which is stored as YAML.

    Parameters
    ----------
    file: str
        Name of the h5 file
    group: str, optional
        The group in the h5 file from which to read the data
    """
    # load the metadata, stored as attributes, from the h5 file and return as a dictionary
    with h5py.File(file, "r") as f:
        meta = dict(f[group].attrs.items())
    if "parameters" in meta:
        # convert from yaml format into dict
        meta["parameters"] = yaml.safe_load(meta["parameters"])
    return meta


class DynamicWriter:
    """Provides a parent class for all writers."""

    # a dict giving for each dataset a tuple, comprising the dataset's maximum size, as a 2D tuple, and its type
    data_types = {}
    # the group in the h5 file to write to
    group = ""
    # compression info
    compression = "gzip"
    compression_opts = 9
    metadata = None

    def __init__(self, file: str):
        self.file = file
        # the metadata is stored as attributes in the h5 file
        if Path(file).exists():
            self.metadata = load_attributes(file)

    def _log(self, message: str, level: str = "warn"):
        # Log messages in the corresponding level
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    def _append(self, data, key, hgroup):
        """
        Append data to existing dataset in the h5 file otherwise create a new one.

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
        Delete and then replace existing dataset in h5 file.

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
            key,
            shape=data_shape,
            dtype=dtype,
            compression=self.compression,
        )
        # write all data, signified by the empty tuple
        hgroup[key][()] = data

    def write(self, data: dict, overwrite: list, meta: dict = {}):
        """
        Write data and metadata to h5 file.

         Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        meta: dict, optional
            Metadata to be written as attributes of the h5 file
        """
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
                        self._log(
                            f"{key}:{value} could not be written: {e}", "error"
                        )
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


##################### Special instances #####################
class TilerWriter(DynamicWriter):
    """Write data stored in a Tiler instance to h5 files."""

    datatypes = {
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        "attrs/tile_size": ((1,), np.uint16),
        "attrs/max_size": ((1,), np.uint16),
    }
    group = "trap_info"

    def write(self, data: dict, overwrite: list, tp: int, meta: dict = {}):
        """
        Write data for time points that have none.

        Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        tp: int
            The time point of interest
        meta: dict, optional
            Metadata to be written as attributes of the h5 file
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


class LinearBabyWriter(DynamicWriter):
    """
    Write data stored in a Baby instance to h5 files.

    Assumes the edgemasks are of form ((None, tile_size, tile_size), bool).
    """

    # TODO make this YAML: Alan: why?
    compression = "gzip"
    _default_tile_size = 117
    datatypes = {
        "centres": ((None, 2), np.uint16),
        "position": ((None,), np.uint16),
        "angles": ((None,), h5py.vlen_dtype(np.float32)),
        "radii": ((None,), h5py.vlen_dtype(np.float32)),
        "edgemasks": ((None, _default_tile_size, _default_tile_size), bool),
        "ellipse_dims": ((None, 2), np.float32),
        "cell_label": ((None,), np.uint16),
        "trap": ((None,), np.uint16),
        "timepoint": ((None,), np.uint16),
        "mother_assign_dynamic": ((None,), np.uint16),
        "volumes": ((None,), np.float32),
    }
    group = "cell_info"

    def write(
        self, data: dict, overwrite: list, tp: int = None, meta: dict = {}
    ):
        """
        Check data does not exist before writing.

        Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        tp: int
            The time point of interest
        meta: dict, optional
            Metadata to be written as attributes of the h5 file
        """
        with h5py.File(self.file, "a") as store:
            hgroup = store.require_group(self.group)
            available_tps = hgroup.get("timepoint", None)
            # write data
            if not available_tps or tp not in np.unique(available_tps[()]):
                super().write(data, overwrite)
            else:
                # data already exists
                print(f"BabyWriter: Skipping tp {tp}")
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


class StateWriter(DynamicWriter):
    """
    Write information summarising the current state of the pipeline - the 'last_state' dataset in the h5 file.

    MOVEDatatypes are specified with the first variable specifying the number of traps and the other specifying the shape of the object.

    """

    datatypes = {
        # the highest cell label assigned for each time point
        "max_lbl": ((None, 1), np.uint16),
        # how far back we go for tracking
        "tp_back": ((None, 1), np.uint16),
        # trap labels
        "trap": ((None, 1), np.int16),
        # cell labels
        "cell_lbls": ((None, 1), np.uint16),
        # previous cell features for tracking
        "prev_feats": ((None, None), np.float32),
        # number of images for which a cell has been present
        "lifetime": ((None, 2), np.uint16),
        # probability of a mother-bud relationship given a bud
        "p_was_bud": ((None, 2), np.float32),
        # probability of a mother-bud relationship given a mother
        "p_is_mother": ((None, 2), np.float32),
        # cumulative matrix, over time, of bud assignments
        "ba_cum": ((None, None), np.float32),
    }
    group = "last_state"
    compression = "gzip"

    @staticmethod
    def format_field(states: list, field: str):
        """Flatten a field in the states list to save as an h5 dataset."""
        fields = [pos_state[field] for pos_state in states]
        return fields

    @staticmethod
    def format_values_tpback(states: list, val_name: str):
        """Unpacks a dict of state data into tp_back, trap, value."""
        # initialise as empty lists
        # Alan: is this initialisation necessary?
        tp_back, trap, value = [
            [[] for _ in states[0][val_name]] for _ in range(3)
        ]
        # store results as a list of tuples
        lbl_tuples = [
            (tp_back, trap, cell_label)
            for trap, state in enumerate(states)
            for tp_back, value in enumerate(state[val_name])
            for cell_label in value
        ]
        # unpack list of tuples to define variables
        if len(lbl_tuples):
            tp_back, trap, value = zip(*lbl_tuples)
        return tp_back, trap, value

    @staticmethod
    def format_values_traps(states: list, val_name: str):
        """Format either lifetime, p_was_bud, or p_is_mother variables as a list."""
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
        """Pad a 2D array with zeros for large indices."""
        padded = np.zeros((pad_size, pad_size)).astype(float)
        length = len(array)
        padded[:length, :length] = array
        return padded

    def format_states(self, states: list):
        """Re-format state data into a dict of lists, with one element per per list per state."""
        formatted_state = {"max_lbl": [state["max_lbl"] for state in states]}
        tp_back, trap, cell_label = self.format_values_tpback(
            states, "cell_lbls"
        )
        _, _, prev_feats = self.format_values_tpback(states, "prev_feats")
        # store lists in a dict
        formatted_state["tp_back"] = tp_back
        formatted_state["trap"] = trap
        formatted_state["cell_lbls"] = cell_label
        formatted_state["prev_feats"] = np.array(prev_feats)
        # one entry per cell label - tp_back independent
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

    def write(self, data: dict, overwrite: list, tp: int = 0):
        """Write the current state of the pipeline."""
        if len(data):
            last_tp = 0
            try:
                with h5py.File(self.file, "r") as f:
                    gr = f.get(self.group, None)
                    if gr:
                        last_tp = gr.attrs.get("tp", 0)
                if tp == 0 or tp > last_tp:
                    # write
                    formatted_data = self.format_states(data)
                    super().write(data=formatted_data, overwrite=overwrite)
                    with h5py.File(self.file, "a") as f:
                        # record that data for the timepoint has been written
                        f[self.group].attrs["tp"] = tp
                elif tp > 0 and tp <= last_tp:
                    # data already present
                    print(f"StateWriter: Skipping timepoint {tp}")
            except Exception as e:
                raise (e)
        else:
            print("Skipping overwriting: no data")


#################### Extraction version ###############################
class Writer(BridgeH5):
    """Class to transform data into compatible structures."""

    # Alan: when is this used?

    def __init__(self, filename, flag=None, compression="gzip"):
        """
        Initialise write.

        Parameters
        ----------
        filename: str
            Name of file to write into
        flag: str, default=None
            Flag to pass to the default file reader. If None the file remains closed.
        compression: str, default="gzip"
            Compression method passed on to h5py writing functions (only used for dataframes and other array-like data).
        """
        super().__init__(filename, flag=flag)
        self.compression = compression

    def write(
        self,
        path: str,
        data: Iterable = None,
        meta: dict = {},
        overwrite: str = None,
    ):
        """
        Write data and metadata to a particular path in the h5 file.

        Parameters
        ----------
        path : str
            Path inside h5 file into which to write.
        data : Iterable, optional
        meta : dict, optional
        overwrite: str, optional
        """
        self.id_cache = {}
        with h5py.File(self.filename, "a") as f:
            # Alan, haven't we already opened the h5 file through BridgeH5's init?
            if overwrite == "overwrite":  # TODO refactor overwriting
                if path in f:
                    del f[path]
            logging.debug(
                "{} {} to {} and {} metadata fields".format(
                    overwrite, type(data), path, len(meta)
                )
            )
            # write data
            if data is not None:
                self.write_dset(f, path, data)
            # write metadata
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, data=metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        """Write data in different ways depending on its type to an open h5 file."""
        # data is a datafram
        if isinstance(data, pd.DataFrame):
            self.write_pd(f, path, data, compression=self.compression)
        # data is a multi-index dataframe
        elif isinstance(data, pd.MultiIndex):
            # Alan: should we still not compress here?
            self.write_index(f, path, data)  # , compression=self.compression)
        # data is a dictionary of dataframes
        elif isinstance(data, Dict) and np.all(
            [isinstance(x, pd.DataFrame) for x in data.values]
        ):
            for k, df in data.items():
                self.write_dset(f, path + f"/{k}", df)
        # data is an iterable
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        # data is a float or integer
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, attr: str, data: Iterable):
        """Write metadata to an open h5 file."""
        obj = f.require_group(path)
        obj.attrs[attr] = data

    @staticmethod
    def write_arraylike(f: h5py.File, path: str, data: Iterable, **kwargs):
        """Write an iterable."""
        if path in f:
            del f[path]
        narray = np.array(data)
        if narray.any():
            chunks = (1, *narray.shape[1:])
        else:
            chunks = None
        # create dset
        dset = f.create_dataset(
            path,
            shape=narray.shape,
            chunks=chunks,
            dtype="int",
            compression=kwargs.get("compression", None),
        )
        # add data to dset
        dset[()] = narray

    @staticmethod
    def write_index(f, path, pd_index, **kwargs):
        """Write a multi-index dataframe."""
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
        """Write a dataframe."""
        values_path = (
            path + "values" if path.endswith("/") else path + "/values"
        )
        if path not in f:

            # create dataset and write data
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

            # create dateset and write indices
            if not len(df):  # Only write more if not empty
                return None

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
            # create dataset and write columns
            if (
                df.columns.dtype == int
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

            # path exists
            dset = f[values_path]

            # filter out repeated timepoints
            new_tps = set(df.columns)
            if path + "/timepoint" in f:
                new_tps = new_tps.difference(f[path + "/timepoint"][()])
            df = df[new_tps]

            if (
                not hasattr(self, "id_cache")
                or df.index.nlevels not in self.id_cache
            ):
                # use cache dict to store previously obtained indices
                self.id_cache[df.index.nlevels] = {}
                existing_ids = self.get_existing_ids(
                    f, [path + "/" + x for x in df.index.names]
                )
                # split indices in existing and additional
                new = df.index.tolist()
                if df.index.nlevels == 1:
                    # cover cases with a single index
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

                # sort indices for h5 indexing
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

            # add found cells
            dset.resize(dset.shape[1] + df.shape[1], axis=1)
            dset[:, ntps:] = np.nan

            # TODO refactor this indices sorting. Could be simpler
            found_indices_sorted = self.id_cache[df.index.nlevels][
                "found_indices"
            ]

            # case when all labels are new
            if found_indices_sorted.any():
                # h5py does not allow bidimensional indexing,
                # so we have to iterate over the columns
                for i, tp in enumerate(df.columns):
                    dset[found_indices_sorted, tp] = existing_values[:, i]
            # add new cells
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
        """Fetch indices and convert them to a (nentries, nlevels) ndarray."""
        return np.array([f[path][()] for path in paths]).T

    @staticmethod
    def find_ids(existing, new):
        """Compare two tuple sets and return the intersection and difference (elements in the 'new' set not in 'existing')."""
        set_existing = set([tuple(*x) for x in zip(existing.tolist())])
        existing_cells = np.array(list(set_existing.intersection(new)))
        new_cells = np.array(list(set(new).difference(set_existing)))
        return existing_cells, new_cells


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
    """Convert tuple to int if it only contains one value."""
    if len(x) == 1:
        return x[0]
    else:
        return x
