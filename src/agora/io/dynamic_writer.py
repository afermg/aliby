"""Writers used for Tiler, Baby, and BabyState."""

import logging
from pathlib import Path

import h5py
import numpy as np
import yaml


def load_meta(file: str, group="/"):
    """
    Load the metadata from an h5 file.

    Convert to a dictionary, including the "parameters" field
    which is stored as YAML.

    Parameters
    ----------
    file: str
        Name of the h5 file
    group: str, optional
        The group in the h5 file from which to read the data
    """
    # load the metadata, stored as attributes, from the h5 file
    with h5py.File(file, "r") as f:
        # return as a dict
        meta = dict(f[group].attrs.items())
    if "parameters" in meta:
        # convert from yaml format into dict
        meta["parameters"] = yaml.safe_load(meta["parameters"])
    return meta


class DynamicWriter:
    """Provide a parent class for all writers."""

    # a dict giving for each dataset a tuple, comprising the
    # dataset's maximum size, as a 2D tuple, and its type
    data_types = {}
    # the group in the h5 file to write to
    group = ""
    # compression info
    compression = "gzip"
    compression_opts = 9
    metadata = None

    def __init__(self, file: str):
        """Define metadata."""
        self.file = file
        # load metadata from the h5 file
        if Path(file).exists():
            self.metadata = load_meta(file)

    def log(self, message: str, level: str = "warn"):
        """Log message."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    def _append(self, data, key, hgroup):
        """
        Append data to dataset in the h5 file or create a new one.

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
                dset = hgroup[key]
                dset.resize(dset.shape[0] + n, axis=0)
                dset[-n:] = data
            except Exception as e:
                logging.debug(
                    "DynamicWriter: Inconsistency between dataset shape and "
                    f"new empty data: {e}."
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
                compression_opts=(
                    self.compression_opts
                    if self.compression is not None
                    else None
                ),
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
                    raise KeyError(f"No defined data type for key {key}.")
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
                        self.log(
                            f"{key}:{value} could not be written: {e}.",
                            "error",
                        )
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


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
            # find xy drift for each time point to check if already processed
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

    Assume edgemasks of form ((None, tile_size, tile_size), bool).
    """

    compression = "gzip"
    datatypes = {
        "centres": ((None, 2), np.uint16),
        "position": ((None,), np.uint16),
        "angles": ((None,), h5py.vlen_dtype(np.float32)),
        "radii": ((None,), h5py.vlen_dtype(np.float32)),
        "ellipse_dims": ((None, 2), np.float32),
        "cell_label": ((None,), np.uint16),
        "trap": ((None,), np.uint16),
        "timepoint": ((None,), np.uint16),
        "mother_assign_dynamic": ((None,), np.uint16),
        "volumes": ((None,), np.float32),
    }
    group = "cell_info"

    def write(
        self,
        data: dict,
        overwrite: list,
        tp: int = None,
        tile_size: int = None,
        meta: dict = {},
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
        self.datatypes["edgemasks"] = ((None, tile_size, tile_size), bool)
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
    Write information summarising the current state of BABY.

    Write to the 'last_state' dataset in the h5 file.

    MOVEDatatypes are specified with the first variable specifying
    number of traps and the other specifying the shape of the object.

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
        """Unpack a dict of state data into tp_back, trap, and value."""
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
        else:
            # set as empty lists
            tp_back, trap, value = [
                [[] for _ in states[0][val_name]] for _ in range(3)
            ]
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
        """
        Re-format state data into a dict of lists.

        Use one element per per list per state.
        """
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
