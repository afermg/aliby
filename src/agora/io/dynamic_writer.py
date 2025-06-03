"""Writers for pipeline steps."""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def load_meta(file: str | Path, group: str = "/") -> dict:
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


class CoreWriter:
    """Provide a parent class for all writers."""

    # a dict giving for each dataset a tuple, comprising the
    # dataset's maximum size, as a 2D tuple, and its type
    datatypes: dict = {}
    # the group to write to in the h5 file
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

    def append(self, data, key, hgroup):
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
            logging.debug(f"Writer: Attributes have no length: {e}")
            n = 1
        if key in hgroup:
            # append to existing dataset
            try:
                dset = hgroup[key]
                dset.resize(dset.shape[0] + n, axis=0)
                dset[-n:] = data
            except Exception as e:
                logging.debug(
                    "Writer: Inconsistency between dataset shape and "
                    f"new empty data: {e}."
                )
        else:
            # create new dataset
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

    def overwrite(self, data, key, hgroup):
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
        data_shape = np.shape(data)
        _, dtype = self.datatypes[key]
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
                            self.overwrite(value, key, hgroup)
                        else:
                            # append or create new dataset
                            self.append(value, key, hgroup)
                    except Exception as e:
                        self.log(
                            f"{key}:{value} could not be written: {e}.",
                            "error",
                        )
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


class TilerWriter(CoreWriter):
    """Write data stored in a Tiler instance to h5 files."""

    datatypes = {
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        "attrs/tile_size": ((1,), np.uint16),
        "attrs/max_size": ((1,), np.uint16),
    }
    group = "trap_info"

    def write(
        self, data: dict, overwrite: list, meta: dict = {}, tp: int = None
    ):
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
        # append to h5 file
        with h5py.File(self.file, "a") as store:
            # open group, creating if necessary
            hgroup = store.require_group(self.group)
            # find xy drift for each time point to check if already processed
            nprev = hgroup.get("drifts", None)
            if nprev and tp < nprev.shape[0]:
                # data already exists
                print(f"Tiler: Skipping timepoint {tp}")
            else:
                super().write(data=data, overwrite=overwrite, meta=meta)


class BabyWriter(CoreWriter):
    """
    Write data stored in a Baby instance to h5 files.

    Assume edgemasks of form ((None, tile_size, tile_size), bool).
    """

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
                super().write(data=data, overwrite=overwrite)
            else:
                # data already exists
                print(f"BabyWriter: Skipping tp {tp}")
            # write metadata
            for key, value in meta.items():
                hgroup.attrs[key] = value


class ExtractorWriter(CoreWriter):
    """Write data frames generated by Extractor to h5 files."""

    compression = "zlib"

    def write(self, extract_dict: dict[str, pd.DataFrame]):
        """Save the extracted data for one position to the h5 file."""
        for extract_name, df in extract_dict.items():
            dset_path = "/extraction/" + extract_name
            self.add_df(dataset=dset_path, df=df)

    def add_df(self, dataset: str, df: pd.DataFrame) -> None:
        """Add data frame to h5 file."""
        if df.empty:
            return
        # convert to simple flat structure
        tp = df.columns.to_list()[0]
        df_flat = df.reset_index()
        df_flat = df_flat.rename(columns={tp: "value"})
        df_flat["time"] = tp
        # first time point has no cell labels
        if "cell_label" not in df_flat.columns:
            df_flat["cell_label"] = -1
        # convert to fixed data types for h5 file
        df_flat = df_flat.astype(
            {
                "trap": np.int16,
                "cell_label": np.int16,
                "time": np.int32,
                "value": np.float32,
            }
        )
        # store
        mode = "a" if Path(self.file).exists() else "w"
        with pd.HDFStore(
            self.file,
            mode=mode,
            complib=self.compression,
            complevel=self.compression_opts,
        ) as store:
            if dataset in store:
                store.append(dataset, df_flat, format="table")
            else:
                store.put(dataset, df_flat, format="table")
