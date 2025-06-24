"""Writers for pipeline steps."""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def write_meta_to_h5(
    file: str | Path, meta: dict[str, str | int | list[int] | list[str]]
):
    """Write minimal metadata to the root of an h5 file."""
    with h5py.File(file, "a") as f:
        root = f
        for att, data in meta.items():
            try:
                if isinstance(data, str):
                    root.attrs[att] = data.encode("utf-8")
                elif isinstance(data, (int, float, bool)):
                    root.attrs[att] = data
                elif isinstance(data, (list, tuple)):
                    if data and isinstance(data[0], str):
                        # create variable-length string array
                        root.attrs[att] = [s.encode("utf-8") for s in data]
                    else:
                        root.attrs[att] = np.array(data)
                elif isinstance(data, np.ndarray):
                    root.attrs[att] = data
                else:
                    # fallback to string representation
                    root.attrs[att] = str(data).encode("utf-8")
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not store attribute '{att}': {e}")


def read_meta_from_h5(file: str | Path) -> dict:
    """
    Read the minimal metadata from an h5 file as a dict.

    Parameters
    ----------
    file: str
        Name of the h5 file
    """
    with h5py.File(file, "r") as f:
        meta = dict(f.attrs.items())
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
    df_compression = "zlib"
    compression_opts = 9

    def __init__(self, file: str):
        """Initialise by reading metadata."""
        self.file = file
        # load metadata from the h5 file
        if Path(file).exists():
            self.metadata = read_meta_from_h5(file)

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
        except ValueError as e:
            logging.debug(f"Writer: Attributes have no length: {e}")
            n = 1
        if key in hgroup:
            # append to existing dataset
            try:
                dset = hgroup[key]
                dset.resize(dset.shape[0] + n, axis=0)
                dset[-n:] = data
            except ValueError as e:
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

    def overwrite(self, data: NDArray, key: str, hgroup: str):
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

    def write(self, data: dict[str, NDArray], overwrite: list[str]):
        """
        Write data to h5 file.

        Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
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
                            # global parameters
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

    def add_df(
        self, dataset: str, df: pd.DataFrame, overwrite: bool = False
    ) -> None:
        """Add data frame to h5 file."""
        if df.empty:
            return
        # convert to tidy structure
        multi_index_names = df.index.names
        df.reset_index(inplace=True)
        df_flat = df.melt(
            id_vars=multi_index_names, var_name="time", value_name="value"
        )
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
            complib=self.df_compression,
            complevel=self.compression_opts,
        ) as store:
            if dataset in store:
                if overwrite:
                    store.put(dataset, df_flat, format="table")
                else:
                    store.append(dataset, df_flat, format="table")
            else:
                store.put(dataset, df_flat, format="table")


class TilerWriter(CoreWriter):
    """Write data from Tiler for one time point."""

    datatypes = {
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        "attrs/tile_size": ((1,), np.uint16),
        "attrs/max_size": ((1,), np.uint16),
    }
    group = "trap_info"

    def write(self, data: dict, overwrite: list[str], tp: int = None):
        """
        Write data for one position and one time point.

        Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        tp: int
            The time point of interest
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
                super().write(data=data, overwrite=overwrite)


class BabyWriter(CoreWriter):
    """Write output from Baby at each time point."""

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
        overwrite: list[str],
        tp: int | None = None,
        tile_size: int | None = None,
    ):
        """
        Write data for one time point and one position.

        Parameters
        ----------
        data: dict
            A dict of datasets and data
        overwrite: list of str
            A list of datasets to overwrite
        tp: int
            The time point of interest
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


class ExtractorWriter(CoreWriter):
    """Write data frames generated by Extractor at each time point."""

    def write(self, data: dict[str, pd.DataFrame]):
        """Write the extracted data for one position."""
        for extract_name, df in data.items():
            dset_path = "/extraction/" + extract_name
            self.add_df(dataset=dset_path, df=df)


class PostProcessorWriter(CoreWriter):
    """Write or overwrite mixed data generated by PostProcessor."""

    datatypes = {
        "merges": ((None, 2), np.uint16),
        "lineage_merged": ((None, 3), np.uint16),
        "picks": ((None, 2), np.uint16),
    }
    group = "modifiers"

    def write(self, data: dict[str, np.ndarray | pd.DataFrame]):
        """Write postprocessed data for whole timelapse overwriting always."""
        merge_pick_dict = {
            key: value for key, value in data.items() if key in self.datatypes
        }
        # write picking and merging
        super().write(
            data=merge_pick_dict, overwrite=list(self.datatypes.keys())
        )
        # write bud data
        bud_data_dict = {
            key: value
            for key, value in data.items()
            if key not in self.datatypes
        }
        for outpath, df in bud_data_dict.items():
            if isinstance(df, pd.DataFrame):
                self.add_df(dataset=outpath, df=df, overwrite=True)
