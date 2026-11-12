"""Writers for pipeline steps."""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sooth import tile_shape

# location in the h5 file of the mask of the PDMS trap within a tile;
# in trap_info, with the other data that is per position rather than
# per cell, and not in extraction, which holds only signals
pdms_mask_path = "trap_info/pdms_mask"


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
                logging.getLogger("aliby").warning(
                    f"Could not store attribute '{att}': {e}"
                )


def tidy(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Return a Signal's data as the columns of a long table.

    The table has a row for each cell at each time point, the first time
    point's cells first, with the levels of the index, the time and the
    value as its columns, in the data types of the h5 file. It is what
    resetting the index and melting gives, which was a quarter of the
    time taken to write a time point's signals.

    Parameters
    ----------
    df: pd.DataFrame
        The data, with time points as columns and trap, and for cells
        cell_label, as the index.
    """
    values = df.to_numpy(dtype=np.float32)
    no_rows, no_times = values.shape
    columns = {}
    for level, name in enumerate(df.index.names):
        level_values = np.asarray(df.index.get_level_values(level))
        if name in ("trap", "cell_label"):
            level_values = level_values.astype(np.int16)
        columns[name] = np.tile(level_values, no_times)
    columns["time"] = np.repeat(
        np.asarray(df.columns).astype(np.int32), no_rows
    )
    columns["value"] = values.ravel(order="F")
    if "cell_label" not in columns:
        # a background signal is of a trap, not of a cell
        columns["cell_label"] = np.full(no_rows * no_times, -1, np.int16)
    return columns


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
    # level 9 took four times as long as level 4 to write edge masks,
    # and a quarter longer to write signals, for files 17% and 2% smaller
    compression_opts = 4
    # rows in a chunk of a dataset whose rows are a few numbers: left to
    # h5py, a chunk is the number of cells at the first time point, and
    # a position that begins with three cells took 111 ms to read three
    # lists that take 1.4 ms
    chunk_rows = 1024
    # rows in a chunk of a dataset whose rows are images, the edge masks,
    # with a whole tile in each chunk. With h5py's chunks, of part of a
    # tile, a position beginning with three cells had a file of masks 2.5
    # times the size; 16 whole tiles are also quicker to write and read
    image_chunk_rows = 16

    def __init__(self, file: str):
        """Initialise with the name of the h5 file."""
        self.file = file

    def log(self, message: str, level: str = "warning"):
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
            if n == 0:
                # a time point with no cells adds nothing
                return
            # append to existing dataset; an error is for write to undo
            dset = hgroup[key]
            dset.resize(dset.shape[0] + n, axis=0)
            dset[-n:] = data
        else:
            # create new dataset
            max_shape, dtype = self.datatypes[key]
            shape = (n,) + max_shape[1:]
            small_rows = int(np.prod(max_shape[1:], dtype=int)) <= 4
            chunk_rows = (
                self.chunk_rows if small_rows else self.image_chunk_rows
            )
            hgroup.create_dataset(
                key,
                shape=shape,
                maxshape=max_shape,
                dtype=dtype,
                chunks=(chunk_rows,) + max_shape[1:],
                compression=self.compression,
                compression_opts=(
                    self.compression_opts
                    if self.compression is not None
                    else None
                ),
            )
            if n:
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
            # only save data with a pre-defined data-type
            for key in data:
                if key not in self.datatypes:
                    raise KeyError(f"No defined data type for key {key}.")
            # the lengths of the datasets added to, to undo a failure
            lengths = {
                key: hgroup[key].shape[0] if key in hgroup else None
                for key in data
                if not key.startswith("attrs/") and key not in overwrite
            }
            # write data
            for key, value in data.items():
                try:
                    if key.startswith("attrs/"):
                        # global parameters
                        hgroup.attrs[key.split("/")[1]] = value
                    elif key in overwrite:
                        # delete and replace existing dataset
                        self.overwrite(value, key, hgroup)
                    else:
                        # append or create new dataset
                        self.append(value, key, hgroup)
                except Exception as e:
                    # leave the datasets as they were: lists of unequal
                    # length, such as trap and cell_label, cannot be read
                    for name, length in lengths.items():
                        if name in hgroup:
                            if length is None:
                                del hgroup[name]
                            else:
                                hgroup[name].resize(length, axis=0)
                    self.log(
                        f"{key}, of shape {np.shape(value)}, could not be"
                        f" written: {e}.",
                        "error",
                    )
                    raise

    def open_store(self) -> pd.HDFStore:
        """Open the h5 file to add data frames to it."""
        mode = "a" if Path(self.file).exists() else "w"
        return pd.HDFStore(
            self.file,
            mode=mode,
            complib=self.df_compression,
            complevel=self.compression_opts,
        )

    def add_df(
        self,
        dataset: str,
        df: pd.DataFrame,
        overwrite: bool = False,
        store: pd.HDFStore | None = None,
    ) -> None:
        """
        Add data frame to h5 file.

        Parameters
        ----------
        dataset: str
            Where in the h5 file to write.
        df: pd.DataFrame
            The data, with time points as columns.
        overwrite: bool
            If True, replace any data already there; if False, add to it.
        store: pd.HDFStore, optional
            The open h5 file, to write several data frames without
            opening it for each.
        """
        if df.empty:
            return
        self.add_columns(dataset, tidy(df), overwrite=overwrite, store=store)

    def add_columns(
        self,
        dataset: str,
        columns: dict[str, np.ndarray],
        overwrite: bool = False,
        store: pd.HDFStore | None = None,
        row_numbers: np.ndarray | None = None,
    ) -> None:
        """
        Add a long table, as tidy returns, to the h5 file.

        Parameters
        ----------
        dataset: str
            Where in the h5 file to write.
        columns: dict
            The columns of the table.
        overwrite: bool
            If True, replace any data already there; if False, add to it.
        store: pd.HDFStore, optional
            The open h5 file, to write several tables without opening it
            for each.
        row_numbers: array, optional
            The table's index, which is stored with it. If None, the rows
            are numbered from zero.
        """
        if store is None:
            with self.open_store() as store:
                self.add_columns(
                    dataset,
                    columns,
                    overwrite=overwrite,
                    store=store,
                    row_numbers=row_numbers,
                )
            return
        df_flat = pd.DataFrame(columns, index=row_numbers)
        # store with no PyTables index: a Signal is read whole, so nothing
        # looks rows up by it, and it took three quarters of the time to
        # write and half the size of the file
        if dataset in store and not overwrite:
            store.append(dataset, df_flat, format="table", index=False)
        else:
            store.put(dataset, df_flat, format="table", index=False)


class TilerWriter(CoreWriter):
    """Write data from Tiler for one time point."""

    datatypes = {
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        # one number for a square tile, two for a rectangle; an attr's
        # shape is h5py's to infer, so this entry only permits the key
        "attrs/tile_size": ((None,), np.uint16),
        # a default, not a measurement: image_size is the field the traps
        # were found in
        "attrs/max_size": ((1,), np.uint16),
        "attrs/image_size": ((2,), np.uint16),
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
                self.log(f"Skipping timepoint {tp}", "info")
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
        tile_size: int | tuple[int, int] | None = None,
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
        tile_size: int or tuple of two ints
            The tile's size, rows first, which is the shape of a stored
            edge mask. One number is a square tile.
        """
        height, width = tile_shape(tile_size)
        self.datatypes["edgemasks"] = ((None, height, width), bool)
        with h5py.File(self.file, "a") as store:
            hgroup = store.require_group(self.group)
            available_tps = hgroup.get("timepoint", None)
            # write data
            if not available_tps or tp not in np.unique(available_tps[()]):
                super().write(data=data, overwrite=overwrite)
            else:
                # data already exists
                self.log(f"Skipping timepoint {tp}", "info")


class ExtractorWriter(CoreWriter):
    """
    Write data frames generated by Extractor at each time point.

    With flush_every over one, keep the time points' signals and write
    them together every flush_every time points. Each append to a table
    costs PyTables about 2 ms whatever its size, which for 100 signals
    was most of the time taken to write a time point. What is kept is
    lost if the process dies, so call flush when the time points end,
    and when an error ends them.
    """

    def __init__(self, file: str, flush_every: int = 1):
        """
        Initialise with the name of the h5 file.

        Parameters
        ----------
        file: str
            The name of the h5 file.
        flush_every: int
            The number of time points to keep before writing them. One,
            the default, writes each time point as it is given.
        """
        super().__init__(file)
        self.flush_every = flush_every
        # the long tables of each signal that are yet to be written
        self.waiting = {}
        self.no_waiting = 0

    def write(self, data: dict[str, pd.DataFrame]):
        """Take the extracted data for one time point."""
        # tidy now, so that what is written is the data as it was given
        for extract_name, df in data.items():
            if not df.empty:
                self.waiting.setdefault(extract_name, []).append(tidy(df))
        self.no_waiting += 1
        if self.no_waiting >= self.flush_every:
            self.flush()

    def flush(self):
        """Write the time points that are waiting to be written."""
        waiting, self.waiting, self.no_waiting = self.waiting, {}, 0
        if not waiting:
            return
        # open the h5 file once for all of the signals
        with self.open_store() as store:
            for extract_name, tables in waiting.items():
                columns = {
                    name: np.concatenate([table[name] for table in tables])
                    for name in tables[0]
                }
                # number each time point's rows from zero, as they are
                # when written one time point at a time, so that the
                # file is the same however many are written together
                row_numbers = np.concatenate(
                    [np.arange(len(table["value"])) for table in tables]
                )
                self.add_columns(
                    "/extraction/" + extract_name,
                    columns,
                    store=store,
                    row_numbers=row_numbers,
                )

    def write_pdms_mask(self, mask: NDArray):
        """
        Write the mask of the PDMS trap within a tile.

        Store the mask both to reuse if the position is run again and
        to allow the masking to be checked by eye. The mask determines
        every background value in the file, and so is kept for the same
        reason as the picker's and merger's choices.

        Write to trap_info, which holds the data that is per position
        rather than per cell, even though the Extractor finds the mask.

        Parameters
        ----------
        mask: array
            A boolean array, shape (Y, X), that is True for the
            pixels of the PDMS trap.
        """
        with h5py.File(self.file, "a") as f:
            if pdms_mask_path in f:
                del f[pdms_mask_path]
            f.create_dataset(
                pdms_mask_path,
                data=np.asarray(mask, dtype=bool),
                compression=self.compression,
            )


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
