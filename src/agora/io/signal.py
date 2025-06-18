"""Signal gets data from an h5 file as a data frame."""

import logging
import typing as t
from copy import copy
from functools import cached_property, lru_cache
from pathlib import Path

import aliby.global_settings as global_settings
import h5py
import numpy as np
import pandas as pd
from agora.io.bridge import BridgeH5
from agora.utils.indexing import validate_lineage
from agora.utils.merge import apply_merges
from agora.utils.multiindex_utils import add_index_levels
from tables import HDF5ExtError


class Signal(BridgeH5):
    """
    Fetch data from h5 files for post-processing.

    Signal assumes that the metadata and data are accessible to
    perform time-adjustments and apply previously recorded
    post-processes.
    """

    def __init__(self, file: t.Union[str, Path]):
        """Initialise defining index names for the dataframe."""
        super().__init__(file, flag=None)
        self.index_names = (
            "experiment",
            "position",
            "trap",
            "cell_label",
            "mother_label",
        )
        self.candidate_channels = global_settings.possible_imaging_channels

    def get(
        self,
        dset: t.Union[str, t.Collection],
        tmax_in_mins: int = None,
    ):
        """Get Signal and apply merging and picking."""
        if isinstance(dset, str):
            record = self.get_raw(dset, tmax_in_mins=tmax_in_mins)
            if record is not None:
                picked_merged = self.apply_merging_picking(record)
                return self.add_name(picked_merged, dset)
        elif isinstance(dset, list):
            return [self.get(d) for d in dset]
        else:
            raise TypeError("Error in Signal.get.")

    @staticmethod
    def add_name(df, name):
        """Add name of the Signal as an attribute to its data frame."""
        df.name = name
        return df

    def cols_in_mins(self, df: pd.DataFrame):
        """Convert numerical columns in a data frame to minutes."""
        df.columns = (df.columns * np.round(self.tinterval / 60)).astype(int)
        if df.columns.max() > 1000:
            breakpoint()
        return df

    @cached_property
    def ntimepoints(self):
        """Find the number of time points for one position, or one h5 file."""
        with h5py.File(self.filename, "r") as f:
            return f["extraction/general/null/area/timepoint"][-1] + 1

    @cached_property
    def tinterval(self) -> int:
        """Find the interval between time points in seconds."""
        tinterval_location = "time_settings/timeinterval"
        with h5py.File(self.filename, "r") as f:
            if tinterval_location in f.attrs:
                res = f.attrs[tinterval_location]
                if isinstance(res, list):
                    tint = res[0]
                else:
                    tint = res
                return tint
            else:
                logging.getLogger("aliby").warn(
                    f"{str(self.filename).split('/')[-1]}: using default time "
                    f"interval of {global_settings.default_time_interval} seconds."
                )
                return global_settings.default_time_interval

    def retained(self, signal, cutoff: float = 0, tmax_in_mins: int = None):
        """Get retained cells for a Signal or list of Signals."""
        # get data frame
        if isinstance(signal, str):
            signal = self.get(signal, tmax_in_mins=tmax_in_mins)
        elif isinstance(signal, list):
            signal = [self.get(s) for s in signal]
        # apply cutoff
        if isinstance(signal, pd.DataFrame):
            return self.apply_cutoff(signal, cutoff)
        elif isinstance(signal, list):
            return [self.apply_cutoff(s, cutoff) for s in signal]

    @staticmethod
    def apply_cutoff(df, cutoff):
        """
        Return sub data frame with retained cells.

        Cells must be present for at least cutoff fraction of the total number
        of time points.
        """
        return df.loc[df.notna().sum(axis=1) > df.shape[1] * cutoff]

    @property
    def channels(self) -> t.Collection[str]:
        """Get channels as an array of strings."""
        with h5py.File(self.filename, "r") as f:
            if "channels" in f.attrs:
                return list(f.attrs["channels"])
            elif "channels/channel" in f.attrs:
                # old defunct h5 format
                return list(f.attrs["channels/channel"])
            else:
                raise KeyError(f"Channels missing in the {self.filename}.")

    @lru_cache(2)
    def lineage(
        self, lineage_location: t.Optional[str] = None, merged: bool = False
    ) -> np.ndarray:
        """
        Get lineage data from the h5 file.

        Return an array with three columns: the tile id, the mother label,
        and the daughter label.
        """
        if lineage_location is None:
            lineage_location = "modifiers/lineage_merged"
        with h5py.File(self.filename, "r") as f:
            if lineage_location not in f:
                lineage_location = "postprocessing/lineage"
            if lineage_location not in f:
                raise KeyError(
                    f"Neither modifiers nor postprocessing in {self.filename}"
                )
            else:
                traps_mothers_daughters = f[lineage_location]
            if isinstance(traps_mothers_daughters, h5py.Dataset):
                lineage = traps_mothers_daughters[()]
            else:
                lineage = np.array(
                    (
                        traps_mothers_daughters["trap"],
                        traps_mothers_daughters["mother_label"],
                        traps_mothers_daughters["daughter_label"],
                    )
                ).T
        return lineage

    def apply_merging_picking(
        self,
        data: t.Union[str, pd.DataFrame],
        merges: t.Union[np.ndarray, bool] = True,
        picks: t.Union[t.Collection, bool] = True,
    ):
        """
        Apply merging and picking to a Signal data frame.

        Parameters
        ----------
        data : t.Union[str, pd.DataFrame]
            A data frame or a path to one.
        merges : t.Union[np.ndarray, bool]
            (optional) An array of pairs of (trap, cell) indices to merge.
            If True, fetch merges from file.
        picks : t.Union[np.ndarray, bool]
            (optional) An array of (trap, cell) indices.
            If True, fetch picks from file.
        """
        if isinstance(merges, bool):
            merges = self.read_merges() if merges else np.array([])
        if merges.any():
            merged = apply_merges(data, merges)
        else:
            merged = copy(data)
        if isinstance(picks, bool):
            if picks is True:
                picks = self.read_picks()
            else:
                return merged
        if len(picks):
            picked_indices = list(
                picks.intersection([tuple(x) for x in merged.index])
            )
            return merged.loc[picked_indices]
        return merged

    @cached_property
    def print_available(self):
        """Print data sets available in h5 file."""
        if not hasattr(self, "_available"):
            self._available = []
            with h5py.File(self.filename, "r") as f:
                f.visititems(self.store_signal_path)
        for sig in self._available:
            print(sig)

    @cached_property
    def available(self):
        """Get data sets available in h5 file."""
        try:
            if not hasattr(self, "_available"):
                self._available = []
            with h5py.File(self.filename, "r") as f:
                f.visititems(self.store_signal_path)
        except KeyError as e:
            self.log(f"Exception when visiting h5: {e}")
        return self._available

    def get_merged(self, dataset):
        """Run merging."""
        return self.apply_merging_picking(dataset, picks=False)

    @cached_property
    def merges(self) -> np.ndarray:
        """Get merges."""
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_merges)
        return dsets

    @cached_property
    def n_merges(self):
        """Get number of merges."""
        return len(self.merges)

    @cached_property
    def picks(self) -> np.ndarray:
        """Get picks."""
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_picks)
        return dsets

    def get_raw(
        self,
        dataset: str or t.List[str],
        in_minutes: bool = True,
        lineage: bool = False,
        tmax_in_mins: int = None,
        **kwargs,
    ) -> pd.DataFrame or t.List[pd.DataFrame]:
        """
        Get raw Signal without merging, picking, and lineage information.

        Parameters
        ----------
        dataset: str or list of strs
            The name of the h5 file or a list of h5 file names.
        in_minutes: boolean
            If True, convert column headings to times in minutes.
        lineage: boolean
            If True, add mother_label to index.
        run_lineage_check: boolean
            If True, raise exception if a likely error in the lineage assignment.
        tmax_in_mins: int (optional)
            Discard data for times > tmax_in_mins. Cells with all NaNs will also
            be discarded to help with assigning lineages.
            Setting tmax_in_mins is a way to ignore parts of the experiment with
            incorrect lineages generated by clogging.
        """
        if isinstance(dataset, str):
            # load from h5 file
            df = self.dataset_to_df(self.filename, dataset)
            if df is not None:
                df = df.sort_index()
                if in_minutes:
                    df = self.cols_in_mins(df)
                # limit data by time and discard NaNs
                if (
                    in_minutes
                    and tmax_in_mins
                    and isinstance(tmax_in_mins, int)
                ):
                    df = df[df.columns[df.columns <= tmax_in_mins]]
                    df = df.dropna(how="all")
                # add mother label to data frame
                if lineage:
                    if "mother_label" in df.index.names:
                        df = df.droplevel("mother_label")
                    mother_label = np.zeros(len(df), dtype=int)
                    lineage = self.lineage()
                    (
                        valid_lineage,
                        valid_indices,
                        lineage,
                    ) = validate_lineage(
                        lineage,
                        indices=np.array(df.index.to_list()),
                        how="daughters",
                    )
                    mother_label[valid_indices] = lineage[valid_lineage, 1]
                    df = add_index_levels(df, {"mother_label": mother_label})
                return df
        elif isinstance(dataset, list):
            return [
                self.get_raw(
                    dset,
                    in_minutes=in_minutes,
                    lineage=lineage,
                    tmax_in_mins=tmax_in_mins,
                )
                for dset in dataset
            ]

    def read_merges(self):
        """Read merges from the h5 file."""
        path = "modifiers/merges"
        with h5py.File(self.filename, "r") as f:
            if path in f:
                merges = f[path][:]
            else:
                merges = np.array([])
        return merges

    def read_picks(self) -> t.Set[t.Tuple[int, str]]:
        """Read picks from the h5 file."""
        path = "modifiers/picks"
        with h5py.File(self.filename, "r") as f:
            if path in f:
                try:
                    picks = f[path][:]
                    picks = set(map(tuple, picks))
                except TypeError:
                    # old h5 file
                    picks = set(
                        zip(
                            *[
                                f[path + "/" + name]
                                for name in ("trap", "cell_label")
                                if name in f[path]
                            ]
                        )
                    )
            else:
                picks = set()
        return picks

    def dataset_to_df(self, f: Path, dataset: str):
        """Get data from h5 file as a data frame."""
        if isinstance(f, str):
            f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"Cannot find {str(f)}.")
        df = None
        try:
            with pd.HDFStore(f, mode="r") as store:
                df = store[dataset]
                # convert to aliby multi-index format
                df = df.pivot(
                    columns="time",
                    index=["trap", "cell_label"],
                    values="value",
                )
        except (HDF5ExtError, KeyError):
            # old h5 file before writer changed
            dataset = (
                dataset.replace("null", "None")
                if "null" in dataset
                else dataset
            )
            with h5py.File(f, "r") as file:
                dset = file[dataset]
                values, index, columns = [], [], []
                index_names = copy(self.index_names)
                valid_names = [
                    lbl for lbl in index_names if lbl in dset.keys()
                ]
                if valid_names:
                    index = pd.MultiIndex.from_arrays(
                        [dset[lbl] for lbl in valid_names], names=valid_names
                    )
                    columns = dset.attrs.get("columns", None)
                    if "timepoint" in dset:
                        columns = file[dataset + "/timepoint"][()]
                    values = file[dataset + "/values"][()]
                df = pd.DataFrame(values, index=index, columns=columns)
        return df

    @property
    def stem(self):
        """Get name of h5 file."""
        return self.filename.stem

    def store_signal_path(
        self,
        name: str,
        node: t.Union[h5py.Dataset, h5py.Group],
    ):
        """Store the name of all signals if leaf nodes."""
        if isinstance(node, h5py.Group) and np.all(
            [isinstance(x, h5py.Dataset) for x in node.values()]
        ):
            if name.startswith("extraction") or name.startswith(
                "postprocessing"
            ):
                if "_i_table" in name:
                    # remove part of path added by pytables
                    self._available.append(name.split("_i_table")[0][:-1])
                else:
                    self._available.append(name)

    @staticmethod
    def _if_merges(name: str, obj):
        if isinstance(obj, h5py.Dataset) and name.startswith(
            "modifiers/merges"
        ):
            return obj[()]

    @staticmethod
    def _if_picks(name: str, obj):
        if isinstance(obj, h5py.Group) and name.endswith("picks"):
            return obj[()]

    @property
    def ntps(self) -> int:
        """Get number of time points from the metadata."""
        return self.meta_h5["time_settings/ntimepoints"][0]
