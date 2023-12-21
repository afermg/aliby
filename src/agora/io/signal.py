import logging
import typing as t
from copy import copy
from functools import cached_property, lru_cache
from pathlib import Path

import bottleneck as bn
import h5py
import numpy as np
import pandas as pd

import aliby.global_parameters as global_parameters
from agora.io.bridge import BridgeH5
from agora.io.decorators import _first_arg_str_to_raw_df
from agora.utils.indexing import validate_lineage
from agora.utils.kymograph import add_index_levels
from agora.utils.merge import apply_merges


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
        self.candidate_channels = global_parameters.possible_imaging_channels

    def __getitem__(self, dsets: t.Union[str, t.Collection]):
        """Get and potentially pre-process data from h5 file and return as a data frame."""
        if isinstance(dsets, str):
            return self.get(dsets)
        elif isinstance(dsets, list):
            is_bgd = [dset.endswith("imBackground") for dset in dsets]
            # Check we are not comparing tile-indexed and cell-indexed data
            assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                dsets
            ), "Tile data and cell data cannot be mixed."
            return [self.get(dset) for dset in dsets]
        else:
            raise Exception(f"Invalid type {type(dsets)} to get datasets")

    def get(self, dset_name: t.Union[str, t.Collection]):
        """Get Signal after merging and picking."""
        if isinstance(dset_name, str):
            dsets = self.get_raw(dset_name)
            if dsets is not None:
                picked_merged = self.apply_merging_picking(dsets)
                return self.add_name(picked_merged, dset_name)
            else:
                return None
        else:
            raise Exception("Error in Signal.get")

    @staticmethod
    def add_name(df, name):
        """Add name of the Signal as an attribute to its data frame."""
        df.name = name
        return df

    def cols_in_mins(self, df: pd.DataFrame):
        """Convert numerical columns in a data frame to minutes."""
        try:
            df.columns = (df.columns * self.tinterval // 60).astype(int)
        except Exception as e:
            self._log(f"Unable to convert columns to minutes: {e}", "debug")
        return df

    @cached_property
    def ntimepoints(self):
        """Find the number of time points for one position, or one h5 file."""
        with h5py.File(self.filename, "r") as f:
            return f["extraction/general/None/area/timepoint"][-1] + 1

    @cached_property
    def tinterval(self) -> int:
        """Find the interval between time points (seconds)."""
        tinterval_location = "time_settings/timeinterval"
        with h5py.File(self.filename, "r") as f:
            if tinterval_location in f.attrs:
                res = f.attrs[tinterval_location]
                if type(res) is list:
                    return res[0]
                else:
                    return res
            else:
                logging.getLogger("aliby").warn(
                    f"{str(self.filename).split('/')[-1]}: using default time interval of 300 seconds."
                )
                return 300

    def retained(self, signal, cutoff=0):
        """Get retained cells for a Signal or list of Signals."""
        if isinstance(signal, str):
            # get data frame
            signal = self.get(signal)
        if isinstance(signal, pd.DataFrame):
            return self.get_retained(signal, cutoff)
        elif isinstance(signal, list):
            return [self.get_retained(d, cutoff=cutoff) for d in signal]

    @staticmethod
    def get_retained(df, cutoff):
        """
        Return sub data frame with retained cells.

        Cells must be present for at least cutoff fraction of the total number
        of time points.
        """
        return df.loc[bn.nansum(df.notna(), axis=1) > df.shape[1] * cutoff]

    @property
    def channels(self) -> t.Collection[str]:
        """Get channels as an array of strings."""
        with h5py.File(self.filename, "r") as f:
            return list(f.attrs["channels"])

    @lru_cache(2)
    def lineage(
        self, lineage_location: t.Optional[str] = None, merged: bool = False
    ) -> np.ndarray:
        """
        Get lineage data from a given location in the h5 file.

        Returns an array with three columns: the tile id, the mother label,
        and the daughter label.
        """
        if lineage_location is None:
            lineage_location = "modifiers/lineage_merged"
        with h5py.File(self.filename, "r") as f:
            if lineage_location not in f:
                lineage_location = "postprocessing/lineage"
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

    # @_first_arg_str_to_raw_df
    def apply_merging_picking(
        self,
        data: t.Union[str, pd.DataFrame],
        merges: t.Union[np.ndarray, bool] = True,
        picks: t.Union[t.Collection, bool] = True,
    ):
        """
        Apply picking and merging to a Signal data frame.

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
            merges = self.load_merges() if merges else np.array([])
        if merges.any():
            merged = apply_merges(data, merges)
        else:
            merged = copy(data)
        if isinstance(picks, bool):
            if picks is True:
                # load picks from h5
                picks = self.get_picks(
                    names=merged.index.names, path="modifiers/picks/"
                )
            else:
                return merged
        if len(picks):
            picked_indices = set(picks).intersection(
                [tuple(x) for x in merged.index]
            )
            return merged.loc[picked_indices]
        else:
            return merged

    @cached_property
    def p_available(self):
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
        except Exception as e:
            self._log("Exception when visiting h5: {}".format(e), "exception")

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
        run_lineage_check: bool = True,
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
        """
        try:
            if isinstance(dataset, str):
                with h5py.File(self.filename, "r") as f:
                    df = self.dataset_to_df(f, dataset)
                    if df is not None:
                        df = df.sort_index()
                        if in_minutes:
                            df = self.cols_in_mins(df)
                        # add mother label to data frame
                        if lineage:
                            mother_label = np.zeros(len(df), dtype=int)
                            lineage = self.lineage()
                            valid_lineage, valid_indices = validate_lineage(
                                lineage,
                                indices=np.array(df.index.to_list()),
                                how="daughters",
                                run_lineage_check=run_lineage_check,
                            )
                            mother_label[valid_indices] = lineage[
                                valid_lineage, 1
                            ]
                            df = add_index_levels(
                                df, {"mother_label": mother_label}
                            )
                    return df
            elif isinstance(dataset, list):
                return [
                    self.get_raw(dset, in_minutes=in_minutes, lineage=lineage)
                    for dset in dataset
                ]
        except Exception as e:
            message = f"Signal could not obtain data {dataset}: {e}."
            self._log(message)

    def load_merges(self):
        """Get merge events going up to the first level."""
        with h5py.File(self.filename, "r") as f:
            merges = f.get("modifiers/merges", np.array([]))
            if not isinstance(merges, np.ndarray):
                merges = merges[()]
        return merges

    def get_picks(
        self,
        names: t.Tuple[str, ...] = ("trap", "cell_label"),
        path: str = "modifiers/picks/",
    ) -> t.Set[t.Tuple[int, str]]:
        """Get picks from the h5 file."""
        with h5py.File(self.filename, "r") as f:
            if path in f:
                picks = set(
                    zip(*[f[path + name] for name in names if name in f[path]])
                )
            else:
                picks = set()
            return picks

    def dataset_to_df(self, f: h5py.File, path: str) -> pd.DataFrame:
        """Get data from h5 file as a dataframe."""
        if path not in f:
            message = f"{path} not in {f}."
            self._log(message)
            return None
        else:
            dset = f[path]
            values, index, columns = [], [], []
            index_names = copy(self.index_names)
            valid_names = [lbl for lbl in index_names if lbl in dset.keys()]
            if valid_names:
                index = pd.MultiIndex.from_arrays(
                    [dset[lbl] for lbl in valid_names], names=valid_names
                )
                columns = dset.attrs.get("columns", None)
                if "timepoint" in dset:
                    columns = f[path + "/timepoint"][()]
                values = f[path + "/values"][()]
            df = pd.DataFrame(values, index=index, columns=columns)
            return df

    @property
    def stem(self):
        """Get name of h5 file."""
        return self.filename.stem

    def store_signal_path(
        self,
        fullname: str,
        node: t.Union[h5py.Dataset, h5py.Group],
    ):
        """Store the name of a signal if it is a leaf node and if it starts with extraction."""
        if isinstance(node, h5py.Group) and np.all(
            [isinstance(x, h5py.Dataset) for x in node.values()]
        ):
            self._if_ext_or_post(fullname, self._available)

    @staticmethod
    def _if_ext_or_post(name: str, siglist: list):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            siglist.append(name)

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
