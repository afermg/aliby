import logging
import typing as t
from copy import copy
from functools import cached_property, lru_cache
from pathlib import PosixPath

import bottleneck as bn
import h5py
import numpy as np
import pandas as pd

from agora.io.bridge import BridgeH5
from agora.io.decorators import _first_arg_str_to_df
from agora.utils.association import validate_association
from agora.utils.kymograph import add_index_levels
from agora.utils.merge import apply_merges


class Signal(BridgeH5):
    """
    Fetch data from h5 files for post-processing.

    Signal assumes that the metadata and data are accessible to perform time-adjustments and apply previously recorded post-processes.
    """

    def __init__(self, file: t.Union[str, PosixPath]):
        """Define index_names for dataframes, candidate fluorescence channels, and composite statistics."""
        super().__init__(file, flag=None)
        self.index_names = (
            "experiment",
            "position",
            "trap",
            "cell_label",
            "mother_label",
        )
        self.candidate_channels = (
            "GFP",
            "GFPFast",
            "mCherry",
            "Flavin",
            "Citrine",
            "mKO2",
            "Cy5",
            "pHluorin405",
        )

    def __getitem__(self, dsets: t.Union[str, t.Collection]):
        """Get and potentially pre-process data from h5 file and return as a dataframe."""
        if isinstance(dsets, str):  # no pre-processing
            df = self.get_raw(dsets)
            return self.add_name(df, dsets)
        elif isinstance(dsets, list):  # pre-processing
            is_bgd = [dset.endswith("imBackground") for dset in dsets]
            # Check we are not comaring tile-indexed and cell-indexed data
            assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                dsets
            ), "Tile data and cell data can't be mixed"
            return [
                self.add_name(self.apply_prepost(dset), dset) for dset in dsets
            ]
        else:
            raise Exception(f"Invalid type {type(dsets)} to get datasets")

    @staticmethod
    def add_name(df, name):
        """Add column of identical strings to a dataframe."""
        df.name = name
        return df

    def cols_in_mins(self, df: pd.DataFrame):
        # Convert numerical columns in a dataframe to minutes
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
        """Find the interval between time points (minutes)."""
        tinterval_location = "time_settings/timeinterval"
        with h5py.File(self.filename, "r") as f:
            if tinterval_location in f.attrs:
                return f.attrs[tinterval_location][0]
            else:
                logging.getlogger("aliby").warn(
                    f"{str(self.filename).split('/')[-1]}: using default time interval of 5 minutes"
                )
                return 5

    @staticmethod
    def get_retained(df, cutoff):
        """Return a fraction of the df, one without later time points."""
        return df.loc[bn.nansum(df.notna(), axis=1) > df.shape[1] * cutoff]

    @property
    def channels(self) -> t.Collection[str]:
        """Get channels as an array of strings."""
        with h5py.File(self.filename, "r") as f:
            return list(f.attrs["channels"])

    @_first_arg_str_to_df
    def retained(self, signal, cutoff=0.8):
        """
        Load data (via decorator) and reduce the resulting dataframe.

        Load data for a signal or a list of signals and reduce the resulting
        dataframes to a fraction of their original size, losing late time
        points.
        """
        if isinstance(signal, pd.DataFrame):
            return self.get_retained(signal, cutoff)
        elif isinstance(signal, list):
            return [self.get_retained(d, cutoff=cutoff) for d in signal]

    @lru_cache(2)
    def lineage(
        self, lineage_location: t.Optional[str] = None, merged: bool = False
    ) -> np.ndarray:
        """
        Get lineage data from a given location in the h5 file.

        Returns an array with three columns: the tile id, the mother label, and the daughter label.
        """
        if lineage_location is None:
            lineage_location = "postprocessing/lineage"
            if merged:
                lineage_location += "_merged"
        with h5py.File(self.filename, "r") as f:
            tile_mo_da = f[lineage_location]
            lineage = np.array(
                (
                    tile_mo_da["trap"],
                    tile_mo_da["mother_label"],
                    tile_mo_da["daughter_label"],
                )
            ).T
        return lineage

    @_first_arg_str_to_df
    def apply_prepost(
        self,
        data: t.Union[str, pd.DataFrame],
        merges: t.Union[np.ndarray, bool] = True,
        picks: t.Union[t.Collection, bool] = True,
    ):
        """
        Apply modifier operations (picker or merger) to a dataframe.

        Parameters
        ----------
        data : t.Union[str, pd.DataFrame]
            DataFrame or path to one.
        merges : t.Union[np.ndarray, bool]
            (optional) 2-D array with three columns: the tile id, the mother label, and the daughter id.
            If True, fetch merges from file.
        picks : t.Union[np.ndarray, bool]
            (optional) 2-D array with two columns: the tiles and
            the cell labels.
            If True, fetch picks from file.

        Examples
        --------
        FIXME: Add docs.

        """
        if isinstance(merges, bool):
            merges: np.ndarray = self.get_merges() if merges else np.array([])
        if merges.any():
            merged = apply_merges(data, merges)
        else:
            merged = copy(data)
        if isinstance(picks, bool):
            picks = (
                self.get_picks(names=merged.index.names)
                if picks
                else set(merged.index)
            )
        with h5py.File(self.filename, "r") as f:
            if "modifiers/picks" in f and picks:
                if picks:
                    return merged.loc[
                        set(picks).intersection(
                            [tuple(x) for x in merged.index]
                        )
                    ]
                else:
                    if isinstance(merged.index, pd.MultiIndex):
                        empty_lvls = [[] for i in merged.index.names]
                        index = pd.MultiIndex(
                            levels=empty_lvls,
                            codes=empty_lvls,
                            names=merged.index.names,
                        )
                    else:
                        index = pd.Index([], name=merged.index.name)
                    merged = pd.DataFrame([], index=index)
        return merged

    # Alan: do we need two similar properties - see below?
    @property
    def datasets(self):
        """Print data sets available in h5 file."""
        if not hasattr(self, "_available"):
            self._available = []
            with h5py.File(self.filename, "r") as f:
                f.visititems(self.store_signal_path)
        for sig in self._available:
            print(sig)

    @cached_property
    def p_available(self):
        """Print data sets available in h5 file."""
        self.datasets

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
        """Run preprocessing for merges."""
        return self.apply_prepost(dataset, picks=False)

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
    ) -> pd.DataFrame or t.List[pd.DataFrame]:
        """
        Load data from a h5 file and return as a dataframe.

        Parameters
        ----------
        dataset: str or list of strs
            The name of the h5 file or a list of h5 file names
        in_minutes: boolean
            If True,
        lineage: boolean
        """
        try:
            if isinstance(dataset, str):
                with h5py.File(self.filename, "r") as f:
                    df = self.dataset_to_df(f, dataset).sort_index()
                    if in_minutes:
                        df = self.cols_in_mins(df)
            elif isinstance(dataset, list):
                return [
                    self.get_raw(dset, in_minutes=in_minutes, lineage=lineage)
                    for dset in dataset
                ]
            if lineage:  # assume that df is sorted
                mother_label = np.zeros(len(df), dtype=int)
                lineage = self.lineage()
                a, b = validate_association(
                    lineage,
                    np.array(df.index.to_list()),
                    match_column=1,
                )
                mother_label[b] = lineage[a, 1]
                df = add_index_levels(df, {"mother_label": mother_label})
            return df
        except Exception as e:
            self._log(f"Could not fetch dataset {dataset}: {e}", "error")
            raise e

    def get_merges(self):
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
        """Get the relevant picks based on names."""
        with h5py.File(self.filename, "r") as f:
            picks = set()
            if path in f:
                picks = set(zip(*[f[path + name] for name in names]))
            return picks

    def dataset_to_df(self, f: h5py.File, path: str) -> pd.DataFrame:
        """Get data from h5 file as a dataframe."""
        assert path in f, f"{path} not in {f}"
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
        """
        Store the name of a signal if it is a leaf node
        (a group with no more groups inside) and if it starts with extraction.
        """
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

    # TODO FUTURE add stages support to fluigent system
    @property
    def ntps(self) -> int:
        """Get number of time points from the metadata."""
        return self.meta_h5["time_settings/ntimepoints"][0]

    @property
    def stages(self) -> t.List[str]:
        """Get the contents of the pump with highest flow rate at each stage."""
        flowrate_name = "pumpinit/flowrate"
        pumprate_name = "pumprate"
        switchtimes_name = "switchtimes"
        main_pump_id = np.concatenate(
            (
                (np.argmax(self.meta_h5[flowrate_name]),),
                np.argmax(self.meta_h5[pumprate_name], axis=0),
            )
        )
        if not self.meta_h5[switchtimes_name][0]:  # Cover for t0 switches
            main_pump_id = main_pump_id[1:]
        return [self.meta_h5["pumpinit/contents"][i] for i in main_pump_id]

    @property
    def nstages(self) -> int:
        return len(self.switch_times) + 1

    @property
    def max_span(self) -> int:
        return int(self.tinterval * self.ntps / 60)

    @property
    def switch_times(self) -> t.List[int]:
        switchtimes_name = "switchtimes"
        switches_minutes = self.meta_h5[switchtimes_name]
        return [
            t_min
            for t_min in switches_minutes
            if t_min and t_min < self.max_span
        ]  # Cover for t0 switches

    @property
    def stages_span(self) -> t.Tuple[t.Tuple[str, int], ...]:
        """Get consecutive stages and their corresponding number of time points."""
        transition_tps = (0, *self.switch_times, self.max_span)
        spans = [
            end - start
            for start, end in zip(transition_tps[:-1], transition_tps[1:])
            if end <= self.max_span
        ]
        return tuple((stage, ntps) for stage, ntps in zip(self.stages, spans))

    @property
    def stages_span_tp(self) -> t.Tuple[t.Tuple[str, int], ...]:
        return tuple(
            [
                (name, (t_min * 60) // self.tinterval)
                for name, t_min in self.stages_span
            ]
        )
