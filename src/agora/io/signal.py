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
from agora.utils.merge import apply_merges


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing

    Signal is works under the assumption that metadata and data are
    accessible, to perform time-adjustments and apply previously-recorded
    postprocesses.
    """

    def __init__(self, file: t.Union[str, PosixPath]):
        super().__init__(file, flag=None)

        self.index_names = (
            "experiment",
            "position",
            "trap",
            "cell_label",
            "mother_label",
        )

        equivalences = {
            "m5m": ("extraction/GFP/max/max5px", "extraction/GFP/max/median")
        }

    def __getitem__(self, dsets: t.Union[str, t.Collection]):

        if isinstance(dsets, str) and dsets.endswith("imBackground"):
            df = self.get_raw(dsets)

        elif isinstance(dsets, str):
            df = self.apply_prepost(dsets)

        elif isinstance(dsets, list):
            is_bgd = [dset.endswith("imBackground") for dset in dsets]
            assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                dsets
            ), "Trap data and cell data can't be mixed"
            return [
                self.add_name(self.apply_prepost(dset), dset) for dset in dsets
            ]
        else:
            raise Exception(f"Invalid type {type(dsets)} to get datasets")

        # return self.cols_in_mins(self.add_name(df, dsets))
        return self.add_name(df, dsets)

    @staticmethod
    def add_name(df, name):
        df.name = name
        return df

    def cols_in_mins(self, df: pd.DataFrame):
        # Convert numerical columns in a dataframe to minutes
        try:
            df.columns = (df.columns * self.tinterval // 60).astype(int)
        except Exception as e:
            print(
                """
                Warning:Can't convert columns to minutes. Signal {}.{}""".format(
                    df.name, e
                )
            )
        return df

    @cached_property
    def ntimepoints(self):
        with h5py.File(self.filename, "r") as f:
            return f["extraction/general/None/area/timepoint"][-1] + 1

    @cached_property
    def tinterval(self) -> int:
        tinterval_location = "time_settings/timeinterval"
        with h5py.File(self.filename, "r") as f:
            return f.attrs[tinterval_location][0]

    @staticmethod
    def get_retained(df, cutoff):
        return df.loc[bn.nansum(df.notna(), axis=1) > df.shape[1] * cutoff]

    @property
    def channels(self):
        with h5py.File(self.filename, "r") as f:
            return f.attrs["channels/channel"]

    @_first_arg_str_to_df
    def retained(self, signal, cutoff=0.8):

        df = signal
        # df = self[signal]
        if isinstance(df, pd.DataFrame):
            return self.get_retained(df, cutoff)

        elif isinstance(df, list):
            return [self.get_retained(d, cutoff=cutoff) for d in df]

    @lru_cache(2)
    def lineage(
        self, lineage_location: t.Optional[str] = None, merged: bool = False
    ) -> np.ndarray:
        """
        Return lineage data from a given location as a matrix where
        the first column is the trap id,
        the second column is the mother label and
        the third column is the daughter label.
        """
        if lineage_location is None:
            lineage_location = "postprocessing/lineage"
            if merged:
                lineage_location += "_merged"

        with h5py.File(self.filename, "r") as f:
            trap_mo_da = f[lineage_location]
            lineage = np.array(
                (
                    trap_mo_da["trap"],
                    trap_mo_da["mother_label"],
                    trap_mo_da["daughter_label"],
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
        """Apply modifier operations (picker, merger) to a given dataframe.


        Parameters
        ----------
        data : t.Union[str, pd.DataFrame]
            DataFrame or url to one.
        merges : t.Union[np.ndarray, bool]
            (optional) 2-D array with three columns and variable length. The
            first column is the trap id, second is mother label and third one is
            daughter id.
            If it is True it fetches merges from file, if false it skips merging step.
        picks : t.Union[np.ndarray, bool]
            (optional) 2-D ndarray where first column is traps and second column
            is cell labels.
            If it is True it fetches picks from file, if false it skips picking step.

        Examples
        --------
        FIXME: Add docs.

        """
        if isinstance(merges, bool):
            merges: np.ndarray = self.get_merges() if merges else np.array([])

        merged = copy(data)

        if merges.any():
            merged = apply_merges(data, merges)

        if isinstance(picks, bool):
            picks = (
                self.get_picks(names=merged.index.names)
                if picks
                else set(merged.index)
            )

        with h5py.File(self.filename, "r") as f:
            if "modifiers/picks" in f and picks:
                # missing_cells = [i for i in picks if tuple(i) not in
                # set(merged.index)]

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

    @property
    def datasets(self):
        if not hasattr(self, "_available"):
            self._available = []

            with h5py.File(self.filename, "r") as f:
                f.visititems(self.store_signal_url)

        for sig in self.siglist:
            print(sig)

    @cached_property
    def p_available(self):
        """Print signal list"""
        self.datasets

    @cached_property
    def available(self):
        """Return list of available signals"""
        try:
            if not hasattr(self, "_available"):
                self._available = []

            with h5py.File(self.filename, "r") as f:
                f.visititems(self.store_signal_url)

        except Exception as e:
            print("Error visiting h5: {}".format(e))

        return self._available

    def get_merged(self, dataset):
        return self.apply_prepost(dataset, picks=False)

    @cached_property
    def merges(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_merges)
        return dsets

    @cached_property
    def n_merges(self):
        return len(self.merges)

    @cached_property
    def picks(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_picks)
        return dsets

    def get_raw(self, dataset: str, in_minutes: bool = True):
        try:
            if isinstance(dataset, str):
                with h5py.File(self.filename, "r") as f:
                    df = self.dataset_to_df(f, dataset)
                    if in_minutes:
                        df = self.cols_in_mins(df)
                    return df
            elif isinstance(dataset, list):
                return [self.get_raw(dset) for dset in dataset]
        except Exception as e:
            print(f"Could not fetch dataset {dataset}")
            print(e)

    def get_merges(self):
        # fetch merge events going up to the first level
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
        """
        Return the relevant picks based on names
        """
        with h5py.File(self.filename, "r") as f:
            picks = set()
            if path in f:
                picks = set(zip(*[f[path + name] for name in names]))

            return picks

    def dataset_to_df(self, f: h5py.File, path: str) -> pd.DataFrame:
        """
        Fetch DataFrame from results storage file.
        """

        assert path in f, f"{path} not in {f}"

        dset = f[path]
        index_names = copy(self.index_names)

        valid_names = [lbl for lbl in index_names if lbl in dset.keys()]
        index = pd.MultiIndex.from_arrays(
            [dset[lbl] for lbl in valid_names], names=valid_names
        )

        columns = dset.attrs.get("columns", None)  # dset.attrs["columns"]
        if "timepoint" in dset:
            columns = f[path + "/timepoint"][()]

        return pd.DataFrame(
            f[path + "/values"][()],
            index=index,
            columns=columns,
        )

    @property
    def stem(self):
        return self.filename.stem

    # def dataset_to_df(self, f: h5py.File, path: str):

    #     all_indices = self.index_names

    #     valid_indices = {
    #         k: f[path][k][()] for k in all_indices if k in f[path].keys()
    #     }

    #     new_index = pd.MultiIndex.from_arrays(
    #         list(valid_indices.values()), names=valid_indices.keys()
    #     )

    #     return pd.DataFrame(
    #         f[path + "/values"][()],
    #         index=new_index,
    #         columns=f[path + "/timepoint"][()],
    #     )

    def store_signal_url(
        self, fullname: str, node: t.Union[h5py.Dataset, h5py.Group]
    ):
        """
        Store the name of a signal it is a leaf node (a group with no more groups inside)
        and starts with extraction
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
        # Return number of time-points according to the metadata
        return self.meta_h5["time_settings/ntimepoints"][0]

    @property
    def stages(self) -> t.List[str]:
        """
        Return the contents of the pump with highest flow rate
        at each stage.
        """
        flowrate_name = "pumpinit/flowrate"
        pumprate_name = "pumprate"
        main_pump_id = np.concatenate(
            (
                (np.argmax(self.meta_h5[flowrate_name]),),
                np.argmax(self.meta_h5[pumprate_name], axis=0),
            )
        )
        return [self.meta_h5["pumpinit/contents"][i] for i in main_pump_id]

    @property
    def nstages(self) -> int:
        switchtimes_name = "switchtimes"
        return self.meta_h5[switchtimes_name] + 1

    @property
    def max_span(self) -> int:
        return int(self.tinterval * self.ntps / 60)

    @property
    def stages_span(self) -> t.Tuple[t.Tuple[str, int], ...]:
        # Return consecutive stages and their corresponding number of time-points
        switchtimes_name = "switchtimes"
        transition_tps = (0, *self.meta_h5[switchtimes_name])
        spans = [
            end - start
            for start, end in zip(transition_tps[:-1], transition_tps[1:])
            if end <= self.max_span
        ]
        return tuple((stage, ntps) for stage, ntps in zip(self.stages, spans))
