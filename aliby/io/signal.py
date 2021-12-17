import numpy as np
from copy import copy
from itertools import accumulate

from numpy import ndarray

# from more_itertools import first_true

import h5py
import pandas as pd
from utils_find_1st import find_1st, cmp_larger

from aliby.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file, flag=None)

        self.names = ["experiment", "position", "trap"]

    @staticmethod
    def add_name(df, name):
        df.name = name
        return df

    def mothers(self, signal, cutoff=0.8):
        df = self[signal]
        get_mothers = lambda df: df.loc[df.notna().sum(axis=1) > df.shape[1] * cutoff]
        if isinstance(df, pd.DataFrame):
            return get_mothers(df)
        elif isinstance(df, list):
            return [get_mothers(d) for d in df]

    def __getitem__(self, dsets):

        if isinstance(dsets, str) and (
            dsets.startswith("postprocessing")
            or dsets.startswith("/postprocessing")
            or dsets.endswith("imBackground")
        ):
            df = self.get_raw(dsets)

        elif isinstance(dsets, str):
            df = self.apply_prepost(dsets)

        elif isinstance(dsets, list):
            is_bgd = [dset.endswith("imBackground") for dset in dsets]
            assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                dsets
            ), "Trap data and cell data can't be mixed"
            with h5py.File(self.filename, "r") as f:
                return [self.add_name(self.apply_prepost(dset), dset) for dset in dsets]

        return self.add_name(df, dsets)

    def apply_prepost(self, dataset: str):
        merges = self.get_merges()
        with h5py.File(self.filename, "r") as f:
            df = self.dset_to_df(f, dataset)

            merged = df
            if merges.any():
                # Split in two dfs, one with rows relevant for merging and one without them
                mergable_ids = pd.MultiIndex.from_arrays(
                    np.unique(merges.reshape(-1, 2), axis=0).T,
                    names=df.index.names,
                )
                merged = self.apply_merge(df.loc[mergable_ids], merges)

                nonmergable_ids = df.index.difference(mergable_ids)

                merged = pd.concat(
                    (merged, df.loc[nonmergable_ids]), names=df.index.names
                )

            search = lambda a, b: np.where(
                np.in1d(
                    np.ravel_multi_index(a.T, a.max(0) + 1),
                    np.ravel_multi_index(b.T, a.max(0) + 1),
                )
            )
            if "modifiers/picks" in f:
                picks = self.get_picks(names=merged.index.names)
                missing_cells = [i for i in picks if tuple(i) not in set(merged.index)]

                if picks:
                    # return merged.loc[
                    #     set(picks).intersection([tuple(x) for x in merged.index])
                    # ]
                    return merged.loc[picks]
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
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_ext_or_post)
        return dsets

    def get_merged(self, dataset):
        return self.apply_prepost(dataset, skip_pick=True)

    @property
    def merges(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_merges)
        return dsets

    @property
    def n_merges(self):
        print("{} merge events".format(len(self.merges)))

    @property
    def merges(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_merges)
        return dsets

    @property
    def picks(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_picks)
        return dsets

    def apply_merge(self, df, changes):
        if len(changes):

            for target, source in changes:
                df.loc[tuple(target)] = self.join_tracks_pair(
                    df.loc[tuple(target)], df.loc[tuple(source)]
                )
                df.drop(tuple(source), inplace=True)

        return df

    def get_raw(self, dataset):
        if isinstance(dataset, str):
            with h5py.File(self.filename, "r") as f:
                return self.dset_to_df(f, dataset)
        elif isinstance(dataset, list):
            return [self.get_raw(dset) for dset in dataset]

    def get_merges(self):
        # fetch merge events going up to the first level
        with h5py.File(self.filename, "r") as f:
            merges = f.get("modifiers/merges", np.array([]))
            if not isinstance(merges, np.ndarray):
                merges = merges[()]

        return merges

    # def get_picks(self, levels):
    def get_picks(self, names, path="modifiers/picks/"):
        with h5py.File(self.filename, "r") as f:
            if path in f:
                return list(zip(*[f[path + name] for name in names]))
                # return f["modifiers/picks"]
            else:
                return None

    def dset_to_df(self, f, dataset):
        dset = f[dataset]
        names = copy(self.names)
        if not dataset.endswith("imBackground"):
            names.append("cell_label")
        lbls = {lbl: dset[lbl][()] for lbl in names if lbl in dset.keys()}
        index = pd.MultiIndex.from_arrays(
            list(lbls.values()), names=names[-len(lbls) :]
        )

        columns = (
            dset["timepoint"][()] if "timepoint" in dset else dset.attrs["columns"]
        )

        df = pd.DataFrame(dset[("values")][()], index=index, columns=columns)

        return df

    @staticmethod
    def dataset_to_df(f: h5py.File, path: str, mode: str = "h5py"):

        if mode is "h5py":
            all_indices = ["experiment", "position", "trap", "cell_label"]
            indices = {k: f[path][k][()] for k in all_indices if k in f[path].keys()}
            return pd.DataFrame(
                f[path + "/values"][()],
                index=pd.MultiIndex.from_arrays(
                    list(indices.values()), names=indices.keys()
                ),
                columns=f[path + "/timepoint"][()],
            )

    @staticmethod
    def _if_ext_or_post(name, *args):
        flag = False
        if name.startswith("extraction") and len(name.split("/")) == 4:
            flag = True
        elif name.startswith("postprocessing") and len(name.split("/")) == 3:
            flag = True

        if flag:
            print(name)

    @staticmethod
    def _if_merges(name: str, obj):
        if isinstance(obj, h5py.Dataset) and name.startswith("modifiers/merges"):
            return obj[()]

    @staticmethod
    def _if_picks(name: str, obj):
        if isinstance(obj, h5py.Group) and name.endswith("picks"):
            return obj[()]

    @staticmethod
    def join_tracks_pair(target, source):
        tgt_copy = copy(target)
        end = find_1st(target.values[::-1], 0, cmp_larger)
        tgt_copy.iloc[-end:] = source.iloc[-end:].values
        return tgt_copy
