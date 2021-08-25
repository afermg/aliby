from copy import copy
from itertools import accumulate

# from more_itertools import first_true

import h5py
import pandas as pd
from utils_find_1st import find_1st, cmp_larger

from core.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file, flag=None)

    def __getitem__(self, dataset):

        if isinstance(dataset, str):
            return self.apply_prepost(dataset)

        elif isinstance(dataset, list):
            is_bgd = [dset.endswith("imBackground") for dset in dataset]
            assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                dataset
            ), "Trap data and cell data can't be mixed"
            with h5py.File(self.filename, "r") as f:
                return [self.apply_prepost(dset) for dset in dataset]

                # indices = [dset.index for dset in datasets]
                # intersection = indices[0]
                # for index in indices[1:]:
                #     intersection = intersection.intersection(index)
                # id_intersect = [dset.loc[intersection] for dset in datasets]

    def apply_prepost(self, dataset: str):
        merges = self.get_merges()  # TODO pass as an argument instead?
        with h5py.File(self.filename, "r") as f:
            print("Reading from ", dataset)
            df = self.dset_to_df(f, dataset)
            merged = self.apply_merge(df, merges)

        picks = self.get_picks(merged.index.names)

        if picks:
            return merged.loc[picks]
        else:
            return merged

    @property
    def datasets(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_ext_or_post)
        return dsets

    @property
    def datasets(self):
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

            tmp = copy(df)
            for target, source in changes:
                df.loc[tuple(target)] = self.join_tracks_pair(
                    df.loc[tuple(target)], tmp.loc[tuple(source)]
                )
                tmp.drop(tuple(source), inplace=True)

                df = tmp

        return df

    def get_merges(self):
        # fetch merge events going up to the first level
        with h5py.File(self.filename, "r") as f:
            merges = f.get("modifiers/merges", [])
            if not isinstance(merges, list):
                merges = merges[()]

        return merges

    def get_picks(self, levels):
        with h5py.File(self.filename, "r") as f:
            if "modifier/picker" in f:
                return zip(*[f["modifier/picker/" + level] for level in levels])
            else:
                return None

    @staticmethod
    def dset_to_df(f, dataset):
        dset = f[dataset]
        names = ["experiment", "position", "trap"]
        if not dataset.endswith("imBackground"):
            names.append("cell_label")
        lbls = {lbl: dset[lbl][()] for lbl in names if lbl in dset.keys()}
        index = pd.MultiIndex.from_arrays(
            list(lbls.values()), names=names[-len(lbls) :]
        )

        columns = dset["timepoint"][()]

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
