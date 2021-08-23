from itertools import accumulate

# from more_itertools import first_true

import h5py
import pandas as pd

from core.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file, flag=None)

    def __getitem__(self, dataset):
        changes = self.get_id_changes()
        with h5py.File(self.filename, "r") as f:
            print("Reading from ", dataset)
            if isinstance(dataset, str):
                df = self.dset_to_df(f, dataset)
                return self.apply_changes(df, changes)

            elif isinstance(dataset, list):
                is_bgd = [dset.endswith("imBackground") for dset in dataset]
                assert sum(is_bgd) == 0 or sum(is_bgd) == len(
                    dataset
                ), "Trap data and cell data can't be mixed"

                datasets = [self.dset_to_df(f, dset) for dset in dataset]

                indices = [dset.index for dset in datasets]
                intersection = indices[0]
                for index in indices[1:]:
                    intersection = intersection.intersection(index)

                return [dset.loc[intersection] for dset in datasets]

    def apply_changes(self, df, changes):
        if changes:
            for change, value in changes:
                if change == "merge":
                    tmp = copy(df)
                    for target, source in value:
                        tmp.loc[target] = join_track_pairs(
                            tmp.loc[target], tmp.loc[source]
                        )

                df = tmp

        return df

    def get_id_changes(self):
        # fetch merge events going up to the first level
        with h5py.File(self.filename, "r") as f:
            id_changes = f.get("/id_changes", [])[()]

        return id_changes

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
    def _if_id_changes(name: str, *args):
        if name.startswith("id_changes"):
            print(name)

    @property
    def datasets(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_ext_or_post)
        return dsets

    @property
    def merge_events(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visititems(self._if_id_changes)
        return dsets


# s = Signal(
#     "/shared_libs/pipeline-core/scripts/pH_calibration_dual_phl__ura8__by4741__01/ph_5_29_025store.h5"
# )

# s.dataset_to_df(s._hdf, "/extraction/em_ratio_bgsub/np_max/median")
