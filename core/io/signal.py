from more_itertools import first_true

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
        with h5py.File(self.filename, "r") as f:
            print("Reading from ", dataset)
            if isinstance(dataset, str):
                return self.dset_to_df(f, dataset)

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

    def get_merge_events(self, signal: str):
        # fetch merge events going up to the first level
        paths = (*accumulate(signal.split("/"), lambda x, y: "".join([x, y])),)
        with h5py.File(self.filename, "r") as f:
            merge_events = first_true(
                (f[path].attrs.get("merge_events") for path in paths)
            )

        return (*previous_merges, *merge_events)

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
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def datasets(self):
        with h5py.File(self.filename, "r") as f:
            dsets = f.visit(self._if_ext_or_post)
        return dsets


# s = Signal(
#     "/shared_libs/pipeline-core/scripts/pH_calibration_dual_phl__ura8__by4741__01/ph_5_29_025store.h5"
# )

# s.dataset_to_df(s._hdf, "/extraction/em_ratio_bgsub/np_max/median")
