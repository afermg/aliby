import pandas as pd

from core.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file)

    def __getitem__(self, dataset):
        dset = self._hdf[dataset]
        lbls = [dset[lbl][()] for lbl in dset.keys() if "axis1_label" in lbl]
        index = pd.MultiIndex.from_arrays(
            lbls, names=["position", "trap", "cell"][-len(lbls) :]
        )

        columns = dset["axis0"][()]

        return pd.DataFrame(dset[("block0_values")][()], index=index, columns=columns)

    @staticmethod
    def dataset_to_df(f, path, mode="h5py"):
        # TODO add support for pytables additionally to h5py?

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
        return self._hdf.visit(self._if_ext_or_post)


s = Signal(
    "/shared_libs/pipeline-core/scripts/pH_calibration_dual_phl__ura8__by4741__01/ph_5_29_025store.h5"
)

s.dataset_to_df(s._hdf, "/extraction/em_ratio_bgsub/np_max/median")
