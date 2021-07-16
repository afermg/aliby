import pandas as pd

from postprocessor.core.io.base import BridgeH5


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
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def datasets(self):
        return self._hdf.visit(self._if_ext_or_post)
