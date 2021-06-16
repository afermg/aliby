from postprocessor.core.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file)

    def __getitem__(self, dataset):
        dset = self._hdf[dataset][()]
        attrs = self._hdf[dataset].attrs
        first_dataset = "/" + dataset.split("/")[0] + "/"
        timepoints = self._hdf[first_dataset].attrs["processed_timepoints"]

        if "cell_label" in self._hdf[dataset].attrs:
            ids = pd.MultiIndex.from_tuples(
                zip(attrs["trap"], attrs["cell_label"]), names=["trap", "cell_label"]
            )
        else:
            ids = pd.Index(attrs["trap"], names=["trap"])

        return pd.DataFrame(dset, index=ids, columns=timepoints)

    @staticmethod
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def datasets(self):
        return signals._hdf.visit(self._if_ext_or_post)
