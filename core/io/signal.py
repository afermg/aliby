import pandas as pd

from postprocessor.core.io.base import BridgeH5


class Signal(BridgeH5):
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        super().__init__(file)
        self._hdf.close()  # Close the file to use pandas hdf functions
        # hdf = pd.HDFStore(file)
        # self.file = file

    def __getitem__(self, dataset):
        return pd.read_hdf(self.file, dataset)

    @staticmethod
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def datasets(self):
        return self._hdf.visit(self._if_ext_or_post)
