import h5py
import pandas as pd

from postprocessor.core.io.base import BridgeH5


def Writer(BridgeH5):
    def __init__(self, filename):
        self._hdf = h5py.File(filename, "a")

    def write(self, address, data):
        self._file.add_group(address)
        if type(data) is pd.DataFrame:
            self.write_df(address, data)
        elif type(data) is np.array:
            self.write_np(address, data)

    def write_df(self, adress, df):
        self._file.get(address)[()] = data
