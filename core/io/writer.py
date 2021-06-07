import numpy as np
import pandas as pd

def Writer(filename):
    def __init__(self, filename):
        self._file = h5py.File(filename)

    def write(self, address, data):
        self._file.add_group(address)
        if type(data) is pd.DataFrame:
            self.write_df(address, data)
        elif type(data) is np.array:
            self.write_np(address, data)

    def write_df(self, adress, df):
        self._file.get(address)[()] = data
