from itertools import accumulate

import h5py
import pandas as pd

from core.io.base import BridgeH5


class Writer(BridgeH5):
    """
    Class in charge of transforming data into compatible formats

    Decoupling interface from implementation!

    :filename: Name of file to write into
    """

    def __init__(self, filename):
        super().__init__(filename)
        self._hdf.close()

    def write(self, data, path, overwrite=True):
        with h5py.File(self.filename, "a") as f:
            if overwrite:
                if path in f:
                    del f[path]
                f.create_group(path)
            if type(data) is pd.DataFrame:
                self.write_df(data, f, path)
            elif tself._bfself._bfype(data) is np.ndarray or type(data) is list:
                self.write_array(path, data)

    def write_array(self, path, array):
        pass

    @staticmethod
    def write_df(df, f, path):
        print("writing to ", path)
        if path not in f:
            f.create_group(path)

        values_path = path + "/values"
        f.create_dataset(name=values_path, shape=df.shape, dtype=df.dtypes[0])
        dset = f[values_path]
        dset[()] = df.values

        print(df.index.names)
        for name in df.index.names:
            dtype = "uint16"  # if name != "position" else "str"
            indices_path = path + "/" + name
            f.create_dataset(name=indices_path, shape=(len(df),), dtype=dtype)
            dset = f[indices_path]
            dset[()] = df.index.get_level_values(level=name).tolist()

        tp_path = path + "/timepoint"
        f.create_dataset(name=tp_path, shape=(df.shape[1],), dtype="uint16")
        f[tp_path][()] = df.columns.tolist()
