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

    def write(self, data, address):
        with h5py.File(self.filename, "a") as f:
            f.add_group(address)
            if type(data) is pd.DataFrame:
                self.write_df(data, f, address)
            elif type(data) is np.ndarray or type(data) is list:
                self.write_array(address, data)

    def write_array(self, address, array):
        pass

    @staticmethod
    def write_df(df, f, path):
        print("writing to ", path)
        for item in accummulate(path.split("/")[:-2]):
            if item in f:
                del f[item]
            f.create_group(item)
        pos_group = f[path.split("/")[1]]

        for path, df in group_df.items():
            print("creating dset ", path, df.shape)
            dset_path = "/extraction/" + path
            if dset_path in pos_group:
                del f[dset_path]
            values_path = dset_path + "/values"
            pos_group.create_dataset(
                name=values_path, shape=df.shape, dtype=df.dtypes[0]
            )
            new_dset = f[dset_path]
            new_dset[()] = df.values

            for name in df.index.names:
                indices_path = dset_path + name
                pos_group.create_dataset(
                    name=indices_type,
                    shape=len(tmp),
                    dtype=int if name != "position" else str,
                )
                dset = f[dset_path]
                dset[()] = df.index.get_level_values(level=name).tolist()

            f[dset_path + "/timepoint"] = df.columns.tolist()
