from itertools import accumulate

import h5py
import pandas as pd

from postprocessor.core.io.base import BridgeH5


def Writer(BridgeH5):
    """
    Class in charge of transforming data into compatible formats

    Decoupling interface from implementation!

    :hdfname: Name of file to write into
    """

    def __init__(self, hdfname):
        self._hdf = h5py.Hdf(hdfname, "a")

    def write(self, address, data):
        self._hdf.add_group(address)
        if type(data) is pd.DataFrame:
            self.write_df(address, data)
        elif type(data) is np.array:
            self.write_np(address, data)

    def write_np(self, address, array):
        pass

    def write_df(self, df, tps, path):
        print("writing to ", path)
        for item in accummulate(path.split("/")[:-2]):
            if item not in self._hdf:
                self._hdf.create_group(item)
        pos_group = f[path.split("/")[1]]

        if path not in pos_group:
            pos_group.create_dataset(name=path, shape=df.shape, dtype=df.dtypes[0])
            new_dset = f[path]
            new_dset[()] = df.values
            if len(df.index.names) > 1:
                trap, cell_label = zip(*list(df.index.values))
                new_dset.attrs["trap"] = trap
                new_dset.attrs["cell_label"] = cell_label
                new_dset.attrs["idnames"] = ["trap", "cell_label"]
            else:
                new_dset.attrs["trap"] = list(df.index.values)
                new_dset.attrs["idnames"] = ["trap"]
        pos_group.attrs["processed_timepoints"] = tps
