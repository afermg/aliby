from collections.abc import Iterable
from typing import Union, List, Dict
from itertools import accumulate

import h5py
import numpy as np
import pandas as pd

from core.io.base import BridgeH5


class Writer(BridgeH5):
    """
    Class in charge of transforming data into compatible formats

    Decoupling interface from implementation!

    Parameters
    ----------
        filename: str Name of file to write into
        flag: str, default=None
            Flag to pass to the default file reader. If None the file remains closed.
    """

    def __init__(self, filename):
        super().__init__(filename, flag=None)

    def write(
        self, path: str, data: Iterable = None, meta: Dict = {}, overwrite: bool = True
    ):
        with h5py.File(self.filename, "a") as f:
            over_txt = ""
            if overwrite:
                over_txt = "over"
                self.deldset(f, path)
            else:  # Add a number if needed
                if path in f:
                    parent, name = path.rsplit("/", maxsplit=1)
                    n = sum([x.startswith(name) for x in f[path]])
                    path = path + str(n).zfill(3)

            # f.create_group(path)
            print(
                "{}writing {} to {} and {} metadata fields".format(
                    over_txt, type(data), path, len(meta)
                )
            )
            if data:
                self.write_dset(f, path, data)
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        if isinstance(data, pd.DataFrame):
            self.write_df(data, f, path)
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, data: Iterable):
        path, attr = path.split(".")

        f[path].attrs[attr] = data

    @staticmethod
    def deldset(f: h5py.File, path: str):
        if path in f:
            del f[path]

    @staticmethod
    def delmeta(f: h5py.File, path: str, attr: str):
        if path in f and attr in f[path].attrs:
            del f[path].attrs[attr]

    @staticmethod
    def write_arraylike(f: h5py.File, path: str, data: Iterable):
        if path in f:
            del f[path]

        narray = np.array(data)
        dset = f.create_dataset(path, shape=narray.shape, dtype="int")
        dset[()] = narray

    @staticmethod
    def write_df(df, f, path):
        print("writing to ", path)
        if path not in f:
            f.create_group(path)  # TODO check if we can remove this

        values_path = path + "/values"
        f.create_dataset(name=values_path, shape=df.shape, dtype=df.dtypes[0])
        dset = f[values_path]
        dset[()] = df.values

        for name in df.index.names:
            dtype = "uint16"  # if name != "position" else "str"
            indices_path = path + "/" + name
            f.create_dataset(name=indices_path, shape=(len(df),), dtype=dtype)
            dset = f[indices_path]
            dset[()] = df.index.get_level_values(level=name).tolist()

        tp_path = path + "/timepoint"
        f.create_dataset(name=tp_path, shape=(df.shape[1],), dtype="uint16")
        f[tp_path][()] = df.columns.tolist()
