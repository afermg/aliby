from collections.abc import Iterable
from typing import Union, List, Dict
from itertools import accumulate

import h5py
import numpy as np
import pandas as pd

from core.io.base import BridgeH5

#################### Dynamic version ##################################

class DynamicWriter:
    data_types = {}
    group = ''
    compression = None

    def __init__(self, file: str):
        self.file = file

    def _append(self, data, key, hgroup):
        """Append data to existing dataset."""
        try:
            n = len(data)
        except:
            # Attributes have no length
            n = 1
        if key not in hgroup:
            # TODO Include sparsity check
            max_shape, dtype = self.datatypes[key]
            shape = (n,) + max_shape[1:]
            hgroup.create_dataset(key, shape=shape, maxshape=max_shape,
                                  dtype=dtype, compression=self.compression)
            hgroup[key][()] = data
        else:
            # The dataset already exists, expand it
            dset = hgroup[key]
            dset.resize(dset.shape[0] + n, axis=0)
            dset[-n:] = data
        return

    def _overwrite(self, data, key, hgroup):
        """Overwrite existing dataset with new data"""
        # We do not append to mother_assign; raise error if already saved
        n = len(data)
        max_shape, dtype = self.datatypes[key]
        if key in hgroup:
            del hgroup[key]
        hgroup.require_dataset(key, shape=(n,),
                               dtype=dtype,
                               compression=self.compression)
        hgroup[key][()] = data

    def _check_key(self, key):
        if key not in self.datatypes:
            raise KeyError(f"No defined data type for key {key}")

    def write(self, data, overwrite: list):
        # Data is a dictionary, if not, make it one
        # Overwrite data is a dictionary
        with h5py.File(self.file, 'a') as store:
            hgroup = store.require_group(self.group)

            for key, value in data.items():
                # We're only saving data that has a pre-defined data-type
                self._check_key(key)
                try:
                    if key.startswith('attrs/'):  # metadata
                        key = key.split('/')[1]  # First thing after attrs
                        hgroup.attrs[key] = value
                    elif key in overwrite:
                        self._overwrite(value, key, hgroup)
                    else:
                        self._append(value, key, hgroup)
                except Exception as e:
                    print(key, value)
                    raise (e)
        return


##################### Special instances #####################
class TilerWriter(DynamicWriter):
    datatypes = {
        'trap_locations': ((None, 2), np.uint16),
        'drifts': ((None, 2), np.float32),
        'attrs/tile_size': ((1,), np.uint16),
        'attrs/max_size': ((1,), np.uint16)
    }
    group = 'trap_info'


tile_size = 117


class BabyWriter(DynamicWriter):
    # TODO make this YAML
    compression = 'gzip'
    datatypes = {
        'centres': ((None, 2), np.uint16),
        'position': ((None,), np.uint16),
        'angles': ((None,), h5py.vlen_dtype(np.float32)),
        'radii': ((None,), h5py.vlen_dtype(np.float32)),
        'edgemasks': ((None, tile_size, tile_size), np.bool),
        'ellipse_dims': ((None, 2), np.float32),
        'cell_label': ((None,), np.uint16),
        'trap': ((None,), np.uint16),
        'timepoint': ((None,), np.uint16),
        'mother_assign': ((None,), h5py.vlen_dtype(np.uint16)),
        'volumes': ((None,), np.float32)
    }
    group = 'cell_info'

#################### Extraction version ###############################
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
            if data is not None:
                self.write_dset(f, path, data)
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, data=metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        if isinstance(data, pd.DataFrame):
            self.write_df(f, path, data)
        elif isinstance(data, pd.MultiIndex):
            self.write_index(f, path, data)
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, attr: str, data: Iterable):
        # path, attr = path.split(".")
        obj = f.require_group(path)

        obj.attrs[attr] = data

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
    def write_dynamic(f: h5py.File, path: str, data: Iterable):
        pass

    @staticmethod
    def write_index(f, path, pd_index):
        if path not in f:
            f.create_group(path)  # TODO check if we can remove this
        for name, ids in zip(pd_index.names, pd_index.values):
            id_path = path + "/" + name
            f.create_dataset(name=id_path, shape=ids.shape, dtype=df.dtypes[0])
            indices = f[id_path]
            indices[()] = ids

    @staticmethod
    def write_df(f, path, df):
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
