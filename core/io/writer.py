import logging

import h5py
import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import Dict

from utils_find_1st import find_1st, cmp_equal

from core.io.base import BridgeH5


#################### Dynamic version ##################################


class DynamicWriter:
    data_types = {}
    group = ""
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
            hgroup.create_dataset(
                key,
                shape=shape,
                maxshape=max_shape,
                dtype=dtype,
                compression=self.compression,
            )
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
        hgroup.require_dataset(
            key, shape=(n,), dtype=dtype, compression=self.compression
        )
        hgroup[key][()] = data

    def _check_key(self, key):
        if key not in self.datatypes:
            raise KeyError(f"No defined data type for key {key}")

    def write(self, data, overwrite: list):
        # Data is a dictionary, if not, make it one
        # Overwrite data is a dictionary
        with h5py.File(self.file, "a") as store:
            hgroup = store.require_group(self.group)

            for key, value in data.items():
                # We're only saving data that has a pre-defined data-type
                self._check_key(key)
                try:
                    if key.startswith("attrs/"):  # metadata
                        key = key.split("/")[1]  # First thing after attrs
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
        "trap_locations": ((None, 2), np.uint16),
        "drifts": ((None, 2), np.float32),
        "attrs/tile_size": ((1,), np.uint16),
        "attrs/max_size": ((1,), np.uint16),
    }
    group = "trap_info"


tile_size = 117


class BabyWriter(DynamicWriter):
    # TODO make this YAML
    compression = "gzip"
    datatypes = {
        "centres": ((None, 2), np.uint16),
        "position": ((None,), np.uint16),
        "angles": ((None,), h5py.vlen_dtype(np.float32)),
        "radii": ((None,), h5py.vlen_dtype(np.float32)),
        "edgemasks": ((None, tile_size, tile_size), np.bool),
        "ellipse_dims": ((None, 2), np.float32),
        "cell_label": ((None,), np.uint16),
        "trap": ((None,), np.uint16),
        "timepoint": ((None,), np.uint16),
        "mother_assign": ((None,), h5py.vlen_dtype(np.uint16)),
        "volumes": ((None,), np.float32),
    }
    group = "cell_info"


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
        compression: str, default=None
            Compression method passed on to h5py writing functions (only used for
        dataframes and other array-like data.)
    """

    def __init__(self, filename, compression=None):
        super().__init__(filename, flag=None)

        if compression is None:
            self.compression = "gzip"

    def write(
        self,
        path: str,
        data: Iterable = None,
        meta: Dict = {},
        overwrite: str = "minimal",
    ):
        """
        Parameters
        ----------
        path : str
            Path inside h5 file to write into.
        data : Iterable, default = None
        meta : Dict, default = {}


        """
        self.id_cache = {}
        with h5py.File(self.filename, "a") as f:
            if overwrite == "overwrite":
                self.deldset(f, path)
            elif overwrite == "accumulate":  # Add a number if needed
                if path in f:
                    parent, name = path.rsplit("/", maxsplit=1)
                    n = sum([x.startswith(name) for x in f[path]])
                    path = path + str(n).zfill(3)
            elif overwrite == "skip":
                if path in f:
                    logging.debug("Skipping dataset {}".format(path))
                    return None

            logging.debug(
                "{} {} to {} and {} metadata fields".format(
                    overwrite, type(data), path, len(meta)
                )
            )
            if data is not None:
                self.write_dset(f, path, data)
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, data=metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        if isinstance(data, pd.DataFrame):
            self.write_df(f, path, data, compression=self.compression)
        elif isinstance(data, pd.MultiIndex):
            self.write_index(f, path, data)  # , compression=self.compression)
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, attr: str, data: Iterable):
        obj = f.require_group(path)

        obj.attrs[attr] = data

    @staticmethod
    def write_arraylike(f: h5py.File, path: str, data: Iterable, **kwargs):
        if path in f:
            del f[path]

        narray = np.array(data)

        dset = f.create_dataset(
            path,
            shape=narray.shape,
            chunks=(1, *narray.shape[1:]),
            dtype="int",
            compression=kwargs.get("compression", None),
        )
        dset[()] = narray

    @staticmethod  # TODO Use this function to implement Diane's dynamic writer
    def write_dynamic(f: h5py.File, path: str, data: Iterable):
        pass

    @staticmethod
    def write_index(f, path, pd_index, **kwargs):
        f.require_group(path)  # TODO check if we can remove this
        for name, ids in zip(pd_index.names, pd_index.values):
            id_path = path + "/" + name
            f.create_dataset(
                name=id_path,
                shape=ids.shape,
                dtype=pd_index.dtypes.iloc[0],
                compression=kwargs.get("compression", None),
            )
            indices = f[id_path]
            indices[()] = ids

    def write_df(self, f, path, df, **kwargs):
        values_path = path + "/values"
        if path not in f:
            max_ncells = 2e5
            max_tps = 1e3
            f.create_dataset(
                name=values_path,
                shape=df.shape,
                # chunks=(min(df.shape[0], 1), df.shape[1]),
                # dtype=df.dtypes.iloc[0], This is making NaN in ints into negative vals
                dtype="float",
                maxshape=(max_ncells, max_tps),
                compression=kwargs.get("compression", None),
            )
            dset = f[values_path]
            dset[()] = df.values

            for name in df.index.names:
                indices_path = "/".join((path, name))
                f.create_dataset(
                    name=indices_path,
                    shape=(len(df),),
                    dtype="uint16",  # Assuming we'll always use int indices
                    chunks=True,
                    maxshape=(max_ncells,),
                )
                dset = f[indices_path]
                dset[()] = df.index.get_level_values(level=name).tolist()

            tp_path = path + "/timepoint"
            f.create_dataset(
                name=tp_path,
                shape=(df.shape[1],),
                maxshape=(max_tps,),
                dtype="uint16",
            )
            tps = df.columns.tolist()
            f[tp_path][tps] = tps

        else:
            dset = f[values_path]

            # Filter out repeated timepoints
            new_tps = set(df.columns).difference(f[path + "/timepoint"][()])
            if len(new_tps) != df.shape[1]:
                print(
                    "Attempting to re-extract existing timepoints. Skipping duplicated"
                )
                df = df[new_tps]

            if (
                not hasattr(self, "id_cache") or not df.index.nlevels in self.id_cache
            ):  # Use cache dict to store previously-obtained indices
                self.id_cache[df.index.nlevels] = {}
                existing_ids = self.get_existing_ids(
                    f, [path + "/" + x for x in df.index.names]
                )
                # Split indices in existing and additional
                new = df.index.tolist()
                if df.index.nlevels == 1:  # Cover for cases with a single index
                    new = [(x,) for x in df.index.tolist()]
                (
                    found_multis,
                    self.id_cache[df.index.nlevels]["additional_multis"],
                ) = self.find_ids(
                    existing=existing_ids,
                    new=new,
                )
                found_indices = np.array(locate_indices(existing_ids, found_multis))

                # we sort the existing indices for some pandas indexing I can't remember
                incremental_existing = np.argsort(found_indices)
                self.id_cache[df.index.nlevels]["found_indices"] = found_indices[
                    incremental_existing
                ]
                self.id_cache[df.index.nlevels]["found_multi"] = found_multis[
                    incremental_existing
                ]

            existing_values = df.loc[
                [
                    _tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels]["found_multi"]
                ]
            ].values
            new_values = df.loc[
                [
                    _tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels]["additional_multis"]
                ]
            ].values
            ncells, ntps = f[values_path].shape

            # Add found cells
            dset.resize(dset.shape[1] + df.shape[1], axis=1)
            dset[:, ntps:] = np.nan
            for i, tp in enumerate(df.columns):
                dset[
                    self.id_cache[df.index.nlevels]["found_indices"], tp
                ] = existing_values[:, i]
            # Add new cells
            n_newcells = len(self.id_cache[df.index.nlevels]["additional_multis"])
            dset.resize(dset.shape[0] + n_newcells, axis=0)
            dset[ncells:, :] = np.nan

            for i, tp in enumerate(df.columns):
                dset[ncells:, tp] = new_values[:, i]

            # save indices
            for i, name in enumerate(df.index.names):
                tmp = path + "/" + name
                dset = f[tmp]
                n = dset.shape[0]
                dset.resize(n + n_newcells, axis=0)
                dset[n:] = (
                    self.id_cache[df.index.nlevels]["additional_multis"][:, i]
                    if len(self.id_cache[df.index.nlevels]["additional_multis"].shape)
                    > 1
                    else self.id_cache[df.index.nlevels]["additional_multis"]
                )

            tmp = path + "/timepoint"
            dset = f[tmp]
            n = dset.shape[0]
            dset.resize(n + df.shape[1], axis=0)
            dset[n:] = df.columns.tolist()

    @staticmethod
    def get_existing_ids(f, paths):
        # Fetch indices and convert them to a (nentries, nlevels) ndarray
        return np.array([f[path][()] for path in paths]).T

    @staticmethod
    def find_ids(existing, new):
        # Compare two tuple sets and return the intersection and difference
        # (elements in the 'new' set not in 'existing')
        set_existing = set([tuple(*x) for x in zip(existing.tolist())])
        existing_cells = np.array(list(set_existing.intersection(new)))
        new_cells = np.array(list(set(new).difference(set_existing)))
        # for i in range(new_cells.shape[1] - 1, -1, -1):
        #     new_cells = new_cells[new_cells[:, i].argsort(kind="mergesort")]

        return (
            existing_cells,
            new_cells,
        )


# @staticmethod
def locate_indices(existing, new):
    if new.any():
        if new.shape[1] > 1:
            return [
                find_1st(
                    (existing[:, 0] == n[0]) & (existing[:, 1] == n[1]), True, cmp_equal
                )
                for n in new
            ]
        else:
            return [find_1st(existing[:, 0] == n, True, cmp_equal) for n in new]
    else:
        return []


# def tuple_or_int(x):
#     if isinstance(x, Iterable):
#         return tuple(x)
#     else:
#         return x
def _tuple_or_int(x):
    # Convert tuple to int if it only contains one value
    if len(x) == 1:
        return x[0]
    else:
        return x
