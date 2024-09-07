"""Writer for Extractor and Postprocessor."""

import logging
from collections.abc import Iterable
from typing import Dict

import aliby.global_settings as global_settings
import h5py
import numpy as np
import pandas as pd
from agora.io.bridge import BridgeH5
from utils_find_1st import cmp_equal, find_1st


class Writer(BridgeH5):
    """
    Class to transform data into compatible structures.

    Use with Extractor and Postprocessor.
    """

    def __init__(self, filename, flag=None, compression="gzip"):
        """
        Initialise write.

        Parameters
        ----------
        filename: str
            Name of file to write into
        flag: str, default=None
            Flag to pass to the default file reader.
            If None the file remains closed.
        compression: str, default="gzip"
            Compression method passed on to h5py writing functions
            (only used for dataframes and other array-like data).
        """
        super().__init__(filename, flag=flag)
        self.compression = compression

    def write(
        self,
        path: str,
        data: Iterable = None,
        meta: dict = {},
        overwrite: str = None,
    ):
        """Write data and metadata to a path in the h5 file."""
        self.id_cache = {}
        with h5py.File(self.filename, "a") as f:
            if overwrite == "overwrite":  # TODO refactor overwriting
                if path in f:
                    del f[path]
            logging.debug(
                f"{overwrite} {type(data)} to {path} and {len(meta)} metadata fields."
            )
            # write data
            if data is not None:
                self.write_dset(f, path, data)
            # write metadata
            if meta:
                for attr, metadata in meta.items():
                    self.write_meta(f, path, attr, data=metadata)

    def write_dset(self, f: h5py.File, path: str, data: Iterable):
        """Write data to the h5 file."""
        # data is a datafram
        if isinstance(data, pd.DataFrame):
            self.write_dataframe(f, path, data, compression=self.compression)
        # data is a multi-index dataframe
        elif isinstance(data, pd.MultiIndex):
            # TODO: benchmark I/O speed when using compression
            self.write_index(f, path, data)  # , compression=self.compression)
        # data is a dictionary of dataframes
        elif isinstance(data, Dict) and np.all(
            [isinstance(x, pd.DataFrame) for x in data.values]
        ):
            for k, df in data.items():
                self.write_dset(f, path + f"/{k}", df)
        # data is an iterable
        elif isinstance(data, Iterable):
            self.write_arraylike(f, path, data)
        # data is a float or integer
        else:
            self.write_atomic(data, f, path)

    def write_meta(self, f: h5py.File, path: str, attr: str, data: Iterable):
        """Write metadata to an open h5 file."""
        obj = f.require_group(path)
        if type(data) is dict:
            # necessary for channels_dict from find_channels_by_position
            for key, vlist in data.items():
                obj.attrs[attr + key] = vlist
        else:
            obj.attrs[attr] = data

    @staticmethod
    def write_arraylike(f: h5py.File, path: str, data: Iterable, **kwargs):
        """Write an iterable."""
        if path in f:
            del f[path]
        narray = np.array(data)
        if narray.any():
            chunks = (1, *narray.shape[1:])
        else:
            chunks = None
        # create dset
        dset = f.create_dataset(
            path,
            shape=narray.shape,
            chunks=chunks,
            dtype="int",
            compression=kwargs.get("compression", None),
        )
        # add data to dset
        dset[()] = narray

    @staticmethod
    def write_index(f, path, pd_index, **kwargs):
        """Write a multi-index dataframe."""
        f.require_group(path)  # TODO check if we can remove this
        for i, name in enumerate(pd_index.names):
            ids = pd_index.get_level_values(i)
            id_path = path + "/" + name
            f.create_dataset(
                name=id_path,
                shape=(len(ids),),
                dtype="uint16",
                compression=kwargs.get("compression", None),
            )
            indices = f[id_path]
            indices[()] = ids

    def write_dataframe(self, f, path, df, **kwargs):
        """Write a dataframe."""
        values_path = (
            path + "values" if path.endswith("/") else path + "/values"
        )
        if path not in f:
            # create dataset and write data
            max_ncells = global_settings.h5_max_ncells
            max_tps = global_settings.h5_max_tps
            f.create_dataset(
                name=values_path,
                shape=df.shape,
                dtype="float",
                maxshape=(max_ncells, max_tps),
                compression=kwargs.get("compression", None),
            )
            dset = f[values_path]
            dset[()] = df.values.astype("float16")
            # create dateset and write indices
            if not len(df):
                return None
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
                # create dataset and write time points as columns
                tp_path = path + "/timepoint"
                if tp_path not in f:
                    f.create_dataset(
                        name=tp_path,
                        shape=(df.shape[1],),
                        maxshape=(max_tps,),
                        dtype="uint16",
                    )
                    tps = list(range(df.shape[1]))
                    f[tp_path][tps] = tps
            else:
                f[path].attrs["columns"] = df.columns.tolist()
        else:
            # path exists
            dset = f[values_path]
            # filter out repeated timepoints
            new_tps = set(df.columns)
            if path + "/timepoint" in f:
                new_tps = new_tps.difference(f[path + "/timepoint"][()])
            df = df[list(new_tps)]
            if (
                not hasattr(self, "id_cache")
                or df.index.nlevels not in self.id_cache
            ):
                # use cache dict to store previously obtained indices
                self.id_cache[df.index.nlevels] = {}
                existing_ids = self.get_existing_ids(
                    f, [path + "/" + x for x in df.index.names]
                )
                # split indices in existing and additional
                new = df.index.tolist()
                if df.index.nlevels == 1:
                    # cover cases with a single index
                    new = [(x,) for x in df.index.tolist()]
                (
                    found_multis,
                    self.id_cache[df.index.nlevels]["additional_multis"],
                ) = self.find_ids(
                    existing=existing_ids,
                    new=new,
                )
                found_indices = np.array(
                    locate_indices(existing_ids, found_multis)
                )
                # sort indices for h5 indexing
                incremental_existing = np.argsort(found_indices)
                self.id_cache[df.index.nlevels]["found_indices"] = (
                    found_indices[incremental_existing]
                )
                self.id_cache[df.index.nlevels]["found_multi"] = found_multis[
                    incremental_existing
                ]
            existing_values = df.loc[
                [
                    tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels]["found_multi"]
                ]
            ].values
            new_values = df.loc[
                [
                    tuple_or_int(x)
                    for x in self.id_cache[df.index.nlevels][
                        "additional_multis"
                    ]
                ]
            ].values
            ncells, ntps = f[values_path].shape
            # add found cells
            dset.resize(dset.shape[1] + df.shape[1], axis=1)
            dset[:, ntps:] = np.nan
            # TODO refactor this indices sorting. Could be simpler
            found_indices_sorted = self.id_cache[df.index.nlevels][
                "found_indices"
            ]
            # case when all labels are new
            if found_indices_sorted.any():
                # h5py does not allow bidimensional indexing,
                # so we iterate over the columns
                for i, tp in enumerate(df.columns):
                    dset[found_indices_sorted, tp] = existing_values[:, i]
            # add new cells
            n_newcells = len(
                self.id_cache[df.index.nlevels]["additional_multis"]
            )
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
                    if len(
                        self.id_cache[df.index.nlevels][
                            "additional_multis"
                        ].shape
                    )
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
        """Fetch indices and convert them to a (nentries, nlevels) ndarray."""
        return np.array([f[path][()] for path in paths]).T

    @staticmethod
    def find_ids(existing, new):
        """Return the intersection and difference of two tuple sets."""
        set_existing = set([tuple(*x) for x in zip(existing.tolist())])
        existing_cells = np.array(list(set_existing.intersection(new)))
        new_cells = np.array(list(set(new).difference(set_existing)))
        return existing_cells, new_cells


def locate_indices(existing, new):
    """Find new indices in existing ones."""
    if new.any():
        if new.shape[1] > 1:
            return [
                find_1st(
                    (existing[:, 0] == n[0]) & (existing[:, 1] == n[1]),
                    True,
                    cmp_equal,
                )
                for n in new
            ]
        else:
            return [
                find_1st(existing[:, 0] == n, True, cmp_equal) for n in new
            ]
    else:
        return []


def tuple_or_int(x):
    """Convert tuple to int if it only contains one value."""
    if len(x) == 1:
        return x[0]
    else:
        return x
