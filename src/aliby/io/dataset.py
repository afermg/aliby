#!/usr/bin/env python3
"""
Dataset is a group of classes to manage multiple types of experiments:
 - Remote experiments on an OMERO server (located in src/aliby/io/omero.py)
 - Local experiments in a multidimensional OME-TIFF image containing the metadata
 - Local experiments in a directory containing multiple positions in independent
images with or without metadata
"""
import os
import shutil
import time
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from itertools import groupby
from pathlib import Path

from aliby.io.image import ImageLocalOME
from aliby.io.omero import Dataset


def dispatch_dataset(expt_id: int or str, **kwargs):
    """
    Find paths to the data.

    Connect to OMERO if data is remotely available.

    Parameters
    ----------
    expt_id: int or str
        To identify the data, either an OMERO ID or an OME-TIFF file
        or a local directory.

    Returns
    -------
    A callable Dataset instance, either network-dependent or local.
    """
    if isinstance(expt_id, int):
        # data available online
        return Dataset(expt_id, **kwargs)
    elif isinstance(expt_id, str):
        # data available locally
        expt_path = Path(expt_id)
        if expt_path.is_dir():
            # data in multiple folders, such as zarr
            return DatasetLocalDir(expt_path)
        else:
            # data in one folder as OME-TIFF files
            return DatasetLocalOME(expt_path)
    else:
        raise Warning(f"{expt_id} is an invalid expt_id.")


class DatasetLocalABC(ABC):
    """
    Abstract Base class to find local files, either OME-XML or raw images.
    """

    _valid_suffixes = ("tiff", "png", "zarr", "tif")
    _valid_meta_suffixes = ("txt", "log")

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        self.path = Path(dpath)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def dataset(self):
        return self.path

    @property
    def name(self):
        return self.path.name

    @property
    def unique_name(self):
        return self.path.name

    @property
    def files(self):
        """Return a dictionary with any available metadata files."""
        if not hasattr(self, "_files"):
            self._files = {
                f: f
                for f in self.path.rglob("*")
                if any(
                    str(f).endswith(suffix)
                    for suffix in self._valid_meta_suffixes
                )
            }
        return self._files

    def cache_logs(self, root_dir):
        """Copy metadata files to results folder."""
        for name, annotation in self.files.items():
            shutil.copy(annotation, root_dir / name.name)
        return True

    @abstractproperty
    def date(self):
        pass

    @abstractmethod
    def get_position_ids(self):
        pass


class DatasetLocalDir(DatasetLocalABC):
    """Find paths to a data set, comprising multiple images in different folders."""

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        super().__init__(dpath)

    @property
    def date(self):
        """Find date when a folder was created."""
        return time.strftime(
            "%Y%m%d", time.strptime(time.ctime(os.path.getmtime(self.path)))
        )

    def get_position_ids(self):
        """
        Return a dict of file paths for each position.

        FUTURE 3.12 use pathlib is_junction to pick Dir or File
        """
        position_ids_dict = {
            item.name: item
            for item in self.path.glob("*/")
            if item.is_dir()
            and any(
                path
                for suffix in self._valid_suffixes
                for path in item.glob(f"*.{suffix}")
            )
            or item.suffix[1:] in self._valid_suffixes
        }
        return position_ids_dict


class DatasetLocalOME(DatasetLocalABC):
    """Find names of images in a folder, assuming images in OME-TIFF format."""

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        super().__init__(dpath)
        assert len(
            self.get_position_ids()
        ), f"No valid files found. Formats are {self._valid_suffixes}"

    @property
    def date(self):
        """Get the date from the metadata of the first position."""
        return ImageLocalOME(list(self.get_position_ids().values())[0]).date

    def get_position_ids(self):
        """Return a dictionary with the names of the image files."""
        return {
            f.name: str(f)
            for suffix in self._valid_suffixes
            for f in self.path.glob(f"*.{suffix}")
        }

class DatasetLocalDir(DatasetLocalABC):
    """Find paths to a data set, comprising multiple images in different folders."""

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        super().__init__(dpath)

    @property
    def date(self):
        """Find date when a folder was created."""
        return time.strftime(
            "%Y%m%d", time.strptime(time.ctime(os.path.getmtime(self.path)))
        )

    def get_position_ids(self) -> dict[str, list[str]]:
        """
        Return a dict of a list for filepaths that define each position.

        """

        position_ids_dict = {
            item.name: item
            for item in self.path.glob("*/")
            if item.is_dir()
            and any(
                path
                for suffix in self._valid_suffixes
                for path in item.glob(f"*.{suffix}")
            )
            or item.suffix[1:] in self._valid_suffixes
        }
        return position_ids_dict

def groupby_regex(dpath:str, regex:str or None = None) -> groupby:
    """
    Use a regex to group filenames of the same field-of-view (or, in
    some cases, well+field-of-view)
    """
    if regex is None:
        regex = re.compile(".+\/(.+)\/_.+([A-P][0-9]{2}).*_T([0-9]{4})F([0-9]{3}).*Z([0-9]{2}).*[0-9].tif")
    sorted_paths = list( map(str, sorted(dpath.rglob("*.tif") ) ) )

    output = dict( zip(sorted_paths, (map(lambda x: regex.findall(x), sorted_paths))) )
    valid = {k:v[0] for k,v in output.items() if len(v)} 
    iterator = groupby(valid.items(), lambda x: (x[1][1],x[1][3]))
    return iterator

def get_fov_id(dpath) -> dict[str,list[str]]:
    iterator = groupby_regex(dpath)
    d = {key: [x for x in group] for key, group in iterator}

    return {k:v[0] for k,v in d.items()}

def get_dimensions(dpath)-> list[int]:
    iterator = groupby_regex(dpath)
    d = {key: [x for x in group] for key, group in iterator}
    pass
