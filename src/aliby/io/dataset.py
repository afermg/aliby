#!/usr/bin/env python3
"""
Dataset is a group of classes to manage multiple types of experiments:
 - Remote experiments on an OMERO server (located in src/aliby/io/omero.py)
 - Local experiments in a multidimensional OME-TIFF image containing the metadata
 - Local experiments in a directory containing multiple positions in independent
images with or without metadata
"""
import os
import re
import shutil
import time
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from itertools import groupby
from pathlib import Path

from aliby.io.image import ImageLocalOME


def dispatch_dataset(expt_id: int or str, custom:str or None = None, **kwargs):
    """
    Find paths to the data.

    Connect to OMERO if data is remotely available.

    Parameters
    ----------
    expt_id: int or str
        To identify the data, either an OMERO ID or an OME-TIFF file
        or a local directory.
    custom: str or None
        Determines whether to use a Omero-based structure or a custom one.

    Returns
    -------
    A callable Dataset instance, either network-dependent or local.
    """
    if not custom:
        if isinstance(expt_id, int):
            from aliby.io.omero import Dataset
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
            # Data requires a special transformation (e.g., an unusual single-file-structure)
        else:
            return DatasetIndFiles(expt_path, **kwargs)



            
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

    @property
    def date(self):
        """Find date when a folder was created."""
        return time.strftime(
            "%Y%m%d", time.strptime(time.ctime(os.path.getmtime(self.path)))
        )

    @abstractmethod
    def get_position_ids(self):
        pass


class DatasetLocalDir(DatasetLocalABC):
    """Find paths to a data set, comprising multiple images in different folders."""

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        super().__init__(dpath)

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

    def get_position_ids(self) -> dict[str,str]:
        """Return a dictionary with the names of the image files."""
        return {
            f.name: str(f)
            for suffix in self._valid_suffixes
            for f in self.path.glob(f"*.{suffix}")
        }

class DatasetIndFiles(DatasetLocalABC):
    """
    Find paths to a data set, comprising of individual files. The
    data can be parsed using a regex and a string to capture the order of dimension.
    """

    def __init__(self, dpath: t.Union[str, Path], regex:str=None, re_dimorder:str=None, *args, **kwargs):
        if regex is None:
            self.regex =  ".+\/(.+)\/_.+([A-P][0-9]{2}).*_T([0-9]{4})F([0-9]{3}).*Z([0-9]{2}).*[0-9].tif"
        if re_dimorder is None:
            self.dimorder = "CFTZ"
        super().__init__(dpath)


    def get_position_ids(self) -> dict[str, list[str]]:
        """
        Return a dict of a list for filepaths that define each position sorted alphabetically.
        The key is the name of the position/field-of-view and the value is
        a list of stings indicated the associated files.
        """

        d = groupby_regex(self.path, self.regex)

        return d


@cache
def groupby_regex(path:str, regex:str, capture_group_indices:tuple[int]=(2,3)) -> dict[str,list[str]]:
    """
    Use a regex to group filenames of the same field-of-view (or, in
    some cases, well+field-of-view)
    """
    regex = re.compile(regex)
    str_paths =  map(str, sorted(Path(path).rglob("*.tif") ) ) 
    captures = list(map(lambda x: regex.findall(x), str_paths))
    valid = [(pth, *capture[0]) for pth,capture in zip(str_paths, captures) if len(capture)]
    key_fn = lambda x: tuple(x[i] for i in capture_group_indices)
    sorted_by = sorted(valid, key=key_fn)
    iterator = groupby(sorted_by, key=key_fn)
    d = { key: [x[0] for x in group] for key, group in iterator}
    return {k:v for k,v in d.items()}
