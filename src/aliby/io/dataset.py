#!/usr/bin/env python3
"""
Dataset is a group of classes to manage multiple types of experiments:
 - Remote experiments on an OMERO server (located in src/aliby/io/omero.py)
 - Local experiments in a multidimensional OME-TIFF image containing the metadata
 - Local experiments in a directory containing multiple positions in independent images with or without metadata
"""
import os
import shutil
import time
import typing as t
from abc import ABC, abstractproperty, abstractmethod
from pathlib import Path, PosixPath


from aliby.io.image import ImageLocalOME


def dispatch_dataset(expt_id: int or str, **kwargs):
    """
    Choose a subtype of dataset based on the identifier.

    Input:
    --------
    expt_id: int or string serving as dataset identifier.

    Returns:
    --------
    Callable Dataset instance, either network-dependent or local.
    """
    if isinstance(expt_id, int):  # Is an experiment online

        from aliby.io.omero import Dataset

        return Dataset(expt_id, **kwargs)

    elif isinstance(expt_id, str):  # Files or Dir
        expt_path = Path(expt_id)
        if expt_path.is_dir():
            return DatasetLocalDir(expt_path)
        else:
            return DatasetLocalOME(expt_path)
    else:
        raise Warning("Invalid expt_id")


class DatasetLocalABC(ABC):
    """
    Abstract Base class to fetch local files, either OME-XML or raw images.
    """

    _valid_suffixes = ("tiff", "png")
    _valid_meta_suffixes = ("txt", "log")

    def __init__(self, dpath: t.Union[str, PosixPath], *args, **kwargs):
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

    @abstractproperty
    def date(self):
        pass

    @property
    def files(self):
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
        # Copy metadata files to results folder
        for name, annotation in self.files.items():
            shutil.copy(annotation, root_dir / name.name)
        return True

    @abstractmethod
    def get_images(self):
        # Return a dictionary with the name of images and their unique identifiers
        pass


class DatasetLocalDir(DatasetLocalABC):
    """
    Organise an entire dataset, composed of multiple images, as a directory containing directories with individual files.
    It relies on ImageDir to manage images.
    """

    def __init__(self, dpath: t.Union[str, PosixPath], *args, **kwargs):
        super().__init__(dpath)

    @property
    def date(self):
        # Use folder creation date, for cases where metadata is minimal
        return time.strftime(
            "%Y%m%d", time.strptime(time.ctime(os.path.getmtime(self.path)))
        )

    def get_images(self):
        return {
            folder.name: folder
            for folder in self.path.glob("*/")
            if any(
                path
                for suffix in self._valid_suffixes
                for path in folder.glob(f"*.{suffix}")
            )
        }


class DatasetLocalOME(DatasetLocalABC):
    """Load a dataset from a folder

    We use a given image of a dataset to obtain the metadata,
    as we cannot expect folders to contain this information.

    It uses the standard OME-TIFF file format.
    """

    def __init__(self, dpath: t.Union[str, PosixPath], *args, **kwargs):
        super().__init__(dpath)
        assert len(self.get_images()), "No .tiff files found"

    @property
    def date(self):
        # Access the date from the metadata of the first position
        return ImageLocalOME(list(self.get_images().values())[0]).date

    def get_images(self):
        # Fetches all valid formats and overwrites if duplicates with different suffix
        return {
            f.name: str(f)
            for suffix in self._valid_suffixes
            for f in self.path.glob(f"*.{suffix}")
        }
