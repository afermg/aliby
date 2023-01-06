#!/usr/bin/env python3
"""
Dataset is a group of classes to manage multiple types of experiments:
 - Remote experiments on an OMERO server
 - Local experiments in a multidimensional OME-TIFF image containing the metadata
 - Local experiments in a directory containing multiple positions in independent images with or without metadata
"""
import os
import shutil
import time
import typing as t
from abc import ABC, abstractproperty, abstractmethod
from pathlib import Path, PosixPath
from typing import Union

import omero

from agora.io.bridge import BridgeH5
from aliby.io.image import ImageLocalOME
from aliby.io.omero import BridgeOmero


class DatasetLocalABC(ABC):
    """
    Abstract Base class to fetch local files, either OME-XML or raw images.
    """

    _valid_suffixes = ("tiff", "png")
    _valid_meta_suffixes = ("txt", "log")

    def __init__(self, dpath: Union[str, PosixPath], *args, **kwargs):
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
        # Return location of images and their unique names
        pass


class DatasetLocalDir(DatasetLocalABC):
    """
    Organise an entire dataset, composed of multiple images, as a directory containing directories with individual files.
    It relies on ImageDir to manage images.
    """

    def __init__(self, dpath: Union[str, PosixPath], *args, **kwargs):
        super().__init__(dpath)

    @property
    def date(self):
        # Use folder creation date, for cases where metadata is minimal
        return time.strftime(
            "%Y%m%d", time.strptime(time.ctime(os.path.getmtime(self.path)))
        )

    def get_images(self):
        return [
            folder
            for folder in self.path.glob("*/")
            if any(
                path
                for suffix in self._valid_meta_suffixes
                for path in folder.glob(f"*.{suffix}")
            )
        ]


class DatasetLocalOME(DatasetLocalABC):
    """Load a dataset from a folder

    We use a given image of a dataset to obtain the metadata,
    as we cannot expect folders to contain this information.

    It uses the standard OME-TIFF file format.
    """

    def __init__(self, dpath: Union[str, PosixPath], *args, **kwargs):
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


class Dataset(BridgeOmero):
    def __init__(self, expt_id, **server_info):
        self.ome_id = expt_id

        super().__init__(**server_info)

    @property
    def name(self):
        return self.ome_class.getName()

    @property
    def date(self):
        return self.ome_class.getDate()

    @property
    def unique_name(self):
        return "_".join(
            (
                str(self.ome_id),
                self.date.strftime("%Y_%m_%d").replace("/", "_"),
                self.name,
            )
        )

    def get_images(self):
        return {
            im.getName(): im.getId() for im in self.ome_class.listChildren()
        }

    @property
    def files(self):
        if not hasattr(self, "_files"):
            self._files = {
                x.getFileName(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        if not len(self._files):
            raise Exception(
                "exception:metadata: experiment has no annotation files."
            )
        elif len(self.file_annotations) != len(self._files):
            raise Exception("Number of files and annotations do not match")

        return self._files

    @property
    def tags(self):
        if self._tags is None:
            self._tags = {
                x.getname(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.TagAnnotationWrapper)
            }
        return self._tags

    def cache_logs(self, root_dir):
        valid_suffixes = ("txt", "log")
        for name, annotation in self.files.items():
            filepath = root_dir / annotation.getFileName().replace("/", "_")
            if (
                any([str(filepath).endswith(suff) for suff in valid_suffixes])
                and not filepath.exists()
            ):
                # save only the text files
                with open(str(filepath), "wb") as fd:
                    for chunk in annotation.getFileInChunks():
                        fd.write(chunk)
        return True

    @classmethod
    def from_h5(
        cls,
        filepath: t.Union[str, PosixPath],
    ):
        """Instatiate Dataset from a hdf5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, PosixPath]
            Location of hdf5 file.

        Examples
        --------
        FIXME: Add docs.

        """
        # metadata = load_attributes(filepath)
        bridge = BridgeH5(filepath)
        dataset_keys = ("omero_id", "omero_id,", "dataset_id")
        for k in dataset_keys:
            if k in bridge.meta_h5:
                return cls(
                    bridge.meta_h5[k], **cls.server_info_from_h5(filepath)
                )
