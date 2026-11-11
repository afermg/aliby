"""
Dataset is a group of classes to manage multiple types of experiments.

Includes
 - Remote experiments on an OMERO server (located in src/aliby/io/omero.py)
 - Local experiments in a multidimensional OME-TIFF image containing the
    metadata
 - Local experiments in a directory containing multiple positions in
    independent images with or without metadata
"""

import logging
import os
import shutil
import time
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from aliby.io.image import ImageLocalOME


def dispatch_dataset(expt_id: int or str, **kwargs):
    """
    Find paths to the data.

    Connect to OMERO if data is available remotely.

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
        # data available from OMERO; omero is optional, so import it here
        from aliby.io.omero import Dataset

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
        """
        Return a dictionary with any available metadata files.

        A metadata file is a file, not a folder, whose suffix is one of
        the valid ones. Look in the data set's folder and in each folder
        within it, where a position's logs may be, but no deeper and never
        in a zarr store: walking every folder listed every chunk of every
        store.
        """
        if not hasattr(self, "_files"):
            found = self.metadata_files_in(self.path)
            for entry in os.scandir(self.path):
                if entry.is_dir() and not entry.name.endswith(".zarr"):
                    found += self.metadata_files_in(entry.path)
            self._files = {f: f for f in sorted(found)}
        return self._files

    def metadata_files_in(self, folder: str | Path) -> list[Path]:
        """Return the metadata files that lie directly in a folder."""
        return [
            Path(entry.path)
            for entry in os.scandir(folder)
            if Path(entry.name).suffix[1:] in self._valid_meta_suffixes
            and entry.is_file()
        ]

    def cache_logs(self, root_dir):
        """
        Copy metadata files to results folder.

        The copies share one folder. Of files with one name, copy the one
        nearest the top of the data set, which is where the microscope's
        logs are looked for, and log a warning.
        """
        chosen = {}
        for file in sorted(self.files, key=lambda f: (len(f.parts), f)):
            if file.name in chosen:
                logging.getLogger("aliby").warning(
                    f"More than one metadata file is called {file.name}; "
                    f"keeping {chosen[file.name]} and not {file}."
                )
            else:
                chosen[file.name] = file
        for name, file in chosen.items():
            shutil.copy(file, root_dir / name)
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
        # sorted, so that a position's number does not depend on the
        # order the file system lists folders in
        position_ids_dict = {
            item.name: item
            for item in sorted(self.path.glob("*/"))
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
        files = sorted(
            f
            for suffix in self._valid_suffixes
            for f in self.path.glob(f"*.{suffix}")
        )
        return {f.name: str(f) for f in files}
