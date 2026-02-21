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
from abc import ABC, abstractmethod
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from tqdm import tqdm


def dispatch_dataset(
    expt_id: int or str, is_zarr: bool = False, is_monozarr: bool = False, **kwargs
):
    """
    Find paths to the data.

    Connect to OMERO if data is remotely available.

    Parameters
    ----------
    expt_id: int or str
        To identify the data, either an OMERO ID or an OME-TIFF file
        or a local directory.
    zarr: str or None
        Determines whether to use a zarr

    Returns
    -------
    A callable Dataset instance, either network-dependent or local.
    """
    if isinstance(expt_id, int):
        from aliby.io.omero import Dataset

        # data available on an Omero server
        return Dataset(expt_id, **kwargs)
    elif isinstance(expt_id, (str, Path)):
        # data available locally
        expt_path = Path(expt_id)
        assert expt_path.exists(), f"Experiment path does not exist: {expt_path}"
        if is_zarr is True:  # data in multiple folders, such as zarr
            if is_monozarr:
                return DatasetMonoZarr(expt_path, **kwargs)
            else:
                return DatasetZarr(expt_path, **kwargs)
        else:  # It is a directory containing all images inside (possibly nested)
            return DatasetDir(expt_path, **kwargs)

        raise Exception(f"Cannot dispatch dataset.. Invalid input path {expt_path}.")
    raise Exception(
        "Invalid experiment id, it must be the id of an omero server or a Path"
    )


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
                if any(str(f).endswith(suffix) for suffix in self._valid_meta_suffixes)
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


class DatasetMonoZarr(DatasetLocalABC):
    """The images are arrays at the root level of a zarr file."""

    def __init__(self, dpath: t.Union[str, Path], *args, **kwargs):
        super().__init__(dpath)

    def get_position_ids(self):
        positions = {}
        with os.scandir(self.path) as it:
            for entry in it:
                # skip hidden folders and files (e.g., .zattrs)
                if entry.is_dir():
                    name = entry.name
                    positions[name] = {"store_path": self.path, "key": name}

        # result = {k:  for k in tqdm(position_ids)}
        return positions


class DatasetZarr(DatasetLocalABC):
    """Find paths to a data set, comprising multiple images in different folders.

    Each position (or site) is one zarr file.
    """

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


class DatasetDir(DatasetLocalABC):
    """
    Find paths to a data set, comprising of individual files. The
    data can be parsed using a regex and a string to capture the order of dimension.
    """

    def __init__(
        self,
        dpath: t.Union[str, Path],
        regex: str,
        capture_order: str,
    ):
        """
                capture_order: determines the order of the regular expression
        .
                - C: Channel
                - W: Well (optional)
                - T: Time point
                - F: Field-of-view (also named position)
                - Z: Z-stack
        """
        super().__init__(dpath)
        self.regex = regex
        self.capture_order = capture_order

    def get_position_ids(
        self, regex: str = None, capture_order: str = None
    ) -> list[dict[str, tuple]]:
        regex = regex or self.regex
        capture_order = capture_order or self.capture_order

        return sort_groups_by_regex(self.path, regex, capture_order)


def sort_groups_by_regex(
    datasets_path: str, regex: str, capture_order, out_dimorder: str = "TCZYX"
) -> dict[str, str | tuple | list]:
    regex_ = re.compile(regex)
    str_paths = scan_directory(datasets_path)

    print("Capturing regex")
    captures = map(lambda x: regex_.match(x), str_paths)
    valid = [
        (*capture.groups(), pth) for pth, capture in zip(str_paths, captures) if capture
    ]

    # sort groups to have all equal values together
    # Keys that define a "site"
    grouper_keys = [
        capture_order.index(x) + 1 for x in capture_order if x not in out_dimorder
    ]
    # Keys that define dimensions in an individual image. Sorted based on dimnames
    dim_keys = tuple(
        capture_order.index(x) + 1
        for x in [y for y in out_dimorder if y in capture_order]
    )
    sorted_keys = multisort(valid, [*grouper_keys, *dim_keys])

    # %%
    iterator = groupby(sorted_keys, key=lambda x: [x[i - 1] for i in grouper_keys])

    position_ids = []
    for key, group in tqdm(iterator, desc="Grouping items"):
        # files are presorted
        files = [x[-1] for x in group]
        position_ids.append({
            "key": key,
            "path": [str(Path(datasets_path) / file) for file in files],
        })

    assert len(position_ids), "No files were found."

    # Returns [{key: (well, site), path: [file1, file2]}]
    return position_ids


def scan_directory(path: str, id_type: str = "name") -> list[str]:
    """Fast directory scanning."""
    paths = []
    with os.scandir(path) as it:
        for entry in tqdm(it, desc="Reading files"):
            if not entry.name.startswith("."):
                path_or_name = getattr(entry, id_type)
                paths.append(path_or_name)

    return paths


def multisort(xs, specs):
    for key in tqdm(specs, desc="Sorting keys"):
        xs.sort(key=itemgetter(key))

    return xs
