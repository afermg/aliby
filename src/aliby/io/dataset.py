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
from pathlib import Path


def dispatch_dataset(expt_id: int or str, is_zarr: bool = True, **kwargs):
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
        if is_zarr == True:  # data in multiple folders, such as zarr
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


class DatasetZarr(DatasetLocalABC):
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

    def get_position_ids(self, regex: str = None, capture_order: str = None):
        regex = regex or self.regex
        capture_order = capture_order or self.capture_order

        return _get_position_ids(self.path, regex, capture_order)


def _get_position_ids(
    path: str, regex: str, capture_order: str
) -> dict[str, list[str]]:
    """
    Return a dict of a list for filepaths that define each position sorted alphabetically.
    The key is the name of the position/field-of-view and the value is
    a list of stings indicated the associated files.
    """

    captured_indices = sorted(
        capture_order.index(x) for x in set("WF").intersection(capture_order)
    )  # Indices of groups to replace with wildcard
    assert len(captured_indices), (
        "capture_order is missing Wells or field-of-view indicator"
    )
    # Sort by time, channel and z-stack
    sort_files_by = tuple(
        capture_order.index(x) for x in [x for x in capture_order if x in "TCZ"]
    )
    sorted_groups = organize_by_regex(
        path, regex, tuple(captured_indices), sort_files_by
    )

    assert len(sorted_groups), "No files were found."

    return sorted_groups


# @cache
def organize_by_regex(
    path: str,
    regex: str,
    group_by_capture: tuple[int],
    sort_by_capture: tuple[int],
) -> dict[str, list[str]]:
    """
    Use a regex to group filenames of the same field-of-view (or, in
    some cases, well+field-of-view).

    It returns the a dictionary where the key is a combination of well+field of view and the value
    is a list with the remaining dimensions (e.g., time point, channel, z-stack) sorted in the order defined by -dimorder_out-.
    """
    regex = re.compile(regex)
    str_paths = list(map(str, sorted(Path(path).rglob("*.tif"))))
    captures = list(map(lambda x: regex.match(x), str_paths))
    valid = [
        (*capture.groups(), pth) for pth, capture in zip(str_paths, captures) if capture
    ]

    def key_fn(x):
        return tuple(x[i] for i in group_by_capture)

    sorted_keys = sorted(valid, key=key_fn)
    iterator = groupby(sorted_keys, key=key_fn)

    sorted_groups = {}
    for key, group in iterator:
        files = [x[-1] for x in group]
        sorted_groups[key] = sort_by_regex_groups(files, regex, sort_by_capture)
    return sorted_groups


def sort_by_regex_groups(
    files: tuple[str], regex: re.Pattern, sort_by_capture: tuple[int]
):
    """
    Sort groups of files based on a given regex. It assumes that they are have the same length
    and format, and sorts based on the captured sections with indexes defined in :sort_by: (for example, if :sort_by=(3,0): the lists are sorted based on the third and first capture group, in that order).
    """
    spans = regex.match(files[0]).regs[1:]
    sorted_files = sorted(
        files, key=lambda x: [x[slice(*spans[i])] for i in sort_by_capture]
    )
    return sorted_files
