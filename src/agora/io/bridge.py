"""
Tools to interact with h5 files.
"""

import logging
import typing as t
from itertools import groupby
from typing import Union

import h5py
import numpy as np


class BridgeH5:
    """
    Base class to interact with h5 files.
    """

    def __init__(self, filename, flag="r"):
        """Initialise with the name of the h5 file."""
        self.filename = filename
        if flag is not None:
            self.hdf = h5py.File(filename, flag)
            assert (
                "cell_info" in self.hdf
            ), "Invalid file. No 'cell_info' found."

    def log(self, message: str, level: str = "warning"):
        """Log messages at the desired level."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    def close(self):
        """Close the h5 file."""
        self.hdf.close()

    @property
    def meta_h5(self) -> t.Dict[str, t.Any]:
        """Return metadata, defining it if necessary."""
        if not hasattr(self, "_meta_h5"):
            with h5py.File(self.filename, "r") as f:
                # load metadata from h5; ensure arrays are lists
                self._meta_h5 = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in f.attrs.items()
                }
        return self._meta_h5

def groupsort(iterable: Union[tuple, list]):
    """
    Sort iterable and return a dict.

    Values are grouped by the first element.
    """
    iterable = sorted(iterable, key=lambda x: x[0])
    grouped = {
        k: [x[1:] for x in v] for k, v in groupby(iterable, lambda x: x[0])
    }
    return grouped
