"""
Utility functions and classes
"""
import itertools
import logging
import operator
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import h5py
import imageio
import numpy as np


def repr_obj(obj, indent=0):
    """
    Helper function to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    string = """%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getAnnotation(),
    )

    return string


def imread(path):
    return cv2.imread(str(path), -1)


class ImageCache:
    """HDF5-based image cache for faster loading of the images once they've
    been read.
    """

    def __init__(self, file, name, shape, remote_fn):
        self.store = h5py.File(file, "a")
        # Create a dataset
        self.dataset = self.store.create_dataset(
            name, shape, dtype=np.float, fill_value=np.nan
        )
        self.remote_fn = remote_fn

    def __getitem__(self, item):
        cached = self.dataset[item]
        if np.any(np.isnan(cached)):
            full = self.remote_fn(item)
            self.dataset[item] = full
            return full
        else:
            return cached


class Cache:
    """
    Fixed-length mapping to use as a cache.
    Deletes items in FIFO manner when maximum allowed length is reached.
    """

    def __init__(self, max_len=5000, load_fn: Callable = imread):
        """
        :param max_len: Maximum number of items in the cache.
        :param load_fn: The function used to load new items if they are not
        available in the Cache
        """
        self._dict = dict()
        self._queue = []
        self.load_fn = load_fn
        self.max_len = max_len

    def __getitem__(self, item):
        if item not in self._dict:
            self.load_item(item)
        return self._dict[item]

    def load_item(self, item):
        self._dict[item] = self.load_fn(item)
        # Clean up the queue
        self._queue.append(item)
        if len(self._queue) > self.max_len:
            del self._dict[self._queue.pop(0)]

    def clear(self):
        self._dict.clear()
        self._queue.clear()


def accumulate(lst: list):
    lst = sorted(lst)
    it = itertools.groupby(lst, operator.itemgetter(0))
    for key, sub_iter in it:
        yield key, [x[1] for x in sub_iter]


def get_store_path(save_dir, store, name):
    """Create a path to a position-specific store.

    This combines the name and the store's base name into a file path within save_dir.
    For example.
    >>> get_store_path('data', 'baby_seg.h5', 'pos001')
    Path(data/pos001baby_seg.h5')

    :param save_dir: The root directory in which to save the file, absolute
    path.
    :param store: The base name of the store
    :param name: The name of the position
    :return: Path(save_dir) / name+store
    """
    store = Path(save_dir) / store
    store = store.with_name(name + store.name)
    return store


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def timed(f, name=None):
    @wraps(f)
    def decorated(*args, **kwargs):
        t = perf_counter()
        res = f(*args, **kwargs)
        to_print = name or f.__name__
        logging.debug(f"Timing:{to_print}:{perf_counter() - t}s")
        return res

    return decorated
