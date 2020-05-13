"""
Utility functions and classes
"""
from typing import Callable

import imageio

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
        obj.getAnnotation())

    return string


# TODO check functools.lru_cache for this purpose
class Cache:
    """
    Fixed-length mapping to use as a cache.
    Deletes items in FIFO manner when maximum allowed length is reached.
    """
    def __init__(self, max_len=200, load_fn: Callable = imageio.imread):
        """
        :param max_len: Maximum number of items in the cache.
        :param load_fn: The function used to load new items if they are not
        available in the Cache
        """
        self._dict = dict()
        self._queue = []
        self.load_fn = load_fn
        self.max_len=max_len

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

