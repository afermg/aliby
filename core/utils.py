"""
Utility functions and classes
"""
import itertools
import operator
from typing import Callable

import imageio
import cv2
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
        obj.getAnnotation())

    return string


# TODO check functools.lru_cache for this purpose
class Cache:
    """
    Fixed-length mapping to use as a cache.
    Deletes items in FIFO manner when maximum allowed length is reached.
    """
    def __init__(self, max_len=5000, load_fn: Callable = lambda x:
    cv2.imread(str(x), cv2.IMREAD_GRAYSCALE)):
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


def accumulate(l: list):
    l = sorted(l)
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, sub_iter in it:
        yield key, [x[1] for x in sub_iter]


import pickle, json, csv, os, shutil

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PersistentDict(dict):
    ''' Persistent dictionary with an API compatible with shelve and anydbm.

    The dict is kept in memory, so the dictionary operations run as fast as
    a regular dictionary.

    Write to disk is delayed until close or sync (similar to gdbm's fast mode).

    Input file format is automatically discovered.
    Output file format is selectable between pickle, json, and csv.
    All three serialization formats are backed by fast C implementations.

    '''

    def __init__(self, filename, flag='c',
                 mode=None, format='json', *args, **kwargs):
        self.flag = flag                    # r=readonly, c=create, or n=new
        self.mode = mode                    # None or an octal triple like 0644
        self.format = format                # 'csv', 'json', or 'pickle'
        self.filename = filename
        if flag != 'n' and os.access(filename, os.R_OK):
            fileobj = open(filename, 'rb' if format=='pickle' else 'r')
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, item):
        if "/" not in item:
            return dict.__getitem__(self, item)
        keys = item.split("/")
        retval = self
        try:
            for key in keys:
                if key == "":
                    pass
                else:
                    retval = dict.__getitem__(retval, key)
            return retval
        except AttributeError as e:
            raise e

    def get(self, item, default=None):
        try:
            self.__getitem__(item)
        except KeyError:
            return default

    def __contains__(self, item):
        try:
            self.__getitem__(item)
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        keys = key.split("/")
        dic = self
        for key in keys[:-1]:
            if key == "":
                continue
            # setdefault: If key is in the dictionary, return its value.
            # If not, insert key with a value of default and return default.
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    def sync(self):
        'Write dict to disk'
        if self.flag == 'r':
            return
        filename = self.filename
        tempname = filename + '.tmp'
        fileobj = open(tempname, 'wb' if self.format=='pickle' else 'w')
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)    # atomic commit
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        if self.format == 'csv':
            csv.writer(fileobj).writerows(self.items())
        elif self.format == 'json':
            json.dump(self, fileobj, separators=(',', ':'), cls=NumpyEncoder) 
        elif self.format == 'pickle':
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError('Unknown format: ' + repr(self.format))

    def load(self, fileobj):
        # try formats from most restrictive to least restrictive
        for loader in (pickle.load, json.load, csv.reader):
            fileobj.seek(0)
            try:
                return self.update(loader(fileobj)) # Create a json decoder
            except Exception:
                pass
        raise ValueError('File not in a supported format')
