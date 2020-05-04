"""
Utility functions and classes
"""
from collections import defaultdict


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


class AttributeDict(defaultdict):
    """
    Dictionary subclass enabling attribute lookup/assignment of keys/values.
    For example::
        >>> m = _AttributeDict({'foo': 'bar'})
        >>> m.foo
        'bar'
        >>> m.foo = 'not bar'
        >>> m['foo']
        'not bar'

    It can also be used for nested dictionaries as follows:
    >>> keys = AttributeDict()
    >>> keys.abc.xyz.x = 123
    >>> keys.abc.xyz.x
    123
    >>> keys.abc.xyz.a.b.c = 234
    >>> keys.abc.xyz.a.b.c
    234
    """
    def __init__(self):
        super(AttributeDict, self).__init__(AttributeDict)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value
