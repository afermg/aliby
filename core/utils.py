"""
Utility functions
"""


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
