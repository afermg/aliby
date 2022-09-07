import typing as t
from inspect import getfullargspec, getmembers, isfunction

import numpy as np

from extraction.core.functions import cell, trap
from extraction.core.functions.custom import localisation
from extraction.core.functions.distributors import trap_apply
from extraction.core.functions.math_utils import div0

"""
Load functions for analysing cells and their background.
Note that inspect.getmembers returns a list of function names and functions, and inspect.getfullargspec returns a function's arguments.
"""


def load_cellfuns_core():
    """
    Load functions from the cell module and return as a dict.
    """
    return {
        f[0]: f[1]
        for f in getmembers(cell)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }


def load_custom_args() -> t.Tuple[
    (t.Dict[str, t.Callable], t.Dict[str, t.List[str]])
]:
    """
    Load custom functions from the localisation module and return the functions and any additional arguments, other than cell_mask and trap_image, as dictionaries.
    """
    # load functions from module
    funs = {
        f[0]: f[1]
        for f in getmembers(localisation)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    # load additional arguments if cell_mask and trap_image are arguments
    args = {
        k: getfullargspec(v).args[2:]
        for k, v in funs.items()
        if set(["cell_mask", "trap_image"]).intersection(
            getfullargspec(v).args
        )
    }
    # return dictionaries of functions and of arguments
    return (
        {k: funs[k] for k in args.keys()},
        {k: v for k, v in args.items() if v},
    )


def load_cellfuns():
    """
    Creates a dict of core functions that can be used on an array of cell_masks.
    The core functions only work on a single mask.
    """
    # create dict of the core functions from cell.py - these functions apply to a single mask
    cell_funs = load_cellfuns_core()
    # create a dict of functions that apply the core functions to an array of cell_masks
    CELLFUNS = {}
    for f_name, f in cell_funs.items():
        if isfunction(f):

            def tmp(f):
                args = getfullargspec(f).args
                if len(args) == 1:
                    # function that applies f to m, an array of masks
                    return lambda m, _: trap_apply(f, m)
                else:
                    # function that applies f to m and img, the trap_image
                    return lambda m, img: trap_apply(f, m, img)

            CELLFUNS[f_name] = tmp(f)
    return CELLFUNS


def load_trapfuns():
    """
    Load functions that are applied to an entire trap or tile or subsection of an image rather than to single cells.
    """
    TRAPFUNS = {
        f[0]: f[1]
        for f in getmembers(trap)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    return TRAPFUNS


def load_funs():
    """
    Combine all automatically loaded functions
    """
    CELLFUNS = load_cellfuns()
    TRAPFUNS = load_trapfuns()
    # return dict of cell funs, dict of trap funs, and dict of both
    return CELLFUNS, TRAPFUNS, {**TRAPFUNS, **CELLFUNS}


def load_redfuns():  # TODO make defining reduction functions more flexible
    """
    Load functions to reduce the z-stack to two dimensions.
    """
    RED_FUNS = {
        "np_max": np.maximum,
        "np_mean": np.mean,
        "np_median": np.median,
        "None": None,
    }
    return RED_FUNS


def load_mergefuns():
    """
    Load functions to merge multiple channels
    """
    MERGE_FUNS = {"div0": div0, "np_add": np.add}
    return MERGE_FUNS
