import typing as t
from types import FunctionType
from inspect import getfullargspec, getmembers, isfunction, isbuiltin

import bottleneck as bn

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


def load_redfuns(
    additional_reducers: t.Optional[
        t.Union[t.Dict[str, t.Callable], t.Callable]
    ] = None,
) -> t.Dict[str, t.Callable]:
    """
    Load functions to reduce a multidimensional image by one dimension.

    It can take custom functions as arguments.
    """
    RED_FUNS = {
        "max": bn.nanmax,
        "mean": bn.nanmean,
        "median": bn.nanmedian,
        "div0": div0,
        "add": bn.nansum,
        "None": None,
    }

    if additional_reducers is not None:
        if isinstance(additional_reducers, FunctionType):
            additional_reducers = [
                (additional_reducers.__name__, additional_reducers)
            ]
        RED_FUNS.update(name, fun)

    return RED_FUNS
