import typing as t
from functools import partial
from inspect import getfullargspec, getmembers, isfunction

import numpy as np
from cp_measure.bulk import get_core_measurements, get_correlation_measurements

from extraction.core.functions import cell, trap

"""
Load functions for analysing cells and their background.

Note that inspect.getmembers returns a list of function names
and functions, and inspect.getfullargspec returns a
function's arguments.
"""


def load_cellfuns_core():
    """Load functions from the cell module and return as a dict."""
    return {
        f[0]: f[1]
        for f in getmembers(cell)
        if isfunction(f[1]) and f[1].__module__.startswith("extraction.core.functions")
    }


def load_cellfuns():
    """
    Create a dict of core functions for use on cell_masks.

    The core functions only work on a single mask.
    """
    # create dict of the core functions from cell.py - these functions apply to a single mask
    cell_funs = load_cellfuns_core()
    # create a dict of functions that apply the core functions to an array of cell_masks
    CELL_FUNS = {}

    for f_name, f in cell_funs.items():
        if isfunction(f):
            args = getfullargspec(f).args
            if len(args) == 1:
                # function that applies f to m, a binary mask
                new_fun = partial(ignore_pixels, cell_fun=f)
            else:
                # function that applies f to m and img, a binary mask and a 2 or 3d image.
                new_fun = f

            CELL_FUNS[f_name] = new_fun

    # Add cp_measure measurements
    for fun_name, f in get_core_measurements().items():
        CELL_FUNS[fun_name] = partial(wrap_cp_measure_features, fun=f)

    for fun_name, f in get_correlation_measurements().items():
        CELL_FUNS[fun_name] = partial(wrap_cp_corr_features, fun=f)

    return CELL_FUNS


def load_trapfuns():
    """Load functions that are applied to an entire tile."""
    TRAP_FUNS = {
        f[0]: f[1]
        for f in getmembers(trap)
        if isfunction(f[1]) and f[1].__module__.startswith("extraction.core.functions")
    }
    return TRAP_FUNS


def load_funs():
    """Combine all automatically loaded functions."""
    CELL_FUNS = load_cellfuns()
    TRAP_FUNS = load_trapfuns()
    # return dict of cell funs, dict of trap funs, and dict of both
    return CELL_FUNS, TRAP_FUNS, {**TRAP_FUNS, **CELL_FUNS}


def load_redfuns() -> t.Dict[str, t.Callable]:
    """
    Load functions to reduce a multidimensional image by one dimension.

    Parameters
    ----------
    additional_reducers: function or a dict of functions (optional)
        Functions to perform the reduction.
    """
    RED_FUNS = {
        "max": np.maximum,
        "mean": np.mean,
        "median": np.median,
        "div": np.divide,
        "add": np.add,
        "None": None,
    }
    return RED_FUNS


# Functional solutions to complex problems
# currently all these wrappers assume that the input mask is a binary array
# In the future we may want to replace this with an array of integer labels instead.


def wrap_cp_measure_features(
    mask: np.ndarray, pixels: np.ndarray, fun: t.Callable = None
) -> t.Callable:
    # results = [
    #     {k: v[0] for k, v in fun(m, pixels).items()} for m in masks.astype(np.uint16)
    # ]
    results = fun(mask.astype(np.uint16), pixels)

    return results


def wrap_cp_corr_features(
    mask: np.ndarray, pixels1: np.ndarray, pixels2: np.ndarray, fun: t.Callable = None
) -> t.Callable:
    # results = [
    #     {k: v[0] for k, v in fun(m, pixels1, pixels2).items()}
    #     for m in masks.astype(np.uint16)
    # ]
    results = fun(pixels1, pixels2, mask)
    return results


def ignore_pixels(mask, pixels, cell_fun):
    return cell_fun(mask)
