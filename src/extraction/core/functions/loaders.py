"""
Load functions for analysing cells and their background.

NB inspect.getmembers returns a list of function names and
functions; inspect.getfullargspec returns a function's arguments.
"""

import typing as t
from types import FunctionType
from inspect import getfullargspec, getmembers, isfunction, isbuiltin

import bottleneck as bn

from extraction.core.functions import cell_functions, trap_functions
from extraction.core.functions.custom import localisation
from extraction.core.functions.distributors import trap_apply
from extraction.core.functions.math_utils import div0


def load_custom_functions_and_args() -> (
    t.Tuple[(t.Dict[str, t.Callable], t.Dict[str, t.List[str]])]
):
    """
    Load custom functions.

    Historically these have been for nuclear localisation.

    Return the functions and any additional arguments other
    than cell_mask and trap_image as dictionaries.
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


def load_cell_functions():
    """Create a dict of functions for use on cell_masks."""
    cell_funs_raw = {
        f[0]: f[1]
        for f in getmembers(cell_functions)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    CELL_FUNS = {}
    for f_name, f in cell_funs_raw.items():
        if isfunction(f):

            def tmp(f):
                args = getfullargspec(f).args
                if len(args) == 1:
                    # function that applies f to m, an array of masks
                    return lambda m, _, __: trap_apply(f, m)
                elif len(args) == 2:
                    # function that applies f to m and img, the fluorescence images
                    return lambda m, img, _: trap_apply(f, m, img)
                else:
                    # function that applies f to m, img - the fluorescence images,
                    # and channels: multichannel functions
                    return lambda m, img, channels: trap_apply(
                        f, m, img, channels
                    )

            CELL_FUNS[f_name] = tmp(f)
    return CELL_FUNS


def load_trap_functions():
    """Load functions that are applied to an entire tile."""
    TRAP_FUNS = {
        f[0]: f[1]
        for f in getmembers(trap_functions)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    return TRAP_FUNS


def load_all_functions():
    """Combine all automatically loaded functions."""
    CELL_FUNS = load_cell_functions()
    TRAP_FUNS = load_trap_functions()
    # return dict of cell funs, dict of trap funs, and dict of both
    return CELL_FUNS, TRAP_FUNS, {**TRAP_FUNS, **CELL_FUNS}


def load_reduction_functions(
    additional_reducers: t.Optional[
        t.Union[t.Dict[str, t.Callable], t.Callable]
    ] = None,
) -> t.Dict[str, t.Callable]:
    """
    Load functions to reduce a multidimensional image by one dimension.

    Parameters
    ----------
    additional_reducers: function or a dict of functions (optional)
        Functions to perform the reduction.
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
        RED_FUNS.update(additional_reducers)
    return RED_FUNS
