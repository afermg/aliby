"""Load functions for analysing cells and their background."""

import typing as t

# inspect.getmembers returns a list of function names and functions
# inspect.getfullargspec returns a function's arguments
from inspect import getfullargspec, getmembers, isfunction
from types import FunctionType

import bottleneck as bn
from extraction.core.functions import cell_functions, trap_functions
from extraction.core.functions.distributors import trap_apply
from extraction.core.functions.math_utils import div0


def load_all_functions():
    """Load cell and trap functions."""
    cell_funs = load_cell_functions()
    trap_funs = load_trap_functions()
    # return dict of cell funs, dict of trap funs, and dict of both
    return cell_funs, {**trap_funs, **cell_funs}


def load_cell_functions():
    """Create a dict of functions for use on cell_masks."""
    cell_funs_raw = {
        f[0]: f[1]
        for f in getmembers(cell_functions)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    cell_funs = {}
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

            cell_funs[f_name] = tmp(f)
    return cell_funs


def load_trap_functions():
    """Load functions that are applied to an entire tile."""
    trap_funs = {
        f[0]: f[1]
        for f in getmembers(trap_functions)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    return trap_funs


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
    red_funs = {
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
        red_funs.update(additional_reducers)
    return red_funs
