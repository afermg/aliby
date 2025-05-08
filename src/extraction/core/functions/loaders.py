"""Load functions for analysing cells and their background."""

import typing as t

# inspect.getmembers returns a list of function names and functions
# inspect.getfullargspec returns a function's arguments
from inspect import getfullargspec, getmembers, isfunction
from types import FunctionType

import bottleneck as bn
from extraction.core.functions import cell_functions, trap_functions
from extraction.core.functions.math_utils import div0


def load_all_functions():
    """Load cell and trap functions."""
    cell_funs = load_cell_functions()
    trap_funs = load_trap_functions()
    # return dict of cell funs and dict of trap and cell funs
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
        # cell functions have different inputs
        if isfunction(f):

            def tmp(f):
                args = getfullargspec(f).args
                if len(args) == 1:
                    # function with input m, an array of masks
                    return lambda m, _, __: call_cell_fun(f, m)
                elif len(args) == 2:
                    # function with inputs m and img, the fluorescence images
                    return lambda m, img, _: call_cell_fun(f, m, img)
                else:
                    # multichannel function with inputs m, img, and channels
                    return lambda m, img, channels: call_cell_fun(
                        f, m, img, channels
                    )

            cell_funs[f_name] = tmp(f)
    return cell_funs


def call_cell_fun(cell_fun, cell_masks, *args, **kwargs):
    """
    Apply a cell_function to a mask and potentially a trap_image.

    Parameters
    ----------
    cell_fun: function
        Function to apply to the cell (from extraction/cell.py)
    cell_masks: 3d array
        Segmentation masks for the cells. Note that cells are in the first dimension (N, Y,X)
    *args: tuple
        Trap_image and any other arguments to pass if needed to custom functions.
    **kwargs: dict
        Keyword arguments to pass if needed to custom functions.
    """
    # apply cell_fun to each cell and return the results as a list
    return [cell_fun(mask, *args, **kwargs) for mask in cell_masks]


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
