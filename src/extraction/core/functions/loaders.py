"""Load functions for analysing cells and their background."""

import typing as t

# inspect.getmembers returns a list of function names and functions
# inspect.getfullargspec returns a function's arguments
from inspect import getfullargspec, getmembers, isfunction
from types import FunctionType

import bottleneck as bn
from extraction.core.functions import background_functions
from extraction.core.functions.cell_functions import (
    _FUNCTION_TO_MODEL,
    CELL_FUNCTIONS,
    PIXEL_FUNCTIONS,
    is_model_available,
)


def load_all_functions():
    """Load cell and trap functions."""
    cell_funs = load_cell_functions()
    trap_funs = load_background_functions()
    # return dict of cell funs and dict of trap and cell funs
    return cell_funs, {**trap_funs, **cell_funs}


def load_cell_functions():
    """
    Create a dict of functions for use on cell_masks.

    Take the functions marked with cell_function, and not the helpers
    beside them. Each is wrapped to take a tile's masks, its image and
    the channels' names, whichever of those it uses, and to return a list
    with a result for each cell. The wrapper of a function that needs
    only a cell's pixels carries that function of the pixels as
    ``of_pixels``.
    """
    cell_funs = {}
    for f_name, f in CELL_FUNCTIONS.items():

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
        if f_name in PIXEL_FUNCTIONS:
            cell_funs[f_name].of_pixels = PIXEL_FUNCTIONS[f_name]
    cell_funs = {
        name: fn
        for name, fn in cell_funs.items()
        if name not in _FUNCTION_TO_MODEL
        or is_model_available(_FUNCTION_TO_MODEL[name])
    }
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


def load_background_functions():
    """Load functions that compute background signals from a tile."""
    trap_funs = {
        f[0]: f[1]
        for f in getmembers(background_functions)
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
        # div0 was here but could reduce nothing: it divides one half of
        # an axis of length two by the other, and the axis reduced is z
        "add": bn.nansum,
        "null": None,
    }
    if additional_reducers is not None:
        if isinstance(additional_reducers, FunctionType):
            additional_reducers = [
                (additional_reducers.__name__, additional_reducers)
            ]
        red_funs.update(additional_reducers)
    return red_funs
