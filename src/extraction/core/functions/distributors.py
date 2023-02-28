import typing as t

import bottleneck as bn
import numpy as np


def trap_apply(cell_fun, cell_masks, *args, **kwargs):
    """
    Apply a cell_function to a mask and a trap_image.

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


def reduce_z(trap_image: np.ndarray, fun: t.Callable, axis: int = 0):
    """
    Reduce the trap_image to 2d.

    Parameters
    ----------
    trap_image: array
        Images for all the channels associated with a trap
    fun: function
        Function to execute the reduction
    axis: int (default 0)
        Axis in which we apply the reduction operation.
    """
    # FUTURE replace with py3.10's match-case.
    if (
        hasattr(fun, "__module__") and fun.__module__[:10] == "bottleneck"
    ):  # Bottleneck type
        return getattr(bn.reduce, fun.__name__)(trap_image, axis=axis)
    elif isinstance(fun, np.ufunc):
        # optimise the reduction function if possible
        return fun.reduce(trap_image, axis=axis)
    else:  # WARNING: Very slow, only use when no alternatives exist
        return np.apply_along_axis(fun, axis, trap_image)
