import typing as t

import numpy as np


def trap_apply(masks, *args, cell_fun=None, **kwargs):
    """
    Apply a cell_function to a mask and a trap_image.

    Parameters
    ----------
    cell_fun: function
        Function to apply to the cell (from extraction/cell.py)
    masks: 3d array
        Segmentation masks for the cells. Note that cells are in the first dimension (N, Y,X)
    *args: tuple
        Trap_image and any other arguments to pass if needed to custom functions.
    **kwargs: dict
        Keyword arguments to pass if needed to custom functions.
    """
    # apply cell_fun to each cell and return the results as a list
    return [cell_fun(mask, *args, **kwargs) for mask in masks]


def reduce_z(pixels: np.ndarray, fun: t.Callable, axis: int = 0):
    """
    Reduce the 3D image to 2d.

    Parameters
    ----------
    pixels: array
        Images for all the channels associated with a trap
    fun: function
        Function to execute the reduction
    axis: int (default 0)
        Axis in which we apply the reduction operation.
    """
    if isinstance(fun, np.ufunc):
        # optimise the reduction function if possible
        return fun.reduce(pixels, axis=axis)
    else:
        # WARNING: Very slow, only use when no alternatives exist
        return np.apply_along_axis(fun, axis, pixels)
