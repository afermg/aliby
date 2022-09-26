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
        Segmentation masks for the cells
    *args: tuple
        Trap_image and any other arguments to pass if needed to custom functions.
    **kwargs: dict
        Keyword arguments to pass if needed to custom functions.
    """
    # find an index for each cell in the trap
    cells_iter = (*range(cell_masks.shape[2]),)
    # apply cell_fun to each cell and return the results as a list
    return [cell_fun(cell_masks[..., i], *args, **kwargs) for i in cells_iter]


def reduce_z(trap_image: np.ndarray, fun: t.Callable):
    """
    Reduce the trap_image to 2d.

    Parameters
    ----------
    trap_image: array
        Images for all the channels associated with a trap
    fun: function
        Function to execute the reduction

    """
    # FUTURE replace with py3.10's match-case.
    if (
        hasattr(fun, "__module__") and fun.__module__[:10] == "bottleneck"
    ):  # Bottleneck type
        return getattr(bn.reduce, fun.__name__)(trap_image, axis=2)
    elif isinstance(fun, np.ufunc):
        # optimise the reduction function if possible
        return fun.reduce(trap_image, axis=2)
    else:  # WARNING: Very slow, only use when no alternatives exist
        return np.apply_along_axis(fun, 2, trap_image)
