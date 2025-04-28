import typing as t

import numpy as np


def reduce_z(pixels: np.ndarray, fun: t.Callable, axis: int = 0) -> np.ndarray:
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
        raise Exception(f"{fun} is an invalid reducer.")
