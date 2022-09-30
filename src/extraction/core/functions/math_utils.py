import numpy as np


def div0(array, fill=0, axis=-1):
    """
    Divide array a by array b.

    If the result is a scalar and infinite, return fill.

    If the result contain elements that are infinite, replace these elements with fill.

    Parameters
    ----------
    a: array
    b: array
    fill: float
    **kwargs: kwargs
    """
    assert array.shape[axis] == 2, f"Array has the wrong shape in axis {axis}"
    slices_0, slices_1 = [[slice(None)] * len(array.shape)] * 2
    slices_0[axis] = 0
    slices_1[axis] = 1

    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(
            array[tuple(slices_0)],
            array[tuple(slices_1)],
        )
    if np.isscalar(c):
        return c if np.isfinite(c) else fill
    else:
        c[~np.isfinite(c)] = fill
        return c
