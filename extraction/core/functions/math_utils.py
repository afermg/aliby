import numpy as np


def div0(a, b, fill=0):
    """
    Divide array a by array b.

    If the result is a scalar and infinite, return fill.

    If the result contain elements that are infinite, replace these elements with fill.

    Parameters
    ----------
    a: array
    b: array
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
    if np.isscalar(c):
        return c if np.isfinite(c) else fill
    else:
        c[~np.isfinite(c)] = fill
        return c
