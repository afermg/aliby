"""Functions to identify cells in lineages."""

import numpy as np
import pandas as pd

# data type to link together trap and cell ids
i_dtype = {"names": ["trap_id", "cell_id"], "formats": [np.int64, np.int64]}


def validate_lineage(
    lineage: np.ndarray,
    indices: np.ndarray,
    how: str = "families",
):
    """
    Identify mother-bud pairs both in lineage and a Signal's indices.

    We expect the lineage information to be unique: a bud should not have
    two mothers.

    Lineage is returned with buds assigned only to their first mother if they
    have multiple.

    Parameters
    ----------
    lineage : np.ndarray
        2D array of lineage associations where columns are
        (trap, mother, daughter)
        or
        a 3D array, which is an array of 2 X 2 arrays comprising
        [[trap_id, mother_label], [trap_id, daughter_label]].
    indices : np.ndarray
        A 2D array of cell indices from a Signal, (trap_id, cell_label).
        This array should not include mother_label.
    how: str
        If "mothers", matches indicate mothers from mother-bud pairs;
        If "daughters", matches indicate daughters from mother-bud pairs;
        If "families", matches indicate mothers and daughters in mother-bud pairs.

    Returns
    -------
    valid_lineage: boolean np.ndarray
        1D array indicating matched elements in lineage.
    valid_indices: boolean np.ndarray
        1D array indicating matched elements in indices.
    lineage: np.ndarray
        Any bud already having a mother that is assigned to another has that
        second assignment discarded.

    Examples
    --------
    >>> import numpy as np
    >>> from agora.utils.indexing import validate_lineage

    >>> lineage = np.array([ [[0, 1], [0, 3]], [[0, 1], [0, 4]], [[0, 1], [0, 6]], [[0, 4], [0, 7]] ])
    >>> indices = np.array([ [0, 1], [0, 2], [0, 3]])

    >>> valid_lineage, valid_indices, lineage = validate_lineage(lineage, indices)

    >>> print(valid_lineage)
     array([ True, False, False, False])
    >>> print(valid_indices)
     array([ True, False, True])

    and

    >>> lineage = np.array([[[0,3], [0,1]], [[0,2], [0,4]]])
    >>> indices = np.array([[0,1], [0,2], [0,3]])
    >>> valid_lineage, valid_indices, lineage = validate_lineage(lineage, indices)
    >>> print(valid_lineage)
     array([ True, False])
    >>> print(valid_indices)
     array([ True, False, True])
    """
    if lineage.ndim == 2:
        # [trap, mother, daughter] becomes [[trap, mother], [trap, daughter]]
        lineage = assoc_indices_to_3d(lineage)
        invert_lineage = True
    if how == "mothers":
        c_index = 0
    elif how == "daughters":
        c_index = 1
    # if buds have two mothers, pick the first one
    lineage = lineage[
        ~pd.DataFrame(lineage[:, 1, :]).duplicated().values, :, :
    ]
    # find valid lineage
    valid_lineages = index_isin(lineage, indices)
    if how == "families":
        # both mother and bud must be in indices
        valid_lineage = valid_lineages.all(axis=1)
    else:
        valid_lineage = valid_lineages[:, c_index, :]
    flat_valid_lineage = valid_lineage.flatten()
    # find valid indices
    selected_lineages = lineage[flat_valid_lineage, ...]
    if how == "families":
        # select only pairs of mother and bud indices
        valid_indices = index_isin(indices, selected_lineages)
    else:
        valid_indices = index_isin(indices, selected_lineages[:, c_index, :])
    flat_valid_indices = valid_indices.flatten()
    # put the corrected lineage in the right format
    if invert_lineage:
        lineage = assoc_indices_to_2d(lineage)
    return flat_valid_lineage, flat_valid_indices, lineage


def index_isin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Find those elements of x that are in y.

    Both arrays must be arrays of integer indices,
    such as (trap_id, cell_id).
    """
    x = np.ascontiguousarray(x, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    xv = x.view(i_dtype)
    inboth = np.intersect1d(xv, y.view(i_dtype))
    x_bool = np.isin(xv, inboth)
    return x_bool


def assoc_indices_to_3d(ndarray: np.ndarray):
    """
    Convert the last column to a new row and repeat first column's values.

    For example: [trap, mother, daughter] becomes
        [[trap, mother], [trap, daughter]].

    Assumes the input array has shape (N,3).
    """
    result = ndarray
    if len(ndarray) and ndarray.ndim > 1:
        # faster indexing for single positions
        if ndarray.shape[1] == 3:
            result = np.transpose(
                np.hstack((ndarray[:, [0]], ndarray)).reshape(-1, 2, 2),
                axes=[0, 2, 1],
            )
        else:
            # 20% slower but more general indexing
            columns = np.arange(ndarray.shape[1])
            result = np.stack(
                (
                    ndarray[:, np.delete(columns, -1)],
                    ndarray[:, np.delete(columns, -2)],
                ),
                axis=1,
            )
    return result


def assoc_indices_to_2d(array: np.ndarray):
    """Convert indices to 2d."""
    result = array
    if len(array):
        result = np.concatenate(
            (array[:, 0, :], array[:, 1, 1, np.newaxis]), axis=1
        )
    return result


def find_1st_greater(arr, limit):
    """Find the first index where the array is larger than limit."""
    indices = np.flatnonzero(arr > limit)
    first_index = indices[0] if len(indices) > 0 else -1
    return first_index


def find_1st_equal(arr, limit):
    """Find the first index where the array equals the limit."""
    indices = np.flatnonzero(arr == limit)
    first_index = indices[0] if len(indices) > 0 else -1
    return first_index


def wrap_int(x):
    """Convert a single integer to a list."""
    return [x] if isinstance(x, int) else x
