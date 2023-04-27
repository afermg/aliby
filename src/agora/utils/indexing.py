#!/usr/bin/env jupyter
"""
Utilities based on association are used to efficiently acquire indices of
tracklets with some kind of relationship.
This can be:
    - Cells that are to be merged.
    - Cells that have a lineage relationship.
"""

import numpy as np
import typing as t


def validate_association(
    association: np.ndarray,
    indices: np.ndarray,
    match_column: t.Optional[int] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Identify matches between two arrays by matching rows.

    We use broadcasting for fast multi-indexing, generalising for lineage dynamics.

    Parameters
    ----------
    association : np.ndarray
        2D array of lineage associations where columns are (trap, mother, daughter)
        or
        a 3D array, which is an array of 2 X 2 arrays comprising  [[trap_id, mother_label], [trap_id, daughter_label]].
    indices : np.ndarray
        a 2D array where each column is a different level, such as (trap_id, cell_label). This should not include mother_label.
    match_column: int
        int indicating a specific column that is required to match (i.e. 0-1 for target-source when trying to merge tracklets or mother-bud for lineage)
        must be present in indices.
        If None, one match suffices for the resultant indices
        vector to be True.

    Returns
    -------
    valid_association: boolean np.ndarray
        1D array indicating valid elements in association.
    valid_indices: boolean np.ndarray
        1D array indicating valid elements in indices.

    Examples
    --------
    >>> import numpy as np
    >>> from agora.utils.indexing import validate_association

    >>> association = np.array([ [[0, 1], [0, 3]], [[0, 1], [0, 4]] ])
    >>> indices = np.array([ [0, 1], [0, 2], [0, 3]])
    >>> print(indices.T)

    >>> valid_associations, valid_indices = validate_association(merges, indices)

    >>> print(valid_associations)
    array([ True, False])
    >>> print(valid_indices)
    array([ True, False, True])
    """
    if association.ndim == 2:
        # reshape into 3D array for broadcasting
        # [trap, mother, daughter] becomes [[trap, mother], [trap, daughter]] for each trap
        association = _assoc_indices_to_3d(association)
    # compare existing association with available indices
    # swap trap and cell_label axes for the indices array to correctly broadcast
    # compare [[trap, mother], [trap, daughter]] with [trap, cell_label] for all traps in association and for all [trap, cell_label] pairs in indices
    valid_ndassociation = (
        association[..., np.newaxis] == indices.T[np.newaxis, ...]
    )
    # broadcasting is confusing (but efficient):
    # first, we check the dimension across trap and cell id to ensure both match
    # 1. find only those comparisons with both trap_ids and cell labels matching - they are now marked as True
    valid_cell_ids = valid_ndassociation.all(axis=2)
    if match_column is None:
        # then, we check the merge tuples to check which have both target and source
        # 2. keep only those comparisons that match at least one row in indices
        # 3. keep those that have a match for both mother and daughter in association
        valid_association = valid_cell_ids.any(axis=2).all(axis=1)
        # finally, we check the dimension that crosses all indices to ensure the pair
        # is present in a valid merge event
        valid_indices = (
            valid_ndassociation[valid_association].all(axis=2).any(axis=(0, 1))
        )
        myversion = valid_cell_ids.any(axis=1).any(axis=0)
        1 / 0
    else:
        # we fetch specific indices if we aim for the ones with one present
        valid_indices = valid_cell_ids[:, match_column].any(axis=0)
        # calid association then becomes a boolean array: True means that there is a
        # match (match_column) between that cell and the index
        valid_association = (
            valid_cell_ids[:, match_column] & valid_indices
        ).any(axis=1)
    return valid_association, valid_indices


def _assoc_indices_to_3d(ndarray: np.ndarray):
    """
    Reorganise an array of shape (N, 3) into one of shape (N, 2, 2).

    Reorganise an array so that the last entry of each row is removed and generates a new row. This new row retains all other entries of the original row.

    Example:
    [ [0, 1, 3], [0, 1, 4] ]
    becomes
    [ [[0, 1], [0, 3]], [[0, 1], [0, 4]] ]

    This is used to convert a signal MultiIndex before comparing association.
    """
    result = ndarray
    if len(ndarray) and ndarray.ndim > 1:
        if ndarray.shape[1] == 3:
            # faster indexing for single positions
            result = np.transpose(
                np.hstack((ndarray[:, [0]], ndarray)).reshape(-1, 2, 2),
                axes=[0, 2, 1],
            )
        else:
            # 20% slower, but more general indexing
            columns = np.arange(ndarray.shape[1])
            result = np.stack(
                (
                    ndarray[:, np.delete(columns, -1)],
                    ndarray[:, np.delete(columns, -2)],
                ),
                axis=1,
            )
    return result


def _3d_index_to_2d(array: np.ndarray):
    """Perform opposite switch to _assoc_indices_to_3d."""
    result = array
    if len(array):
        result = np.concatenate(
            (array[:, 0, :], array[:, 1, 1, np.newaxis]), axis=1
        )
    return result


def compare_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fetch two 2-D indices and return a binary 2-D matrix
    where a True value links two cells where all cells are the same.
    """
    return (x[..., None] == y.T[None, ...]).all(axis=1)
