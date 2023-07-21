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


def validate_lineage(
    lineage: np.ndarray, indices: np.ndarray, how: str = "families"
):
    """
    Identify mother-bud pairs that exist both in lineage and a Signal's indices.

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

    Examples
    --------
    >>> import numpy as np
    >>> from agora.utils.indexing import validate_lineage

    >>> lineage = np.array([ [[0, 1], [0, 3]], [[0, 1], [0, 4]], [[0, 1], [0, 6]], [[0, 4], [0, 7]] ])
    >>> indices = np.array([ [0, 1], [0, 2], [0, 3]])

    >>> valid_lineage, valid_indices = validate_lineage(lineage, indices)

    >>> print(valid_lineage)
     array([ True, False, False, False])
    >>> print(valid_indices)
     array([ True, False, True])

    and

    >>> lineage = np.array([[[0,3], [0,1]], [[0,2], [0,4]]])
    >>> indices = np.array([[0,1], [0,2], [0,3]])
    >>> valid_lineage, valid_indices = validate_lineage(lineage, indices)
    >>> print(valid_lineage)
     array([ True, False])
    >>> print(valid_indices)
     array([ True, False, True])
    """
    if lineage.ndim == 2:
        # [trap, mother, daughter] becomes [[trap, mother], [trap, daughter]]
        lineage = _assoc_indices_to_3d(lineage)
    if how == "mothers":
        c_index = 0
    elif how == "daughters":
        c_index = 1
    # dtype links together trap and cell ids
    dtype = {"names": ["trap_id", "cell_id"], "formats": [np.int64, np.int64]}
    lineage = np.ascontiguousarray(lineage, dtype=np.int64)
    # find (trap, cell_ids) in intersection
    inboth = np.intersect1d(lineage.view(dtype), indices.view(dtype))
    # find valid lineage
    valid_lineage = np.isin(lineage.view(dtype), inboth)
    if how == "families":
        # both mother and bud must be in indices
        valid_lineage = valid_lineage.all(axis=1)
    else:
        valid_lineage = valid_lineage[:, c_index, :]
    # find valid indices
    possible_indices = lineage[valid_lineage.flatten(), ...]
    if how == "families":
        # select only pairs of mother and bud indices
        valid_indices = np.isin(
            indices.view(dtype), possible_indices.view(dtype)
        )
    else:
        valid_indices = np.isin(
            indices.view(dtype), possible_indices.view(dtype)[:, c_index, :]
        )
    return valid_lineage.flatten(), valid_indices.flatten()


def validate_association(
    association: np.ndarray,
    indices: np.ndarray,
    match_column: t.Optional[int] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Identify mother-bud pairs that exist both in lineage and a Signal's indices.

    """
    if association.ndim == 2:
        # reshape into 3D array for broadcasting
        # for each trap, [trap, mother, daughter] becomes
        #  [[trap, mother], [trap, daughter]]
        association = _assoc_indices_to_3d(association)

    valid_association, valid_indices = validate_lineage(association, indices)

    # Alan's working code
    # Compare existing association with available indices
    # Swap trap and label axes for the association array to correctly cast
    valid_ndassociation_a = association[..., None] == indices.T[None, ...]
    # Broadcasting is confusing (but efficient):
    # First we check the dimension across trap and cell id, to ensure both match
    valid_cell_ids_a = valid_ndassociation_a.all(axis=2)
    if match_column is None:
        # Then we check the merge tuples to check which cases have both target and source
        valid_association_a = valid_cell_ids_a.any(axis=2).all(axis=1)

        # Finally we check the dimension that crosses all indices, to ensure the pair
        # is present in a valid merge event.
        valid_indices_a = (
            valid_ndassociation_a[valid_association_a]
            .all(axis=2)
            .any(axis=(0, 1))
        )
    else:  # We fetch specific indices if we aim for the ones with one present
        valid_indices_a = valid_cell_ids_a[:, match_column].any(axis=0)
        # Valid association then becomes a boolean array, true means that there is a
        # match (match_column) between that cell and the index
        valid_association_a = (
            valid_cell_ids_a[:, match_column] & valid_indices
        ).any(axis=1)

    assert np.array_equal(
        valid_association, valid_association_a
    ), "valid_association error"
    assert np.array_equal(
        valid_indices, valid_indices_a
    ), "valid_indices error"

    return valid_association, valid_indices


def validate_association_old(
    association: np.ndarray,
    indices: np.ndarray,
    match_column: t.Optional[int] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Identify mother-bud pairs that exist both in lineage and a Signal's indices.


    Parameters
    ----------
    association : np.ndarray
        2D array of lineage associations where columns are (trap, mother, daughter)
        or
        a 3D array, which is an array of 2 X 2 arrays comprising [[trap_id, mother_label], [trap_id, daughter_label]].
    indices : np.ndarray
        A 2D array where each column is a different level, such as (trap_id, cell_label), which typically is an index of a Signal
        dataframe. This array should not include mother_label.
    match_column: int
        If 0, matches indicate mothers from mother-bud pairs;
        If 1, matches indicate daughters from mother-bud pairs;
        If None, matches indicate either mothers or daughters in mother-bud pairs.

    Returns
    -------
    valid_association: boolean np.ndarray
        1D array indicating elements in association with matches.
    valid_indices: boolean np.ndarray
        1D array indicating elements in indices with matches.

    Examples
    --------
    >>> import numpy as np
    >>> from agora.utils.indexing import validate_association

    >>> association = np.array([ [[0, 1], [0, 3]], [[0, 1], [0, 4]], [[0, 1], [0, 6]], [[0, 4], [0, 7]] ])
    >>> indices = np.array([ [0, 1], [0, 2], [0, 3]])
    >>> print(indices.T)

    >>> valid_association, valid_indices = validate_association(association, indices)

    >>> print(valid_association)
     array([ True, False, False, False])
    >>> print(valid_indices)
     array([ True, False, True])

    and

    >>> association = np.array([[[0,3], [0,1]], [[0,2], [0,4]]])
    >>> indices = np.array([[0,1], [0,2], [0,3]])
    >>> valid_association, valid_indices = validate_association(association, indices)
    >>> print(valid_association)
     array([ True, False])
    >>> print(valid_indices)
     array([ True, False, True])
    """
    if association.ndim == 2:
        # reshape into 3D array for broadcasting
        # for each trap, [trap, mother, daughter] becomes
        #  [[trap, mother], [trap, daughter]]
        association = _assoc_indices_to_3d(association)
    # use broadcasting to compare association with indices
    # swap trap and cell_label axes for correct broadcasting
    indicesT = indices.T
    # compare each of [[trap, mother], [trap, daughter]] for all traps
    # in association with [trap, cell_label] for all traps in indices
    # association is no_traps x 2 x 2; indices is no_traps X 2
    # valid_ndassociation is no_traps_association x 2 x 2 x no_traps_indices
    valid_ndassociation = (
        association[..., np.newaxis] == indicesT[np.newaxis, ...]
    )
    # find matches in association
    ###
    # make True comparisons with both trap_ids and cell labels matching
    # compare trap_ids and cell_ids for each pair of traps
    valid_cell_ids = valid_ndassociation.all(axis=2)
    if match_column is None:
        # make True comparisons match at least one row in indices
        # at least one cell_id matches
        va_intermediate = valid_cell_ids.any(axis=2)
        # make True comparisons have both mother and bud matching rows in indices
        valid_association = va_intermediate.all(axis=1)
    else:
        # match_column selects mothers if 0 and daughters if 1
        # make True match at least one row in indices
        valid_association = valid_cell_ids[:, match_column].any(axis=1)
    # find matches in indices
    ###
    # make True comparisons have a validated association for both the mother and bud
    # make True comparisons have both trap_ids and cell labels matching
    valid_cell_ids_va = valid_ndassociation[valid_association].all(axis=2)
    if match_column is None:
        # make True comparisons match either a mother or a bud in association
        valid_indices = valid_cell_ids_va.any(axis=(0, 1))
    else:
        valid_indices = valid_cell_ids_va[:, match_column][0]

    # Alan's working code
    # Compare existing association with available indices
    # Swap trap and label axes for the association array to correctly cast
    valid_ndassociation_a = association[..., None] == indices.T[None, ...]
    # Broadcasting is confusing (but efficient):
    # First we check the dimension across trap and cell id, to ensure both match
    valid_cell_ids_a = valid_ndassociation_a.all(axis=2)
    if match_column is None:
        # Then we check the merge tuples to check which cases have both target and source
        valid_association_a = valid_cell_ids_a.any(axis=2).all(axis=1)

        # Finally we check the dimension that crosses all indices, to ensure the pair
        # is present in a valid merge event.
        valid_indices_a = (
            valid_ndassociation_a[valid_association_a]
            .all(axis=2)
            .any(axis=(0, 1))
        )
    else:  # We fetch specific indices if we aim for the ones with one present
        valid_indices_a = valid_cell_ids_a[:, match_column].any(axis=0)
        # Valid association then becomes a boolean array, true means that there is a
        # match (match_column) between that cell and the index
        valid_association_a = (
            valid_cell_ids_a[:, match_column] & valid_indices
        ).any(axis=1)

    assert np.array_equal(
        valid_association, valid_association_a
    ), "valid_association error"
    assert np.array_equal(
        valid_indices, valid_indices_a
    ), "valid_indices error"

    return valid_association, valid_indices


def _assoc_indices_to_3d(ndarray: np.ndarray):
    """
    Reorganise an array of shape (N, 3) into one of shape (N, 2, 2).

    Reorganise an array so that the last entry of each row is removed
    and generates a new row. This new row retains all other entries of
    the original row.

    Example:
    [ [0, 1, 3], [0, 1, 4] ]
    becomes
    [ [[0, 1], [0, 3]], [[0, 1], [0, 4]] ]
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
    """Revert switch from _assoc_indices_to_3d."""
    result = array
    if len(array):
        result = np.concatenate(
            (array[:, 0, :], array[:, 1, 1, np.newaxis]), axis=1
        )
    return result


def compare_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compare two 2D arrays using broadcasting.

    Return a binary array where a True value links two cells where
    all cells are the same.
    """
    return (x[..., np.newaxis] == y.T[np.newaxis, ...]).all(axis=1)
