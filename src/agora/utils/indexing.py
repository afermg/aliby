#!/usr/bin/env jupyter
"""
Utilities based on association are used to efficiently acquire indices of tracklets with some kind of relationship.
This can be:
    - Cells that are to be merged
    - Cells that have a linear relationship
"""

import numpy as np
import typing as t

# data type to link together trap and cell ids
i_dtype = {"names": ["trap_id", "cell_id"], "formats": [np.int64, np.int64]}


def validate_lineage(
    lineage: np.ndarray, indices: np.ndarray, how: str = "families"
):
    """
    Identify mother-bud pairs that exist both in lineage and a Signal's
    indices.

    We expect the lineage information to be unique: a bud should not have
    two mothers.

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
        lineage = assoc_indices_to_3d(lineage)
    if how == "mothers":
        c_index = 0
    elif how == "daughters":
        c_index = 1
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
    if (
        indices[flat_valid_indices, :].size
        != np.unique(
            lineage[flat_valid_lineage, :].reshape(-1, 2), axis=0
        ).size
    ):
        # all unique indices in valid_lineages should be in valid_indices
        raise Exception(
            "Error in validate_lineage: "
            "lineage information is likely not unique."
        )
    return flat_valid_lineage, flat_valid_indices


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


###################################################################


def validate_association(
    association: np.ndarray,
    indices: np.ndarray,
    match_column: t.Optional[int] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:

    """Select rows from the first array that are present in both.
        We use casting for fast multiindexing, generalising for lineage dynamics


        Parameters
        ----------
        association : np.ndarray
            2-D array where columns are (trap, mother, daughter) or 3-D array where
            dimensions are (X,trap,2), containing tuples ((trap,mother), (trap,daughter))
            across the 3rd dimension.
        indices : np.ndarray
            2-D array where each column is a different level. This should not include mother_label.
        match_column: int
            int indicating a specific column is required to match (i.e.
            0-1 for target-source when trying to merge tracklets or mother-bud for lineage)
            must be present in indices. If it is false one match suffices for the resultant indices
            vector to be True.

        Returns
        -------
        np.ndarray
            1-D boolean array indicating valid merge events.
        np.ndarray
            1-D boolean array indicating indices with an association relationship.

        Examples
        --------

        >>> import numpy as np
        >>> from agora.utils.indexing import validate_association
        >>> merges = np.array(range(12)).reshape(3,2,2)
        >>> indices = np.array(range(6)).reshape(3,2)

        >>> print(merges, indices)
        >>> print(merges); print(indices)
        [[[ 0  1]
          [ 2  3]]

         [[ 4  5]
          [ 6  7]]

         [[ 8  9]
          [10 11]]]

        [[0 1]
         [2 3]
         [4 5]]

        >>> valid_associations, valid_indices  = validate_association(merges, indices)
        >>> print(valid_associations, valid_indices)
    [ True False False] [ True  True False]

    """
    if association.ndim == 2:
        # Reshape into 3-D array for broadcasting if neded
        # association = np.stack(
        #     (association[:, [0, 1]], association[:, [0, 2]]), axis=1
        # )
        association = _assoc_indices_to_3d(association)

    # Compare existing association with available indices
    # Swap trap and label axes for the association array to correctly cast
    valid_ndassociation = association[..., None] == indices.T[None, ...]

    # Broadcasting is confusing (but efficient):
    # First we check the dimension across trap and cell id, to ensure both match
    valid_cell_ids = valid_ndassociation.all(axis=2)

    if match_column is None:
        # Then we check the merge tuples to check which cases have both target and source
        valid_association = valid_cell_ids.any(axis=2).all(axis=1)

        # Finally we check the dimension that crosses all indices, to ensure the pair
        # is present in a valid merge event.
        valid_indices = (
            valid_ndassociation[valid_association].all(axis=2).any(axis=(0, 1))
        )
    else:  # We fetch specific indices if we aim for the ones with one present
        valid_indices = valid_cell_ids[:, match_column].any(axis=0)
        # Valid association then becomes a boolean array, true means that there is a
        # match (match_column) between that cell and the index
        valid_association = (
            valid_cell_ids[:, match_column] & valid_indices
        ).any(axis=1)

    return valid_association, valid_indices


def compare_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fetch two 2-D indices and return a binary 2-D matrix
    where a True value links two cells where all cells are the same
    """
    return (x[..., None] == y.T[None, ...]).all(axis=1)
