#!/usr/bin/env jupyter
"""
Utilities based on association are used to efficiently acquire indices of tracklets with some kind of relationship.
This can be:
    - Cells that are to be merged
    - Cells that have a linear relationship
"""

import numpy as np
import typing as t


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
            dimensions are (X, (trap,mother), (trap,daughter))
        indices : np.ndarray
            2-D array where each column is a different level.
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
        >>> from agora.utils.association import validate_association
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
    if association.ndim < 3:
        # Reshape into 3-D array for broadcasting if neded
        association = np.stack(
            (association[:, [0, 1]], association[:, [0, 2]]), axis=1
        )

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
