#!/usr/bin/env jupyter
"""
Functions to efficiently merge rows in DataFrames.
"""
import typing as t
from copy import copy

import numpy as np
import pandas as pd
from utils_find_1st import cmp_larger, find_1st


def apply_merges(data: pd.DataFrame, merges: np.ndarray):
    """Split data in two, one subset for rows relevant for merging and one
    without them. It uses an array of source tracklets and target tracklets
    to efficiently merge them.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    merges : np.ndarray
        3-D ndarray where dimensions are (X,2,2): nmerges, source-target
        pair and single-cell identifiers, respectively.

    Examples
    --------
    FIXME: Add docs.

    """

    valid_merges, indices = validate_merges(merges, np.array(list(data.index)))

    # Assign non-merged
    merged = data.loc[~indices]

    # Implement the merges and drop source rows.
    if valid_merges.any():
        to_merge = data.loc[indices]
        for target, source in merges[valid_merges]:
            target, source = tuple(target), tuple(source)
            to_merge.loc[target] = join_tracks_pair(
                to_merge.loc[target].values,
                to_merge.loc[source].values,
            )
            to_merge.drop(source, inplace=True)

        merged = pd.concat((merged, to_merge), names=data.index.names)
    return merged


def validate_merges(
    merges: np.ndarray, indices: np.ndarray
) -> t.Tuple[np.ndarray, np.ndarray]:

    """Select rows from the first array that are present in both.
    We use casting for fast multiindexing.




    Parameters
    ----------
    merges : np.ndarray
        2-D array where columns are (trap, mother, daughter) or 3-D array where
        dimensions are (X, (trap,mother), (trap,daughter))
    indices : np.ndarray
        2-D array where each column is a different level.

    Returns
    -------
    np.ndarray
        1-D boolean array indicating valid merge events.
    np.ndarray
        1-D boolean array indicating indices involved in merging.

    Examples
    --------
    FIXME: Add docs.

    """
    if merges.ndim < 3:
        # Reshape into 3-D array for broadcasting if neded
        merges = np.stack((merges[:, [0, 1]], merges[:, [0, 2]]), axis=1)

    # Compare existing merges with available indices
    # Swap trap and label axes for the merges array to correctly cast
    # valid_ndmerges = merges.swapaxes(1, 2)[..., None] == indices.T[:, None, :]
    valid_ndmerges = merges[..., None] == indices.T[None, ...]

    # Broadcasting is confusing (but efficient):
    # First we check the dimension across trap and cell id, to ensure both match
    valid_cell_ids = valid_ndmerges.all(axis=2)

    # Then we check the merge tuples to check which cases have both target and source
    valid_merges = valid_cell_ids.any(axis=2).all(axis=1)

    # Finalle we check the dimension that crosses all indices, to ensure the pair
    # is present in a valid merge event.
    valid_indices = valid_ndmerges[valid_merges].all(axis=2).any(axis=(0, 1))

    return valid_merges, valid_indices


def join_tracks_pair(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Join two tracks and return the new value of the target.
    TODO replace this with arrays only.
    """
    target_copy = copy(target)
    end = find_1st(target_copy[::-1], 0, cmp_larger)
    target_copy[-end:] = source[-end:]
    return target_copy
