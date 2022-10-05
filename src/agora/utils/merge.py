#!/usr/bin/env jupyter
"""
Functions to efficiently merge rows in DataFrames.
"""
import typing as t
from copy import copy

import numpy as np
import pandas as pd
from utils_find_1st import cmp_larger, find_1st

from agora.utils.association import validate_association


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

    valid_merges, indices = validate_association(
        merges, np.array(list(data.index))
    )

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


def join_tracks_pair(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Join two tracks and return the new value of the target.
    TODO replace this with arrays only.
    """
    target_copy = copy(target)
    end = find_1st(target_copy[::-1], 0, cmp_larger)
    target_copy[-end:] = source[-end:]
    return target_copy
