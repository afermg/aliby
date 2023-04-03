#!/usr/bin/env jupyter
"""
Functions to efficiently merge rows in DataFrames.
"""
import typing as t
from copy import copy

import numpy as np
import pandas as pd
from utils_find_1st import cmp_larger, find_1st

from agora.utils.indexing import compare_indices, validate_association


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

    indices = data.index
    if "mother_label" in indices.names:
        indices = indices.droplevel("mother_label")
    valid_merges, indices = validate_association(
        merges, np.array(list(indices))
    )

    # Assign non-merged
    merged = data.loc[~indices]

    # Implement the merges and drop source rows.
    # TODO Use matrices to perform merges in batch
    # for ecficiency
    if valid_merges.any():
        to_merge = data.loc[indices]
        targets, sources = zip(*merges[valid_merges])
        for source, target in zip(sources, targets):
            target = tuple(target)
            to_merge.loc[target] = join_tracks_pair(
                to_merge.loc[target].values,
                to_merge.loc[tuple(source)].values,
            )
        to_merge.drop(map(tuple, sources), inplace=True)

        merged = pd.concat((merged, to_merge), names=data.index.names)
    return merged


def join_tracks_pair(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Join two tracks and return the new value of the target.
    """
    target_copy = target
    end = find_1st(target_copy[::-1], 0, cmp_larger)
    target_copy[-end:] = source[-end:]
    return target_copy


def group_merges(merges: np.ndarray) -> t.List[t.Tuple]:
    # Return a list where the cell is present as source and target
    # (multimerges)

    sources_targets = compare_indices(merges[:, 0, :], merges[:, 1, :])
    is_multimerge = sources_targets.any(axis=0) | sources_targets.any(axis=1)
    is_monomerge = ~is_multimerge

    multimerge_subsets = union_find(zip(*np.where(sources_targets)))
    merge_groups = [merges[np.array(tuple(x))] for x in multimerge_subsets]

    sorted_merges = list(map(sort_association, merge_groups))

    # Ensure that source and target are at the edges
    return [
        *sorted_merges,
        *[[event] for event in merges[is_monomerge]],
    ]


def union_find(lsts):
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def sort_association(array: np.ndarray):
    # Sort the internal associations

    order = np.where(
        (array[:, 0, ..., None] == array[:, 1].T[None, ...]).all(axis=1)
    )

    res = []
    [res.append(x) for x in np.flip(order).flatten() if x not in res]
    sorted_array = array[np.array(res)]
    return sorted_array


def merge_association(
    association: np.ndarray, merges: np.ndarray
) -> np.ndarray:
    grouped_merges = group_merges(merges)

    flat_indices = association.reshape(-1, 2)
    comparison_mat = compare_indices(merges[:, 0], flat_indices)

    valid_indices = comparison_mat.any(axis=0)

    if valid_indices.any():  # Where valid, perform transformation
        replacement_d = {}
        for dataset in grouped_merges:
            for k in dataset:
                replacement_d[tuple(k[0])] = dataset[-1][1]

        flat_indices[valid_indices] = [
            replacement_d[tuple(i)] for i in flat_indices[valid_indices]
        ]

    merged_indices = flat_indices.reshape(-1, 2, 2)
    return merged_indices
