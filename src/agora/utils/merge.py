#!/usr/bin/env jupyter
"""
Functions to efficiently merge rows in DataFrames.
"""
import typing as t
from copy import copy

import numpy as np
import pandas as pd
from utils_find_1st import cmp_larger, find_1st

from agora.utils.indexing import (
    index_isin,
    compare_indices,
    validate_association,
)


def apply_merges(data: pd.DataFrame, merges: np.ndarray):
    """
    Split data in two, one subset for rows relevant for merging and one
    without them.

    Use an array of source tracklets and target tracklets
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
    # for efficiency
    if valid_merges.any():
        to_merge = data.loc[indices].copy()
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
    """
    Convert merges into a list of merges for traps requiring multiple
    merges and then for traps requiring single merges.
    """
    left_track = merges[:, 0]
    right_track = merges[:, 1]
    # find traps requiring multiple merges
    linr = merges[index_isin(left_track, right_track).flatten(), :]
    rinl = merges[index_isin(right_track, left_track).flatten(), :]
    # make unique and order merges for each trap
    multi_merge = np.unique(np.concatenate((linr, rinl)), axis=0)
    # find traps requiring a singe merge
    single_merge = merges[
        ~index_isin(merges, multi_merge).all(axis=1).flatten(), :
    ]
    # convert to lists of arrays
    single_merge_list = [[sm] for sm in single_merge]
    multi_merge_list = [
        multi_merge[multi_merge[:, 0, 0] == trap_id, ...]
        for trap_id in np.unique(multi_merge[:, 0, 0])
    ]
    res = [*multi_merge_list, *single_merge_list]
    # #
    # sources_targets = compare_indices(merges[:, 0, :], merges[:, 1, :])
    # is_multimerge = sources_targets.any(axis=0) | sources_targets.any(axis=1)
    # is_monomerge = ~is_multimerge
    # multimerge_subsets = union_find(zip(*np.where(sources_targets)))
    # merge_groups = [merges[np.array(tuple(x))] for x in multimerge_subsets]
    # sorted_merges = list(map(sort_association, merge_groups))
    # res = [
    #     *sorted_merges,
    #     *[[event] for event in merges[is_monomerge]],
    # ]
    # #
    return res


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


def merge_association(lineage: np.ndarray, merges: np.ndarray) -> np.ndarray:
    """Use merges to update lineage information."""
    flat_lineage = lineage.reshape(-1, 2)
    left_track = merges[:, 0]
    # comparison_mat = compare_indices(left_track, flat_lineage)
    # valid_indices = comparison_mat.any(axis=0)
    valid_lineages = index_isin(flat_lineage, left_track).flatten()
    # group into multi- and single merges
    grouped_merges = group_merges(merges)
    # perform merges
    if valid_lineages.any():
        # indices of each left track -> indices of rightmost track
        replacement_dict = {
            tuple(contig_pair[0]): merge[-1][1]
            for merge in grouped_merges
            for contig_pair in merge
        }
        # correct lineage information
        # replace mother or bud index with index of rightmost track
        flat_lineage[valid_lineages] = [
            replacement_dict[tuple(i)] for i in flat_lineage[valid_lineages]
        ]
    # reverse flattening
    new_lineage = flat_lineage.reshape(-1, 2, 2)
    # remove any duplicates
    new_lineage = np.unique(new_lineage, axis=0)
    # buds should have only one mother
    buds = new_lineage[:, 1]
    ubuds, counts = np.unique(buds, axis=0, return_counts=True)
    duplicate_buds = ubuds[counts > 1, :]
    # duplicates
    new_lineage[index_isin(buds, duplicate_buds).flatten(), ...]
    # original
    lineage[index_isin(lineage[:, 1], duplicate_buds).flatten(), ...]
    breakpoint()
    return new_lineage
