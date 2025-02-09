#!/usr/bin/env jupyter
"""
Functions to efficiently merge rows in DataFrames.
"""

import typing as t

import numpy as np
import pandas as pd

from agora.utils.indexing import find_1st_greater, index_isin


def group_merges(merges: np.ndarray) -> t.List[t.Tuple]:
    """
    Convert merges into a list of merges for traps requiring multiple
    merges and then for traps requiring single merges.
    """
    left_tracks = merges[:, 0]
    right_tracks = merges[:, 1]
    # find traps requiring multiple merges
    linr = merges[index_isin(left_tracks, right_tracks).flatten(), :]
    rinl = merges[index_isin(right_tracks, left_tracks).flatten(), :]
    # make unique and order merges for each trap
    multi_merge = np.unique(np.concatenate((linr, rinl)), axis=0)
    # find traps requiring a singe merge
    single_merge = merges[~index_isin(merges, multi_merge).all(axis=1).flatten(), :]
    # convert to lists of arrays
    single_merge_list = [[sm] for sm in single_merge]
    multi_merge_list = [
        multi_merge[multi_merge[:, 0, 0] == trap_id, ...]
        for trap_id in np.unique(multi_merge[:, 0, 0])
    ]
    res = [*multi_merge_list, *single_merge_list]
    return res


def merge_lineage(lineage: np.ndarray, merges: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Use merges to update lineage information.

    Check if merging causes any buds to have multiple mothers and discard
    those incorrect merges.

    Return updated lineage and merge arrays.
    """
    flat_lineage = lineage.reshape(-1, 2)
    bud_mother_dict = {
        tuple(bud): mother for bud, mother in zip(lineage[:, 1], lineage[:, 0])
    }
    left_tracks = merges[:, 0]
    # find left tracks that are in lineages
    valid_lineages = index_isin(flat_lineage, left_tracks).flatten()
    # group into multi- and then single merges
    grouped_merges = group_merges(merges)
    # perform merges
    if valid_lineages.any():
        # indices of each left track -> indices of rightmost right track
        replacement_dict = {
            tuple(contig_pair[0]): merge[-1][1]
            for merge in grouped_merges
            for contig_pair in merge
        }
        # if both key and value are buds, they must have the same mother
        buds = lineage[:, 1]
        incorrect_merges = [
            key
            for key in replacement_dict
            if np.any(index_isin(buds, replacement_dict[key]).flatten())
            and np.any(index_isin(buds, key).flatten())
            and not np.array_equal(
                bud_mother_dict[key],
                bud_mother_dict[tuple(replacement_dict[key])],
            )
        ]
        if incorrect_merges:
            # reassign incorrect merges so that they have no affect
            for key in incorrect_merges:
                replacement_dict[key] = key
            # find only correct merges
            new_merges = merges[
                ~index_isin(merges[:, 0], np.array(incorrect_merges)).flatten(),
                ...,
            ]
        else:
            new_merges = merges
        # correct lineage information
        # replace mother or bud index with index of rightmost track
        flat_lineage[valid_lineages] = [
            replacement_dict[tuple(index)] for index in flat_lineage[valid_lineages]
        ]
    else:
        new_merges = merges
    # reverse flattening
    new_lineage = flat_lineage.reshape(-1, 2, 2)
    # remove any duplicates
    new_lineage = np.unique(new_lineage, axis=0)
    return new_lineage, new_merges


def apply_merges(data: pd.DataFrame, merges: np.ndarray):
    """
    Generate a new data frame containing merged tracks.

    Parameters
    ----------
    data : pd.DataFrame
        A Signal data frame.
    merges : np.ndarray
        An array of pairs of (trap, cell) indices to merge.
    """
    indices = data.index
    if "mother_label" in indices.names:
        indices = indices.droplevel("mother_label")
    indices = np.array(list(indices))
    # merges in the data frame's indices
    valid_merges = index_isin(merges, indices).all(axis=1).flatten()
    # corresponding indices for the data frame in merges
    selected_merges = merges[valid_merges, ...]
    valid_indices = index_isin(indices, selected_merges).flatten()
    # data not requiring merging
    merged = data.loc[~valid_indices]
    # merge tracks
    if valid_merges.any():
        to_merge = data.loc[valid_indices].copy()
        left_indices = merges[valid_merges, 0]
        right_indices = merges[valid_merges, 1]
        # join left track with right track
        for left_index, right_index in zip(left_indices, right_indices):
            to_merge.loc[tuple(left_index)] = join_two_tracks(
                to_merge.loc[tuple(left_index)].values,
                to_merge.loc[tuple(right_index)].values,
            )
        # drop indices for right tracks
        to_merge.drop(map(tuple, right_indices), inplace=True)
        # add to data not requiring merges
        merged = pd.concat((merged, to_merge), names=data.index.names)
    return merged


def join_two_tracks(left_track: np.ndarray, right_track: np.ndarray) -> np.ndarray:
    """Join two tracks and return the new one."""
    new_track = left_track.copy()
    # find last positive element by inverting track
    end = find_1st_greater(left_track[::-1], 0)
    # merge tracks into one
    new_track[-end:] = right_track[-end:]
    return new_track


##################################################################


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

    order = np.where((array[:, 0, ..., None] == array[:, 1].T[None, ...]).all(axis=1))

    res = []
    [res.append(x) for x in np.flip(order).flatten() if x not in res]
    sorted_array = array[np.array(res)]
    return sorted_array
