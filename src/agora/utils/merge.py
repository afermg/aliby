"""Functions to efficiently merge rows in DataFrames."""

import typing as t

import numpy as np
import pandas as pd
from agora.utils.indexing import find_1st_greater, index_isin


def chain_ends(merges: np.ndarray) -> t.Dict[t.Tuple, t.Tuple]:
    """
    Find the last track of the chain of merges that each left track begins.

    A merge joins a left track to the right track that carries on from it.
    Merges chain: a cell lost twice by the segmenter is three tracks and
    two merges. Each chain is followed to its own end, so two chains in one
    trap stay two cells.

    Parameters
    ----------
    merges: np.ndarray
        An array of pairs of (trap, cell) indices to merge.

    Returns
    -------
    dict
        The (trap, cell) index of the last track of its chain for each
        left track.
    """
    right_of = {tuple(left): tuple(right) for left, right in merges.tolist()}
    ends = {}
    for track in right_of:
        end = track
        seen = {track}
        # stop if a chain comes back on itself
        while end in right_of and right_of[end] not in seen:
            end = right_of[end]
            seen.add(end)
        ends[track] = end
    return ends


def find_incorrect_merges(
    merges: np.ndarray, bud_mother_dict: t.Dict[t.Tuple, t.Tuple]
) -> t.List[t.Tuple]:
    """
    Find merges that give a bud two mothers.

    Each chain of merges is followed from its first track. Where a track is
    a bud of a different mother from the bud before it in the chain, the two
    are not one cell and the merge that joins the chain to that track is
    incorrect. The tracks between them need not be buds.

    Two mothers are the same if they are tracks of one merged cell.

    Parameters
    ----------
    merges: np.ndarray
        An array of pairs of (trap, cell) indices to merge.
    bud_mother_dict: dict
        The (trap, cell) index of the mother of each bud.

    Returns
    -------
    list
        The left track of each incorrect merge.
    """
    ends = chain_ends(merges)
    right_of = {tuple(left): tuple(right) for left, right in merges.tolist()}

    def mother_of(track):
        """Return the merged mother of a track, or None if not a bud."""
        mother = bud_mother_dict.get(track)
        return None if mother is None else ends.get(mother, mother)

    incorrect_merges = []
    for track in set(right_of) - set(right_of.values()):
        mother = mother_of(track)
        while track in right_of:
            next_track = right_of[track]
            next_mother = mother_of(next_track)
            if None not in (mother, next_mother) and mother != next_mother:
                incorrect_merges.append(track)
            if next_mother is not None:
                mother = next_mother
            track = next_track
    return incorrect_merges


def merge_lineage(
    lineage: np.ndarray, merges: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Use merges to update lineage information.

    Every track of a merged cell takes the index of the last track of its
    own chain of merges.

    Check if merging causes any buds to have multiple mothers and discard
    these incorrect merges. A discarded merge ends the chain it was part
    of: a track is renamed only through merges that are kept, so that the
    lineage and the merges returned describe the same cells.

    Return updated lineage and merge arrays.
    """
    flat_lineage = lineage.reshape(-1, 2)
    bud_mother_dict = {
        tuple(bud): tuple(mother)
        for bud, mother in zip(lineage[:, 1].tolist(), lineage[:, 0].tolist())
    }
    new_merges = merges
    # discarding a merge of mothers can make another merge incorrect
    while len(new_merges):
        incorrect_merges = find_incorrect_merges(new_merges, bud_mother_dict)
        if not incorrect_merges:
            break
        new_merges = new_merges[
            ~index_isin(
                new_merges[:, 0], np.array(incorrect_merges)
            ).flatten(),
            ...,
        ]
    if len(new_merges):
        # indices of each left track -> indices of last track of its chain
        replacement_dict = chain_ends(new_merges)
        # find left tracks that are in lineages
        valid_lineages = index_isin(flat_lineage, new_merges[:, 0]).flatten()
        if valid_lineages.any():
            # replace mother or bud index with index of last track
            flat_lineage[valid_lineages] = [
                replacement_dict[tuple(index)]
                for index in flat_lineage[valid_lineages].tolist()
            ]
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


def join_two_tracks(
    left_track: np.ndarray, right_track: np.ndarray
) -> np.ndarray:
    """Join two tracks and return the new one."""
    new_track = left_track.copy()
    # find last positive element by inverting track
    end = find_1st_greater(left_track[::-1], 0)
    # merge tracks into one
    new_track[-end:] = right_track[-end:]
    return new_track
