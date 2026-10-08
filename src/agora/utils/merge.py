"""Functions to efficiently merge rows in DataFrames."""

import typing as t

import numpy as np
import pandas as pd
from agora.utils.indexing import index_isin


def find_chains(merges: np.ndarray) -> t.List[t.List[t.Tuple]]:
    """
    Find the chains of tracks that merges join into single cells.

    A merge joins a left track to the right track that carries on from it.
    Merges chain: a cell lost twice by the segmenter is three tracks and
    two merges. Each chain is followed from its own first track to its own
    last, in whatever order the merges are listed, so two chains in one
    trap stay two cells.

    Parameters
    ----------
    merges: np.ndarray
        An array of pairs of (trap, cell) indices to merge.

    Returns
    -------
    list
        The (trap, cell) indices of the tracks of each chain, from its
        first track to its last.

    Raises
    ------
    ValueError
        If a track is the left of two merges or the right of two, or if
        merges form a loop: a track carries on into one track and from
        one track, and never into itself.
    """
    # a merge listed twice is one merge
    pairs = dict.fromkeys(
        (tuple(left), tuple(right)) for left, right in merges.tolist()
    )
    right_of, left_of = {}, {}
    for left, right in pairs:
        if left in right_of:
            raise ValueError(
                f"Track {left} is merged with both {right_of[left]}"
                f" and {right}."
            )
        if right in left_of:
            raise ValueError(
                f"Tracks {left_of[right]} and {left} are both merged"
                f" with {right}."
            )
        right_of[left] = right
        left_of[right] = left
    chains = []
    for track in right_of:
        if track in left_of:
            # not the first track of its chain
            continue
        chain = [track]
        while chain[-1] in right_of:
            chain.append(right_of[chain[-1]])
        chains.append(chain)
    # a loop has no first track, so no chain reaches its merges
    in_chains = {track for chain in chains for track in chain[:-1]}
    in_loops = [track for track in right_of if track not in in_chains]
    if in_loops:
        raise ValueError(f"Merges form a loop through tracks {in_loops}.")
    return chains


def chain_starts(merges: np.ndarray) -> t.Dict[t.Tuple, t.Tuple]:
    """
    Find the first track of its chain of merges for each right track.

    A merged cell is known by the index of its first track, which is the
    index apply_merges keeps for its merged Signal.
    """
    return {
        track: chain[0]
        for chain in find_chains(merges)
        for track in chain[1:]
    }


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
    starts = chain_starts(merges)

    def mother_of(track):
        """Return the merged mother of a track, or None if not a bud."""
        mother = bud_mother_dict.get(track)
        return None if mother is None else starts.get(mother, mother)

    incorrect_merges = []
    for chain in find_chains(merges):
        mother = mother_of(chain[0])
        for track, next_track in zip(chain, chain[1:]):
            next_mother = mother_of(next_track)
            if None not in (mother, next_mother) and mother != next_mother:
                incorrect_merges.append(track)
            if next_mother is not None:
                mother = next_mother
    return incorrect_merges


def merge_lineage(
    lineage: np.ndarray, merges: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Use merges to update lineage information.

    Every track of a merged cell takes the index of the first track of its
    own chain of merges, which is the index its merged Signal has.

    Check if merging causes any buds to have multiple mothers and discard
    these incorrect merges. A discarded merge ends the chain it was part
    of: a track is renamed only through merges that are kept, so that the
    lineage and the merges returned describe the same cells.

    Return updated lineage and merge arrays.
    """
    # a copy, so that the lineage passed is not renamed too
    flat_lineage = lineage.reshape(-1, 2).copy()
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
        # indices of each right track -> indices of first track of its chain
        replacement_dict = chain_starts(new_merges)
        # find right tracks that are in lineages
        valid_lineages = index_isin(flat_lineage, new_merges[:, 1]).flatten()
        if valid_lineages.any():
            # replace mother or bud index with index of first track
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

    The tracks of each chain of merges are joined into one, which keeps the
    index of the chain's first track. A chain is joined whole, from its
    first track to its last, in whatever order its merges are listed.

    Only merges of tracks that are both in the data frame are applied.

    Parameters
    ----------
    data : pd.DataFrame
        A Signal data frame, with trap and cell_label in its index and
        optionally mother_label.
    merges : np.ndarray
        An array of pairs of (trap, cell) indices to merge.
    """
    indices = data.index
    if "mother_label" in indices.names:
        indices = indices.droplevel("mother_label")
    # the row of each track
    rows = {tuple(index): row for row, index in enumerate(indices)}
    # merges in the data frame's indices
    selected_merges = np.array(
        [
            merge
            for merge in merges.tolist()
            if tuple(merge[0]) in rows and tuple(merge[1]) in rows
        ]
    )
    if not len(selected_merges):
        return data.copy()
    values = data.to_numpy(copy=True)
    is_merged = np.zeros(len(data), dtype=bool)
    is_right = np.zeros(len(data), dtype=bool)
    # join each chain's tracks into its first track
    for chain in find_chains(selected_merges):
        first = rows[chain[0]]
        is_merged[first] = True
        for track in chain[1:]:
            values[first] = join_two_tracks(values[first], values[rows[track]])
            is_right[rows[track]] = True
    joined = pd.DataFrame(values, index=data.index, columns=data.columns)
    # data not requiring merging and then the merged tracks
    in_a_merge = is_merged | is_right
    merged = pd.concat(
        (joined.loc[~in_a_merge], joined.loc[is_merged]),
        names=data.index.names,
    )
    return merged


def join_two_tracks(
    left_track: np.ndarray, right_track: np.ndarray
) -> np.ndarray:
    """
    Join two tracks and return the new one.

    The left track is kept up to the last time point at which its cell
    was seen, and the right track is taken from there on. A cell that was
    not seen has NaN, so a value of zero or less is a measurement like any
    other and is kept.
    """
    new_track = left_track.copy()
    # find the last time point at which the left track has a value
    seen = np.flatnonzero(~pd.isna(left_track))
    end = seen[-1] + 1 if len(seen) else 0
    # merge tracks into one
    new_track[end:] = right_track[end:]
    return new_track
