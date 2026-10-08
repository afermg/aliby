"""
Check that merging tracks gives one cell one name, and all of its data.

The postprocessor finds tracks that are one cell, lost by the segmenter
and found again under a new label. apply_merges joins their Signals and
merge_lineage renames the lineage to match: both know the cell by the
label of its first track. Tracks chain, and a trap may hold more than
one chain.
"""

import numpy as np
import pandas as pd
import pytest

from agora.utils.indexing import assoc_indices_to_2d, assoc_indices_to_3d
from agora.utils.merge import apply_merges, find_chains, merge_lineage

nan = np.nan


def merges_of(*triples):
    """Return merges from (trap, left, right) triples."""
    return np.array(
        [[[trap, left], [trap, right]] for trap, left, right in triples]
    )


def merged(lineage, merges):
    """Return the merged lineage as rows and the merges that were kept."""
    new_lineage, new_merges = merge_lineage(
        assoc_indices_to_3d(np.array(lineage)), merges
    )
    kept = [(m[0][0], m[0][1], m[1][1]) for m in new_merges.tolist()]
    return assoc_indices_to_2d(new_lineage).tolist(), kept


def signal(rows, names=("trap", "cell_label")):
    """Return a Signal data frame from {index: values}."""
    index = pd.MultiIndex.from_tuples(list(rows), names=names)
    return pd.DataFrame(list(rows.values()), index=index, dtype=float)


def tracks(data):
    """Return a data frame's rows as {index: values}, NaN as None."""
    return {
        index: [None if np.isnan(value) else value for value in row]
        for index, row in zip(data.index.tolist(), data.to_numpy().tolist())
    }


def test_a_cell_lost_twice_takes_its_first_label():
    """Follow a chain of merges back to its start."""
    # mother 1 has bud 2, which is lost and found as 3, and again as 4
    lineage, kept = merged(
        [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        merges_of((0, 2, 3), (0, 3, 4)),
    )
    assert lineage == [[0, 1, 2]]
    assert kept == [(0, 2, 3), (0, 3, 4)]


def test_two_cells_each_lost_twice_stay_two_cells():
    """Follow each chain of a trap to its own start."""
    # one trap, two mothers, each lost twice: 2 -> 3 -> 4 and 5 -> 6 -> 7,
    # and each has a bud at the end. Renamed as a group, as they once
    # were, both mothers are one cell and buds 8 and 9 share a mother
    lineage, _kept = merged(
        [[0, 4, 8], [0, 7, 9]],
        merges_of((0, 2, 3), (0, 3, 4), (0, 5, 6), (0, 6, 7)),
    )
    assert lineage == [[0, 2, 8], [0, 5, 9]]


def test_a_merge_of_buds_with_different_mothers_is_dropped():
    """Discard a merge that would give a bud two mothers."""
    # buds 2 and 3 have mothers 1 and 5, so they are not one cell
    lineage, kept = merged(
        [[0, 1, 2], [0, 5, 3]],
        merges_of((0, 2, 3)),
    )
    assert lineage == [[0, 1, 2], [0, 5, 3]]
    assert kept == []


def test_a_dropped_merge_ends_the_chain_it_was_part_of():
    """Rename a track only through merges that were kept."""
    # 2 -> 3 -> 4, and bud 3 has another mother than bud 4: the second
    # merge is dropped, so 3 becomes 2 and 4 stays 4
    lineage, kept = merged(
        [[0, 1, 2], [0, 1, 3], [0, 5, 4]],
        merges_of((0, 2, 3), (0, 3, 4)),
    )
    assert kept == [(0, 2, 3)]
    assert lineage == [[0, 1, 2], [0, 5, 4]]


def test_buds_of_different_mothers_are_parted_across_other_tracks():
    """Discard a merge where a chain's buds change mother."""
    # 21 -> 25 -> 29, where 21 is a bud of 19, 29 a bud of 11 and 25 is
    # no bud at all: no two tracks merged directly disagree, and the chain
    # still gives one cell two mothers
    lineage, kept = merged(
        [[0, 19, 21], [0, 11, 29]],
        merges_of((0, 21, 25), (0, 25, 29)),
    )
    assert kept == [(0, 21, 25)]
    assert lineage == [[0, 11, 29], [0, 19, 21]]


def test_a_bud_lost_with_its_mother_is_still_one_bud():
    """Compare mothers as merged cells."""
    # mother 1 is found again as 3 and her bud 2 as 5: the bud's two
    # tracks name mothers 1 and 3, which are one cell
    lineage, kept = merged(
        [[0, 1, 2], [0, 3, 5]],
        merges_of((0, 1, 3), (0, 2, 5)),
    )
    assert kept == [(0, 1, 3), (0, 2, 5)]
    assert lineage == [[0, 1, 2]]


def test_the_lineage_passed_is_left_as_it_was():
    """Rename a copy of the lineage."""
    lineage = np.array([[[0, 1], [0, 3]]])
    new_lineage, _kept = merge_lineage(lineage, merges_of((0, 2, 3)))
    assert lineage.tolist() == [[[0, 1], [0, 3]]]
    assert new_lineage.tolist() == [[[0, 1], [0, 2]]]


def test_the_lineage_names_cells_the_merged_signal_holds():
    """Name a merged cell alike in its lineage and in its Signal."""
    # the lineage named a merged cell by its last track and the Signal by
    # its first, so a quarter of real mother-bud pairs named a cell the
    # merged Signal did not hold
    merges = merges_of((0, 1, 3), (0, 2, 5))
    data = signal(
        {
            (0, 1): [9, 9, nan, nan],
            (0, 2): [1, 2, nan, nan],
            (0, 3): [nan, nan, 9, 9],
            (0, 5): [nan, nan, 3, 4],
        }
    )
    lineage, kept = merged([[0, 3, 5]], merges)
    held = set(apply_merges(data, np.array(merges)).index.tolist())
    assert kept == [(0, 1, 3), (0, 2, 5)]
    assert all(
        (trap, mother) in held and (trap, bud) in held
        for trap, mother, bud in lineage
    )


def test_a_merged_track_holds_both_tracks_under_the_first_label():
    """Join two tracks and leave the others alone."""
    data = signal(
        {
            (0, 1): [9, 9, 9, 9, 9, 9],
            (0, 2): [1, 2, 3, nan, nan, nan],
            (0, 5): [nan, nan, nan, 4, 5, 6],
        }
    )
    assert tracks(apply_merges(data, merges_of((0, 2, 5)))) == {
        (0, 1): [9, 9, 9, 9, 9, 9],
        (0, 2): [1, 2, 3, 4, 5, 6],
    }


def test_a_cell_lost_twice_is_joined_whatever_the_order_of_its_merges():
    """Join a whole chain, first track to last."""
    # joined a merge at a time, with 2 -> 3 before 3 -> 4, the first track
    # was given the second before the second had been given the third
    data = signal(
        {
            (0, 2): [1, 2, nan, nan, nan, nan],
            (0, 3): [nan, nan, 3, 4, nan, nan],
            (0, 4): [nan, nan, nan, nan, 5, 6],
        }
    )
    whole = {(0, 2): [1, 2, 3, 4, 5, 6]}
    assert tracks(apply_merges(data, merges_of((0, 2, 3), (0, 3, 4)))) == whole
    assert tracks(apply_merges(data, merges_of((0, 3, 4), (0, 2, 3)))) == whole


def test_a_value_of_zero_or_less_is_a_measurement():
    """Keep the left track to the last time point its cell was seen."""
    # a background-subtracted signal: where the left track ended was found
    # from its last positive value, so these two became NaN
    data = signal(
        {
            (0, 2): [1, -2, 0, nan, nan, nan],
            (0, 5): [nan, nan, nan, 4, 5, 6],
        }
    )
    assert tracks(apply_merges(data, merges_of((0, 2, 5)))) == {
        (0, 2): [1, -2, 0, 4, 5, 6]
    }


def test_a_frame_missing_between_two_tracks_stays_missing():
    """Take nothing from a track for a time point it has no value at."""
    data = signal(
        {
            (0, 2): [1, 2, nan, nan, nan, nan],
            (0, 5): [nan, nan, nan, 4, 5, 6],
        }
    )
    assert tracks(apply_merges(data, merges_of((0, 2, 5)))) == {
        (0, 2): [1, 2, None, 4, 5, 6]
    }


def test_tracks_are_joined_when_the_index_holds_the_mother():
    """Join tracks of a Signal that has mother_label in its index."""
    # found by trap and cell_label alone, a row of this data frame came
    # back as a table of one row, and the left track was lost whole
    data = signal(
        {
            (0, 1, 0): [9, 9, 9, 9, 9, 9],
            (0, 2, 1): [1, 2, 3, nan, nan, nan],
            (0, 5, 0): [nan, nan, nan, 4, 5, 6],
        },
        names=("trap", "cell_label", "mother_label"),
    )
    assert tracks(apply_merges(data, merges_of((0, 2, 5)))) == {
        (0, 1, 0): [9, 9, 9, 9, 9, 9],
        (0, 2, 1): [1, 2, 3, 4, 5, 6],
    }


def test_a_merge_of_a_track_the_signal_lacks_is_passed_over():
    """Apply only merges of tracks that are both in the data frame."""
    data = signal({(0, 2): [1, 2, nan], (0, 7): [5, 5, 5]})
    assert tracks(apply_merges(data, merges_of((0, 2, 5)))) == tracks(data)


@pytest.mark.parametrize(
    "merges, message",
    [
        # one track carries on into two
        (merges_of((0, 2, 3), (0, 2, 4)), "both"),
        # two tracks carry on into one
        (merges_of((0, 2, 4), (0, 3, 4)), "both"),
        # a loop, which once was passed over in silence
        (merges_of((0, 2, 3), (0, 3, 2)), "loop"),
        # a loop with a track leading into it, which once never returned
        (merges_of((0, 1, 2), (0, 2, 3), (0, 3, 2)), "both"),
        # a track merged with itself
        (merges_of((0, 2, 2)), "loop"),
        # a loop beside a sound chain
        (merges_of((0, 5, 6), (0, 2, 3), (0, 3, 2)), "loop"),
    ],
)
def test_merges_that_are_not_chains_are_refused(merges, message):
    """Raise an error for merges that do not join tracks end to end."""
    # the merger pairs tracks one to one and forwards in time, so these
    # come only from merges made elsewhere
    with pytest.raises(ValueError, match=message):
        find_chains(merges)
    with pytest.raises(ValueError, match=message):
        merge_lineage(assoc_indices_to_3d(np.array([[0, 1, 2]])), merges)


def test_a_merge_listed_twice_is_one_merge():
    """Follow a chain whose merges are repeated."""
    merges = merges_of((0, 2, 3), (0, 3, 4), (0, 2, 3))
    assert find_chains(merges) == [[(0, 2), (0, 3), (0, 4)]]
