"""
Check that merging tracks renames a lineage cell by cell.

The postprocessor finds tracks that are one cell, lost by the segmenter
and found again under a new label, and merge_lineage renames the lineage
to match: every track of a cell takes the label of the last. Tracks
chain, and a trap may hold more than one chain.
"""

import numpy as np

from agora.utils.indexing import assoc_indices_to_2d, assoc_indices_to_3d
from agora.utils.merge import merge_lineage


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


def test_a_cell_lost_twice_takes_its_last_label():
    """Follow a chain of merges to its end."""
    # mother 1 has bud 2, which is lost and found as 3, and again as 4
    lineage, kept = merged(
        [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        merges_of((0, 2, 3), (0, 3, 4)),
    )
    assert lineage == [[0, 1, 4]]
    assert kept == [(0, 2, 3), (0, 3, 4)]


def test_two_cells_each_lost_twice_stay_two_cells():
    """Follow each chain of a trap to its own end."""
    # one trap, two mothers, each lost twice: 2 -> 3 -> 4 and 5 -> 6 -> 7.
    # Given the right-hand label of the trap's last merge, both mothers
    # are cell 7 and buds 8 and 9 have the same mother
    lineage, _kept = merged(
        [[0, 2, 8], [0, 5, 9]],
        merges_of((0, 2, 3), (0, 3, 4), (0, 5, 6), (0, 6, 7)),
    )
    assert lineage == [[0, 4, 8], [0, 7, 9]]


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
    # merge is dropped, so 2 becomes 3 and no more
    lineage, kept = merged(
        [[0, 1, 2], [0, 1, 3], [0, 5, 4]],
        merges_of((0, 2, 3), (0, 3, 4)),
    )
    assert kept == [(0, 2, 3)]
    assert lineage == [[0, 1, 3], [0, 5, 4]]


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
    assert lineage == [[0, 11, 29], [0, 19, 25]]


def test_a_bud_lost_with_its_mother_is_still_one_bud():
    """Compare mothers as merged cells."""
    # mother 1 is found again as 3 and her bud 2 as 5: the bud's two
    # tracks name mothers 1 and 3, which are one cell
    lineage, kept = merged(
        [[0, 1, 2], [0, 3, 5]],
        merges_of((0, 1, 3), (0, 2, 5)),
    )
    assert kept == [(0, 1, 3), (0, 2, 5)]
    assert lineage == [[0, 3, 5]]
