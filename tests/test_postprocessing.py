"""
Check merging and picking, and that a file can be processed twice.

The postprocessor finds tracks that are one cell (merger), says which
cells are worth keeping (picker), and then counts buddings and measures
buds from the Signals so merged and picked. Each step must work from
what this run found, not from what an earlier run left in the h5 file.
"""

import logging

import h5py
import numpy as np
import pandas as pd
import pytest

from agora.io.cells import Cells
from agora.io.signal import Signal
from agora.io.writers import PostProcessorWriter
from agora.utils.indexing import validate_lineage
from postprocessor.core.postprocessing import (
    PostProcessor,
    PostProcessorParameters,
)
from postprocessor.core.reshapers.bud_metric import BudMetric
from postprocessor.core.reshapers.buddings import buddings, buddingsParameters
from postprocessor.core.reshapers.picker import Picker, PickerParameters
from postprocessor.core.reshapers.tracks import get_merges
from postprocessor.grouper import Grouper

nan = np.nan
NTPS = 10


def signal(rows):
    """Return a Signal data frame from {(trap, cell_label): values}."""
    index = pd.MultiIndex.from_tuples(list(rows), names=("trap", "cell_label"))
    return pd.DataFrame(list(rows.values()), index=index, dtype=float)


def seen(value, start, end):
    """Return a track with a value from start up to end and NaN elsewhere."""
    return [value if start <= tp < end else nan for tp in range(NTPS)]


def merged_pairs(data):
    """Return the merges found as (left label, right label) pairs."""
    return sorted((m[0][1], m[1][1]) for m in get_merges(data).tolist())


def test_two_tracks_end_and_two_begin():
    """Join each track that ends to the track that carries on from it."""
    # cell 1 carries on as cell 4, and cell 2 is followed by cell 3, of
    # twice its size and so another cell. With as many tracks ending as
    # beginning, left and right were exchanged: 2 was joined to 3
    data = signal(
        {
            (0, 1): seen(100, 0, 5),
            (0, 2): seen(200, 0, 5),
            (0, 3): seen(400, 5, 10),
            (0, 4): seen(100, 5, 10),
        }
    )
    assert merged_pairs(data) == [(1, 4)]


def test_three_tracks_end_and_three_begin():
    """Join three cells each to its own continuation."""
    data = signal(
        {
            (0, 1): seen(100, 0, 5),
            (0, 2): seen(200, 0, 5),
            (0, 3): seen(300, 0, 5),
            (0, 4): seen(300, 5, 10),
            (0, 5): seen(100, 5, 10),
            (0, 6): seen(200, 5, 10),
        }
    )
    assert merged_pairs(data) == [(1, 5), (2, 6), (3, 4)]


@pytest.mark.parametrize(
    "tracks, wanted",
    [
        # more tracks end than begin
        ({1: (100, 0, 5), 2: (200, 0, 5), 3: (200, 5, 10)}, [(2, 3)]),
        # more tracks begin than end
        ({1: (200, 0, 5), 2: (100, 5, 10), 3: (200, 5, 10)}, [(1, 3)]),
    ],
)
def test_unequal_numbers_of_tracks_end_and_begin(tracks, wanted):
    """Join the right pair when the numbers of tracks differ."""
    data = signal(
        {(0, label): seen(*track) for label, track in tracks.items()}
    )
    assert merged_pairs(data) == wanted


def test_a_picker_with_no_lineage_still_picks_by_condition():
    """Pick by a condition in a movie too short to have a lineage."""
    # the picker's own advice for short movies; it picked nothing
    data = signal({(0, 1): seen(1, 0, 10), (0, 2): seen(1, 0, 2)})
    picker = Picker(
        PickerParameters.from_dict(
            {"picker_sequence": [["condition", "present", 3]]}
        )
    )
    picker.lineage = np.array([])
    assert picker.run(data) == [(0, 1)]


def test_a_picker_asked_for_a_lineage_picks_none_without_one():
    """Pick no cell by lineage when no cell is in one."""
    data = signal({(0, 1): seen(1, 0, 10), (0, 2): seen(1, 0, 10)})
    picker = Picker(PickerParameters.default())
    picker.lineage = np.array([])
    assert picker.run(data) == []


def with_mothers(rows):
    """Return a Signal from {(trap, cell_label, mother_label): values}."""
    index = pd.MultiIndex.from_tuples(
        list(rows), names=("trap", "cell_label", "mother_label")
    )
    return pd.DataFrame(list(rows.values()), index=index, dtype=float)


def test_a_signals_index_gives_its_lineage_with_the_daughter_last():
    """Read a cell that names its mother as that mother's daughter."""
    # cell 2 is a bud of cell 1, and cells 1 and 3 have no mother. The
    # index was taken as it stood, (trap, cell, mother), and read as
    # (trap, mother, daughter): every cell was a mother, cell 2 of cell 1
    # and cells 1 and 3 of a cell 0
    data = with_mothers(
        {
            (0, 1, 0): seen(9, 0, 10),
            (0, 2, 1): seen(1, 4, 10),
            (0, 3, 0): seen(5, 0, 10),
        }
    )
    picker = Picker(
        PickerParameters.from_dict(
            {"picker_sequence": [["lineage", "mothers"]]}
        )
    )
    assert picker.get_lineage_information(data).tolist() == [[0, 1, 2]]
    assert picker.run(data) == [(0, 1, 0)]


def test_validate_lineage_takes_the_lineage_its_docstring_shows():
    """Validate a lineage of [[trap, mother], [trap, daughter]] pairs."""
    lineage = np.array(
        [
            [[0, 1], [0, 3]],
            [[0, 1], [0, 4]],
            [[0, 1], [0, 6]],
            [[0, 4], [0, 7]],
        ]
    )
    indices = np.array([[0, 1], [0, 2], [0, 3]])
    valid_lineage, valid_indices, returned = validate_lineage(lineage, indices)
    assert valid_lineage.tolist() == [True, False, False, False]
    assert valid_indices.tolist() == [True, False, True]
    assert returned.shape == (4, 2, 2)


def test_validate_lineage_finds_no_cells_in_an_empty_lineage():
    """Match nothing when there are no mother-bud pairs."""
    # a position with no buds raised an IndexError, which wela caught
    # and reported as an error in the lineage
    indices = np.array([[0, 1], [0, 2]])
    for lineage in (np.array([]), np.empty((0, 3), dtype=int)):
        valid_lineage, valid_indices, _returned = validate_lineage(
            lineage, indices, "daughters"
        )
        assert valid_lineage.tolist() == []
        assert valid_indices.tolist() == [False, False]


def test_validate_lineage_refuses_a_how_it_does_not_know():
    """Raise an error that names the mistake."""
    with pytest.raises(ValueError, match="mother"):
        validate_lineage(np.array([[0, 1, 2]]), np.array([[0, 1]]), "mother")


def test_buddings_takes_a_lineage():
    """Find buddings from a lineage passed as an array."""
    data = signal({(0, 1): seen(9, 0, 10), (0, 2): seen(1, 4, 10)})
    process = buddings(buddingsParameters.default())
    found = process.run(data, lineage=np.array([[0, 1, 2]]))
    assert found.loc[(0, 1)].tolist() == [tp == 4 for tp in range(NTPS)]


@pytest.fixture
def position_h5(tmp_path):
    """
    Write an h5 of one trap: a mother and a bud that is lost and found.

    The mother is cell 1. Her bud is cell 2 for three time points and
    cell 3 for the next four, and BABY says each is a bud of cell 1.
    """
    path = tmp_path / "position.h5"
    tracks = {(0, 1): (300, 0, 10), (0, 2): (50, 3, 6), (0, 3): (50, 6, 10)}
    for name in ("area", "volume"):
        PostProcessorWriter(path).add_df(
            f"/extraction/general/null/{name}",
            signal({index: seen(*track) for index, track in tracks.items()}),
        )
    rows = [
        (index, tp, 0 if index == (0, 1) else 1)
        for tp in range(NTPS)
        for index, (_value, start, end) in tracks.items()
        if start <= tp < end
    ]
    with h5py.File(path, "a") as f:
        f.attrs["time_settings/timeinterval"] = 300
        f.create_group("trap_info").create_dataset(
            "trap_locations", data=np.array([[10.0, 10.0]])
        )
        cell_info = f.create_group("cell_info")
        for name, values in (
            ("trap", [index[0] for index, _tp, _mother in rows]),
            ("cell_label", [index[1] for index, _tp, _mother in rows]),
            ("timepoint", [tp for _index, tp, _mother in rows]),
            ("mother_assign_dynamic", [mother for *_rest, mother in rows]),
        ):
            cell_info.create_dataset(name, data=np.array(values, "uint16"))
    return path


def postprocess(path):
    """Run the postprocessor, write what it finds and return it as lists."""
    result = PostProcessor(path, PostProcessorParameters.default()).run()
    found = {
        "merges": np.asarray(result["merges"]).tolist(),
        "lineage": np.asarray(result["lineage_merged"]).tolist(),
        "picks": sorted(map(tuple, np.asarray(result["picks"]).tolist())),
        "others": {
            key: value.sort_index().to_numpy().tolist()
            for key, value in result.items()
            if isinstance(value, pd.DataFrame)
        },
    }
    found["others"] = {
        key: [[None if np.isnan(v) else v for v in row] for row in rows]
        for key, rows in found["others"].items()
    }
    PostProcessorWriter(path).write(data=result)
    return found


def test_cells_are_picked_from_the_merged_signal(position_h5):
    """Pick a cell for what its merged track is."""
    # the bud's first track is three time points long, too few to be
    # picked, and its second is four. Merged it is seven and is cell 2.
    # Picked before merging, as a new file was, the bud is lost: its
    # first track is too short, and its second no longer has a name
    found = postprocess(position_h5)
    assert found["merges"] == [[[0, 2], [0, 3]]]
    assert found["lineage"] == [[0, 1, 2]]
    assert found["picks"] == [(0, 1), (0, 2)]


def test_a_file_processed_twice_gives_the_same_results(position_h5):
    """Work from this run's merges and picks, not those in the file."""
    first = postprocess(position_h5)
    second = postprocess(position_h5)
    assert first["others"]
    assert second == first


def test_the_unmerged_lineage_of_a_file_is_read_when_asked_for(position_h5):
    """Read modifiers/lineage when the merged lineage is not wanted."""
    # the name looked for had a bracket in it and was never found
    with h5py.File(position_h5, "a") as f:
        modifiers = f.create_group("modifiers")
        modifiers.create_dataset("lineage", data=np.array([[0, 1, 3]]))
        modifiers.create_dataset("lineage_merged", data=np.array([[0, 1, 2]]))
    picker = Picker(
        PickerParameters.default(), cells=Cells.from_source(position_h5)
    )
    assert picker.get_lineage_information().tolist() == [[0, 1, 2]]
    assert picker.get_lineage_information(merged=False).tolist() == [[0, 1, 3]]


def test_a_raw_signal_gives_each_bud_its_own_mother(position_h5):
    """Find a bud's mother by the bud, not by its place in the lineage."""
    # a merged lineage is sorted by mother and a Signal by cell: given
    # out in order, the mothers of a third of real buds went to other
    # buds of their trap
    index = [(0, 1), (0, 2), (0, 3), (0, 5)]
    PostProcessorWriter(position_h5).add_df(
        "/extraction/general/null/eccentricity",
        signal({cell: seen(1, 0, 10) for cell in index}),
    )
    with h5py.File(position_h5, "a") as f:
        f.create_group("modifiers").create_dataset(
            "lineage_merged", data=np.array([[0, 1, 5], [0, 2, 3]])
        )
    data = Signal(position_h5).get_raw(
        "extraction/general/null/eccentricity", lineage=True
    )
    assert data.index.tolist() == [(0, 1, 0), (0, 2, 0), (0, 3, 2), (0, 5, 1)]


def test_a_bud_in_the_first_image_processed_is_not_a_budding():
    """Know the first image by the first column, not by time point zero."""
    # a movie processed from time point 5: the bud was there when
    # processing began, and was counted as budding at 5
    data = signal({(0, 1): seen(9, 0, 10), (0, 2): seen(1, 0, 10)})
    data.columns = range(5, 5 + NTPS)
    found = buddings(buddingsParameters.default()).run(
        data, lineage=np.array([[0, 1, 2]])
    )
    assert not found.to_numpy().any()
    # and one that appears later is
    data = signal({(0, 1): seen(9, 0, 10), (0, 2): seen(1, 4, 10)})
    data.columns = range(5, 5 + NTPS)
    found = buddings(buddingsParameters.default()).run(
        data, lineage=np.array([[0, 1, 2]])
    )
    assert found.columns[found.loc[(0, 1)]].tolist() == [9]


def test_bud_metric_leaves_the_signal_as_it_was():
    """Find the buds' data without adding mother_label to the Signal."""
    data = signal({(0, 1): seen(9, 0, 10), (0, 2): seen(1, 4, 10)})
    before = data.copy()
    found = BudMetric.as_function(data, lineage=np.array([[0, 1, 2]]))
    pd.testing.assert_frame_equal(data, before)
    assert found.index.tolist() == [(0, 1)]
    assert found.loc[(0, 1)].dropna().tolist() == [1] * 6


def test_a_bud_process_takes_parameters(position_h5):
    """Run a bud process with the parameters it is given."""
    # they were looked for as an attribute named for the process, which
    # there never is
    parameters = PostProcessorParameters.default()
    parameters.param_sets["bud_processes"] = {
        "buddings": {"lineage_location": "elsewhere"}
    }
    result = PostProcessor(position_h5, parameters).run()
    assert "/postprocessing/buddings/extraction_general_null_volume" in result


def picking(sequence):
    """Return postprocessor parameters with a picker sequence."""
    parameters = PostProcessorParameters.default().to_dict()
    parameters["param_sets"]["merging_picking"]["picker_params"] = {
        "picker_sequence": sequence
    }
    return PostProcessorParameters.from_dict(parameters)


def test_a_position_with_no_cell_picked_has_no_cells(position_h5):
    """Keep no cell when the picker picked none."""
    # no cell is seen at more than 50 time points. Picks of no cells were
    # read as no picking, and the file gave back every cell it held
    result = PostProcessor(
        position_h5, picking([["condition", "present", 50]])
    ).run()
    assert len(result["picks"]) == 0
    assert all(
        value.empty
        for value in result.values()
        if isinstance(value, pd.DataFrame)
    )
    PostProcessorWriter(position_h5).write(data=result)
    signal = Signal(position_h5)
    assert signal.read_picks() == set()
    assert signal.get("/extraction/general/null/volume").empty
    assert len(signal.get_raw("/extraction/general/null/volume")) == 3


def test_a_file_that_was_never_picked_gives_every_cell(position_h5):
    """Tell a file with no picks from one whose picker picked none."""
    signal = Signal(position_h5)
    assert signal.read_picks() is None
    assert len(signal.get("/extraction/general/null/volume")) == 3


def test_an_empty_picker_sequence_keeps_every_cell(position_h5):
    """Pick every cell when asked for no picking."""
    result = PostProcessor(position_h5, picking([])).run()
    picks = sorted(map(tuple, np.asarray(result["picks"]).tolist()))
    # the bud's two tracks are merged into cell 2
    assert picks == [(0, 1), (0, 2)]


def test_picked_cells_keep_the_order_of_the_signal():
    """Pick rows without reordering them, whatever the index holds."""
    picks = {(0, 2), (0, 3)}
    data = signal({(0, 3): [1.0], (0, 1): [2.0], (0, 2): [3.0]})
    picked = Signal.apply_merging_picking(
        None, data, merges=False, picks=picks
    )
    assert picked.index.tolist() == [(0, 3), (0, 2)]
    # picks are trap and cell, so with the mother in the index too no
    # row matched and none was kept
    data = with_mothers({(0, 3, 1): [1.0], (0, 1, 0): [2.0], (0, 2, 1): [3.0]})
    picked = Signal.apply_merging_picking(
        None, data, merges=False, picks=picks
    )
    assert picked.index.tolist() == [(0, 3, 1), (0, 2, 1)]


def test_a_picker_says_which_pick_left_no_cells():
    """Warn when a pick removes the last cells, naming the pick."""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("aliby")
    logger.addHandler(handler)
    data = signal({(0, 1): seen(1, 0, 10), (0, 2): seen(1, 0, 10)})
    picker = Picker(PickerParameters.default())
    picker.lineage = np.array([])
    try:
        assert picker.run(data) == []
    finally:
        logger.removeHandler(handler)
    messages = [record.getMessage() for record in records]
    assert sum("No cells picked" in message for message in messages) == 1
    assert any("['lineage', 'families']" in message for message in messages)


def test_signals_are_concatenated_without_a_cutoff(position_h5):
    """Keep every picked cell when no cutoff is given."""
    postprocess(position_h5)
    grouper = Grouper(position_h5.parent)
    volume = grouper.concat_signal("extraction/general/null/volume")
    assert len(volume) == 2
