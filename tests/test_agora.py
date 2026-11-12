"""
Check agora's writers, Cells, Signal, metadata and parameters.

The writers append to an h5 file one time point at a time, and Cells and
Signal read it back. A write that fails must leave the file as it was;
what is read must be what the file holds now, not what it held when the
reader was made.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from agora.abc import ParametersABC
from agora.io.cells import Cells, fill_outline
from agora.io.metadata import parse_microscopy_logs
from agora.io.signal import Signal
from agora.io.writers import (
    BabyWriter,
    ExtractorWriter,
    PostProcessorWriter,
    tidy,
)
from postprocessor.core.reshapers.picker import PickerParameters

TILE = 9
AREA = "extraction/general/null/area"


def cells_at(tp, labels, tile=TILE):
    """Return what BABY gives the writer for cells in trap 0."""
    n = len(labels)
    return {
        "trap": np.zeros(n, dtype=int),
        "cell_label": np.array(labels, dtype=int),
        "timepoint": np.full(n, tp),
        "mother_assign_dynamic": np.zeros(n, dtype=int),
        "edgemasks": np.zeros((n, tile, tile), dtype=bool),
    }


def lengths(path):
    """Return the length of each dataset of cell_info."""
    with h5py.File(path, "r") as f:
        return {key: len(value) for key, value in f["cell_info"].items()}


def areas(rows):
    """Return a Signal data frame from {(trap, cell_label): values}."""
    index = pd.MultiIndex.from_tuples(list(rows), names=("trap", "cell_label"))
    return pd.DataFrame(list(rows.values()), index=index, dtype=float)


@pytest.fixture
def signal_h5(tmp_path):
    """Write an h5 with the areas of two cells and a lineage."""
    path = tmp_path / "position.h5"
    PostProcessorWriter(path).add_df(
        "/" + AREA, areas({(0, 1): [9, 9, 9], (0, 2): [1, 1, 1]})
    )
    with h5py.File(path, "a") as f:
        f.attrs["time_settings/timeinterval"] = 300
        f.create_dataset("modifiers/lineage_merged", data=[[0, 1, 2]])
    return path


def test_a_write_that_fails_leaves_the_file_as_it_was(tmp_path):
    """Undo the datasets already written and raise."""
    # the datasets before the one that failed had the time point and
    # those after had not: trap, cell_label and timepoint then differed
    # in length and no time point could be read. The pipeline was told
    # only by a logged warning, and carried on
    path = tmp_path / "position.h5"
    writer = BabyWriter(path)
    writer.write(cells_at(0, [1, 2]), overwrite=[], tp=0, tile_size=TILE)
    before = lengths(path)
    bad = cells_at(1, [1, 2, 3])
    # masks of another tile's shape cannot be appended
    bad["edgemasks"] = np.zeros((3, TILE + 2, TILE + 2), dtype=bool)
    with pytest.raises(Exception):
        writer.write(bad, overwrite=[], tp=1, tile_size=TILE)
    assert lengths(path) == before
    assert Cells(path).labels_at_time(0) == {0: [1, 2]}
    # and the time point can be written when it is right
    writer.write(cells_at(1, [1, 2, 3]), overwrite=[], tp=1, tile_size=TILE)
    assert Cells(path).labels_at_time(1) == {0: [1, 2, 3]}


def test_a_first_write_that_fails_leaves_no_datasets(tmp_path):
    """Remove the datasets a failed first time point created."""
    path = tmp_path / "position.h5"
    bad = cells_at(0, [1, 2])
    bad["timepoint"] = np.zeros((2, 3, 4))
    with pytest.raises(Exception):
        BabyWriter(path).write(bad, overwrite=[], tp=0, tile_size=TILE)
    assert lengths(path) == {}


@pytest.mark.parametrize("no_cells", [0, 3, 500])
def test_chunks_do_not_depend_on_the_first_time_point(tmp_path, no_cells):
    """Give the lists of cells the same chunks however a movie begins."""
    # h5py made a chunk the size of the first time point, and a
    # position that began with three cells was read in chunks of three
    path = tmp_path / "position.h5"
    labels = list(range(1, no_cells + 1))
    BabyWriter(path).write(
        cells_at(0, labels), overwrite=[], tp=0, tile_size=TILE
    )
    with h5py.File(path, "r") as f:
        for name in ("trap", "cell_label", "timepoint"):
            assert f["cell_info"][name].chunks == (1024,)
            assert len(f["cell_info"][name]) == no_cells
        # and the masks, 16 whole tiles to a chunk
        assert f["cell_info/edgemasks"].chunks == (16, TILE, TILE)
        assert f["cell_info/edgemasks"].shape == (no_cells, TILE, TILE)


def test_writing_a_data_frame_leaves_it_as_it_was(tmp_path):
    """Write a Signal without resetting the index of the one passed."""
    data = areas({(0, 1): [9, 9, 9], (0, 2): [1, 1, 1]})
    before = data.copy()
    PostProcessorWriter(tmp_path / "position.h5").add_df("/" + AREA, data)
    pd.testing.assert_frame_equal(data, before)


def test_an_outline_is_filled_as_the_whole_tile_was():
    """Fill within the outline's bounding box and get the same mask."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        outline = np.zeros((30, 40), dtype=bool)
        top, left = rng.integers(0, 15, 2)
        height, width = rng.integers(1, 15, 2)
        outline[top : top + height, [left, left + width - 1]] = True
        outline[[top, top + height - 1], left : left + width] = True
        # a break in some outlines, which then enclose nothing
        if rng.random() < 0.3:
            outline[top, left] = False
        np.testing.assert_array_equal(
            fill_outline(outline), ndimage.binary_fill_holes(outline)
        )
    empty = np.zeros((30, 40), dtype=bool)
    assert not fill_outline(empty).any()


def write_cells(path, traps, labels, timepoints, mothers, ntraps=2):
    """Write the lists of cell_info that Cells finds lineages from."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "trap_info/trap_locations", data=np.zeros((ntraps, 2))
        )
        for name, values in (
            ("trap", traps),
            ("cell_label", labels),
            ("timepoint", timepoints),
            ("mother_assign_dynamic", mothers),
        ):
            f.create_dataset(
                "cell_info/" + name, data=np.array(values, dtype="uint16")
            )
    return path


def test_mothers_are_found_whatever_their_labels_sum_to(tmp_path):
    """Find the mother-bud pairs when the mothers' labels sum to 65536."""
    # the labels are uint16 and so was their sum, which was here zero:
    # the position was said to have no mothers
    path = write_cells(
        tmp_path / "position.h5",
        traps=[0, 0, 1, 1],
        labels=[40000, 40001, 25536, 25537],
        timepoints=[0, 0, 0, 0],
        mothers=[0, 40000, 0, 25536],
    )
    assert Cells(path).mothers_daughters.tolist() == [
        [0, 40000, 40001],
        [1, 25536, 25537],
    ]


def test_a_position_with_no_cells_has_no_time_points(tmp_path):
    """Count no time points and no cells, and raise nothing."""
    path = write_cells(tmp_path / "position.h5", [], [], [], [])
    cells = Cells(path)
    assert cells.ntimepoints == 0
    assert len(cells.mothers_daughters) == 0


def test_a_signal_reads_the_lineage_the_file_has_now(signal_h5):
    """Read a lineage written after the Signal last read one."""
    signal = Signal(signal_h5)
    assert signal.lineage().tolist() == [[0, 1, 2]]
    with h5py.File(signal_h5, "a") as f:
        del f["modifiers/lineage_merged"]
        f.create_dataset("modifiers/lineage_merged", data=[[0, 2, 1]])
    assert signal.lineage().tolist() == [[0, 2, 1]]


def test_a_lineage_asked_for_where_there_is_none_is_an_error(signal_h5):
    """Raise for a missing location and do not read another in its place."""
    with h5py.File(signal_h5, "a") as f:
        f.create_dataset("postprocessing/lineage", data=[[0, 7, 8]])
    signal = Signal(signal_h5)
    with pytest.raises(KeyError, match="modifiers/nowhere"):
        signal.lineage("modifiers/nowhere")
    assert signal.lineage("postprocessing/lineage").tolist() == [[0, 7, 8]]
    assert signal.lineage().tolist() == [[0, 1, 2]]


def test_an_old_files_lineage_is_read_by_default(signal_h5):
    """Fall back to postprocessing/lineage when no location is given."""
    with h5py.File(signal_h5, "a") as f:
        del f["modifiers/lineage_merged"]
        f.create_dataset("postprocessing/lineage", data=[[0, 7, 8]])
    assert Signal(signal_h5).lineage().tolist() == [[0, 7, 8]]
    with h5py.File(signal_h5, "a") as f:
        del f["postprocessing/lineage"]
    with pytest.raises(KeyError, match="lineage_merged"):
        Signal(signal_h5).lineage()


def test_a_signal_lists_signals_added_since_it_last_looked(signal_h5):
    """Find the signals and the time points the file has now."""
    signal = Signal(signal_h5)
    assert signal.available == [AREA]
    assert signal.ntimepoints == 3
    # the list is the caller's to change
    signal.available.append("nothing")
    assert signal.available == [AREA]
    writer = PostProcessorWriter(signal_h5)
    writer.add_df("/postprocessing/buddings/area", areas({(0, 1): [0, 1, 0]}))
    writer.add_df(
        "/" + AREA, areas({(0, 1): [9, 9, 9, 9]}), overwrite=True
    )
    assert signal.available == [AREA, "postprocessing/buddings/area"]
    assert signal.ntimepoints == 4


def test_an_unchanged_file_is_listed_once(signal_h5, monkeypatch):
    """Visit the file again only when it has changed."""
    signal = Signal(signal_h5)
    visits = []
    find_available = signal.find_available
    monkeypatch.setattr(
        signal,
        "find_available",
        lambda: visits.append(1) or find_available(),
    )
    for _ in range(3):
        assert signal.available == [AREA]
    assert len(visits) == 1


def test_a_raw_signal_refuses_an_argument_it_does_not_know(signal_h5):
    """Raise for a misspelt argument, which was passed over in silence."""
    with pytest.raises(TypeError):
        Signal(signal_h5).get_raw(AREA, in_minute=False)


def test_a_text_file_that_is_no_log_is_no_metadata(tmp_path):
    """Find no metadata in a folder whose only text file is some notes."""
    # any .txt was taken for a legacy log, and parsing the log and acq
    # files that were not there raised KeyError
    (tmp_path / "notes.txt").write_text("the lamp was replaced today")
    assert parse_microscopy_logs(tmp_path) is None


class SomeParameters(ParametersABC):
    """Parameters with a list and a dict among their defaults."""

    _defaults = {"sequence": [["a", 1]], "tree": {"reduce": "null"}}


def test_default_parameters_are_not_shared():
    """Change one instance's parameters and leave the defaults alone."""
    first = SomeParameters.default()
    first.sequence.append(["b", 2])
    first.tree["reduce"] = "max"
    assert SomeParameters.default().sequence == [["a", 1]]
    assert SomeParameters.default().tree == {"reduce": "null"}
    picker = PickerParameters.default()
    picker.picker_sequence.append(["condition", "present", 9])
    assert ["condition", "present", 9] not in (
        PickerParameters.default().picker_sequence
    )


def test_parameters_with_the_value_null_become_a_dict():
    """Convert parameters holding "null", which once recursed for ever."""
    assert SomeParameters.default().to_dict() == {
        "sequence": [["a", 1]],
        "tree": {"reduce": "null"},
    }


def signals_at(tp, labels):
    """Return what the extractor gives the writer at one time point."""
    index = pd.MultiIndex.from_tuples(
        [(0, label) for label in labels], names=("trap", "cell_label")
    )
    traps = pd.Index([0, 1], name="trap")
    return {
        "general/null/area": pd.DataFrame(
            {tp: [10.5 * label + tp for label in labels]}, index=index
        ),
        "GFP/max/median_background": pd.DataFrame(
            {tp: [100.0 + tp, 200.0 + tp]}, index=traps
        ),
    }


def tables(path):
    """Return each signal's table in the h5 file, as it is stored."""
    with pd.HDFStore(path, mode="r") as store:
        return {key: store[key] for key in store.keys()}


# cells come and go, and one time point has none
LABELS_AT = [[1, 2], [1, 2, 3], [], [2, 3], [3], [1, 3, 4], [4]]


def test_signals_written_together_are_those_written_one_by_one(tmp_path):
    """Store the same tables whether time points are kept or not."""
    one_by_one = ExtractorWriter(tmp_path / "one.h5")
    together = ExtractorWriter(tmp_path / "together.h5", flush_every=3)
    for tp, labels in enumerate(LABELS_AT):
        one_by_one.write(signals_at(tp, labels))
        together.write(signals_at(tp, labels))
    together.flush()
    first = tables(tmp_path / "one.h5")
    second = tables(tmp_path / "together.h5")
    assert set(first) == set(second) and len(first) == 2
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    area = Signal(tmp_path / "together.h5").get_raw(AREA, in_minutes=False)
    assert area.columns.tolist() == list(range(7))
    assert area.loc[(0, 3)].dropna().index.tolist() == [1, 3, 4, 5]
    assert area.loc[(0, 3), 4] == 10.5 * 3 + 4


def test_signals_are_kept_until_enough_time_points_or_a_flush(tmp_path):
    """Write nothing until flush_every time points, and all at a flush."""
    path = tmp_path / "position.h5"
    writer = ExtractorWriter(path, flush_every=3)
    writer.write(signals_at(0, [1]))
    writer.write(signals_at(1, [1]))
    assert not path.exists()
    writer.write(signals_at(2, [1]))
    assert Signal(path).ntimepoints == 3
    writer.write(signals_at(3, [1]))
    assert Signal(path).ntimepoints == 3
    writer.flush()
    assert Signal(path).ntimepoints == 4
    # a flush with nothing waiting writes nothing more
    writer.flush()
    assert len(tables(path)["/extraction/general/null/area"]) == 4


def test_what_is_written_is_the_data_as_it_was_given(tmp_path):
    """Keep a time point's values, not the data frame they came in."""
    path = tmp_path / "position.h5"
    writer = ExtractorWriter(path, flush_every=2)
    data = signals_at(0, [1])
    writer.write(data)
    data["general/null/area"].iloc[0, 0] = -1.0
    writer.flush()
    written = tables(path)["/extraction/general/null/area"]
    assert written["value"].tolist() == [10.5]


def test_a_tidy_signal_is_the_melted_data_frame():
    """Give the long table that resetting the index and melting gave."""
    data = areas({(0, 1): [9, np.nan, 7], (2, 5): [1, 2, np.nan]})
    data.columns = [4, 5, 6]
    melted = data.reset_index().melt(
        id_vars=["trap", "cell_label"], var_name="time", value_name="value"
    )
    tidied = pd.DataFrame(tidy(data))
    assert tidied.columns.tolist() == ["trap", "cell_label", "time", "value"]
    assert tidied.dtypes.astype(str).tolist() == [
        "int16",
        "int16",
        "int32",
        "float32",
    ]
    np.testing.assert_array_equal(
        tidied.to_numpy(dtype=float), melted.to_numpy(dtype=float)
    )
    # a signal of traps has a cell_label of -1, as its last column
    background = pd.DataFrame(
        {3: [1.0, 2.0]}, index=pd.Index([0, 1], name="trap")
    )
    tidied = pd.DataFrame(tidy(background))
    assert tidied.columns.tolist() == ["trap", "time", "value", "cell_label"]
    assert tidied["cell_label"].tolist() == [-1, -1]
