"""
Check that a time point's masks, traps and labels stay in step.

Cells.at_time matches masks to traps by position, and the extractor
matches those masks to Cells.labels_at_time by position again
(extraction/core/extractor.py, get_outlines). A cell dropped from one
list and kept in the other therefore hands every later cell in the time
point the trap and the label of the cell before it.
"""

import h5py
import numpy as np
import pytest

from agora.io.cells import Cells

TILE = 8


def ring(box=(2, 2, 6, 6)):
    """Return a cell's boundary, as BABY stores it."""
    mask = np.zeros((TILE, TILE), dtype=bool)
    top, left, bottom, right = box
    mask[top:bottom, left] = True
    mask[top:bottom, right - 1] = True
    mask[top, left:right] = True
    mask[bottom - 1, left:right] = True
    return mask


def write_h5(path, rows, ntraps=3):
    """Write an h5 holding one time point's cells, in trap order."""
    with h5py.File(path, "w") as f:
        trap_info = f.create_group("trap_info")
        trap_info.create_dataset(
            "trap_locations",
            data=np.arange(ntraps * 2, dtype=float).reshape(ntraps, 2),
        )
        cell_info = f.create_group("cell_info")
        cell_info.create_dataset(
            "trap", data=np.array([r[0] for r in rows], dtype="uint16")
        )
        cell_info.create_dataset(
            "timepoint", data=np.array([r[1] for r in rows], dtype="uint16")
        )
        cell_info.create_dataset(
            "cell_label", data=np.array([r[2] for r in rows], dtype="uint16")
        )
        cell_info.create_dataset(
            "edgemasks", data=np.array([r[3] for r in rows], dtype=bool)
        )
    return path


@pytest.fixture
def empty_mask_h5(tmp_path):
    """One time point whose second cell has an empty edge mask."""
    rows = [
        (0, 0, 1, ring()),
        (0, 0, 2, np.zeros((TILE, TILE), dtype=bool)),
        (1, 0, 1, ring(box=(1, 1, 5, 5))),
        (2, 0, 1, ring(box=(3, 3, 7, 7))),
    ]
    return write_h5(tmp_path / "position.h5", rows)


@pytest.mark.parametrize("kind", ["mask", "edgemask"])
def test_cells_stay_in_their_own_trap(empty_mask_h5, kind):
    """Give each trap its own cells, empty edge mask or not."""
    cells = Cells.from_source(empty_mask_h5)
    at_time = cells.at_time(0, kind=kind)
    assert [len(at_time[trap]) for trap in (0, 1, 2)] == [2, 1, 1]
    # trap 1's cell is the one drawn at (1, 1): it must not be trap 0's
    assert at_time[1][0][1, 1]
    assert not at_time[2][0][1, 1]
    assert at_time[2][0][3, 3]


def test_masks_and_labels_match_cell_for_cell(empty_mask_h5):
    """Keep at_time and labels_at_time countable against each other."""
    cells = Cells.from_source(empty_mask_h5)
    at_time = cells.at_time(0, kind="edgemask")
    labels = cells.labels_at_time(0)
    assert {trap: len(masks) for trap, masks in at_time.items()} == {
        trap: len(trap_labels) for trap, trap_labels in labels.items()
    }
    assert labels[0] == [1, 2]
    assert not at_time[0][1].any()


def test_a_time_point_without_cells_is_empty(empty_mask_h5):
    cells = Cells.from_source(empty_mask_h5)
    assert cells.at_time(5) == {0: [], 1: [], 2: []}
    assert cells.labels_at_time(5) == {0: [], 1: [], 2: []}


def test_mask_kind_fills_the_boundary(empty_mask_h5):
    """A filled mask holds its interior; an edge mask holds the ring."""
    cells = Cells.from_source(empty_mask_h5)
    filled = cells.at_time(0, kind="mask")[1][0]
    outline = cells.at_time(0, kind="edgemask")[1][0]
    assert filled[2, 2] and not outline[2, 2]
    assert filled.sum() > outline.sum()


def test_a_rectangular_tile_stores_masks_of_its_own_shape(tmp_path):
    """
    Check the h5 a tall tile writes is self-consistent.

    The edgemask dataset's shape is the tile's, so a mother machine's
    channel has to reach it as two numbers; declared square, every stored
    mask would be clipped to the shorter side. Cells reads the size back off
    that raster rather than off the attr, so the two must agree.
    """
    from agora.io.writers import BabyWriter

    path = tmp_path / "pos001.h5"
    masks = np.zeros((2, 240, 80), dtype=bool)
    masks[0, 10:20, 5:15] = True
    masks[1, 200:210, 60:70] = True
    BabyWriter(path).write(
        data={
            "edgemasks": masks,
            "trap": np.array([0, 0]),
            "cell_label": np.array([1, 2]),
            "timepoint": np.array([0, 0]),
        },
        overwrite=[],
        tp=0,
        tile_size=(240, 80),
    )

    with h5py.File(path, "r") as h5:
        assert h5["cell_info/edgemasks"].shape == (2, 240, 80)
    assert Cells(path).tile_size == (240, 80)
