"""
Check where extraction says a cell sits in the image.

A cell's centroid is measured in its tile, and the tile moves, so turning
one into an image coordinate means adding the tile's origin. That origin is
sooth's, the definition of record: it is where the tile was actually cut.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sooth

from extraction.core.extractor import Extractor

CENTROIDS_YX = [(10.5, 3.25), (30.0, 12.0)]
CENTRES_YX = [[100, 60], [140, 200]]


class FakeTileLocs:
    """Stand in for TileLocations, holding centres that do not drift."""

    def __init__(self, centres_yx):
        self.centres = np.asarray(centres_yx, dtype=float)

    def centres_at_time(self, tp):
        """Return every trap's centre at a time point."""
        return self.centres


def extractor_with(tile_size):
    """Return something with just the attributes the method reads."""
    return SimpleNamespace(
        tiler=SimpleNamespace(
            tile_size=tile_size,
            tile_locs=FakeTileLocs(CENTRES_YX),
            spatial_location=None,
        )
    )


def extract_dict(tp=0):
    """Return the centroid frames extraction builds, one cell per trap."""
    index = pd.MultiIndex.from_arrays(
        [[0, 1], [1, 1]], names=["trap", "cell_label"]
    )
    return {
        "general/null/centroid_x": pd.DataFrame(
            {tp: [centroid[1] for centroid in CENTROIDS_YX]}, index=index
        ),
        "general/null/centroid_y": pd.DataFrame(
            {tp: [centroid[0] for centroid in CENTROIDS_YX]}, index=index
        ),
    }


@pytest.mark.parametrize("tile_size", [117, 96, (48, 16)])
def test_a_cell_is_placed_where_its_tile_was_cut(tile_size):
    """
    Check the image coordinate is the tile's origin plus the centroid.

    The origin is ``centre - size // 2`` on each axis. Extraction used the
    tile's *geometric* centre, ``(size - 1) / 2``, which is half a pixel
    away from it for an even size -- so every cell of an even-sized tile
    was placed half a pixel down and to the right of where its tile sat.
    Every aliby run so far has used 117, which is odd, and for an odd size
    the two agree exactly.
    """
    result = extract_dict()
    Extractor.add_spatial_locations_of_cells(
        extractor_with(tile_size), result
    )

    for row, (centroid, centre) in enumerate(zip(CENTROIDS_YX, CENTRES_YX)):
        wanted_y, wanted_x = sooth.tile_to_image_yx(
            np.asarray(centroid), centre, tile_size
        )
        assert result["general/null/image_x"][0].iloc[row] == wanted_x
        assert result["general/null/image_y"][0].iloc[row] == wanted_y


def test_a_position_whose_only_trap_is_trap_zero_is_placed_too():
    """
    Check a single trap named zero still gets image coordinates.

    A trap is a name, not a count. Asking ``np.any(traps)`` asks whether any
    trap is named something other than zero, so a position holding one trap
    -- or a per-trap crop, which holds exactly one -- had its cells left at
    their tile-local centroids, silently. A normal run has traps 0..N, so
    some name is non-zero and the question happened to answer correctly.
    """
    index = pd.MultiIndex.from_arrays(
        [[0], [1]], names=["trap", "cell_label"]
    )
    result = {
        "general/null/centroid_x": pd.DataFrame({0: [3.25]}, index=index),
        "general/null/centroid_y": pd.DataFrame({0: [10.5]}, index=index),
    }
    Extractor.add_spatial_locations_of_cells(extractor_with(117), result)

    wanted_y, wanted_x = sooth.tile_to_image_yx(
        np.asarray(CENTROIDS_YX[0]), CENTRES_YX[0], 117
    )
    assert result["general/null/image_x"][0].iloc[0] == wanted_x
    assert result["general/null/image_y"][0].iloc[0] == wanted_y


class FakeTile:
    """Stand in for a Tile, cut at a fixed place."""

    def __init__(self, slices):
        self.slices = slices

    def as_range(self, tp):
        """Return the tile's slices, which do not drift."""
        return self.slices


def square(top, left, side=6, shape=(20, 20)):
    """Return a square cell mask."""
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def extractor_for_obscured(slices, image_yx=(100, 100), exclude=True):
    """Return something with what find_obscured and the functions read."""
    return SimpleNamespace(
        params=SimpleNamespace(exclude_obscured=exclude),
        tiler=SimpleNamespace(
            shape=(1, 1, 1, *image_yx),
            tile_size=20,
            tile_locs=SimpleNamespace(tiles=[FakeTile(slices)]),
        ),
        obscured={},
        pdms_mask=None,
        cell_fun_names={"area"},
        all_funs={
            "area": lambda masks, trap, channels: masks.sum(axis=(1, 2)),
            "background_area": lambda masks, trap, channels, exclude_mask: (
                ~masks.any(axis=0)
            ).sum(),
        },
    )


def test_a_cell_cut_off_by_its_tile_is_obscured():
    """
    Check a cell touching the tile border is set aside.

    Its area is that of the part inside the tile, not of the cell.
    """
    masks = [np.stack([square(7, 7), square(0, 7), square(7, 14)])]
    labels = {0: [1, 2, 3]}
    extractor = extractor_for_obscured((slice(40, 60), slice(40, 60)))

    obscured = Extractor.find_obscured(extractor, 0, masks, labels)

    assert obscured == {0: {2, 3}}


def test_a_cell_over_a_tile_padded_beyond_the_image_is_obscured():
    """
    Check a cell drawn over padding is set aside.

    The tile hangs three columns off the image's right edge, where tiler
    repeats the last column, so a cell ending two columns from the tile's
    border is drawn over pixels that were never imaged.
    """
    masks = [np.stack([square(7, 7), square(7, 12)])]
    labels = {0: [1, 2]}
    extractor = extractor_for_obscured((slice(40, 60), slice(83, 103)))

    obscured = Extractor.find_obscured(extractor, 0, masks, labels)

    assert obscured == {0: {2}}


def test_obscured_cells_are_kept_when_asked():
    masks = [np.stack([square(0, 7)])]
    extractor = extractor_for_obscured(
        (slice(40, 60), slice(40, 60)), exclude=False
    )

    assert Extractor.find_obscured(extractor, 0, masks, {0: [1]}) == {}


def test_an_obscured_cell_is_not_measured_but_is_not_background():
    """
    Check an obscured cell is skipped by cell functions alone.

    Its pixels are still a cell's, so a background estimate that counted
    them would be contaminated by its fluorescence.
    """
    masks = [np.stack([square(7, 7), square(0, 7)])]
    extractor = extractor_for_obscured((slice(40, 60), slice(40, 60)))
    extractor.obscured = {0: {2}}
    tile = np.zeros((20, 20))

    areas, cells = Extractor.apply_extraction_function(
        extractor, [tile], masks, "area", {0: [1, 2]}, ["GFP"]
    )
    (background,), _ = Extractor.apply_extraction_function(
        extractor, [tile], masks, "background_area", {0: [1, 2]}, ["GFP"]
    )

    assert cells == ((0, 1),)
    assert areas == (36,)
    assert background == 400 - 2 * 36
