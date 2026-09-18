"""
Unit tests for aliby.tile.tiler and aliby.tile.tiles.

Uses small synthetic dask arrays; no file I/O, no OMERO connection,
no neural-network inference.
"""

import sys
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
from aliby.tile.tiler import (
    Tiler,
    TilerParameters,
    find_channel_index,
    find_channel_name,
)
from aliby.tile.tiles import TileLocations

# omero is an optional dependency — mock before any aliby.tile import
sys.modules.setdefault("omero", MagicMock())
sys.modules.setdefault("omero.gateway", MagicMock())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_image(T=3, C=2, Z=1, Y=128, X=128):
    arr = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    return da.from_array(arr, chunks=(1, 1, -1, -1, -1))


def _make_tiler(T=3, C=2, Y=128, X=128, channels=None):
    if channels is None:
        channels = ["Brightfield", "GFP"]
    image = _fake_image(T=T, C=C, Y=Y, X=X)
    params = TilerParameters.default()
    return Tiler(image, {"channels": channels}, params)


# ---------------------------------------------------------------------------
# find_channel_index / find_channel_name
# ---------------------------------------------------------------------------


def test_channel_index_exact():
    assert find_channel_index(["Brightfield", "GFP"], "Brightfield") == 0


def test_channel_index_exact_second():
    assert find_channel_index(["Brightfield", "GFP"], "GFP") == 1


def test_channel_index_case_insensitive():
    assert find_channel_index(["Brightfield", "GFP"], "brightfield") == 0


def test_channel_index_prefix_regex():
    assert find_channel_index(["Brightfield", "GFP"], "Bright") == 0


def test_channel_index_no_match():
    assert find_channel_index(["Brightfield", "GFP"], "mCherry") is None


def test_channel_name_returns_string():
    assert find_channel_name(["Brightfield", "GFP"], "GFP") == "GFP"


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------


def _tile_with_drifts(centre, tile_size, drifts):
    locs = TileLocations([centre], tile_size=tile_size, drifts=drifts)
    return locs.tiles[0]


def test_tile_centre_no_drift():
    tile = _tile_with_drifts([64, 64], 32, [[0, 0]])
    assert tile.centre_at_time(0) == [64, 64]


def test_tile_centre_cumulative_drift():
    # sum([[2,3],[1,-1]]) = [3, 2] → [64,64] - [3,2] = [61,62]
    tile = _tile_with_drifts([64, 64], 32, [[2, 3], [1, -1]])
    assert tile.centre_at_time(1) == [61, 62]


def test_tile_as_range_symmetric():
    tile = _tile_with_drifts([64, 64], 32, [[0, 0]])
    s0, s1 = tile.as_range(0)
    assert s0 == slice(48, 80)
    assert s1 == slice(48, 80)


# ---------------------------------------------------------------------------
# TileLocations
# ---------------------------------------------------------------------------


def test_tile_locations_len():
    locs = TileLocations([[10, 10], [50, 50]], tile_size=16)
    assert len(locs) == 2


def test_tile_locations_iter():
    locs = TileLocations([[10, 10], [50, 50]], tile_size=16)
    assert len(list(locs)) == 2


def test_tile_locations_centres_at_time():
    locs = TileLocations([[10, 20], [30, 40]], tile_size=16, drifts=[[1, 1]])
    centres = locs.centres_at_time(0)
    np.testing.assert_array_equal(centres, [[9, 19], [29, 39]])


def test_tile_locations_to_dict_tp0():
    locs = TileLocations([[10, 10]], tile_size=16, drifts=[[0, 0]])
    d = locs.to_dict(0)
    assert "trap_locations" in d
    assert "attrs/tile_size" in d
    assert d["attrs/tile_size"] == 16


def test_tile_locations_to_dict_tp_nonzero_no_locations():
    locs = TileLocations([[10, 10]], tile_size=16, drifts=[[0, 0], [1, 0]])
    d = locs.to_dict(1)
    assert "trap_locations" not in d
    assert "drifts" in d


# ---------------------------------------------------------------------------
# Tiler.get_tile_and_pad (static)
# ---------------------------------------------------------------------------


def test_get_tile_and_pad_no_padding():
    img = da.ones((1, 64, 64))
    slices = (slice(16, 48), slice(16, 48))
    tile = Tiler.get_tile_and_pad(img, slices, tile_size=32)
    arr = tile.compute()
    assert arr.shape == (1, 32, 32)
    assert not np.isnan(arr).any()


def test_get_tile_and_pad_edge_padding():
    img = da.ones((1, 64, 64))
    # 4 px outside the left edge, well within 25 % limit
    slices = (slice(-4, 28), slice(16, 48))
    tile = Tiler.get_tile_and_pad(img, slices, tile_size=32)
    arr = tile.compute()
    assert arr.shape == (1, 32, 32)
    assert not np.isnan(arr).any()


def test_get_tile_and_pad_nan_fill():
    img = da.ones((1, 64, 64))
    # 20 px outside of 32 px tile = 62.5 % → NaN fill
    slices = (slice(-20, 12), slice(16, 48))
    tile = Tiler.get_tile_and_pad(img, slices, tile_size=32)
    arr = tile.compute()
    assert arr.shape == (1, 32, 32)
    assert np.isnan(arr).all()


# ---------------------------------------------------------------------------
# Tiler.__init__
# ---------------------------------------------------------------------------


def test_tiler_channels_from_metadata():
    tiler = _make_tiler(channels=["Brightfield", "GFP"])
    assert tiler.channels == ["Brightfield", "GFP"]


def test_tiler_ref_channel_index_non_default_order():
    # Brightfield is second in the list
    tiler = _make_tiler(channels=["GFP", "Brightfield"])
    assert tiler.ref_channel_index == 1


def test_tiler_shape():
    tiler = _make_tiler(T=5, C=2, Y=64, X=64)
    assert tiler.shape == (5, 2, 1, 64, 64)


# ---------------------------------------------------------------------------
# Tiler.initialise_tiles
# ---------------------------------------------------------------------------


def test_initialise_tiles_filters_edge_traps():
    tiler = _make_tiler(Y=256, X=256)
    # one trap at centre, one too close to the corner
    with patch(
        "aliby.tile.tiler.segment_traps",
        return_value=[[128, 128], [5, 5]],
    ):
        tiler.initialise_tiles(tile_size=64)
    assert tiler.no_tiles == 1


def test_initialise_tiles_no_size_one_central_tile():
    tiler = _make_tiler(Y=128, X=128)
    tiler.initialise_tiles(tile_size=None)
    assert tiler.no_tiles == 1


# ---------------------------------------------------------------------------
# Tiler.find_drift
# ---------------------------------------------------------------------------


def test_find_drift_first_tp_appends():
    tiler = _make_tiler()
    tiler.tile_locs = TileLocations([[64, 64]], tile_size=32, drifts=[])
    with patch(
        "tiler.drift.phase_cross_correlation",
        return_value=(np.array([1.0, 2.0]), None, None),
    ):
        tiler.find_drift(0)
    assert len(tiler.tile_locs.drifts) == 1
    assert tiler.tile_locs.drifts[0] == [1.0, 2.0]


def test_find_drift_second_tp_appends():
    tiler = _make_tiler()
    tiler.tile_locs = TileLocations(
        [[64, 64]], tile_size=32, drifts=[[0.0, 0.0]]
    )
    with patch(
        "tiler.drift.phase_cross_correlation",
        return_value=(np.array([0.5, -1.0]), None, None),
    ):
        tiler.find_drift(1)
    assert len(tiler.tile_locs.drifts) == 2
    assert tiler.tile_locs.drifts[1] == [0.5, -1.0]


# ---------------------------------------------------------------------------
# bugs: non-square images, initial_tp, metadata
# ---------------------------------------------------------------------------


def test_get_tile_and_pad_interior_tile_of_tall_image():
    # clipping to the width called rows 250-290 of a 300-row image outside
    img = da.ones((1, 300, 200))
    slices = (slice(250, 290), slice(50, 90))
    arr = Tiler.get_tile_and_pad(img, slices, tile_size=40).compute()
    assert arr.shape == (1, 40, 40)
    assert not np.isnan(arr).any()


def test_get_tile_and_pad_pads_past_bottom_of_wide_image():
    # clipping to the width left a tile past the bottom 10 rows high
    img = da.ones((1, 200, 300))
    slices = (slice(190, 230), slice(250, 290))
    arr = Tiler.get_tile_and_pad(img, slices, tile_size=40).compute()
    assert arr.shape == (1, 40, 40)
    assert np.isnan(arr).all()
    slices = (slice(165, 205), slice(250, 290))
    arr = Tiler.get_tile_and_pad(img, slices, tile_size=40).compute()
    assert arr.shape == (1, 40, 40)
    assert not np.isnan(arr).any()


def test_initialise_tiles_keeps_traps_along_longer_axis():
    tiler = _make_tiler(Y=128, X=512)
    # both traps are well inside; the second is past x=128
    with patch(
        "aliby.tile.tiler.segment_traps",
        return_value=[[64, 64], [64, 400]],
    ):
        tiler.initialise_tiles(tile_size=64)
    assert tiler.no_tiles == 2


def test_initialise_tiles_drops_trap_near_edge_of_shorter_axis():
    tiler = _make_tiler(Y=512, X=128)
    # rows first: the second trap's column is too near the right edge
    with patch(
        "aliby.tile.tiler.segment_traps",
        return_value=[[256, 64], [256, 110]],
    ):
        tiler.initialise_tiles(tile_size=64)
    assert tiler.no_tiles == 1


def _moving_square_tiler(shifts, initial_processing_tp):
    T = len(shifts)
    arr = np.zeros((T, 1, 1, 64, 64), dtype=np.float32)
    for tp, shift in enumerate(shifts):
        arr[tp, 0, 0, 10 + shift : 20 + shift, 10:20] = 1
    params = TilerParameters.default().to_dict()
    params["initial_processing_tp"] = initial_processing_tp
    tiler = Tiler(
        da.from_array(arr),
        {"channels": ["Brightfield"]},
        TilerParameters.from_dict(params),
    )
    tiler.tile_locs = TileLocations([[32, 32]], tile_size=20, drifts=[])
    return tiler


def test_drifts_are_indexed_by_image_from_the_first_processed():
    # the square moves before image 2, which is ignored, and between
    # images 2 and 3; time points are the images' own indices
    tiler = _moving_square_tiler([0, 7, 0, 5], initial_processing_tp=2)
    tiler.find_drift(2)
    tiler.find_drift(3)
    assert tiler.tile_locs.drifts == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-5.0, 0.0],
    ]


def test_find_drift_refuses_a_time_point_before_the_first_processed():
    tiler = _moving_square_tiler([0, 0, 0, 5], initial_processing_tp=2)
    with pytest.raises(ValueError, match="before the first to process"):
        tiler.find_drift(1)


def test_run_tiles_from_the_first_processed_image():
    tiler = _moving_square_tiler([0, 0, 0, 5], initial_processing_tp=2)
    tiler.tile_size = 20
    with patch(
        "aliby.tile.tiler.segment_traps", return_value=[[32, 32]]
    ) as segment:
        tiler.run()
    # traps are found in image 2, not image 0
    np.testing.assert_array_equal(
        np.asarray(segment.call_args[0][0]),
        np.asarray(tiler.image[2, 0, 0]),
    )
    assert len(tiler.tile_locs.drifts) == 4


def test_first_processed_time_point_exports_every_drift_up_to_it():
    tiler = _moving_square_tiler([0, 0, 0, 5], initial_processing_tp=2)
    tiler.tile_size = 20
    with patch(
        "aliby.tile.tiler.segment_traps", return_value=[[32, 32]]
    ):
        first = tiler.run_tp(2)
        second = tiler.run_tp(3)
    assert "trap_locations" in first
    np.testing.assert_array_equal(first["drifts"], np.zeros((3, 2)))
    assert "trap_locations" not in second
    np.testing.assert_array_equal(second["drifts"], [[-5.0, 0.0]])


def test_tile_data_is_read_from_the_image_of_its_time_point():
    tiler = _moving_square_tiler([0, 0, 0, 5], initial_processing_tp=2)
    tiler.tile_locs = TileLocations([[15, 15]], tile_size=10, drifts=[])
    tiler.tile_locs.drifts = [[0, 0]] * 4
    tile = tiler.get_tile_data(0, tp=3, c=0, lazy=False)
    np.testing.assert_array_equal(
        tile, np.asarray(tiler.image[3, 0, :, 10:20, 10:20])
    )


def test_initial_tp_is_refused_with_its_new_name():
    with pytest.raises(ValueError, match="initial_processing_tp"):
        TilerParameters.default(initial_tp=2)


def test_tiler_takes_channels_from_image_metadata_without_microscopy():
    # a zarr image's data is not a dask array
    image = np.zeros((1, 2, 1, 32, 32))
    tiler = Tiler(
        image, {"channels": ["GFP", "Brightfield"]}, TilerParameters.default()
    )
    assert tiler.channels == ["GFP", "Brightfield"]
    assert tiler.ref_channel_index == 1


def test_tiler_imports_without_omero():
    import subprocess

    code = (
        "import sys; sys.modules['omero'] = None; "
        "sys.modules['omero.gateway'] = None; "
        "import aliby.tile.tiler"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# drift registered to the first image, checked against the previous
# ---------------------------------------------------------------------------


def _labelled_tiler(n_frames, initial_processing_tp=0, **parameters):
    # each image is filled with its own frame number, so a fake registration
    # can tell which images it was given
    arr = np.ones((n_frames, 1, 1, 8, 8), dtype=np.float32)
    arr *= np.arange(n_frames, dtype=np.float32)[:, None, None, None, None]
    params = TilerParameters.default(
        initial_processing_tp=initial_processing_tp, **parameters
    )
    tiler = Tiler(da.from_array(arr), {"channels": ["Brightfield"]}, params)
    tiler.tile_locs = TileLocations([[4, 4]], tile_size=4, drifts=[])
    return tiler


def _fake_registration(from_first, steps):
    # return from_first[frame] against the first image, steps[frame]
    # against the one before
    def register(reference, moving):
        ref, frame = int(np.asarray(reference)[0, 0]), int(
            np.asarray(moving)[0, 0]
        )
        if ref == frame:
            return np.zeros(2), 0.0, 0.0
        if frame - ref == 1 and not (ref == 0 and frame in from_first):
            return np.asarray(steps[frame], dtype=float), 0.0, 0.0
        return np.asarray(from_first[frame], dtype=float), 0.0, 0.0

    return register


def test_drifts_sum_to_each_image_displacement_from_the_first():
    tiler = _labelled_tiler(4)
    from_first = {1: [2, 0], 2: [3, -1], 3: [5, -1]}
    steps = {1: [2, 0], 2: [1, -1], 3: [2, 0]}
    with patch(
        "tiler.drift.phase_cross_correlation",
        side_effect=_fake_registration(from_first, steps),
    ):
        for tp in range(4):
            tiler.find_drift(tp)
    cumulative = np.cumsum(tiler.tile_locs.drifts, axis=0)
    np.testing.assert_array_equal(
        cumulative, [[0, 0], [2, 0], [3, -1], [5, -1]]
    )
    assert tiler.drift_disagreements == []


def test_drift_is_registered_to_the_first_processed_image():
    tiler = _labelled_tiler(5, initial_processing_tp=2)
    pairs = []

    def register(reference, moving):
        pairs.append(
            (int(np.asarray(reference)[0, 0]), int(np.asarray(moving)[0, 0]))
        )
        return np.zeros(2), 0.0, 0.0

    with patch(
        "tiler.drift.phase_cross_correlation", side_effect=register
    ):
        for tp in range(2, 5):
            tiler.find_drift(tp)
    # each image registers to the one before, never before image 2, and
    # to image 2
    assert pairs == [(2, 2), (2, 2), (2, 3), (2, 3), (3, 4), (2, 4)]


def test_one_bad_registration_is_flagged_and_does_not_persist():
    tiler = _labelled_tiler(5)
    # registration to the first image fails at time point 2 only
    from_first = {1: [1, 0], 2: [40, 30], 3: [3, 0], 4: [4, 0]}
    steps = {1: [1, 0], 2: [1, 0], 3: [1, 0], 4: [1, 0]}
    with patch(
        "tiler.drift.phase_cross_correlation",
        side_effect=_fake_registration(from_first, steps),
    ):
        for tp in range(5):
            tiler.find_drift(tp)
    cumulative = np.cumsum(tiler.tile_locs.drifts, axis=0)
    # the failure moves time point 2 only
    np.testing.assert_array_equal(cumulative[3:], [[3, 0], [4, 0]])
    assert tiler.drift_disagreements == [2, 3]


def test_small_disagreement_is_not_flagged():
    tiler = _labelled_tiler(3, drift_check_px=3)
    from_first = {1: [3, 0], 2: [3, 0]}
    steps = {1: [0, 0], 2: [0, 0]}
    with patch(
        "tiler.drift.phase_cross_correlation",
        side_effect=_fake_registration(from_first, steps),
    ):
        for tp in range(3):
            tiler.find_drift(tp)
    assert tiler.drift_disagreements == []


def test_previous_reference_sums_steps():
    tiler = _labelled_tiler(4, drift_reference="previous")
    # at time point 1 the previous image is the first image
    from_first = {1: [1, 0], 2: [9, 9], 3: [9, 9]}
    steps = {1: [1, 0], 2: [1, 0], 3: [1, 0]}
    with patch(
        "tiler.drift.phase_cross_correlation",
        side_effect=_fake_registration(from_first, steps),
    ):
        for tp in range(4):
            tiler.find_drift(tp)
    assert tiler.tile_locs.drifts == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]


def test_find_drift_refuses_a_gap_in_drifts():
    tiler = _labelled_tiler(4)
    with pytest.raises(ValueError, match="drifts of the 2 before"):
        tiler.find_drift(2)


# ---------------------------------------------------------------------------
# rectangular tiles
# ---------------------------------------------------------------------------


def test_a_tall_tile_keeps_both_of_its_sides():
    locs = TileLocations([[120, 40]], tile_size=(240, 80), drifts=[[0, 0]])
    tile = locs.tiles[0]
    y, x, height, width = tile.as_tile(0)
    rows, columns = tile.as_range(0)

    assert locs.tile_size == (240, 80)
    assert tile.size == (240, 80)
    assert tile.half_size == (120, 40)
    assert (y, x) == (0, 0)
    assert (height, width) == (240, 80)
    assert (rows.stop - rows.start, columns.stop - columns.start) == (240, 80)


def test_a_square_tile_is_still_written_as_one_number():
    # every h5 already written holds a bare number, and a reader that casts
    # one would break on a pair
    written = TileLocations(
        [[10, 10]], tile_size=16, drifts=[[0, 0]]
    ).to_dict(0)["attrs/tile_size"]

    assert written == 16
    assert np.ndim(written) == 0


def test_a_rectangular_tile_is_written_as_its_height_and_width():
    written = TileLocations(
        [[120, 40]], tile_size=(240, 80), drifts=[[0, 0]]
    ).to_dict(0)["attrs/tile_size"]

    np.testing.assert_array_equal(written, [240, 80])


def test_a_size_nobody_set_stays_unset():
    # the whole-image path gives TileLocations no tile size, and a size that
    # was never set must not be invented here
    locs = TileLocations([[10, 10]], max_size=32, drifts=[[0, 0]])
    assert locs.tile_size is None
    assert locs.to_dict(0)["attrs/tile_size"] is None
    # the tile itself falls back to the maximum size, as it always did
    assert locs.tiles[0].size == (32, 32)


def test_detection_refuses_a_rectangle_rather_than_looking_for_one():
    # the detector is built on one length, so looking for a tall trap would
    # find something that is not there. Such a layout comes with its centres
    tiler = _make_tiler(Y=400, X=400)
    with pytest.raises(ValueError, match="square traps"):
        tiler.initialise_tiles(tile_size=(240, 80))
