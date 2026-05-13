"""
Unit tests for aliby.tile.tiler and aliby.tile.tiles.

Uses small synthetic dask arrays; no file I/O, no OMERO connection,
no neural-network inference.
"""

import sys
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
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
        "aliby.tile.tiler.phase_cross_correlation",
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
        "aliby.tile.tiler.phase_cross_correlation",
        return_value=(np.array([0.5, -1.0]), None, None),
    ):
        tiler.find_drift(1)
    assert len(tiler.tile_locs.drifts) == 2
    assert tiler.tile_locs.drifts[1] == [0.5, -1.0]
