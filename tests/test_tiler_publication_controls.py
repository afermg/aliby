import dask.array as da
import numpy as np
import pytest

from aliby.tile.tiler import Tiler, TilerParameters, dispatch_tiler
from aliby.tile.tiles import TileLocations


def make_shifted_tiler(*, track_drift: bool | None = None) -> Tiler:
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(64, 64)).astype(np.float32)
    shifted = np.roll(reference, shift=(3, -4), axis=(0, 1))
    shifted_again = np.roll(reference, shift=(5, -6), axis=(0, 1))
    pixels = da.from_array(
        np.stack([reference, shifted, shifted_again])[:, np.newaxis, np.newaxis],
        chunks=(1, 1, 1, 64, 64),
    )
    tile_locs = TileLocations.from_tiler_init([[32, 32]], 16, 64)
    tile_locs.drifts.append([0.0, 0.0])
    parameter_overrides = {"tile_size": 16}
    if track_drift is not None:
        parameter_overrides["track_drift"] = track_drift
    tiler = Tiler(
        pixels,
        meta={},
        parameters=TilerParameters.default(**parameter_overrides),
        tile_locs=tile_locs,
    )
    tiler.no_processed = 1
    return tiler


def test_default_disables_drift_and_keeps_crop_centre_fixed():
    tiler = make_shifted_tiler()

    result = tiler._run_tp(1)

    assert TilerParameters._defaults["track_drift"] is False
    assert tiler.tile_locs.drifts == [[0.0, 0.0], [0.0, 0.0]]
    assert tiler.tile_locs.tiles[0].centre_at_time(1) == [32, 32]
    assert result["drift"]["drifts"].tolist() == [[0.0, 0.0]]


def test_track_drift_records_phase_correlation_and_moves_dask_backed_crop():
    tiler = make_shifted_tiler(track_drift=True)

    initial_pixels = tiler.get_fczyx(0)
    shifted = tiler._run_tp(1)
    shifted_again = tiler._run_tp(2)

    assert tiler.tile_locs.drifts == [
        [0.0, 0.0],
        [-3.0, 4.0],
        [-2.0, 2.0],
    ]
    assert tiler.tile_locs.tiles[0].centre_at_time(2) == [37, 26]
    np.testing.assert_array_equal(shifted["pixels"], initial_pixels)
    np.testing.assert_array_equal(shifted_again["pixels"], initial_pixels)


def test_dispatch_accepts_publication_controls_without_mutating_config():
    class Image:
        data = np.zeros((1, 1, 1, 32, 32), dtype=np.float32)
        meta = {}

    config = {
        "tile_size": 16,
        "ref_channel": 0,
        "ref_z": 0,
        "track_drift": True,
        "fallback_to_center": False,
    }

    tiler = dispatch_tiler(None, config)(Image())

    assert tiler.track_drift is True
    assert tiler.fallback_to_center is False
    assert config == {
        "tile_size": 16,
        "ref_channel": 0,
        "ref_z": 0,
        "track_drift": True,
        "fallback_to_center": False,
    }


def test_dispatch_maps_legacy_calculate_drift_instead_of_dropping_it():
    class Image:
        data = np.zeros((1, 1, 1, 32, 32), dtype=np.float32)
        meta = {}

    with pytest.warns(DeprecationWarning, match="track_drift"):
        tiler = dispatch_tiler(
            None,
            {
                "tile_size": 16,
                "ref_channel": 0,
                "ref_z": 0,
                "calculate_drift": True,
            },
        )(Image())

    assert tiler.track_drift is True
