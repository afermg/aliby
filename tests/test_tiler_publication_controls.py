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


def test_track_drift_seeds_zero_at_featureless_tp0_without_phase_correlation(
    monkeypatch,
):
    pixels = da.zeros((2, 1, 1, 64, 64), chunks=(1, 1, 1, 64, 64))
    monkeypatch.setattr("aliby.tile.tiler.segment_traps", lambda *_args: [[32, 32]])

    def fail_phase_correlation(*args, **kwargs):
        raise AssertionError("phase correlation must not run at tp0")

    monkeypatch.setattr(
        "aliby.tile.tiler.phase_cross_correlation",
        fail_phase_correlation,
    )
    tiler = Tiler(
        pixels,
        meta={},
        parameters=TilerParameters.default(tile_size=16, track_drift=True),
    )

    result = tiler._run_tp(0)

    assert tiler.no_processed == 1
    assert tiler.tile_locs.drifts == [[0.0, 0.0]]
    assert result["drift"]["drifts"].tolist() == [[0.0, 0.0]]


def test_direct_parameters_use_defaults_and_allow_legacy_override(monkeypatch):
    pixels = np.zeros((3, 1, 1, 64, 64), dtype=np.float32)

    def fail_trap_detection(*args, **kwargs):
        raise RuntimeError("forced detector failure")

    monkeypatch.setattr("aliby.tile.tiler.segment_traps", fail_trap_detection)
    parameters = TilerParameters(tile_size=16, ref_channel=0, ref_z=0)
    tiler = Tiler(pixels, meta={}, parameters=parameters)

    assert not hasattr(tiler, "track_drift")
    assert not hasattr(tiler, "fallback_to_center")
    with pytest.warns(UserWarning, match="falling back to center tile"):
        tiler._run_tp(0)
    tiler._run_tp(1)

    assert tiler.tile_locs.drifts == [[0.0, 0.0], [0.0, 0.0]]

    drift_calls = []

    def record_drift(tp):
        drift_calls.append(tp)
        tiler.tile_locs.drifts.append([-1.0, 2.0])

    tiler.calculate_drift = True
    monkeypatch.setattr(tiler, "find_drift", record_drift)
    tiler._run_tp(2)

    assert drift_calls == [2]
    assert tiler.tile_locs.drifts[2] == [-1.0, 2.0]


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


def test_dispatch_only_injects_new_control_defaults():
    constructor = dispatch_tiler("crop", {})
    parameters = constructor.keywords["parameters"]

    assert parameters.to_dict() == {
        "track_drift": False,
        "fallback_to_center": True,
    }
    assert not hasattr(parameters, "tile_size")
    assert not hasattr(parameters, "ref_channel")
    assert not hasattr(parameters, "ref_z")


def test_dispatch_rejects_legacy_and_current_drift_controls_together():
    with pytest.raises(TypeError, match="Specify only 'track_drift'"):
        dispatch_tiler(
            None,
            {"track_drift": True, "calculate_drift": True},
        )


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
