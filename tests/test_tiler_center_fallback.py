import numpy as np
import pytest

from aliby.tile.tiler import (
    Tiler,
    TilerParameters,
    if_out_of_bounds_pad,
    set_areas_of_interest,
)


def test_trap_detection_exception_preserves_requested_center_tile(monkeypatch):
    pixels = np.arange(1 * 1 * 1 * 201 * 221, dtype=np.float32).reshape(
        1, 1, 1, 201, 221
    )

    def fail_trap_detection(*args, **kwargs):
        raise RuntimeError("forced detector failure")

    monkeypatch.setattr("aliby.tile.tiler.segment_traps", fail_trap_detection)
    parameters = TilerParameters.default(tile_size=117)
    tiler = Tiler(
        pixels,
        meta={},
        parameters=parameters,
    )

    assert parameters.fallback_to_center is True
    with pytest.warns(UserWarning, match="forced detector failure"):
        result = tiler._run_tp(0)

    tile = tiler.tile_locs.tiles[0]
    assert tuple(tile.centre) == (100, 110)
    assert tiler.tile_locs.tile_size == (117, 117)
    assert tile.size == (117, 117)
    assert tile.as_range(0) == (slice(42, 159), slice(52, 169))
    assert result["drift"]["attrs/tile_size"] == (117, 117)
    assert result["pixels"].shape == (1, 1, 1, 117, 117)


def test_trap_detection_exception_fails_closed_when_fallback_disabled(monkeypatch):
    pixels = np.zeros((1, 1, 1, 201, 221), dtype=np.float32)

    def fail_trap_detection(*args, **kwargs):
        raise ValueError("forced detector failure")

    monkeypatch.setattr("aliby.tile.tiler.segment_traps", fail_trap_detection)
    tiler = Tiler(
        pixels,
        meta={},
        parameters=TilerParameters.default(
            tile_size=117,
            fallback_to_center=False,
        ),
    )

    with pytest.raises(RuntimeError, match="fallback_to_center is disabled") as error:
        tiler._run_tp(0)

    assert isinstance(error.value.__cause__, ValueError)
    assert tiler.tile_locs is None


@pytest.mark.parametrize(
    ("tile_size", "expected_size"),
    [
        (117, (117, 117)),
        ([80, 100], (80, 100)),
        ((80, 100), (80, 100)),
    ],
)
def test_configured_center_fallback_geometry(tile_size, expected_size):
    pixels = np.zeros(expected_size)

    configured = set_areas_of_interest(pixels, tile_size=tile_size)
    tile = configured.tiles[0]
    tiled_pixels = if_out_of_bounds_pad(pixels[np.newaxis], tile.as_range(0))

    assert tuple(configured.tile_size) == expected_size
    assert tuple(tile.size) == expected_size
    assert tile.as_range(0) == tuple(slice(0, size) for size in expected_size)
    assert tiled_pixels.shape == (1, *expected_size)


def test_none_tile_size_returns_centered_full_rectangular_fov():
    pixels = np.zeros((80, 120))

    full_fov = set_areas_of_interest(pixels, tile_size=None)
    tile = full_fov.tiles[0]
    tiled_pixels = if_out_of_bounds_pad(pixels[np.newaxis], tile.as_range(0))

    assert tuple(tile.centre) == (40, 60)
    assert full_fov.tile_size is None
    assert tile.size == (80, 120)
    assert tile.as_range(0) == (slice(0, 80), slice(0, 120))
    assert tiled_pixels.shape == (1, 80, 120)
