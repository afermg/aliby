import numpy as np
import pytest

from aliby.tile.tiler import Tiler, TilerParameters, get_center, set_areas_of_interest


def test_trap_detection_exception_preserves_requested_center_tile(monkeypatch):
    pixels = np.arange(1 * 1 * 1 * 201 * 221, dtype=np.float32).reshape(
        1, 1, 1, 201, 221
    )

    def fail_trap_detection(*args, **kwargs):
        raise RuntimeError("forced detector failure")

    monkeypatch.setattr("aliby.tile.tiler.segment_traps", fail_trap_detection)
    tiler = Tiler(
        pixels,
        meta={},
        parameters=TilerParameters.default(tile_size=117),
    )

    with pytest.warns(UserWarning, match="forced detector failure"):
        result = tiler._run_tp(0)

    tile = tiler.tile_locs.tiles[0]
    assert tuple(tile.centre) == (100, 110)
    assert tiler.tile_locs.tile_size == (117, 117)
    assert tile.size == (117, 117)
    assert tile.as_range(0) == (slice(42, 159), slice(52, 169))
    assert result["drift"]["attrs/tile_size"] == (117, 117)
    assert result["pixels"].shape == (1, 1, 1, 117, 117)


def test_configured_center_fallback_and_full_fov_geometry():
    configured = set_areas_of_interest(np.zeros((117, 117)), tile_size=117)
    full_fov = get_center((1, 1, 1, 80, 120))

    assert configured.tile_size == (117, 117)
    assert configured.tiles[0].as_range(0) == (slice(0, 117), slice(0, 117))
    assert full_fov.tile_size is None
    assert full_fov.tiles[0].size == (80, 120)
    assert full_fov.tiles[0].as_range(0) == (slice(0, 80), slice(0, 120))
