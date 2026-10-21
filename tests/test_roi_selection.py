import numpy as np
import pytest

from aliby.tile.tiler import Tiler, TilerParameters, dispatch_tiler
from aliby.tile.tiles import TileLocations


def make(monkeypatch, indices):
    monkeypatch.setattr(
        "aliby.tile.tiler.set_areas_of_interest",
        lambda *a, **k: TileLocations.from_tiler_init(
            [[12, 12], [20, 20], [32, 32]], 16, 64, roi_source="trap_detection"
        ),
    )
    return Tiler(
        np.ones((3, 1, 1, 64, 64), dtype=np.uint16),
        {},
        TilerParameters.default(tile_size=16, roi_indices=indices),
    )


@pytest.mark.parametrize("indices", [[0, 2], (0, 2)])
def test_explicit_selection_retains_order_pixels_and_original_source(
    monkeypatch, indices
):
    tiler = make(monkeypatch, indices)
    out = tiler._run_tp(0)
    np.testing.assert_array_equal(
        tiler.tile_locs.initial_location, [[12, 12], [32, 32]]
    )
    assert tiler.tile_locs.roi_source == "trap_detection"
    assert out["pixels"].shape == (2, 1, 1, 16, 16)
    assert out["pixels"].dtype == np.uint16
    tiler._run_tp(1)
    assert len(tiler.tile_locs) == 2 and len(tiler.tile_locs.drifts) == 2


@pytest.mark.parametrize("indices", [[], [2, 0], [0, 0], [-1], [3], [True], [0.0], "0"])
def test_invalid_selection_rejected(monkeypatch, indices):
    with pytest.raises(ValueError, match="roi_indices"):
        make(monkeypatch, indices)._run_tp(0)


def test_default_preserves_all_rois(monkeypatch):
    assert make(monkeypatch, None)._run_tp(0)["pixels"].shape[0] == 3


def test_dispatch_retains_explicit_selection():
    assert dispatch_tiler(None, {"roi_indices": [0, 2]}).keywords[
        "parameters"
    ].roi_indices == [0, 2]
