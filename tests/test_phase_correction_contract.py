"""Independent known-motion checks; not a biological segmentation assessment."""

import numpy as np
import pytest

from aliby.tile.tiler import Tiler, TilerParameters
from aliby.tile.tiles import TileLocations


@pytest.fixture
def moving(monkeypatch):
    rng = np.random.default_rng(19)
    brightfield = rng.integers(1, 1000, size=(96, 96), dtype=np.uint16)
    labels = np.zeros((96, 96), dtype=np.uint16)
    labels[44:49, 47:51] = 7
    base = np.stack([brightfield, labels])[:, None]
    displacement = [(0, 0), (3, -5), (-2, 4)]
    pixels = np.stack([np.roll(base, shift, axis=(-2, -1)) for shift in displacement])
    monkeypatch.setattr(
        "aliby.tile.tiler.set_areas_of_interest",
        lambda *a, **kw: TileLocations.from_tiler_init(np.array([[48, 48]]), 16, 96),
    )
    tiler = Tiler(
        pixels,
        {},
        TilerParameters.default(tile_size=16, ref_channel=0, track_drift=True),
    )
    outputs = [tiler._run_tp(t)["pixels"] for t in range(3)]
    return tiler, pixels, base, outputs, displacement


def test_signed_cumulative_registration_preserves_label_plane(moving):
    tiler, _, base, outputs, displacement = moving
    for tp, shift in enumerate(displacement):
        np.testing.assert_array_equal(
            tiler.tile_locs.centres_at_time(tp), [[48 + shift[0], 48 + shift[1]]]
        )
        np.testing.assert_array_equal(outputs[tp][0], base[..., 40:56, 40:56])
        assert set(np.unique(outputs[tp][0, 1])) == {0, 7}


def test_uncorrected_single_channel_uses_initial_geometry(moving):
    tiler, pixels, _, _, _ = moving
    np.testing.assert_array_equal(
        tiler.get_tp_channel(1, 0, drift=False)[0], pixels[1, 0, :, 40:56, 40:56]
    )


def test_uncorrected_all_channels_forwards_flag(moving):
    tiler, pixels, _, _, _ = moving
    np.testing.assert_array_equal(
        tiler.get_fczyx(1, drift=False)[0], pixels[1, :, :, 40:56, 40:56]
    )
