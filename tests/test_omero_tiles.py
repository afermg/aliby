"""
Unit tests for loading tiles from OMERO in aliby.io.omero.

A fake OMERO image serves each requested region from a numpy array and
refuses to serve a whole plane, so the tests fail if a tile is ever cut from
a downloaded plane again.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest

# omero is an optional dependency — mock before importing aliby.io.omero
sys.modules.setdefault("omero", MagicMock())
sys.modules.setdefault("omero.gateway", MagicMock())
sys.modules.setdefault("omero.model", MagicMock())

from aliby.io.omero import load_tiles_lazy  # noqa: E402
from aliby.tile.tiler import Tiler  # noqa: E402

T, C, Z, Y, X = 3, 2, 2, 64, 64
TILE_SIZE = 32


class FakePixels:
    """Serve regions of a TCZYX array and record each request."""

    def __init__(self, data):
        self.data = data
        self.requests = []

    def getPixelsType(self):
        return SimpleNamespace(getValue=lambda: "uint16")

    def getTile(self, z, c, t, tile):
        x, y, width, height = tile
        assert x >= 0 and y >= 0, "a region starts outside the image"
        assert x + width <= X and y + height <= Y, "a region ends outside"
        self.requests.append((z, c, t, tile))
        return self.data[t, c, z, y : y + height, x : x + width].copy()

    def getPlane(self, *args):
        raise AssertionError("a whole plane was downloaded")


class FakeImage:
    """Stand in for an OMERO ImageWrapper."""

    def __init__(self):
        values = np.arange(T * C * Z * Y * X) % 60000
        self.data = values.reshape(T, C, Z, Y, X).astype(np.uint16)
        self.pixels = FakePixels(self.data)

    def getSizeT(self):
        return T

    def getSizeC(self):
        return C

    def getSizeZ(self):
        return Z

    def getSizeY(self):
        return Y

    def getSizeX(self):
        return X

    def getPrimaryPixels(self):
        return self.pixels


def tiler_tile(image, slices, tp, c, z):
    """Return the tile the tiler cuts from the whole plane."""
    stack = da.from_array(image.data[tp, c])
    tile = Tiler.get_tile_and_pad(stack, slices, tile_size=TILE_SIZE)
    return tile.compute()[z]


def test_interior_tile_requests_only_its_region():
    image = FakeImage()
    slices = (slice(10, 42), slice(20, 52))
    (tile,) = load_tiles_lazy(image, [slices], [1], [1], [0])
    result = tile.compute()
    np.testing.assert_array_equal(result, image.data[1, 1, 0, 10:42, 20:52])
    assert image.pixels.requests == [(0, 1, 1, (20, 10, 32, 32))]


@pytest.mark.parametrize(
    "slices",
    [
        # past the bottom and right edges
        (slice(40, 72), slice(36, 68)),
        # before the top and left edges: a negative start
        (slice(-4, 28), slice(-2, 30)),
    ],
)
def test_edge_tile_is_padded_as_the_tiler_pads(slices):
    image = FakeImage()
    (tile,) = load_tiles_lazy(image, [slices], [2], [0], [1])
    result = tile.compute()
    assert result.shape == (TILE_SIZE, TILE_SIZE)
    np.testing.assert_array_equal(
        result, tiler_tile(image, slices, tp=2, c=0, z=1)
    )
    assert len(image.pixels.requests) == 1


def test_tile_mostly_outside_is_nan_and_requests_nothing():
    image = FakeImage()
    # 20 px outside of a 32 px tile is more than a quarter
    slices = (slice(-20, 12), slice(16, 48))
    (tile,) = load_tiles_lazy(image, [slices], [0], [0], [0])
    result = tile.compute()
    assert result.shape == (TILE_SIZE, TILE_SIZE)
    assert np.isnan(result).all()
    assert image.pixels.requests == []


def test_tiles_are_ordered_by_time_then_channel_then_z():
    image = FakeImage()
    slices = [(slice(0, 32), slice(0, 32)), (slice(8, 40), slice(16, 48))]
    tps, channels, zs = [0, 2], [1, 0], [0, 1]
    tiles = load_tiles_lazy(image, slices, tps, channels, zs)
    expected = [
        image.data[tp, c, z][tile_slice]
        for tp, tile_slice in zip(tps, slices)
        for c in channels
        for z in zs
    ]
    assert len(tiles) == len(expected)
    for tile, want in zip(tiles, expected):
        np.testing.assert_array_equal(tile.compute(), want)


def test_nothing_is_requested_until_computed():
    image = FakeImage()
    load_tiles_lazy(image, [(slice(0, 32), slice(0, 32))], [0], [0], [0, 1])
    assert image.pixels.requests == []


def test_one_tile_location_is_needed_for_each_time_point():
    with pytest.raises(ValueError):
        load_tiles_lazy(
            FakeImage(), [(slice(0, 32), slice(0, 32))], [0, 1], [0], [0]
        )
