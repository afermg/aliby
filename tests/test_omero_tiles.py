"""
Unit tests for loading images and tiles from OMERO in aliby.io.omero.

A fake OMERO image serves each requested region from a numpy array and
refuses to serve a whole plane for a tile, so the tests fail if a tile is
ever cut from a downloaded plane again.
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

from aliby.io.omero import Image, load_tiles_lazy  # noqa: E402
from aliby.tile.tiler import Tiler  # noqa: E402
from tiler import OmeroSource  # noqa: E402

T, C, Z, Y, X = 3, 2, 2, 64, 64
TILE_SIZE = 32


class FakePixels:
    """Serve regions of a TCZYX array and record each request."""

    def __init__(self, data):
        self.data = data
        self.requests = []
        self.planes_allowed = False

    def getPixelsType(self):
        return SimpleNamespace(getValue=lambda: "uint16")

    def getTile(self, z, c, t, tile):
        x, y, width, height = tile
        assert x >= 0 and y >= 0, "a region starts outside the image"
        assert x + width <= X and y + height <= Y, "a region ends outside"
        self.requests.append((z, c, t, tile))
        return self.data[t, c, z, y : y + height, x : x + width].copy()

    def getPlane(self, z, c, t):
        if not self.planes_allowed:
            raise AssertionError("a whole plane was downloaded")
        self.requests.append((z, c, t))
        return self.data[t, c, z].copy()


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

    def getName(self):
        return "pos_001"

    def getChannelLabels(self):
        return ["Brightfield", "cy5"]

    def getPixelSizeX(self, units=None):
        return SimpleNamespace(getValue=lambda: 0.182)


class FakeGateway:
    """Stand in for a BlitzGateway holding one image, id 5."""

    def __init__(self, image):
        self.image = image
        self.closed = False

    def getObject(self, kind, object_id):
        return self.image if (kind, object_id) == ("Image", 5) else None

    def close(self):
        self.closed = True


def source_of(image):
    """Return a tiler source reading the fake image."""
    return OmeroSource(5, connect=lambda: FakeGateway(image))


def tiler_tile(image, slices, tp, c, z):
    """Return the tile the tiler cuts from the whole plane."""
    stack = da.from_array(image.data[tp, c])
    tile = Tiler.get_tile_and_pad(stack, slices, tile_size=TILE_SIZE)
    return tile.compute()[z]


def test_interior_tile_requests_only_its_region():
    image = FakeImage()
    slices = (slice(10, 42), slice(20, 52))
    (tile,) = load_tiles_lazy(source_of(image), [slices], [1], [1], [0])
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
    (tile,) = load_tiles_lazy(source_of(image), [slices], [2], [0], [1])
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
    (tile,) = load_tiles_lazy(source_of(image), [slices], [0], [0], [0])
    result = tile.compute()
    assert result.shape == (TILE_SIZE, TILE_SIZE)
    assert np.isnan(result).all()
    assert image.pixels.requests == []


def test_tiles_are_ordered_by_time_then_channel_then_z():
    image = FakeImage()
    slices = [(slice(0, 32), slice(0, 32)), (slice(8, 40), slice(16, 48))]
    tps, channels, zs = [0, 2], [1, 0], [0, 1]
    tiles = load_tiles_lazy(source_of(image), slices, tps, channels, zs)
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
    load_tiles_lazy(
        source_of(image), [(slice(0, 32), slice(0, 32))], [0], [0], [0, 1]
    )
    assert image.pixels.requests == []


def test_one_tile_location_is_needed_for_each_time_point():
    with pytest.raises(ValueError):
        load_tiles_lazy(
            source_of(FakeImage()),
            [(slice(0, 32), slice(0, 32))],
            [0, 1],
            [0],
            [0],
        )


def test_an_image_is_read_through_one_login():
    image = FakeImage()
    image.pixels.planes_allowed = True
    gateways = []

    def connect():
        gateways.append(FakeGateway(image))
        return gateways[-1]

    with Image(5, connect=connect) as omero_image:
        assert omero_image.name == "pos_001"
        assert omero_image.metadata == {
            "size_x": X,
            "size_y": Y,
            "size_z": Z,
            "size_c": C,
            "size_t": T,
            "channels": ["Brightfield", "cy5"],
            "name": "pos_001",
        }
        assert omero_image.pixel_size_um == 0.182
        data = omero_image.data
        assert isinstance(data, da.Array)
        assert data.chunksize == (1, 1, 1, Y, X)
        np.testing.assert_array_equal(
            data[2, 1, 0].compute(), image.data[2, 1, 0]
        )
        (tile,) = omero_image.tiles([(slice(0, 32), slice(0, 32))], 1, 0, 1)
        np.testing.assert_array_equal(
            tile.compute(), image.data[1, 0, 1, :32, :32]
        )
    assert len(gateways) == 1 and gateways[0].closed
