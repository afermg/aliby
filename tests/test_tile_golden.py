"""
Golden tests pinning the tiler's behaviour before it moves into ``tiler``.

Every later step -- fixing bugs, delegating tile placement to ``sooth``,
extracting ``tiler`` -- must reproduce these arrays, or say in the test why a
case changed. They pin what the code does, not what it should do: a case
that is wrong today is pinned as wrong, and changed only with the fix.

Regenerate with ``pytest tests/test_tile_golden.py --update-golden``.

The layouts are real: trap locations and drifts from four aliby h5 files,
chosen for the largest cumulative drift and for centres drifting below
zero. Regenerating them needs ``ALIBY_OUTPUT`` to point at the directory
holding those experiments. The zarr test runs the tiler on real brightfield
frames and needs ``ALIBY_GOLDEN_ZARR``; without it the test skips, and a skip
is not a pass.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest
from scipy import ndimage

try:
    import omero.gateway  # noqa: F401
except ImportError:
    # omero is optional; the tiler imports it only for a type hint
    sys.modules.setdefault("omero", MagicMock())
    sys.modules.setdefault("omero.gateway", MagicMock())

from aliby.tile.process_traps import segment_traps  # noqa: E402
from aliby.tile.tiler import Tiler, TilerParameters  # noqa: E402
from aliby.tile.tiles import TileLocations  # noqa: E402

DATA = Path(__file__).parent / "data"
LAYOUT_SOURCES = [
    "1706_2023_06_13_Hxt_vacuole_00/YST_1444_024s24.h5",
    "2859_2025_08_01_NADH_Switch_02/YST_1145_001.h5",
    "2859_2025_08_01_NADH_Switch_02/YST_229_001.h5",
    "3308_2025_10_16_Switch_2to0pGlc_Msn2_dIra12_00/1704_002.h5",
]
CROP_TILE_SIZE = 16
CROP_SLICES = [
    # interior
    (slice(16, 32), slice(16, 32)),
    # partly outside each edge, within a quarter of the tile
    (slice(-4, 12), slice(16, 32)),
    (slice(36, 52), slice(16, 32)),
    (slice(16, 32), slice(-4, 12)),
    (slice(16, 32), slice(36, 52)),
    # a corner
    (slice(-3, 13), slice(36, 52)),
    # exactly a quarter outside: still padded
    (slice(-4, 12), slice(-4, 12)),
    # just over a quarter outside: NaN
    (slice(-5, 11), slice(16, 32)),
    (slice(16, 32), slice(37, 53)),
    # wholly outside
    (slice(60, 76), slice(60, 76)),
]
DRIFT_SHIFTS = [(0, 0), (3, -2), (-1, 4), (0, 0), (7, 5), (-6, 1)]


def load_golden(name: str) -> dict[str, np.ndarray]:
    """Load a golden file as a dict of arrays."""
    with np.load(DATA / name) as golden:
        return {key: golden[key] for key in golden.files}


def save_golden(name: str, **arrays: np.ndarray) -> None:
    """Write a golden file."""
    np.savez_compressed(DATA / name, **arrays)


def tile_ranges(locations: np.ndarray, drifts: np.ndarray) -> np.ndarray:
    """
    Return every tile's ranges at every time point.

    Parameters
    ----------
    locations: array
        Initial trap centres, as stored in ``trap_info/trap_locations``.
    drifts: array
        Per-time-point drifts, as stored in ``trap_info/drifts``.

    Returns
    -------
    ranges: array
        Shape ``(time points, traps, 4)`` holding the start and stop of the
        first slice then of the second, as ``Tile.as_range`` returns them.
    """
    tile_locs = TileLocations(locations, tile_size=117, max_size=1200)
    tile_locs.drifts = drifts.tolist()
    ranges = np.empty((len(drifts), len(locations), 4), dtype=np.int32)
    for tp in range(len(drifts)):
        for index, tile in enumerate(tile_locs):
            first, second = tile.as_range(tp)
            ranges[tp, index] = (
                first.start,
                first.stop,
                second.start,
                second.stop,
            )
    return ranges


def crop_image() -> da.Array:
    """Return a seeded two-plane image to crop tiles from."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 4096, size=(2, 48, 48), dtype=np.uint16)
    return da.from_array(image)


def drift_sequence() -> np.ndarray:
    """Return a textured sequence moved by ``DRIFT_SHIFTS``, with noise."""
    rng = np.random.default_rng(2)
    texture = ndimage.gaussian_filter(rng.normal(size=(128, 128)), 3)
    frames = []
    offset = np.zeros(2, dtype=int)
    for shift in DRIFT_SHIFTS:
        offset += shift
        frame = np.roll(texture, tuple(offset), axis=(0, 1))
        frames.append(frame + rng.normal(scale=0.01, size=frame.shape))
    return np.stack(frames)[:, None, None].astype(np.float32)


def ring_grid_image() -> np.ndarray:
    """Return a noisy, non-square image of a six by eight grid of rings."""
    from test_process_traps import _make_ring_image

    rng = np.random.default_rng(0)
    image, _ = _make_ring_image(6, 8, 117, 35)
    return image + rng.normal(0, 0.05, image.shape).astype(np.float32)


def test_tile_layouts_match_golden(update_golden):
    """Place every tile of four real layouts at every time point."""
    name = "golden_tile_layouts.npz"
    if update_golden:
        root = os.environ.get("ALIBY_OUTPUT")
        if root is None:
            pytest.fail("set ALIBY_OUTPUT to regenerate the tile layouts")
        import h5py

        arrays = {}
        for index, source in enumerate(LAYOUT_SOURCES):
            with h5py.File(Path(root) / source, "r") as h5:
                locations = h5["trap_info/trap_locations"][()]
                drifts = h5["trap_info/drifts"][()]
            arrays[f"locations_{index}"] = locations
            arrays[f"drifts_{index}"] = drifts
            arrays[f"ranges_{index}"] = tile_ranges(locations, drifts)
        save_golden(name, **arrays)
    golden = load_golden(name)
    for index in range(len(LAYOUT_SOURCES)):
        ranges = tile_ranges(
            golden[f"locations_{index}"], golden[f"drifts_{index}"]
        )
        np.testing.assert_array_equal(
            ranges, golden[f"ranges_{index}"], err_msg=LAYOUT_SOURCES[index]
        )


def test_tile_crops_match_golden(update_golden):
    """Cut interior, padded and NaN tiles from a square image."""
    name = "golden_tile_crops.npz"
    image = crop_image()
    tiles = np.stack(
        [
            Tiler.get_tile_and_pad(image, slices, CROP_TILE_SIZE)
            .compute()
            .astype(np.float64)
            for slices in CROP_SLICES
        ]
    )
    if update_golden:
        save_golden(name, tiles=tiles)
    np.testing.assert_array_equal(tiles, load_golden(name)["tiles"])


def test_tiles_timepoint_shape_is_pinned():
    """Pin the extra axis that callers of get_tiles_timepoint index away."""
    image = da.zeros((2, 2, 3, 64, 64))
    tiler = Tiler(
        image, {"channels": ["Brightfield", "GFP"]}, TilerParameters.default()
    )
    tiler.tile_locs = TileLocations([[32, 32], [20, 40]], tile_size=16)
    tiler.tile_locs.drifts = [[0, 0], [0, 0]]
    tiles = tiler.get_tiles_timepoint(
        0, channels=["Brightfield", "GFP"], z=[0, 2]
    )
    assert tiles.shape == (2, 2, 1, 2, 16, 16)


def test_drifts_match_golden(update_golden):
    """Measure drift over a textured sequence with known shifts."""
    name = "golden_tile_drifts.npz"
    tiler = Tiler(
        da.from_array(drift_sequence()),
        {"channels": ["Brightfield"]},
        TilerParameters.default(),
    )
    tiler.tile_locs = TileLocations([[64, 64]], tile_size=32)
    for tp in range(len(DRIFT_SHIFTS)):
        tiler.find_drift(tp)
    drifts = np.array(tiler.tile_locs.drifts)
    if update_golden:
        save_golden(name, drifts=drifts)
    np.testing.assert_array_equal(drifts, load_golden(name)["drifts"])


@pytest.mark.slow
def test_trap_detection_matches_golden(update_golden):
    """Detect traps in a synthetic, non-square ring grid."""
    name = "golden_trap_detection.npz"
    traps = np.asarray(segment_traps(ring_grid_image(), 117))
    if update_golden:
        save_golden(name, traps=traps)
    np.testing.assert_array_equal(traps, load_golden(name)["traps"])


@pytest.mark.slow
def test_real_zarr_layout_matches_golden(update_golden):
    """Detect traps and measure drift on six real brightfield frames."""
    name = "golden_tiler_zarr.npz"
    path = os.environ.get("ALIBY_GOLDEN_ZARR")
    if path is None:
        pytest.skip("set ALIBY_GOLDEN_ZARR to htb2mCherry_001.zarr")
    from aliby.io.image import dispatch_image

    with dispatch_image(Path(path))(Path(path)) as image:
        data = da.from_array(np.asarray(image.data[:6]))
    parameters = TilerParameters.default().to_dict()
    parameters["ref_z"] = 2
    tiler = Tiler(
        data,
        {"channels": ["Brightfield", "Flavin", "mCherry"]},
        TilerParameters.from_dict(parameters),
    )
    tiler.initialise_tiles(tiler.tile_size)
    for tp in range(6):
        tiler.find_drift(tp)
    locations = np.asarray(tiler.tile_locs.initial_location)
    drifts = np.asarray(tiler.tile_locs.drifts)
    if update_golden:
        save_golden(name, locations=locations, drifts=drifts)
    golden = load_golden(name)
    np.testing.assert_array_equal(locations, golden["locations"])
    np.testing.assert_array_equal(drifts, golden["drifts"])
