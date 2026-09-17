"""
Tests pinning how the image readers read pixels before they move into
``tiler``.

The golden tests pin what the readers do today. The tests marked
``xfail(strict=True)`` are confirmed bugs: each is fixed when the reader moves
to tiler, and the mark is then removed. A strict xfail fails if the bug
disappears unnoticed.

Two tests read a real experiment and need ``ALIBY_GOLDEN_ZARR`` set to
``htb2mCherry_001.zarr`` and ``ALIBY_GOLDEN_TIFFS`` to the same position
downloaded as a folder of TIFFs; without them the tests skip, and a skip is
not a pass.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from aliby.io.image import ImageDir, ImageZarr, dispatch_image

CHANNELS = ["Brightfield", "GFP", "mCherry", "cy5"]
# a Swain lab log listing channels in acquisition order, which is not
# alphabetical: sorted, cy5 comes before mCherry
SWAINLAB_LOG = """\
2025-05-02 16:09:45,230 - INFO
Swain Lab microscope experiment log file
Microscope name: Batman
Date: 02-May-2025
-----Acquisition settings-----
2025-05-02 16:09:45,230 - INFO Image Configs:
Image config,Channel,Description,Exposure (ms),Number of Z sections,\
Z spacing (um),Sectioning method
Brightfield,Brightfield,Default bright field config,5,2,0.6,Default
GFP,GFPFast,Default GFP,30,2,0.6,PIFOC
mCherry,mCherry,mCherry imaging,100,2,0.6,PIFOC
cy5,cy5,Default cy5,100,2,0.6,PIFOC

Device properties:
Image config,device,property,value
GFP,DTOL-DAC-2,Volts,4


2025-05-02 16:09:45,230 - INFO
group: pos field: position
Name,X,Y,Z,Autofocus offset
pos_001,24836,2429,3881.85,134.6

group: pos field: time
start: 0
interval: 300
frames: 3

"""


def plane_value(t: int, channel: str, z: int) -> int:
    """Return the value filling one plane, encoding where it came from."""
    return 100 * t + 10 * CHANNELS.index(channel) + z


def write_tiff_folder(
    directory: Path, n_t: int = 3, n_z: int = 2, shape=(6, 9)
) -> Path:
    """Write a position as one TIFF per plane, named as wela downloads."""
    directory.mkdir(parents=True)
    for t in range(n_t):
        for channel in CHANNELS:
            for z in range(n_z):
                plane = np.full(
                    shape, plane_value(t, channel, z), dtype=np.uint16
                )
                name = f"pos_001_t{t:04d}_{channel}_z{z:02d}.tiff"
                tifffile.imwrite(directory / name, plane)
    return directory


def test_a_tiff_folder_is_tczyx_with_channels_in_sorted_file_order(tmp_path):
    folder = write_tiff_folder(tmp_path / "pos_001")
    image = ImageDir(folder)
    data = np.asarray(image.data)
    assert data.shape == (3, 4, 2, 6, 9)
    assert data.dtype == np.uint16
    for c, channel in enumerate(sorted(CHANNELS)):
        for t in range(3):
            for z in range(2):
                assert (data[t, c, z] == plane_value(t, channel, z)).all()
    # with no log there are no channel names, only a size
    assert "channels" not in image.metadata
    assert image.metadata["size_c"] == 4
    assert image.name == "pos_001"
    assert image.data.chunksize == (1, 1, 1, 6, 9)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "bug: planes are stacked in sorted file order but named in the "
        "log's order, so with channels Brightfield, GFP, mCherry, cy5 "
        "(as in experiments 2801 and 2809) mCherry and cy5 swap"
    ),
)
def test_a_tiff_folder_names_each_channel_by_its_own_planes(tmp_path):
    folder = write_tiff_folder(tmp_path / "expt" / "pos_001")
    (tmp_path / "expt" / "pos.log").write_text(SWAINLAB_LOG)
    image = ImageDir(folder)
    assert image.metadata["channels"] == CHANNELS
    data = np.asarray(image.data)
    for c, channel in enumerate(image.metadata["channels"]):
        assert (data[0, c, 0] == plane_value(0, channel, 0)).all()


def write_zarr(path: Path) -> np.ndarray:
    """Write a random TCZYX zarr and return its array."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 4096, size=(3, 2, 2, 5, 7), dtype=np.uint16)
    zarr.save_array(str(path), array)
    return array


def test_a_zarr_is_read_as_stored(tmp_path):
    (tmp_path / "pos.log").write_text(SWAINLAB_LOG)
    path = tmp_path / "pos_001.zarr"
    array = write_zarr(path)
    image = dispatch_image(path)(path)
    assert isinstance(image, ImageZarr)
    np.testing.assert_array_equal(np.asarray(image.data), array)
    assert image.name == "pos_001"
    # pinned as found: upper-case keys, unlike every other reader
    assert {
        key: image.metadata[key]
        for key in ("size_T", "size_C", "size_Z", "size_Y", "size_X")
    } == dict(size_T=3, size_C=2, size_Z=2, size_Y=5, size_X=7)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "bug: with no log beside it, a zarr's shape is sought from TIFF "
        "files inside it, and opening it raises IndexError"
    ),
)
def test_a_zarr_without_a_log_can_be_read(tmp_path):
    path = tmp_path / "pos_001.zarr"
    array = write_zarr(path)
    image = dispatch_image(path)(path)
    np.testing.assert_array_equal(np.asarray(image.data), array)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "bug: ImageLocalOME cannot read an OME-TIFF; its dimorder "
        "property recurses without end"
    ),
)
def test_an_ome_tiff_is_read_as_tczyx(tmp_path):
    path = tmp_path / "pos_001.ome.tif"
    rng = np.random.default_rng(1)
    array = rng.integers(0, 4096, size=(3, 2, 2, 5, 7), dtype=np.uint16)
    tifffile.imwrite(
        path,
        array,
        metadata={"axes": "TCZYX", "Channel": {"Name": ["GFP", "cy5"]}},
    )
    image = dispatch_image(path)(path)
    np.testing.assert_array_equal(np.asarray(image.data), array)
    assert image.metadata["channels"] == ["GFP", "cy5"]


def test_local_readers_import_without_omero():
    code = (
        "import sys\n"
        "sys.modules['omero'] = None\n"
        "from aliby.io.image import ImageDir, ImageZarr, dispatch_image\n"
        "from aliby.io.dataset import dispatch_dataset\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def real_paths() -> tuple[Path, Path]:
    """Return the real zarr and TIFF folder, or skip."""
    zarr_path = os.environ.get("ALIBY_GOLDEN_ZARR")
    tiff_path = os.environ.get("ALIBY_GOLDEN_TIFFS")
    if zarr_path is None or tiff_path is None:
        pytest.skip(
            "set ALIBY_GOLDEN_ZARR to htb2mCherry_001.zarr and "
            "ALIBY_GOLDEN_TIFFS to its TIFF folder"
        )
    return Path(zarr_path), Path(tiff_path)


def test_a_real_zarr_is_pinned():
    zarr_path, _ = real_paths()
    image = dispatch_image(zarr_path)(zarr_path)
    assert image.data.shape == (288, 3, 5, 1200, 1200)
    assert image.metadata["channels"] == ["Brightfield", "Flavin", "mCherry"]
    plane = np.asarray(image.data[0, 0, 2])
    assert plane.dtype == np.uint16


def test_a_real_tiff_folder_matches_the_same_position_as_zarr():
    zarr_path, tiff_path = real_paths()
    zarr_image = dispatch_image(zarr_path)(zarr_path)
    tiff_image = dispatch_image(tiff_path)(tiff_path)
    assert tiff_image.data.shape == (100, 3, 5, 1200, 1200)
    assert (
        tiff_image.metadata["channels"] == zarr_image.metadata["channels"]
    )
    for t in (0, 1):
        for c in range(3):
            np.testing.assert_array_equal(
                np.asarray(tiff_image.data[t, c]),
                np.asarray(zarr_image.data[t, c]),
            )
