"""
Evaluate the different ways supported to load datasets.

Datasets are a collection of image sets.
Each image set is comprised of a position or field of view imaged, either once or over multiple time points.
Individual images should be identified with a key and a path. The key is a string and the path is one path (for multidimensional images), a list of paths (for imagesets comprised of multiple images) or a string (for zarr arrays).
"""

from pathlib import Path
import pytest

from aliby.io.dataset import DatasetDir, DatasetMonoZarr, DatasetZarr, dispatch_dataset
from aliby.io.image import ImageZarrArray


# --- DatasetDir Tests ---

input_path = Path("/datastore/alan/aliby/test_dataset/data/tif_256/")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


@pytest.mark.skipif(not input_path.exists(), reason="Test dataset not found")
def test_dataset_multifile():
    dif = DatasetDir(
        input_path,
        regex=regex,
        capture_order=capture_order,
    )

    imagesets = (
        dif.get_position_ids()
    )  # This asserts that at least one set of images is found.

    assert len(imagesets), "Image set not found."
    print(imagesets)


# --- DatasetMonoZarr Tests ---


@pytest.mark.parametrize(
    "store_path",
    [
        "/work/datasets/jump_toy/std.zarr",
    ],
)
@pytest.mark.skip(reason="Needs local dataset")
def test_monozarr_position(store_path: str):
    dataset = DatasetMonoZarr(store_path, is_monozarr=True)

    for _, arr_zarr in dataset.get_position_ids().items():
        img = ImageZarrArray(arr_zarr, capture_order="CYX", dimorder="TCZYX")
        pixels = img.get_lazy_data().compute()
        assert pixels.ndim == 5


# --- Dispatcher Tests ---


@pytest.mark.skipif(True, reason="Zarr IO test requires unavailable datastore path")
def test_zarr_io():
    fpath = "/datastore/alan/aliby/test_dataset/data/256.zarr/"

    dset = dispatch_dataset(fpath, is_zarr=True, is_monozarr=True)
    positions = dset.get_position_ids()

    image = ImageZarrArray(positions[0])
    assert image.data.shape == (1, 5, 1, 256, 256)


# --- DatasetZarr Tests ---

@pytest.mark.skip(reason="Test stub: Needs local DatasetZarr dataset")
def test_dataset_zarr():
    """Stub for testing DatasetZarr."""
    pass


