"""
Evaluate the different ways supported to load datasets.

Datasets are a collection of image sets.
Each image set is comprised of a position or field of view imaged, either once or over multiple time points.
Individual images should be identified with a key and a path. The key is a string and the path is one path (for multidimensional images), a list of paths (for imagesets comprised of multiple images) or a string (for zarr arrays).
"""

from pathlib import Path
import pytest

import pooch
from aliby.io.dataset import DatasetDir, DatasetMonoZarr, dispatch_dataset, DatasetZarr


try:
    DATA_DIR = Path("/datastore/alan/aliby/test_dataset/data/")
except Exception as _:
    print("Missing local files, pulling from Zenodo")
    marker = "aliby_tests/data/"
    TEST_FILES = pooch.retrieve(
        url="https://zenodo.org/api/records/19228474/files/aliby_test_dataset.tar.gz/content",
        known_hash="f8c59009ad5addfe7fa9175a23496884121119c84372b6e07225d0a4924b5daa",
        fname="aliby_test_dataset.tar.gz",
        processor=pooch.Untar(extract_dir="aliby_tests"),
    )
    DATA_DIR = Path(TEST_FILES[0].split(marker)[0] + marker)
    print(f"DATA_DIR is {DATA_DIR} and exists? {DATA_DIR.exists()}")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


# --- Dispatcher test ---


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Test dataset not found")
@pytest.mark.parametrize("dataset", ["tif_256"])
def test_dispatch_dataset_types(dataset):
    """Test that dispatch_dataset returns the expected types."""
    dset_dir = dispatch_dataset(DATA_DIR / dataset, regex=regex, capture_order="WFC")
    assert isinstance(dset_dir, DatasetDir)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Test dataset not found")
@pytest.mark.parametrize("dataset", ["256.zarr"])
def test_dispatch_dataset_zarr(dataset):
    dset_zarr = dispatch_dataset(DATA_DIR / dataset, is_zarr=True)
    assert isinstance(dset_zarr, DatasetZarr)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Test dataset not found")
@pytest.mark.parametrize("dataset", ["256.zarr"])
def test_dispatch_dataset_monozarr(dataset):
    dset_monozarr = dispatch_dataset(DATA_DIR / dataset, is_zarr=True, is_monozarr=True)
    assert isinstance(dset_monozarr, DatasetMonoZarr)


# --- DatasetDir Tests ---


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Test dataset not found")
@pytest.mark.parametrize("dataset", ["tif_256"])
def test_dataset_multifile(dataset):
    dif = DatasetDir(
        DATA_DIR / dataset,
        regex=regex,
        capture_order=capture_order,
    )

    imagesets = dif.get_position_ids()

    assert len(imagesets), "Image set not found."
    print(imagesets)


# --- DatasetMonoZarr Tests ---


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Test dataset not found")
@pytest.mark.parametrize("dataset", ["256.zarr"])
def test_dataset_monozarr(dataset):
    dif = DatasetMonoZarr(DATA_DIR / dataset)

    imagesets = dif.get_position_ids()

    assert len(imagesets), "Image set not found."
    print(imagesets)


# --- DatasetZarr Tests ---


@pytest.mark.skip(reason="DatasetZarr test data not yet available")
def test_dataset_zarr():
    """Stub for DatasetZarr tests."""
    pass
