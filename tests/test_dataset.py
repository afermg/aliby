"""
Evaluate the different ways supported to load datasets.

Datasets are a collection of image sets.
Each image set is comprised of a position or field of view imaged, either once or over multiple time points.
Individual images should be identified with a key and a path. The key is a string and the path is one path (for multidimensional images), a list of paths (for imagesets comprised of multiple images) or a string (for zarr arrays).
"""

from pathlib import Path
import pytest

import pooch
from aliby.io.dataset import DatasetDir, DatasetZarr, dispatch_dataset


DATA_DIR = Path("/datastore/alan/aliby/test_dataset/data/")
if not DATA_DIR.exists()
    marker = "aliby_tests/"
    FILES = pooch.retrieve(
        url="https://zenodo.org/api/records/19411429/files/aliby_test_dataset.tar.gz/content",
        known_hash="3a8b1b7b362f002098ba44e65622862057cfe46f0b459514bf270349c8bce4a7",
        fname="aliby_test_dataset.tar.gz",
        processor=pooch.Untar(extract_dir="aliby_tests"),
    )
    DATA_DIR = Path(FILES[0].split(marker)[0] + marker)
REGEX_PARAMETERS = (
    (
        "crop_cellpainting_256",
        ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif",
        "WFC",
    ),
    (
        "crop_timeseries_alcatras_round_diff_dims_293",
        ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "FTCZ",
    ),
    (
        "crop_timeseries_alcatras_square_same_channels_293",
        ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "FTCZ",
    ),
)


# --- Dispatcher test ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_dispatch_dataset_types(dataset, regex, capture_order):
    """Test that dispatch_dataset returns the expected types."""
    dset_dir = dispatch_dataset(
        DATA_DIR / dataset, regex=regex, capture_order=capture_order
    )
    assert isinstance(dset_dir, DatasetDir)


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_dispatch_dataset_zarr(dataset):
    dset_monozarr = dispatch_dataset(DATA_DIR / dataset, is_zarr=True, is_monozarr=True)
    assert isinstance(dset_monozarr, DatasetZarr)


# --- DatasetDir Tests ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_dataset_multifile(dataset, regex, capture_order):
    dataset = DatasetDir(
        DATA_DIR / dataset,
        regex=regex,
        capture_order=capture_order,
    )

    dataset.get_position_ids()  # This has an internal assert


# --- DatasetoZarr Tests ---


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_dataset_zarr(dataset):
    dataset = DatasetZarr(DATA_DIR / dataset)

    dataset.get_position_ids()
