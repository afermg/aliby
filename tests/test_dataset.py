"""
Evaluate the different ways supported to load datasets.

Datasets are a collection of image sets.
Each image set is comprised of a position or field of view imaged, either once or over multiple time points.
Individual images should be identified with a key and a path. The key is a string and the path is one path (for multidimensional images), a list of paths (for imagesets comprised of multiple images) or a string (for zarr arrays).
"""

import pytest

from aliby.io.dataset import DatasetDir, DatasetZarr, dispatch_dataset

from conftest import REGEX_PARAMETERS

# --- Dispatcher test ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_dispatch_dataset_types(data_dir, dataset, regex, capture_order):
    """Test that dispatch_dataset returns the expected types."""
    dset_dir = dispatch_dataset(
        data_dir / dataset, regex=regex, capture_order=capture_order
    )
    assert isinstance(dset_dir, DatasetDir)


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_dispatch_dataset_zarr(data_dir, dataset):
    dset_monozarr = dispatch_dataset(data_dir / dataset, is_zarr=True, is_monozarr=True)
    assert isinstance(dset_monozarr, DatasetZarr)


# --- DatasetDir Tests ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_dataset_multifile(data_dir, dataset, regex, capture_order):
    dataset = DatasetDir(
        data_dir / dataset,
        regex=regex,
        capture_order=capture_order,
    )

    dataset.get_position_ids()  # This has an internal assert


# --- DatasetoZarr Tests ---


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_dataset_zarr(data_dir, dataset):
    dataset = DatasetZarr(data_dir / dataset)

    dataset.get_position_ids()
