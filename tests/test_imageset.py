"""
Evaluate the different ways supported to load images.
"""

import pytest

from aliby.io.dataset import DatasetDir, DatasetZarr
from aliby.io.image import (
    dispatch_image,
    ImageList,
    ImageZarr,
    ImageDir,
    ImageMultiTiff,
)

from common import REGEX_PARAMETERS

# --- Dispatcher test ---


def test_dispatch_image_types():
    assert dispatch_image({"path": ["a.tif", "b.tif"]}) is ImageList
    assert dispatch_image(["a.tif", "b.tif"]) is ImageList
    assert dispatch_image({"path": "/path.zarr", "key": "1"}) is ImageZarr
    assert dispatch_image("*.tif") is ImageList
    assert dispatch_image("img.tif") is ImageMultiTiff
    assert dispatch_image("/tmp") is ImageDir


# --- ImageList tests ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_image_list(data_dir, dataset, regex, capture_order):
    dataset_obj = DatasetDir(
        data_dir / dataset,
        regex=regex,
        capture_order=capture_order,
    )
    positions = dataset_obj.get_position_ids()

    img = ImageList(
        source=positions[0]["path"],
        regex=regex,
        capture_order=capture_order,
    )

    assert img.name is not None
    data = img.get_data_lazy()
    assert data is not None
    assert len(data.shape) == 5
    assert img.dimorder is not None


# --- ImageZarr tests ---


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_image_zarr(data_dir, dataset):
    dataset_obj = DatasetZarr(data_dir / dataset)
    positions = dataset_obj.get_position_ids()

    img = ImageZarr(source=positions[0])

    data = img.get_data_lazy()
    assert data is not None
    assert len(data.shape) == 5
    assert img.dimorder is not None
    assert img.name is not None
