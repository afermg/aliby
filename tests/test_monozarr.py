"""Ensure that datasets and images can be loaded via zarr."""

import pytest

from aliby.io.dataset import DatasetMonoZarr
from aliby.io.image import ImageZarrArray

store_path = "/work/datasets/jump_toy/zstd.zarr/"


@pytest.mark.parametrize(
    "store_path",
    [
        "/work/datasets/jump_toy/std.zarr",
    ],
)
def test_monozarr_position(store_path: str):
    dataset = DatasetMonoZarr(store_path, is_monozarr=True)

    for _, arr_zarr in dataset.get_position_ids().items():
        img = ImageZarrArray(arr_zarr, capture_order="CYX", dimorder="TCZYX")
        pixels = img.get_lazy_data().compute()
        assert pixels.ndim == 5
