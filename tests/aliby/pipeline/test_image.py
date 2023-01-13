#!/usr/bin/env python3
import numpy as np
import dask.array as da
import pytest

from aliby.io.image import ImageDummy

tiler_parameters = {"tile_size": 117, "ref_channel": "Brightfield", "ref_z": 0}

sample_da = da.from_array(np.array([[1, 2], [3, 4]]))
# Make it 5-dimensional
sample_da = da.reshape(
    sample_da, (1, 1, sample_da.shape[-2], sample_da.shape[-1])
)


@pytest.mark.parametrize("sample_da", [sample_da])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("n_empty_slices", [4])
def test_pad_array(sample_da, dim, n_empty_slices):
    """Test ImageDummy.pad_array() method"""
    # create object
    imgdmy = ImageDummy(tiler_parameters)
    # pads array
    padded_da = imgdmy.pad_array(
        sample_da, dim=dim, n_empty_slices=n_empty_slices
    )

    # select which dimension to index the multidimensional array
    indices = {dim: n_empty_slices}
    ix = [
        indices.get(dim, slice(None))
        for dim in range(padded_da.compute().ndim)
    ]

    # Checks that original image array is there and is at the last index
    assert padded_da.compute()[ix] == sample_da
    # Checks that the additional axis is extended correctly
    assert padded_da.compute.shape[dim] == n_empty_slices + 1
