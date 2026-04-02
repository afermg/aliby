import pytest

from aliby.io.dataset import dispatch_dataset
from aliby.io.image import ImageZarrArray


@pytest.mark.skipif(True, reason="Zarr IO test requires unavailable datastore path")
def test_zarr_io():
    fpath = "/datastore/alan/aliby/test_dataset/data/256.zarr/"

    dset = dispatch_dataset(fpath, is_zarr=True, is_monozarr=True)
    positions = dset.get_position_ids()

    image = ImageZarrArray(positions[0])
    assert image.data.shape == (1, 5, 1, 256, 256)
