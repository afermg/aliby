import pytest
from matplotlib import pyplot as plt

from aliby.io.dataset import dispatch_dataset
from aliby.io.image import ImageZarrArray


fpath = "/datastore/alan/aliby/test_dataset/data/256.zarr/"

dset = dispatch_dataset(fpath, is_zarr=True, is_monozarr=True)
positions = dset.get_position_ids()

image = ImageZarrArray(positions[0])
assert image.data.shape == (1, 5, 1, 256, 256)

for i, data in enumerate(image.data.compute()[0]):
    plt.close()
    plt.imshow(data[0])
    plt.savefig(f"test_{i}.png")
