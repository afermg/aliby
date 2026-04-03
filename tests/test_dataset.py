"""
Evaluate the different ways supported to load datasets.

Datasets are a collection of image sets.
Each image set is comprised of a position or field of view imaged, either once or over multiple time points.
Individual images should be identified with a key and a path. The key is a string and the path is one path (for multidimensional images), a list of paths (for imagesets comprised of multiple images) or a string (for zarr arrays).
"""

from pathlib import Path
import pytest

from aliby.io.dataset import DatasetDir


input_path = Path("/datastore/alan/aliby/test_dataset/data/tif_256/")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


@pytest.mark.skipif(not input_path.exists(), reason="Test dataset not found")
def test_imageset():
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
