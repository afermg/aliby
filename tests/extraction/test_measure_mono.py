"""
Test extraction of measurements with one mask and image.
"""

import numpy as np
import pytest

from extraction.extract import extract_tree, process_tree_masks


@pytest.mark.skip(reason="broken")
def test_measure_mono():
    masks = [np.zeros((6, 100, 100), dtype=int)]
    ones = np.ones((15, 15))
    for i in range(6):
        masks[0][i, 15 * i : 15 * (i + 1), 15 * i : 15 * (i + 1)] = i + 1
    rng = np.random.default_rng(1)
    pixels = rng.standard_normal(size=(3, 5, 100, 100))
    tree = {
        None: {
            "max": [
                "area",
                "centroid_x",
                "centroid_y",
            ]
        },
        1: {
            "max": [
                "intensity",
            ]
        },
    }
    result = process_tree_masks(tree, masks, pixels, extract_tree)
    assert result is not None
