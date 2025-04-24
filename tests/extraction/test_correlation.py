"""
Test extraction of measurements with one mask and two images.
"""

import dask.array as da

from extraction.extract import extract_tree_multi, process_tree_masks

masks = [da.zeros((6, 100, 100), dtype=int)]
ones = da.ones((15, 15))
for i in range(6):
    masks[0][i, 15 * i : 15 * (i + 1), 15 * i : 15 * (i + 1)] = i + 1
masks[0] = masks[0].max(axis=0)
rng = da.random.default_rng(1)
pixels = rng.standard_normal(size=(3, 5, 100, 100)).compute()
tree = {
    (0, 1): {
        "None": {
            "max": [
                "pearson",
                "costes",
                "manders_fold",
                "rwc",
            ]
        },
    }
}
result = process_tree_masks(tree, masks, pixels, extract_tree_multi)
