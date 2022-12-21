from itertools import product

import pytest

from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.loaders import (
    load_funs,
    load_redfuns,
)
from extraction import local_data_loaders as data

dsets1z = data.load_1z()
dsets = data.load()
masks = [d["segoutlines"] for d in dsets1z]
functions = load_funs()[2].values()
tree = {
    c: {r: list(load_funs()[2].keys()) for r in load_redfuns()}
    for c in dsets[0]
    if c != "segoutlines"
}


@pytest.mark.parametrize(
    ["imgs", "masks", "f"], list(product(dsets1z, masks, functions))
)
def test_metrics_run(imgs, masks, f):
    """
    Test all core cell functions using pre-flattened images
    """

    for ch, img in imgs.items():
        if ch != "segoutlines":
            assert tuple(masks.shape[:2]) == tuple(imgs[ch].shape)
            f(masks, img)


@pytest.mark.parametrize(
    ["imgs", "masks", "tree"], list(product(dsets, masks, tree))
)
def test_extractor(imgs, masks, tree):
    """
    Test a tiler-less extractor using an instance built using default parameters.


    Tests reduce-extract
    """
    extractor = Extractor(
        ExtractorParameters.from_meta({"channels": ["Brightfield", "GFP"]})
    )
    # Load all available functions
    extractor.load_funs()
    extractor.tree = tree
    traps = imgs["GFP"]
    # Generate mock labels
    labels = list(range(masks.shape[2]))
    for ch_branches in extractor.params.tree.values():
        print(
            extractor.reduce_extract(
                red_metrics=ch_branches,
                traps=[traps],
                masks=[masks],
                labels={0: labels},
            )
        )
