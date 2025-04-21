from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from itertools import product

import dask.array as da
import numpy as np

from extraction.core.functions.distributors import reduce_z
from extraction.core.functions.loaders import (
    load_funs,
    load_redfuns,
)

CELL_FUNS, TRAP_FUNS, ALL_FUNS = load_funs()
REDUCTION_FUNS = load_redfuns()


def flatten(d, pref=""):
    return reduce(
        lambda new_d, kv: isinstance(kv[1], dict)
        and {**new_d, **flatten(kv[1], (*pref, kv[0]))}
        or {**new_d, (*pref, kv[0]): kv[1]},
        d.items(),
        {},
    )


def kv(flat: dict[tuple, list]):
    return [(*k1, v1) for k, v in flat.items() for k1, v1 in product((k,), v)]


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
            "max2p5pc",
        ]
    },
}


def measure(
    mask: da.array,
    pixels: da.array or None,
    reduction: Callable,
    metric: Callable,
) -> da.array:
    # Core function: Apply a metric on a z-reduced image and mask pairs
    result = da.array([])
    if len(mask):
        if pixels is not None:
            pixels = reduce_z(pixels, reduction)
        result = metric(mask, pixels)

    return result


def measure_multichannels(
    mask: da.array,
    pixels: da.array,
    reduction: Callable,
    metric: Callable,
    channels_reductor: Callable,
) -> da.array:
    """
    Apply reduction to an array over the first axis, then combine these arrays using `channels_reductor` and call the normal `measure`.
    """
    result = da.array([])
    if len(mask):
        new_pixels = channels_reductor(pixels).compute()
        result = measure(mask, new_pixels, reduction, metric)

    return result


def extract_tree(
    tree: dict[int, dict[str, list[str]]],
    masks: da.array,
    pixels: da.array,
) -> dict[str, da.array]:
    """
    Apply functions based on tree on the input dask arrays
    """
    instructions = kv(flatten(tree))  # Channel, reduction, metric
    with ThreadPoolExecutor() as ex:
        result = list(
            ex.map(
                lambda x: when_da_compute(
                    measure(
                        masks,
                        pixels[x[0]] if x[0] is not None else None,
                        REDUCTION_FUNS[x[1]],
                        CELL_FUNS[x[2]],
                    )
                ),
                instructions,
            )
        )
    return instructions, result


def extract_tree_multi(
    tree: dict[tuple[int, int], dict[str, dict[str, list[str]]]],
    masks: da.array,
    pixels: da.array,
) -> dict[str, da.array]:
    """
    Similar to extract_tree, but it pulls two channels and combines them before following the rest of the instructions.

    """
    instructions = kv(flatten(tree))  # Channel, reduction, metric
    with ThreadPoolExecutor() as ex:
        result = list(
            ex.map(
                lambda x: when_da_compute(
                    measure_multichannels(
                        masks,  # sk-like masks
                        da.stack(
                            (pixels[x[0][0]], pixels[x[0][1]]), axis=-1
                        ),  # pixels to combine on a new axis (at the end)
                        REDUCTION_FUNS[x[2]],  # Function to reduce new channel
                        CELL_FUNS[x[3]],  # Metric
                        REDUCTION_FUNS[
                            x[1]
                        ],  # Function to combine pixels into new channel
                    )
                ),
                instructions,
            )
        )
    return instructions, result


def when_da_compute(msmts: list[da.array or np.ndarray]):
    """
    Compute array if it is a dask one.
    """
    with ThreadPoolExecutor() as ex:
        return list(
            ex.map(lambda x: x.compute() if isinstance(x, da.core.Array) else x, msmts)
        )


# tree_multi = {
#     (0, 1): {"div0": {"max": ["mean"]}},
# }

# tree = {i: {"max": ["mean"]} for i in range(2)}

# masks = da.zeros((6, 100, 100), dtype=bool)
# ones = da.ones((15, 15))
# for i in range(6):
#     masks[i, 15 * i : 15 * (i + 1), 15 * i : 15 * (i + 1)] = ones
# rng = da.random.default_rng(1)
# pixels = rng.standard_normal(size=(3, 5, 100, 100)).compute()
# result = extract_tree(tree, masks, pixels)
# result_multi = extract_tree_multi(tree_multi, masks, pixels)
# print(result)

# TASKS
# Add dual imports to aliby's function loader
# Expand new extractor to support them
# Integrate them into the pipeline
# Run e coli pipeline
# Run aliby pipeline
