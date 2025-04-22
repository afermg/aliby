"""
Obtain measurements from images and masks using a functional programming approach.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import product

import dask.array as da
import numpy as np
import pyarrow as pa

from agora.utils.masks import transform_2d_to_3d
from extraction.core.functions.distributors import reduce_z
from extraction.core.functions.loaders import (
    load_funs,
    load_redfuns,
)

CELL_FUNS, TRAP_FUNS, ALL_FUNS = load_funs()
REDUCTION_FUNS = load_redfuns()


def flatten(d: dict, pref: str = "") -> dict:
    """
    Flattens a nested dictionary into a single level dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to be flattened.
    pref : str, optional
        The prefix for the keys in the flattened dictionary. Defaults to "".

    Returns
    -------
    dict
        The flattened dictionary.
    """
    return reduce(
        lambda new_d, kv: isinstance(kv[1], dict)
        and {**new_d, **flatten(kv[1], (*pref, kv[0]))}
        or {**new_d, (*pref, kv[0]): kv[1]},
        d.items(),
        {},
    )


def kv(flat: dict[tuple, list]) -> list:
    """
    Creates a list of (key, *values) tuples from a flattened dictionary.

    Parameters
    ----------
    flat : dict
        The flattened dictionary.

    Returns
    -------
    list
        A list of tuples.
    """
    return [(*k1, v1) for k, v in flat.items() for k1, v1 in product((k,), v)]


def measure(
    mask: da.array,
    pixels: da.array or None,
    reduction: Callable,
    metric: Callable,
) -> da.core.Array:
    """
    Apply a metric on a z-reduced image and mask pairs.


    The inputs of this function are data or callable elements.

    Parameters
    ----------
    mask : da.array
        Input mask.
    pixels : da.array or None, optional
        Input pixels (default is None).
    reduction : Callable
        Reduction function to apply.
    metric : Callable
        Metric function to apply.

    Returns
    -------
    result : da.array
        Result of applying the metric.
    """
    result = da.array([])
    if len(mask):
        if pixels is not None:
            pixels = reduce_z(pixels, reduction)
        result = metric(mask, pixels)

    return result


def measure_mono(
    tileid_x: tuple[int, tuple],
    masks: da.core.Array,
    pixels: da.core.Array,
    REDUCTION_FUNS: dict[str, Callable],
    CELL_FUNS: dict[str, Callable],
) -> da.core.Array:
    """
    Applies a metric on a z-reduced image and mask pairs.

    This function coordinates indices, data and callables.

    Parameters
    ----------
    mask : dask array
        The mask to be applied.
    pixels : dask array or None
        The pixel values. If None, no pixels are used.
    reduction : callable
        The reduction function to apply.
    metric : callable
        The metric function to apply.

    Returns
    -------
    dask array
        The result of the metric application.
    """

    tileid, x = tileid_x
    return when_da_compute(
        measure(
            masks[tileid],  # TODO formalise how to handle this
            pixels[x[0]] if x[0] != "None" else None,
            REDUCTION_FUNS[x[1]],
            CELL_FUNS[x[2]],
        )
    )


def measure_multichannels(
    mask: da.array,
    pixels: da.array,
    reduction: Callable,
    metric: Callable,
    channels_reductor: Callable,
) -> da.array:
    """
    Parameters
    ----------
    mask : da.array
        Input array.
    pixels : da.array
        Array of pixel values.
    reduction : Callable
        Reduction function to apply along the first axis.
    metric : Callable
        Metric function to calculate.
    channels_reductor : Callable
        Function to reduce channels.

    Returns
    -------
    result : da.array
        Result of applying reduction and combining arrays using channels_reductor.
    """
    result = da.array([])
    if len(mask):
        new_pixels = channels_reductor(pixels).compute()
        result = measure_mono(mask, new_pixels, reduction, metric)

    return result


def process_tree_masks(
    tree: dict,
    masks: da.array,
    pixels: np.ndarray,
    function: Callable,
) -> tuple[list, list]:
    """
    Processes a tree of masks and applies the given function.

    Parameters
    ----------
    tree : dict
        The tree to be processed.
    masks : dask array
        The mask values.
    pixels : numpy array
        The pixel values.
    function : callable
        The function to apply.

    Returns
    -------
    tuple
        A tuple containing the instructions and results.
    """
    instructions = kv(flatten(tree))
    tileid_instructions = tuple(x for x in product(range(len(masks)), instructions))
    result = list(function(tileid_instructions, masks, pixels))

    return tileid_instructions, result


def when_da_compute(msmts: list[da.array or np.ndarray]) -> list[np.ndarray]:
    """
    Computes a dask array if it exists.

    Parameters
    ----------
    msmts : list of dask arrays or numpy arrays
        The values to be computed.

    Returns
    -------
    list
        A list of computed values.
    """
    with ThreadPoolExecutor() as ex:
        return list(
            ex.map(lambda x: x.compute() if isinstance(x, da.core.Array) else x, msmts)
        )


def format_extraction(
    instructions_result: tuple[list, list],
) -> pa.lib.Table:
    """
    Formats the extraction results into a pyarrow table.

    Parameters
    ----------
    instructions_result : tuple of lists
        The instructions and results to be formatted.

    Returns
    -------
    pyarrow Table
        The formatted table.
    """
    formatted = {k: [] for k in ("tile", "branch", "metric", "values")}
    for inst, measurements in zip(*instructions_result):
        if len(measurements):
            tileid = inst[0]
            branch = "/".join(str(x) for x in inst[1])
            if isinstance(measurements[0], dict):
                for measurement_set in measurements:
                    for k, values in measurement_set.items():
                        formatted["branch"].append(branch)
                        formatted["metric"].append(k)
                        formatted["values"].append(values)
                        formatted["tile"].append(tileid)
            else:
                for v in measurements:
                    formatted["tile"].append(tileid)
                    formatted["branch"].append(branch)
                    formatted["metric"].append(inst[1][-1])
                    formatted["values"].append(v)

    arrow_table = pa.Table.from_pydict(formatted)
    return arrow_table


def extract_tree(
    tileid_instructions: tuple[da.array, tuple[int or str, str, str, str]],
    masks: list[da.array],
    pixels: da.array,
    threaded: bool = True,
) -> dict[str, da.array]:
    """
    Extracts features from one channels.

    Parameters
    ----------
    tileid_instructions : tuple
        A tuple containing an array and instructions for tile extraction.
    masks : list[dask array]
        A list of mask values for feature extraction.
    pixels : dask array
        The pixel values used in the extraction process.

    Returns
    -------
    list
        A list of extracted features from the tree branches.
    """
    if threaded:  # Threaded or not
        with ThreadPoolExecutor() as ex:
            binmasks = list(ex.map(lambda x: transform_2d_to_3d(x)[1], masks))
            result = ex.map(
                partial(
                    measure_mono,
                    masks=binmasks,
                    pixels=pixels,
                    REDUCTION_FUNS=REDUCTION_FUNS,
                    CELL_FUNS=CELL_FUNS,
                ),
                tileid_instructions,
            )
    else:
        binmasks = [transform_2d_to_3d(mask)[1] for mask in masks]
        result = [
            measure_mono(
                tileid_x,
                masks=binmasks,
                pixels=pixels,
                REDUCTION_FUNS=REDUCTION_FUNS,
                CELL_FUNS=CELL_FUNS,
            )
            for tileid_x in tileid_instructions
        ]
    return result


def extract_tree_multi(
    tileid_instructions: tuple[da.array, tuple[tuple[int, int], str, str, str]],
    masks: list[da.array],
    pixels: da.array,
) -> list:
    """
    Extracts features from multiple channels.

    Parameters
    ----------
    tileid_instructions : tuple
        A tuple containing an array and instructions for tile extraction.
    masks : list[dask array]
        A list of mask values for feature extraction.
    pixels : dask array
        The pixel values used in the extraction process.

    Returns
    -------
    list
        A list of extracted features from the tree branches.
    """
    with ThreadPoolExecutor() as ex:
        binmasks = ex.map(
            lambda tileid, _: transform_2d_to_3d(masks[tileid]), tileid_instructions
        )
        result = ex.map(
            lambda tileid, x: when_da_compute(
                measure_multichannels(
                    binmasks[tileid],  # sk-like masks
                    da.stack(
                        (pixels[x[0][0]], pixels[x[0][1]]), axis=-1
                    ),  # pixels to combine on a new axis (at the end)
                    REDUCTION_FUNS[x[2]],  # Function to reduce new channel
                    CELL_FUNS[x[3]],  # Metric
                    REDUCTION_FUNS[x[1]],  # Function to combine pixels into new channel
                )
            ),
            tileid_instructions,
        )
    return result
