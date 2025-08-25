"""
Obtain measurements from images and masks using a functional programming approach.

Types of measurements:
- One mask (e.g., area)
- One mask, one image (e.g., intensity)
- Two images combined into a new channel, one mask (e.g., ratiometric probes)
- Two images, one mask (e.g., correlation)
- TODO Two masks (e.g., neighbours)
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import product

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
    mask: np.ndarray,
    pixels: np.ndarray or None,
    reduction: Callable,
    metric: Callable,
) -> np.ndarray:
    """
    Apply a metric on a z-reduced image and mask pairs.


    The inputs of this function are data or callable elements.

    Parameters
    ----------
    mask : np.ndarray
        Input mask.
    pixels : np.ndarray or None, optional
        Input pixels (default is None).
    reduction : Callable
        Reduction function to apply.
    metric : Callable
        Metric function to apply.

    Returns
    -------
    result : np.ndarray
        Result of applying the metric.
    """
    # result = np.ndarray([])
    # if len(mask):
    if pixels is not None:
        pixels = reduce_z(pixels, reduction)
    result = metric(mask, pixels)

    return result


def measure_mono(
    tileid_x: tuple[int, tuple],
    masks: np.ndarray,
    pixels: np.ndarray,
    REDUCTION_FUNS: dict[str, Callable] = REDUCTION_FUNS,
    CELL_FUNS: dict[str, Callable] = CELL_FUNS,
) -> np.ndarray:
    """
    Applies a metric on a z-reduced image and mask pairs.

    This function coordinates indices, data and callables.

    Parameters
    ----------
    mask : numpy array
        The mask to be applied.
    pixels : numpy array or None
        The pixel values. If None, no pixels are used.
    reduction : callable
        The reduction function to apply.
    metric : callable
        The metric function to apply.

    Returns
    -------
    numpy array
        The result of the metric application.
    """

    (tile_i, mask_i), (ch, red_z, metric) = tileid_x
    return measure(
        masks[tile_i][mask_i],  # TODO formalise how to handle this
        pixels[ch] if ch != "None" else None,
        REDUCTION_FUNS[red_z],
        CELL_FUNS[metric],
    )


def measure_multi(
    tileid_x: tuple[int, tuple],
    masks: list[np.ndarray],
    pixels: np.ndarray,
    REDUCTION_FUNS: dict[str, Callable],
    CELL_FUNS: dict[str, Callable],
) -> np.ndarray:
    """
    Parameters
    ----------
    mask : np.ndarray
        Input array.
    pixels : np.ndarray
        Array of pixel values. Dimensions are (Z,Y,X,Channels)
    reduction : Callable
        Reduction function to apply along the first axis.

    Returns
    -------
    result : np.ndarray
        Result of applying reduction and combining arrays using channels_reductor.
    """
    (tile_i, mask_i), ((ch0, ch1), red_ch, red_z, metric) = tileid_x
    if red_ch == "None":  # This is a multi-image measurement
        pixels_redz = reduce_z(pixels[[ch0, ch1]], REDUCTION_FUNS[red_z], axis=1)
        result = CELL_FUNS[metric](masks[tile_i][mask_i], *pixels_redz)
    else:  # This is a monoimage measurement, but with a combination of channels
        new_pixels = reduce_z(
            np.stack((pixels[ch0], pixels[ch1])), REDUCTION_FUNS[red_ch], axis=0
        )[np.newaxis, ...]
        tileid_x_new = ((tile_i, mask_i), (0, red_z, metric))
        # Treat as a normal metric from here on
        result = measure_mono(tileid_x_new, masks=masks, pixels=new_pixels)

    return result


def process_tree_masks(
    tree: dict,
    masks: list[np.ndarray],
    pixels: np.ndarray,
    measure_fn: Callable,
) -> tuple[list, list]:
    """
    Orchestrates the processing of all masks using a tree of instructions.

    Parameters
    ----------
    tree : dict
        The tree to be processed.
    masks : numpy array
        The mask values.
    pixels : numpy array
        The pixel values.
    measure_fn : callable
        The measurement function to apply, it can be `measure_mono` or `measure_multi`.

    Returns
    -------
    tuple
        A tuple containing the instructions and results.
    """
    instructions = kv(flatten(tree))
    tileid_instructions = tuple(
        product(
            [
                (tile_i, mask_i)
                for tile_i, masks_in_tile in enumerate(masks)
                for mask_i in range(masks_in_tile.max() - 1)
            ],
            instructions,
        )
    )
    result = measure_fn(tileid_instructions, masks, pixels)
    return tileid_instructions, result


def extract_tree(
    tileid_instructions: tuple[np.ndarray, tuple[int or str, str, str, str]],
    masks: list[np.ndarray],
    pixels: np.ndarray,
    threaded: bool = False,
) -> dict[str, np.ndarray]:
    """
    Extracts features from one channels.

    Parameters
    ----------
    tileid_instructions : tuple
        A tuple containing an array and instructions for tile extraction.
    masks : list[numpy array]
        A list of mask values for feature extraction.
    pixels : numpy array
        The pixel values used in the extraction process.

    Returns
    -------
    list
        A list of extracted features from the tree branches.
    """
    if threaded:  # Threaded or not
        with ThreadPoolExecutor() as ex:
            binmasks = list(x for x in ex.map(transform_2d_to_3d, masks))
            result = list(
                ex.map(
                    partial(
                        measure_mono,
                        masks=binmasks,
                        pixels=pixels,
                        REDUCTION_FUNS=REDUCTION_FUNS,
                        CELL_FUNS=CELL_FUNS,
                    ),
                    tileid_instructions,
                )
            )
    else:
        binmasks = [transform_2d_to_3d(mask) for mask in masks]
        # These should be a list of binary masks
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
    tileid_instructions: tuple[
        int, tuple[tuple[int, int], str or None, str or None, str]
    ],
    masks: list[np.ndarray],
    pixels: np.ndarray,
    threaded: bool = False,
) -> list:
    """
    Extracts features from multiple channels.

    tile
    Parameters
    ----------
    tileid_instructions : tuple
        A tuple containing an array and instructions for tile extraction.
    - index of tile
    - tuple of channels indices
    - reduction over channels
    - reduction over z-stack
    - measurement
    masks : list[numpy array]
        A list of mask values for feature extraction.
    pixels : numpy array
        The pixel values used in the extraction process.

    Returns
    -------
    list
        A list of extracted features from the tree branches.
    """
    assert isinstance(masks, list) or masks.ndim >= 3, (
        "Masks dimensions < 2. It should include batch/tile dimension."
    )
    if threaded:
        with ThreadPoolExecutor() as ex:
            binmasks = list(x for x in ex.map(transform_2d_to_3d, masks))
            result = ex.map(
                partial(
                    measure_multi,
                    masks=binmasks,
                    pixels=pixels,
                    REDUCTION_FUNS=REDUCTION_FUNS,
                    CELL_FUNS=CELL_FUNS,
                ),
                tileid_instructions,
            )
    else:
        binmasks = [transform_2d_to_3d(mask) for mask in masks]
        result = [
            measure_multi(
                ids_instructions,
                masks=binmasks,
                pixels=pixels,
                REDUCTION_FUNS=REDUCTION_FUNS,
                CELL_FUNS=CELL_FUNS,
            )
            for ids_instructions in tileid_instructions
        ]

    return result


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
    formatted = {k: [] for k in ("tile", "label", "branch", "metric", "values")}
    for inst, metrics in zip(*instructions_result):
        tileid, label = inst[0]
        branch = "/".join(str(x) for x in inst[1])
        if isinstance(metrics, int):
            formatted["tile"].append(tileid)
            formatted["label"].append(label)
            formatted["branch"].append(branch)
            formatted["metric"].append(inst[1][-1])
            formatted["values"].append(metrics)
        elif isinstance(metrics, dict):
            for k, values in metrics.items():
                formatted["branch"].append(branch)
                formatted["metric"].append(k)
                formatted["values"].append(values)
                formatted["tile"].append(tileid)
                formatted["label"].append(label)
        elif isinstance(metrics, list):
            for v in metrics:
                formatted["tile"].append(tileid)
                formatted["label"].append(label)
                formatted["branch"].append(branch)
                formatted["metric"].append(inst[1][-1])
                formatted["values"].append(v)

    arrow_table = pa.Table.from_pydict(formatted)
    return arrow_table
