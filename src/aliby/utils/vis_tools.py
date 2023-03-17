#!/usr/bin/env jupyter
"""
Visualisation tools useful to generate figures cell pictures and figures from scripts.

These do not depend on matplotlib to work, they focus on array processing.
To check plot-related functions look at plots.py in this folder.
"""
import typing as t
from copy import copy

import numpy as np

from agora.io.cells import Cells
from aliby.io.image import instantiate_image
from aliby.tile.tiler import Tiler, TilerParameters


def fetch_tc(
    image_path: str, results_path: str, t: int = 0, c: int = 0
) -> np.ndarray:
    """
    Return 3D ndarray with (Z,Y,X) for a given pair of time point and channel.
    """
    with instantiate_image(image_path) as img:
        tiler = Tiler.from_h5(img, results_path, TilerParameters.default())
        tc = tiler.get_tp_data(t, c)
    return tc


def get_tiles_at_times(
    image_path: str,
    results_path: str,
    timepoints: t.List[int] = [0],
    tile_reduction: t.Union[
        int, t.List[int], str, t.Callable
    ] = lambda x: concatenate_dims(x, 1, -1),
    channel: int = 1,
) -> np.ndarray:
    """Use Image and tiler to get tiled position for specific time points.

    Parameters
    ----------
    image_path : str
        hdf5 index
    timepoints : t.List[int]
        list of timepoints to fetch
    tile_reduction : t.Union[int, t.List[int], str, t.Callable]
        Reduce dimensionality. Generally used to collapse z-stacks into one

    Examples
    --------
    FIXME: Add docs.


    """

    # Get the correct tile in space and time
    with instantiate_image(image_path) as image:
        tiler = Tiler.from_h5(image, results_path, TilerParameters.default())
        tp_channel_stack = [
            _dispatch_tile_reduction(tile_reduction)(
                tiler.get_tp_data(tp, channel)
            )
            for tp in timepoints
        ]
    return tp_channel_stack


def get_cellmasks_at_times(
    results_path: str, timepoints: t.List[int] = [0]
) -> t.List[t.List[np.ndarray]]:
    return Cells(results_path).at_times(timepoints)


def concatenate_dims(ndarray, axis1: int, axis2: int) -> np.ndarray:
    axis2 = len(ndarray.shape) + axis2 if axis2 < 0 else axis2
    return np.concatenate(np.moveaxis(ndarray, axis1, 0), axis=axis2 - 1)


def get_tile_mask_pairs(
    image_path: str,
    results_path: str,
    timepoints: t.List[int] = [0],
    tile_reduction=lambda x: concatenate_dims(x, 1, -1),
) -> t.Tuple[np.ndarray, t.List[t.List[np.ndarray]]]:

    return (
        get_tiles_at_times(
            image_path, results_path, timepoints, tile_reduction
        ),
        get_cellmasks_at_times(results_path, timepoints),
    )


def _dispatch_tile_reduction(how: t.Union[int, str, t.List[int]], axis=1):
    """
    Return an appropriate dimensional reduction based on the input on a specified axis.
    If "how" is a string, it operates in dimension 1  (to match tile dimension standard Tile, Z, Y, X)
    how: int, str or list of int
        if int or list of int those numbers are indexed;
        if str it assumes it is a numpy function such as np max.
        if it is a callable it applies that operation to the array.
        if None it returns the result as-is
    axis: Only used when "how" is string. Determines the dimension to which the
        standard operation is applied.
    """
    # FUTURE use match case when migrating to python 3.10

    if how is None:
        return lambda x: x
    elif isinstance(how, (int, list)):
        return lambda x: x.take(how, axis=axis)
    elif isinstance(how, str):
        return lambda x: getattr(x, how)(axis=axis)
    elif isinstance(how, t.Callable):
        return lambda x: how(x)
    else:
        raise Exception(f"Invalid reduction {how}")


def tile_like(arr1: np.ndarray, arr2: np.ndarray):
    """
    Tile the first two dimensions of arr1 (ND) to match arr2 (2D)
    """

    result = arr1
    ratio = np.divide(arr2.shape, arr1.shape[-2:]).astype(int)
    if reps := max(ratio - 1):
        tile_ = (
            lambda x, n: np.tile(x, n)
            if ratio.argmax()
            else np.tile(x.T, n + 1).T
        )
        result = np.stack([tile_(mask, reps + 1) for mask in arr1])
    return result


def centre_mask(image: np.ndarray, mask: np.ndarray):
    """Roll image to the centre of the image based on a mask of equal size"""

    cell_centroid = (
        np.max(np.where(mask), axis=1) + np.min(np.where(mask), axis=1)
    ) // 2
    tile_centre = np.array(image.shape) // 2
    return np.roll(image, (tile_centre - cell_centroid), (0, 1))


def long_side_vertical(arr):
    result = arr
    if np.subtract(*arr.shape):
        result = arr.T
    return result


def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def crop_mask(img: np.ndarray, mask: np.ndarray):
    img = copy(img).astype(float)
    img[~mask] = np.nan
    return img


def _sample_n_tiles_masks(
    image_path: str,
    results_path: str,
    n: int,
    seed: int = 0,
    interval=None,
) -> t.Tuple[t.Tuple, t.Tuple[np.ndarray, np.ndarray]]:

    cells = Cells(results_path)
    indices, masks = cells._sample_masks(n, seed=seed, interval=interval)

    # zipped_indices = zip(*indices)
    for index, (background, cropped_fg) in zip(
        zip(*indices),
        _gen_overlay_masks_tiles(
            image_path,
            results_path,
            masks,
            [indices[i] for i in (0, 2)],
        ),
    ):
        yield index, (background, cropped_fg)


def _overlay_mask_tile(
    image_path: str,
    results_path: str,
    mask: np.ndarray,
    index: t.Tuple[int, int, int],
    bg_channel: int = 0,
    fg_channel: int = 1,
    reduce_z: t.Union[None, t.Callable] = np.max,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Return a tuplw with two channels
    """

    tc = np.stack(
        [
            fetch_tc(image_path, results_path, index[1], i)
            for i in (bg_channel, fg_channel)
        ]
    )  # Returns C(tile)ZYX

    tiles = tc[:, index[0]].astype(float)

    reduced_z = (
        reduce_z(tiles, axis=1) if reduce_z else concatenate_dims(tiles, 1, -2)
    )

    repeated_mask = tile_like(mask, reduced_z[0])

    cropped_fg = crop_mask(reduced_z[1], repeated_mask)

    return reduced_z[0], cropped_fg


def _gen_overlay_masks_tiles(
    image_path: str,
    results_path: str,
    masks: np.ndarray,
    indices: t.Tuple[t.Tuple[int], t.Tuple[int], t.Tuple[int]],
    bg_channel: int = 0,
    fg_channel: int = 1,
    reduce_z: t.Union[None, t.Callable] = np.max,
) -> t.Generator[t.Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator of mask tiles
    """

    for mask, index in zip(masks, zip(*indices)):
        yield _overlay_mask_tile(
            image_path,
            results_path,
            mask,
            index,
            bg_channel,
            fg_channel,
            reduce_z,
        )
