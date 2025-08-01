#!/usr/bin/env jupyter
"""
Trackers get a pair of masks as inputs and return a list of labels linking the first set ot the second (previous) one, as well as a max_label to keep track of label number over the entire time lapse.
"""

from itertools import cycle

import numpy as np
from cellpose.utils import stitch3D

from agora.utils.masks import labels_from_masks, transform_2d_to_3d


def stitch_rois(
    masks: list[list[np.ndarray]],
    track_info: list[list[int]] = [],
) -> dict[int, np.ndarray]:
    """
    Apply the stitch function to multiple regions of interest.

    Parameters
    ----------
    masks : list[list[np.ndarray]]
        A list of numpy integer arrays representing the regions of interest. The outermost nesting level is time (..., t-2, t-1)
        and the next one is tiles. The arrays are in a non-overlapping integer representation (i.e., skimage-like).

    track_info : list[list[int]], optional
        A list containing information about the tracker state (default is an empty list).

    Returns
    -------
    dict[int, np.ndarray]
        A dictionary where each key corresponds to the index of the input mask and the value is the result of applying the stitch function.

    Notes
    -----
    If track_info is not provided, it is assumed that there is no previous tracker state.
    """
    # Process the masks to fetch the last two masks, and the previous tracker state

    prev_labels, max_labels = [cycle((None,))] * 2
    if len(track_info):
        prev_labels, max_labels = [
            [v[f] for v in track_info.values()] for f in ("labels", "max_label")
        ]

    result = {}

    for k, (masks_in_tile_pairs, labels_in_tile, max_in_tile) in enumerate(
        zip(masks, prev_labels, max_labels)
    ):
        pair_of_masks = np.array(masks_in_tile_pairs)
        assert pair_of_masks.ndim == 3, "Masks are in wrong dimensions"

        result[k] = stitch(pair_of_masks, labels_in_tile, max_in_tile)
    return result


def stitch(
    masks: np.ndarray, prev_labels: np.ndarray = None, max_label: int = 0
) -> dict[str, int or list[int]]:
    """
    Wrapper that returns the new masks. It only returns the latest mask
    and performs no error correction.

    Its inputs are a (2, Z, Y, X) int numpy array with the non-overlapping masks where the 1st axis is time.
    """
    if prev_labels is None:
        tracked_mask = masks
        max_label = masks.max()
    else:
        masks[0] = update_labels(masks[0], prev_labels)
        tracked_mask = stitch3D(masks)[-1]
        max_label = max(max_label, tracked_mask.max())

    result = {"labels": labels_from_masks(tracked_mask), "max_label": max_label}
    return result


def update_labels(masks: np.ndarray, prev_labels: list[int] = []) -> np.ndarray:
    """
    Update the labels in masks to become prev_labels.

    Masks must have the same unique numbers as labels (ignoring zero).
    """
    updated_labels = masks
    if len(prev_labels):
        _, masks_3d = transform_2d_to_3d(masks)
        updated_labels = (np.moveaxis(masks_3d, 0, -1) * prev_labels).max(axis=-1)

    return updated_labels
