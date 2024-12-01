#!/usr/bin/env jupyter
"""
Trackers get a pair of masks as inputs and return a list of labels linking the first set ot the second (previous) one, as well as a max_label to keep track of label number over the entire time lapse.
"""

import numpy as np
from agora.utils.masks import labels_from_masks, transform_2d_to_3d
from cellpose.utils import stitch3D


def stitch_rois(
    masks: list[list[np.ndarray]],
    track_info: list[list[int]] = [],
) -> list[np.ndarray]:
    """
    Apply the stitch function to multiple regions of interest.
    """
    # Process the masks to fetch the last two masks, and the previous tracker state
    masks = np.moveaxis(np.array(masks[-2:]), 0, 1)

    if not len(track_info):
        prev_labels = [None for _ in masks]
        max_labels = prev_labels
    else:
        prev_labels = [v["labels"] for v in track_info[-1].values()]
        max_labels = [v["max_label"] for v in track_info[-1].values()]

    result = {}

    for k, (masks_set, labels_set, max_set) in enumerate(
        zip(masks, prev_labels, max_labels)
    ):
        result[k] = stitch(masks_set, labels_set, max_set)
    return result


def stitch(
    masks: np.ndarray, prev_labels: np.ndarray = None, max_label: int = 0
) -> (np.ndarray, int):
    """
    Wrapper that returns the new masks. It only returns the latest mask
    and performs no error correction.

    Its inputs are masks, and optionally the previous mask and max_labels.
    """
    if prev_labels is None:
        tracked_mask = masks
        max_label = masks.max()
    else:
        masks[0] = update_labels(masks[0], prev_labels)
        tracked_mask = stitch3D(masks)[-1]
        max_label = max(max_label, masks.max())

    return {"labels": labels_from_masks(tracked_mask), "max_label": max_label}


def update_labels(masks: np.ndarray, prev_labels: list[int] = []) -> np.ndarray:
    """
    Update the labels in masks to become prev_labels.

    Masks must have the same unique numbers as labels (ignoring zero).s
    """
    updated_labels = masks
    if len(prev_labels):
        _, masks_3d = transform_2d_to_3d(masks)
        updated_labels = (np.moveaxis(masks_3d, 0, -1) * prev_labels).max(axis=-1)

    return updated_labels
