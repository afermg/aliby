#!/usr/bin/env jupyter
import numpy as np


def transform_2d_to_3d(masks: np.array) -> (tuple[int], np.array):
    """
    Convert a 2d (int) to its equivalent cell labels and 3d bool image.

    This assumes that labels in mask are 1->n and background is zero.
    All labels must be present for this to be correct.


    Example:

    > masks = np.array([[0, 2], [5, 0], [4, 4], [0, 0]])
    > transform_2d_to_3d(masks)

        (array([2, 4, 5]),
         array([[[False,  True],
                 [False, False],
                 [False, False],
                 [False, False]],

                [[False, False],
                 [False, False],
                 [ True,  True],
                 [False, False]],

                [[False, False],
                 [ True, False],
                 [False, False],
                 [False, False]]]))
    """

    cell_labels = np.arange(1, masks.max())
    masks_3d = np.equal.outer(cell_labels, masks)
    return masks_3d


def labels_from_masks(masks: np.ndarray):
    """
    Provides a consistent way to transform 2d masks to a list of labels.
    """
    return transform_2d_to_3d(masks)[0]
