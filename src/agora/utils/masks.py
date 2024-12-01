#!/usr/bin/env jupyter
import numpy as np


def transform_2d_to_3d(masks: np.ndarray) -> (tuple[int], np.ndarray):
    """
    Convert a 2d (int) to its equivalent cell labels and 3d bool image.


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

    cell_labels = np.unique(masks)
    cell_labels = cell_labels[cell_labels > 0]

    to_compare = np.ones((*masks.shape, 1)) * cell_labels

    masks_3d = masks == np.moveaxis(to_compare, -1, 0)
    return cell_labels, masks_3d


def labels_from_masks(masks: np.ndarray):
    """
    Provides a consistent way to transform 2d masks to a list of labels.
    """
    return transform_2d_to_3d(masks)[0]
