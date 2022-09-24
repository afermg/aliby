## Trap-wise calculations

import numpy as np


def imBackground(cell_masks, trap_image):
    """
    Finds the median background (pixels not comprising cells) from trap_image

    Parameters
    ----------
    cell_masks: 3d array
       Segmentation masks for cells
    trap_image:
        The image (all channels) for the tile containing the cell
    """
    if not len(cell_masks):
        # create cell_masks if none are given
        cell_masks = np.zeros_like(trap_image)
    # find background pixels
    # sum over all cells identified at a trap - one mask for each cell
    background = ~cell_masks.sum(axis=2).astype(bool)
    return np.median(trap_image[np.where(background)])


def background_max5(cell_masks, trap_image):
    """
    Finds the mean of the maximum five pixels of the background (pixels not comprising cells) from trap_image

    Parameters
    ----------
    cell_masks: 3d array
        Segmentation masks for cells
    trap_image:
        The image (all channels) for the tile containing the cell
    """
    if not len(cell_masks):
        # create cell_masks if none are given
        cell_masks = np.zeros_like(trap_image)
    # find background pixels
    # sum over all cells identified at a trap - one mask for each cell
    background = ~cell_masks.sum(axis=2).astype(bool)
    return np.mean(np.sort(trap_image[np.where(background)])[-5:])
