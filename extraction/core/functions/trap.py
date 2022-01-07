## Trap-wise calculations

import numpy as np


def imBackground(cell_masks, trap_image):
    '''
    :param cell_masks: (numpy 3d array) cells' segmentation mask
    :param trap_image: the image for the trap in which the cell is (all
    channels)
    '''
    if not len(cell_masks):
        cell_masks = np.zeros_like(trap_image)

    background = ~cell_masks.sum(axis=2).astype(bool)
    return (np.median(trap_image[np.where(background)]))
