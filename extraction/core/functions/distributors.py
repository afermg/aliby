import numpy as np


def trap_apply(cell_fun, cell_masks, trap_image, **kwargs):
    """
    Apply a cell_function to a mask, trap_image pair

    :param cell_fun: function to apply to a cell (from extraction/cell.py)
    :param cell_masks: (numpy 3d array) cells' segmentation mask
    :param trap_image: the image for the trap in which the cell is (all
    channels)
    :**kwargs: parameters to pass if needed for custom functions
    """

    cells_iter = (*range(cell_masks.shape[2]),)
    return [cell_fun(cell_masks[..., i], trap_image, **kwargs) for i in cells_iter]


def reduce_z(trap_image, fun):
    # Optimise the reduction function if possible
    if isinstance(fun, np.ufunc):
        return fun.reduce(trap_image, axis=2)
    else:
        return np.apply_along_axis(fun, 2, trap_image)
