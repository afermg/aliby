"""Functions to compute background signals from a tile."""

import numpy as np


def _background_pixels(cell_masks, trap_image):
    """
    Return the background pixels of trap_image.

    Parameters
    ----------
    cell_masks: array
        Segmentation masks for cells, shape (N_cells, Y, X), which
        may be empty if the trap contains no cells.
    trap_image: array
        The z-reduced image for the tile, shape (Y, X).
    """
    if cell_masks is None or not len(cell_masks):
        # with no cells, the whole tile is background
        return trap_image.ravel()
    # any() over axis=0 collapses (N_cells, Y, X) → (Y, X)
    background = ~cell_masks.any(axis=0)
    return trap_image[background]


def median_background(cell_masks, trap_image, channels=None):
    """
    Find the median of background pixels (outside all cells) in trap_image.

    Parameters
    ----------
    cell_masks: array
        Segmentation masks for cells, shape (N_cells, Y, X).
    trap_image: array
        The z-reduced image for the tile, shape (Y, X).
    channels: list, optional
        Not used; present for interface consistency.
    """
    pixels = _background_pixels(cell_masks, trap_image)
    # cells may cover the entire tile
    return np.nanmedian(pixels) if pixels.size else np.nan


def mean_background(cell_masks, trap_image, channels=None):
    """
    Find the mean of background pixels (outside all cells) in trap_image.

    Parameters
    ----------
    cell_masks: array
        Segmentation masks for cells, shape (N_cells, Y, X).
    trap_image: array
        The z-reduced image for the tile, shape (Y, X).
    channels: list, optional
        Not used; present for interface consistency.
    """
    pixels = _background_pixels(cell_masks, trap_image)
    # cells may cover the entire tile
    return np.nanmean(pixels) if pixels.size else np.nan


def std_background(cell_masks, trap_image, channels=None):
    """
    Find the standard deviation of background pixels in trap_image.

    Use as a noise estimate for signal-to-noise calculations.

    Parameters
    ----------
    cell_masks: array
        Segmentation masks for cells, shape (N_cells, Y, X).
    trap_image: array
        The z-reduced image for the tile, shape (Y, X).
    channels: list, optional
        Not used; present for interface consistency.
    """
    pixels = _background_pixels(cell_masks, trap_image)
    # cells may cover the entire tile
    return np.nanstd(pixels) if pixels.size else np.nan
