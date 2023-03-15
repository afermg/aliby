#!/usr/bin/env jupyter
"""
Basic plotting functions for cell visualisation
"""

import typing as t

import numpy as np
from grid_strategy import strategies
from matplotlib import pyplot as plt


def plot_overlay(
    bg: np.ndarray, fg: np.ndarray, alpha: float = 1.0, ax=plt
) -> None:
    """
    Plot two images, one on top of the other.
    """

    ax1 = ax.imshow(
        bg, cmap=plt.cm.gray, interpolation="none", interpolation_stage="rgba"
    )
    ax2 = ax.imshow(
        stretch(fg),
        alpha=alpha,
        interpolation="none",
        interpolation_stage="rgba",
    )
    plt.axis("off")
    return ax1, ax2


def plot_overlay_in_square(data: t.Tuple[np.ndarray, np.ndarray]):
    """
    Plot images in an automatically-arranged grid.
    """
    specs = strategies.SquareStrategy("center").get_grid(len(data))
    for i, (gs, (tile, mask)) in enumerate(zip(specs, data)):
        ax = plt.subplot(gs)
        plot_overlay(tile, mask, ax=ax)


def plot_in_square(data: t.Iterable):
    """
    Plot images in an automatically-arranged grid. Only takes one mask
    """
    specs = strategies.SquareStrategy("center").get_grid(len(data))
    for i, (gs, datum) in enumerate(zip(specs, data)):
        ax = plt.subplot(gs)
        ax.imshow(datum)


def stretch_clip(image, clip=True):
    """
    Performs contrast stretching on an input image.

    This function takes an array-like input image and enhances its contrast by adjusting
    the dynamic range of pixel values. It first scales the pixel values between 0 and 255,
    then clips the values that are below the 2nd percentile or above the 98th percentile.
    Finally, the pixel values are scaled to the range between 0 and 1.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    stretched : ndarray
        Contrast-stretched version of the input image.

    Examples
    --------
    FIXME: Add docs.
    """
    from copy import copy

    image = image[~np.isnan(image)]
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    if clip:
        minval = np.percentile(image, 2)
        maxval = np.percentile(image, 98)
        image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image


def stretch(image):

    nona = image[~np.isnan(image)]

    return (image - nona.min()) / (nona.max() - nona.min())
