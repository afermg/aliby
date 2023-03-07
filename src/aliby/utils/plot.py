#!/usr/bin/env jupyter
"""
Basic plotting functions for cell visualisation
"""

import typing as t

import numpy as np
from grid_strategy import strategies
from matplotlib import pyplot as plt


def plot_overlay(
    bg: np.ndarray, fg: np.ndarray, alpha: float = 0.5, ax=plt
) -> None:
    """
    Plot two images, one on top of the other.
    """

    ax1 = ax.imshow(bg, cmap=plt.cm.gray, interpolation="none")
    ax2 = ax.imshow(fg, alpha=alpha, interpolation="none")
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
