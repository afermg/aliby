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

    ax.imshow(bg, cmap=plt.cm.gray, interpolation="none")
    ax.imshow(fg, alpha=alpha, interpolation="none")
    ax.axis("off")


def plot_overlay_in_square(data: t.Tuple[np.ndarray, np.ndarray]):
    specs = strategies.SquareStrategy("center").get_grid(len(data))
    for i, (gs, (tile, mask)) in enumerate(zip(specs, data)):
        ax = plt.subplot(gs)
        plot_overlay(tile, mask, ax=ax)


def plot_in_square(data: t.Iterable):
    specs = strategies.SquareStrategy("center").get_grid(len(data))
    for i, (gs, datum) in enumerate(zip(specs, data)):
        ax = plt.subplot(gs)
        ax.imshow(datum)
