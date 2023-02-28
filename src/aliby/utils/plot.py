#!/usr/bin/env jupyter
"""
Basic plotting functions for cell visualisation
"""

from matplotlib import pyplot as plt


def plot_overlay(bg, fg, alpha=0.5, ax=plt) -> None:
    ax.imshow(bg, cmap=plt.cm.gray, interpolation="none")
    ax.imshow(fg, alpha=alpha, interpolation="none")
    ax.axis("off")
