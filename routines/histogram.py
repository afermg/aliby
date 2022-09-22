#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


class _HistogramPlotter:
    """Draw histogram"""

    def __init__(
        self,
        values,
        label,
        color,
        binsize,
        lognormal,
        lognormal_base,
        xlabel,
        ylabel,
        plot_title,
    ):
        # Define attributes from arguments
        self.values = values
        self.label = label
        self.color = color
        self.binsize = binsize
        self.lognormal = lognormal
        self.lognormal_base = lognormal_base
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_title = plot_title

        # Define median
        self.median = np.median(self.values)

        # Define bins
        if self.lognormal:
            self.bins = np.logspace(
                0,
                np.ceil(
                    np.log(np.nanmax(values)) / np.log(self.lognormal_base)
                ),
                base=self.lognormal_base,
            )  # number of bins will be 50 by default, as it's the default in np.logspace
        else:
            self.bins = np.arange(
                self.binsize * (np.nanmin(values) // self.binsize - 2),
                self.binsize * (np.nanmax(values) // self.binsize + 2),
                self.binsize,
            )

    def plot(self, ax):
        """Plot histogram onto specified Axes."""
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        ax.set_title(self.plot_title)

        if self.lognormal:
            ax.set_xscale("log")
        ax.hist(
            self.values,
            self.bins,
            alpha=0.5,
            color=self.color,
            label=self.label,
        )
        ax.axvline(
            self.median,
            color=self.color,
            alpha=0.75,
            label="median " + self.label,
        )
        ax.legend(loc="upper right")


def histogram(
    values,
    label,
    color="b",
    binsize=5,
    lognormal=False,
    lognormal_base=10,
    xlabel="Time (min)",
    ylabel="Number of occurences",
    plot_title="Distribution",
    ax=None,
):
    """Draw histogram with median indicated

    Parameters
    ----------
    values : array_like
        Input values for histogram
    label : string
        Name of value being plotting, e.g. cell division cycle length.
    color : string
        Colour of bars.
    binsize : float
        Bin size.
    lognormal : bool
        Whether to use a log scale for the horizontal axis.
    lognormal_base : float
        Base of the log scale, if lognormal is True.
    xlabel : string
        x axis label.
    ylabel : string
        y axis label.
    plot_title : string
        Plot title.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently active Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the histogram.

    Examples
    --------
    FIXME: Add docs.

    """
    plotter = _HistogramPlotter(
        values,
        label,
        color,
        binsize,
        lognormal,
        lognormal_base,
        xlabel,
        ylabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
