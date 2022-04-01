#!/usr/bin/env python3

import matplotlib.pyplot as plt


class _SinglePlotter:
    """Draw a line plot of a single time series."""

    def __init__(
        self,
        trace_timepoints,
        trace_values,
        trace_name,
        sampling_period,
        trace_color,
        trace_linestyle,
        xlabel,
        plot_title,
    ):
        # Define attributes from arguments
        self.trace_timepoints = trace_timepoints
        self.trace_values = trace_values
        self.trace_name = trace_name
        self.sampling_period = sampling_period
        self.trace_color = trace_color
        self.trace_linestyle = trace_linestyle
        self.xlabel = xlabel
        self.plot_title = plot_title

        # Define some labels
        self.ylabel = "Normalised " + self.trace_name + " fluorescence (AU)"

    def plot(self, ax):
        """Draw the line plot on the provided Axes."""
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        ax.set_title(self.plot_title)
        ax.plot(
            self.trace_timepoints * self.sampling_period,
            self.trace_values,
            color=self.trace_color,
            linestyle=self.trace_linestyle,
            label=self.trace_name + " fluorescence",
        )


def single_plot(
    trace_timepoints,
    trace_values,
    trace_name="flavin",
    sampling_period=5,
    trace_color="b",
    trace_linestyle="-",
    xlabel="Time (min)",
    plot_title="",
    ax=None,
):
    """Plot time series of trace.

    Parameters
    ----------
    trace_timepoints : array_like
        Time points (as opposed to the actual times in time units).
    trace_values : array_like
        Trace to plot.
    trace_name : string
        Name of trace being plotted, e.g. 'flavin'.
    sampling_period : int or float
        Sampling period, in unit time.
    trace_color : string
        matplotlib colour string, specifies colour of line plot.
    trace_linestyle : string
        matplotlib linestyle argument.
    xlabel : string
        x axis label.
    plot_title : string
        Plot title.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently active Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the plot.

    Examples
    --------
    FIXME: Add docs.

    """
    plotter = _SinglePlotter(
        trace_timepoints,
        trace_values,
        trace_name,
        sampling_period,
        trace_color,
        trace_linestyle,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
