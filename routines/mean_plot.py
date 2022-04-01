#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from postprocessor.core.processes.standardscaler import standardscaler


class _MeanPlotter:
    """Draw mean time series plus standard error."""

    def __init__(
        self,
        trace_df,
        trace_name,
        sampling_period,
        label,
        mean_color,
        error_color,
        mean_linestyle,
        scale,
        xlabel,
        plot_title,
    ):
        # Define attributes from arguments
        self.trace_df = trace_df
        self.trace_name = trace_name
        self.sampling_period = sampling_period
        self.label = label
        self.mean_color = mean_color
        self.error_color = error_color
        self.mean_linestyle = mean_linestyle
        self.scale = scale
        self.xlabel = xlabel
        self.plot_title = plot_title

        # Define some labels
        self.ylabel = "Normalised " + self.trace_name + " fluorescence (AU)"

        # Scale
        if self.scale:
            self.trace_scaled = standardscaler.as_function(trace_df)
        else:
            self.trace_scaled = trace_df

        # Mean and standard error
        self.trace_time = np.array(self.trace_df.columns) * self.sampling_period
        self.mean_ts = self.trace_scaled.mean(axis=0)
        self.stderr = self.trace_scaled.std(axis=0) / np.sqrt(len(self.trace_scaled))

    def plot(self, ax):
        """Draw lines and shading on provided Axes."""
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.plot_title)

        ax.plot(
            self.trace_time,
            self.mean_ts,
            color=self.mean_color,
            alpha=0.75,
            linestyle=self.mean_linestyle,
            label="Mean, " + self.label,
        )
        ax.fill_between(
            self.trace_time,
            self.mean_ts - self.stderr,
            self.mean_ts + self.stderr,
            color=self.error_color,
            alpha=0.5,
            label="Standard error, " + self.label,
        )
        ax.legend(loc="upper right")


def mean_plot(
    trace_df,
    trace_name="flavin",
    sampling_period=5,
    label="wild type",
    mean_color="b",
    error_color="lightblue",
    mean_linestyle="-",
    scale=True,
    xlabel="Time (min)",
    plot_title="",
    ax=None,
):
    """Plot mean time series of a DataFrame, with standard error shading.

    Parameters
    ----------
    trace_df : pandas.DataFrame
        Time series of traces (rows = cells, columns = time points).
    trace_name : string
        Name of trace being plotted, e.g. 'flavin'.
    sampling_period : int or float
        Sampling period, in unit time.
    label : string
        Name of group being plotted, e.g. a strain name.
    mean_color : string
        matplotlib colour string for the mean trace.
    error_color : string
        matplotlib colour string for the standard error shading.
    mean_linestyle : string
        matplotlib linestyle argument for the mean trace.
    scale : bool
        Whether to use standard scaler to scale the trace time series.
    xlabel : string
        x axis label.
    plot_title : string
        Plot title.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently active Axes.

    Examples
    --------
    FIXME: Add docs.

    """
    plotter = _MeanPlotter(
        trace_df,
        trace_name,
        sampling_period,
        label,
        mean_color,
        error_color,
        mean_linestyle,
        scale,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
