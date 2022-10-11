#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from postprocessor.routines.plottingabc import BasePlotter


class _MeanPlotter(BasePlotter):
    """Draw mean time series plus standard error."""

    def __init__(
        self,
        trace_df,
        trace_name,
        unit_scaling,
        label,
        mean_color,
        error_color,
        mean_linestyle,
        mean_marker,
        xlabel,
        ylabel,
        plot_title,
    ):
        super().__init__(trace_name, unit_scaling, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.label = label
        self.mean_color = mean_color
        self.error_color = error_color
        self.mean_linestyle = mean_linestyle
        self.mean_marker = mean_marker

        # Define some labels
        self.ylabel = ylabel

        # Mean and standard error
        self.trace_time = (
            np.array(self.trace_df.columns, dtype=float) * self.unit_scaling
        )
        self.mean_ts = self.trace_df.mean(axis=0)
        self.stderr = self.trace_df.std(axis=0) / np.sqrt(len(self.trace_df))

    def plot(self, ax):
        """Draw lines and shading on provided Axes."""
        super().plot(ax)
        ax.plot(
            self.trace_time,
            self.mean_ts,
            color=self.mean_color,
            alpha=0.75,
            linestyle=self.mean_linestyle,
            marker=self.mean_marker,
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
    unit_scaling=1,
    label="wild type",
    mean_color="b",
    error_color="lightblue",
    mean_linestyle="-",
    mean_marker="",
    xlabel="Time (min)",
    ylabel="Normalised flavin fluorescence (AU)",
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
    unit_scaling : int or float
        Unit scaling factor, e.g. 1/60 to convert minutes to hours.
    label : string
        Name of group being plotted, e.g. a strain name.
    mean_color : string
        matplotlib colour string for the mean trace.
    error_color : string
        matplotlib colour string for the standard error shading.
    mean_linestyle : string
        matplotlib linestyle argument for the mean trace.
    mean_marker : string
        matplotlib marker argument for the mean trace.
    xlabel : string
        x axis label.
    ylabel : string
        y axis label.
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
        unit_scaling,
        label,
        mean_color,
        error_color,
        mean_linestyle,
        mean_marker,
        xlabel,
        ylabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
