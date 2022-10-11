#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from postprocessor.routines.plottingabc import BasePlotter


class _MedianPlotter(BasePlotter):
    """Draw median time series plus interquartile range."""

    def __init__(
        self,
        trace_df,
        trace_name,
        unit_scaling,
        label,
        median_color,
        error_color,
        median_linestyle,
        median_marker,
        xlabel,
        ylabel,
        plot_title,
    ):
        super().__init__(trace_name, unit_scaling, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.label = label
        self.median_color = median_color
        self.error_color = error_color
        self.median_linestyle = median_linestyle
        self.median_marker = median_marker

        # Define some labels
        self.ylabel = ylabel

        # Median and interquartile range
        self.trace_time = (
            np.array(self.trace_df.columns, dtype=float) * self.unit_scaling
        )
        self.median_ts = self.trace_df.median(axis=0)
        self.quartile1_ts = self.trace_df.quantile(0.25)
        self.quartile3_ts = self.trace_df.quantile(0.75)

    def plot(self, ax):
        """Draw lines and shading on provided Axes."""
        super().plot(ax)
        ax.plot(
            self.trace_time,
            self.median_ts,
            color=self.median_color,
            alpha=0.75,
            linestyle=self.median_linestyle,
            marker=self.median_marker,
            label="Median, " + self.label,
        )
        ax.fill_between(
            self.trace_time,
            self.quartile1_ts,
            self.quartile3_ts,
            color=self.error_color,
            alpha=0.5,
            label="Interquartile range, " + self.label,
        )
        ax.legend(loc="upper right")


def median_plot(
    trace_df,
    trace_name="flavin",
    unit_scaling=1,
    label="wild type",
    median_color="b",
    error_color="lightblue",
    median_linestyle="-",
    median_marker="",
    xlabel="Time (min)",
    ylabel="Normalised flavin fluorescence (AU)",
    plot_title="",
    ax=None,
):
    """Plot median time series of a DataFrame, with interquartile range
    shading.

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
    median_color : string
        matplotlib colour string for the median trace.
    error_color : string
        matplotlib colour string for the interquartile range shading.
    median_linestyle : string
        matplotlib linestyle argument for the median trace.
    median_marker : string
        matplotlib marker argument for the median trace.
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
    plotter = _MedianPlotter(
        trace_df,
        trace_name,
        unit_scaling,
        label,
        median_color,
        error_color,
        median_linestyle,
        median_marker,
        xlabel,
        ylabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
