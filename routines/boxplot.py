#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from postprocessor.routines.plottingabc import BasePlotter


class _BoxplotPlotter(BasePlotter):
    """Draw boxplots over time"""

    def __init__(
        self,
        trace_df,
        trace_name,
        unit_scaling,
        box_color,
        xtick_step,
        xlabel,
        plot_title,
    ):
        super().__init__(trace_name, unit_scaling, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.box_color = box_color
        self.xtick_step = xtick_step

        # Define some labels
        self.ylabel = "Normalised " + self.trace_name + " fluorescence (AU)"

        # Define horizontal axis ticks and labels
        # hacky! -- redefine column names
        trace_df.columns = trace_df.columns * self.unit_scaling
        self.fmt = ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(
                x / (self.xtick_step / self.unit_scaling)
            )
        )

    def plot(self, ax):
        """Draw the heatmap on the provided Axes."""
        super().plot(ax)
        ax.xaxis.set_major_formatter(self.fmt)
        sns.boxplot(
            data=self.trace_df,
            color=self.box_color,
            linewidth=1,
            ax=ax,
        )
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(self.xtick_step / self.unit_scaling)
        )


def boxplot(
    trace_df,
    trace_name,
    unit_scaling=1,
    box_color="b",
    xtick_step=60,
    xlabel="Time (min)",
    plot_title="",
    ax=None,
):
    """Draw series of boxplots from an array of time series of traces

    Draw series of boxplots from an array of time series of traces, showing the
    distribution of values at each time point over time.

    Parameters
    ----------
    trace_df : pandas.DataFrame
        Time series of traces (rows = cells, columns = time points).
    trace_name : string
        Name of trace being plotted, e.g. 'flavin'.
    unit_scaling : int or float
        Unit scaling factor, e.g. 1/60 to convert minutes to hours.
    box_color : string
        matplolib colour string, specifies colour of boxes in boxplot
    xtick_step : int or float
        Interval length, in unit time, to draw x axis ticks.
    xlabel : string
        x axis label.
    plot_title : string
        Plot title.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently active Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    Examples
    --------
    FIXME: Add docs.

    """

    plotter = _BoxplotPlotter(
        trace_df,
        trace_name,
        unit_scaling,
        box_color,
        xtick_step,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
