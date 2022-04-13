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
        sampling_period,
        box_color,
        xtick_step,
        xlabel,
        plot_title,
    ):
        super().__init__(trace_name, sampling_period, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.box_color = box_color
        self.xtick_step = xtick_step

        # Define some labels
        self.ylabel = "Normalised " + self.trace_name + " fluorescence (AU)"

        # Define horizontal axis ticks and labels
        # hacky! -- redefine column names
        trace_df.columns = trace_df.columns * self.sampling_period
        self.fmt = ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(x / (self.xtick_step / self.sampling_period))
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
            ticker.MultipleLocator(self.xtick_step / self.sampling_period)
        )


def boxplot(
    trace_df,
    trace_name,
    sampling_period=5,
    box_color="b",
    xtick_step=60,
    xlabel="Time (min)",
    plot_title="",
    ax=None,
):
    plotter = _BoxplotPlotter(
        trace_df,
        trace_name,
        sampling_period,
        box_color,
        xtick_step,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
