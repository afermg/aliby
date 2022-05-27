#!/usr/bin/env python3

import matplotlib.pyplot as plt

from postprocessor.routines.single_plot import _SinglePlotter


class _SingleBirthPlotter(_SinglePlotter):
    """Draw a line plot of a single time series, but with births overlaid"""

    def __init__(
        self,
        trace_timepoints,
        trace_values,
        trace_name,
        birth_mask,
        sampling_period,
        trace_color,
        birth_color,
        trace_linestyle,
        birth_linestyle,
        xlabel,
        plot_title,
    ):
        # Define attributes from arguments
        super().__init__(
            trace_timepoints,
            trace_values,
            trace_name,
            sampling_period,
            trace_color,
            trace_linestyle,
            xlabel,
            plot_title,
        )
        # Add some more attributes useful for births
        self.birth_mask = birth_mask
        self.birth_color = birth_color
        self.birth_linestyle = birth_linestyle

    def plot(self, ax):
        """Draw the line plots on the provided Axes."""
        trace_time = self.trace_timepoints * self.sampling_period
        super().plot(ax)
        birth_mask_bool = self.birth_mask.astype(bool)
        for occurence, birth_time in enumerate(trace_time[birth_mask_bool]):
            if occurence == 0:
                label = "birth event"
            else:
                label = None
            ax.axvline(
                birth_time,
                color=self.birth_color,
                linestyle=self.birth_linestyle,
                label=label,
            )
        ax.legend()


def single_birth_plot(
    trace_timepoints,
    trace_values,
    trace_name="flavin",
    birth_mask=None,
    sampling_period=5,
    trace_color="b",
    birth_color="k",
    trace_linestyle="-",
    birth_linestyle="--",
    xlabel="Time (min)",
    plot_title="",
    ax=None,
):
    """Plot time series of trace, overlaid with births

    Parameters
    ----------
    trace_timepoints : array_like
        Time points (as opposed to the actual times in time units)
    trace_values : array_like
        Trace to plot
    trace_name : string
        Name of trace being plotted, e.g. 'flavin'.
    birth_mask : array_like
        Mask to indicate where births are. Expect values of '0' and '1' or
        'False' and 'True' in the elements.
    sampling_period : int or float
        Sampling period, in unit time.
    trace_color : string
        matplotlib colour string for the trace
    birth_color : string
        matplotlib colour string for the vertical lines indicating births
    trace_linestyle : string
        matplotlib linestyle argument for the trace
    birth_linestyle : string
        matplotlib linestyle argument for the vertical lines indicating births
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
    plotter = _SingleBirthPlotter(
        trace_timepoints,
        trace_values,
        trace_name,
        birth_mask,
        sampling_period,
        trace_color,
        birth_color,
        trace_linestyle,
        birth_linestyle,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax
