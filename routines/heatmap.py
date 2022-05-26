#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from postprocessor.core.processes.standardscaler import standardscaler
from postprocessor.routines.plottingabc import BasePlotter


class _HeatmapPlotter(BasePlotter):
    """Draw heatmap"""

    def __init__(
        self,
        trace_df,
        trace_name,
        births_df,
        cmap,
        sampling_period,
        xtick_step,
        scale,
        robust,
        xlabel,
        plot_title,
    ):
        super().__init__(trace_name, sampling_period, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.births_df = births_df
        self.cmap = cmap
        self.xtick_step = xtick_step
        self.scale = scale
        self.robust = robust

        # Define some labels
        self.colorbarlabel = "Normalised " + self.trace_name + " fluorescence (AU)"
        self.ylabel = "Cell"

        # Scale
        if self.scale:
            self.trace_scaled = standardscaler.as_function(self.trace_df)
        else:
            self.trace_scaled = self.trace_df

        # If robust, redefine colormap scale to remove outliers
        if self.robust:
            self.vmin = np.nanpercentile(self.trace_scaled, 2)
            self.vmax = np.nanpercentile(self.trace_scaled, 98)
            # Make axes even
            if self.scale:
                if np.abs(self.vmin) > np.abs(self.vmax):
                    self.vmax = -self.vmin
                else:
                    self.vmin = -self.vmax
        else:
            self.vmin = None
            self.vmax = None

        # Define horizontal axis ticks and labels
        time_axis = self.sampling_period * self.trace_df.columns.to_numpy()
        xtick_min = self.xtick_step * np.ceil(np.min(time_axis) / self.xtick_step)
        xtick_max = self.xtick_step * np.ceil(np.max(time_axis) / self.xtick_step)
        xticklabels = np.arange(xtick_min, xtick_max, self.xtick_step)
        self.xticks = [
            int(np.where(time_axis == label)[0].item()) for label in xticklabels
        ]
        self.xticklabels = list(map(str, xticklabels.tolist()))

    def plot(self, ax, cax):
        """Draw the heatmap on the provided Axes."""
        super().plot(ax)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels)

        # Draw trace heatmap
        trace_heatmap = ax.imshow(
            self.trace_scaled,
            cmap=self.cmap,
            interpolation="none",
            vmin=self.vmin,
            vmax=self.vmax,
        )

        # Overlay births, if present
        if self.births_df is not None:
            # Must be masked array for transparency
            births_array = self.births_df.to_numpy()
            births_heatmap_mask = np.ma.masked_where(births_array == 0, births_array)
            # Overlay
            births_heatmap = ax.imshow(
                births_heatmap_mask,
                interpolation="none",
            )

        # Draw colour bar
        colorbar = ax.figure.colorbar(
            mappable=trace_heatmap, cax=cax, ax=ax, label=self.colorbarlabel
        )


def heatmap(
    trace_df,
    trace_name,
    births_df=None,
    cmap=cm.RdBu,
    sampling_period=5,
    xtick_step=60,
    scale=True,
    robust=True,
    xlabel="Time (min)",
    plot_title="",
    ax=None,
    cbar_ax=None,
):
    """Draw heatmap from an array of time series of traces

    Parameters
    ----------
    trace_df : pandas.DataFrame
        Time series of traces (rows = cells, columns = time points).
    trace_name : string
        Name of trace being plotted, e.g. 'flavin'.
    births_df : pandas.DataFrame
        Birth mask (rows = cells, columns = time points).  Elements should be
        0 or 1.
    cmap : matplotlib ColorMap
        Colour map for heatmap.
    sampling_period : int or float
        Sampling period, in unit time.
    xtick_step : int or float
        Interval length, in unit time, to draw x axis ticks.
    scale : bool
        Whether to use standard scaler to scale the trace time series.
    robust : bool
        If True, the colour map range is computed with robust quantiles instead
        of the extreme values.
    xlabel : string
        x axis label.
    plot_title : string
        Plot title.
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently active Axes.
    cbar_ax : matplotlib Axes
        Axes in which to draw the colour bar, otherwise take space from the main
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    Examples
    --------
    FIXME: Add docs.

    """
    plotter = _HeatmapPlotter(
        trace_df,
        trace_name,
        births_df,
        cmap,
        sampling_period,
        xtick_step,
        scale,
        robust,
        xlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax, cbar_ax)
    return ax
