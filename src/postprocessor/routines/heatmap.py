#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker

from postprocessor.core.processes.standardscaler import standardscaler
from postprocessor.routines.plottingabc import BasePlotter


class _HeatmapPlotter(BasePlotter):
    """Draw heatmap"""

    def __init__(
        self,
        trace_df,
        trace_name,
        buddings_df,
        cmap,
        unit_scaling,
        xtick_step,
        scale,
        robust,
        xlabel,
        ylabel,
        cbarlabel,
        plot_title,
    ):
        super().__init__(trace_name, unit_scaling, xlabel, plot_title)
        # Define attributes from arguments
        self.trace_df = trace_df
        self.buddings_df = buddings_df
        self.cmap = cmap
        self.xtick_step = xtick_step
        self.scale = scale
        self.robust = robust

        # Define some labels
        self.cbarlabel = cbarlabel
        self.ylabel = ylabel

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
        # hacky! -- redefine column names
        trace_df.columns = trace_df.columns * self.unit_scaling
        self.fmt = ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(x * self.unit_scaling)
        )

    def plot(self, ax, cax):
        """Draw the heatmap on the provided Axes."""
        super().plot(ax)
        ax.xaxis.set_major_formatter(self.fmt)
        # Draw trace heatmap
        trace_heatmap = ax.imshow(
            self.trace_scaled,
            cmap=self.cmap,
            interpolation="none",
            vmin=self.vmin,
            vmax=self.vmax,
        )
        # Horizontal axis labels as multiples of xtick_step, taking
        # into account unit scaling
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(self.xtick_step / self.unit_scaling)
        )
        # Overlay buddings, if present
        if self.buddings_df is not None:
            # Must be masked array for transparency
            buddings_array = self.buddings_df.to_numpy()
            buddings_heatmap_mask = np.ma.masked_where(
                buddings_array == 0, buddings_array
            )
            # Overlay
            ax.imshow(
                buddings_heatmap_mask,
                interpolation="none",
            )
        # Draw colour bar
        ax.figure.colorbar(
            mappable=trace_heatmap, cax=cax, ax=ax, label=self.cbarlabel
        )


def heatmap(
    trace_df,
    trace_name,
    buddings_df=None,
    cmap=cm.RdBu,
    unit_scaling=1,
    xtick_step=60,
    scale=True,
    robust=True,
    xlabel="Time (min)",
    ylabel="Cell",
    cbarlabel="Normalised fluorescence (AU)",
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
    buddings_df : pandas.DataFrame
        Birth mask (rows = cells, columns = time points).  Elements should be
        0 or 1.
    cmap : matplotlib ColorMap
        Colour map for heatmap.
    unit_scaling : int or float
        Unit scaling factor, e.g. 1/60 to convert minutes to hours.
    xtick_step : int or float
        Interval length, in unit time, to draw x axis ticks.
    scale : bool
        Whether to use standard scaler to scale the trace time series.
    robust : bool
        If True, the colour map range is computed with robust quantiles instead
        of the extreme values.
    xlabel : string
        x axis label.
    ylabel : string
        y axis label.
    cbarlabel : string
        Colour bar label.
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
        buddings_df,
        cmap,
        unit_scaling,
        xtick_step,
        scale,
        robust,
        xlabel,
        ylabel,
        cbarlabel,
        plot_title,
    )
    if ax is None:
        ax = plt.gca()
    plotter.plot(ax, cbar_ax)
    return ax
