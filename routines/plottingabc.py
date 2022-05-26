#!/usr/bin/env python3

from abc import ABC


class BasePlotter(ABC):
    """Base class for plotting handler classes"""

    def __init__(self, trace_name, sampling_period, xlabel, plot_title):
        """Common attributes"""
        self.trace_name = trace_name
        self.sampling_period = sampling_period

        self.xlabel = xlabel
        self.ylabel = None
        self.plot_title = plot_title

    def plot(self, ax):
        """Template for drawing on provided Axes"""
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        ax.set_title(self.plot_title)
        # Derived classes extends this with plotting functions


# TODO: something about the plotting functions at the end of the modules.
# Decorator?
