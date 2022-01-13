#!/usr/bin/env python3

"""
Main dataframe structure

| position | group | ntraps |robustness index | initial_ncells | final_ncells
"""
dir = "/home/alan/Documents/dev/skeletons/data/2021_06_15_pypipeline_unit_test_00/2021_06_15_pypipeline_unit_test_00/"

from abc import abstractclassmethod, abstractmethod
from typing import Iterable, Union, Dict
from pathlib import PosixPath
import warnings
from collections import Counter

import numpy as np
from numpy import ndarray
import pandas as pd
from scipy.signal import find_peaks

import seaborn as sns
import matplotlib.pyplot as plt

from agora.abc import ProcessABC, ParametersABC
from postprocessor.grouper import NameGrouper


# group_pos_trap_ncells = (
#     concat.dropna().groupby(["group", "position", "trap"]).apply(len)
# )
# group_pos_trapswcell = (
#     group_pos_trap_ncells.dropna().groupby(["group", "position"]).apply(len)
# )


class Compiler(ProcessABC):
    def __init__(self, parameters):
        pass
        # super().__init__(parameters)

    @abstractmethod
    def load_data(self):
        """Abstract function that must be reimplemented"""
        pass

    def ntraps(self, pos_path: PosixPath):
        with h5py.File(pos_path, "r") as f:
            return f["trap_info/trap_locations"].shape[0]

    @abstractmethod
    def run():
        pass


class ExperimentCompiler(Compiler):
    def __init__(self, CompilerParameters, exp_path: PosixPath):
        super().__init__(CompilerParameters)
        self.load_data(exp_path)

    def run(self):

        return {
            method: getattr(self, "compile_" + method)()
            for method in ("slice", "delta_traps")
        }

    def load_data(self, path: PosixPath):
        self.grouper = NameGrouper(path)

    @property
    def ntraps(self) -> dict:
        """Get the number of traps in each position

        Returns
        -------
        dict str -> int

        Examples
        --------
        FIXME: Add docs.


        """
        return {pos: coords.shape[0] for pos, coords in self.grouper.traplocs().items()}

    def concat_signal(self, sigloc=None, raw=None, *args, **kwargs) -> pd.DataFrame:

        if sigloc == None:
            sigloc = "extraction/general/None/volume"
        self.sigloc = sigloc

        if raw == None:
            raw = True

        if not hasattr(self, "_concat") or self.sigloc != sigloc:
            self._concat = self.grouper.concat_signal(self.sigloc, pool=7, raw=raw)

        return self._concat

    def get_tp(self, sigloc=None, tp=None, raw=None, *args, **kwargs) -> pd.Series:

        if tp is None:
            tp = 0

        if raw == None:
            raw = True

        return self.concat_signal(sigloc=sigloc, raw=raw, *args, **kwargs).iloc[:, tp]

    def compile_slice(
        self, sigloc=None, tp=None, metrics=None, raw=None, *args, **kwargs
    ) -> pd.DataFrame:

        if sigloc == None:
            self.sigloc = "extraction/general/None/volume"

        if tp == None:
            tp = 0

        if metrics == None:
            metrics = ("max", "mean", "median", "count", "std", "sem")

        if raw == None:
            raw = True

        df = pd.concat(
            [
                getattr(pd.core.groupby.generic.SeriesGroupBy, met)(
                    self.get_tp(sigloc=sigloc, tp=tp, raw=raw, *args, **kwargs)
                    .groupby(["group", "position", "trap"])
                    .max()
                    .groupby(["group", "position"])
                )
                for met in metrics
            ],
            axis=1,
        )

        df.columns = metrics

        merged = self.add_column(df, self.ntraps, name="ntraps")

        return merged

    @staticmethod
    def add_column(df: pd.DataFrame, new_values_d: dict, name="new_col"):

        if name in df.columns:
            warnings.warn("ExpCompiler: Replacing existing column in compilation")
        df[name] = [new_values_d[pos] for pos in df.index.get_level_values("position")]

        return df

    @staticmethod
    def traploc_diffs(traplocs: ndarray) -> list:
        """Obtain metrics for trap localisation.

        Parameters
        ----------
        traplocs : ndarray
            (x,2) 2-dimensional array with the x,y coordinates of traps in each
            column

        Examples
        --------
        FIXME: Add docs.

        """
        signal = np.zeros((traplocs.max(), 2))
        for i in range(2):
            counts = Counter(traplocs[:, i])
            for j, v in counts.items():
                signal[j - 1, i] = v

        where_x = np.where(signal[:, 0])[0]
        where_y = np.where(signal[:, 1])[0]

        diffs = [
            np.diff(x)
            for x in np.apply_along_axis(find_peaks, 0, signal, distance=10)[0]
        ]
        return diffs

    def compile_delta_traps(self):
        tups = [
            (pos, axis, val)
            for pos, coords in self.grouper.traplocs().items()
            for axis, vals in zip(("x", "y"), self.traploc_diffs(coords))
            for val in vals
        ]

        return pd.DataFrame(tups, columns=["position", "axis", "value"])


# df2 = pd.DataFrame(tups, columns=["position", "axis", "value"])

# fig = plt.figure(tight_layout=True)
# gs = Grid_plot = plt.GridSpec(3, 2, wspace=0.8, hspace=0.6)

# sns.stripplot(
#     data=merged.reset_index(),
#     x="group",
#     y="count",
#     hue="ntraps",
#     ax=fig.add_subplot(gs[0, 0]),
# )

# g = NameGrouper(dir)


class PageOrganiser(object):
    def __init__(self, data: Dict[str, pd.DataFrame], grid_spec: tuple = None):
        if grid_spec is None:
            grid_spec = plt.GridSpec(1, 1)
            self.fig = plt.figure(tight_layout=True)
            self.gs = plt.GridSpec(*grid_spec, wspace=0.8, hspace=0.6)

        def place_plot(self, func, xloc=None, yloc=None, *args, **kwargs):
            if xloc is None:
                xloc = slice(0, gs.ncols)
            if yloc is None:
                yloc = slice(0, gs.nrows)

            return func(
                *args,
                ax=self.fig.add_subplot(self.gs[xloc, yloc]),
                **kwargs,
            )

    # plot.set_title("Trap identification robustness")
    # plot.set_xlabel("Axis")
    # plot.set_ylabel("Distance (pixels)")
    # plt.show()


# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np


# plt1 = dummyplot()
# plt2 = dummyplot()
# pp = PdfPages("foo.pdf")
# for i in [plt1, plt2]:
#     pp.savefig(i)
# pp.close()
# # df = (
# #     pd.DataFrame({ax: signal[:, i] for i, ax in enumerate(("x", "y"))})
# #     .reset_index()
# #     .melt("index")
# # )
# # sns.lineplot(data=df, x="index", y="value", hue="variable")

# # inverted_errors = {
# #     k: {pos: v[k] for pos, v in errors.items()} for k in list(errors.values())[0].keys()
# # }
# # for_addition = {
# #     k: [v[pos] for pos in merged.index.get_level_values("position")]
# #     for k, v in inverted_errors.items()
# # }
# # for k, v in for_addition.items():
# #     merged[k] = v

# # fig, axes = plt.subplots(2, 1, sharex=True)
# # for i in range(2):
# #     axes[i].plot(signal[:, i])
# # plt.show()
