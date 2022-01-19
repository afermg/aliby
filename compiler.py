#!/usr/bin/env python3

"""
Main dataframe structure

| position | group | ntraps |robustness index | initial_ncells | final_ncells
"""
# dir = "/home/alan/Documents/dev/skeletons/data/2021_06_15_pypipeline_unit_test_00/2021_06_15_pypipeline_unit_test_00/"
# dir = "/home/alan/Documents/dev/libs/aliby/data/2021_08_24_2Raf_00/2021_08_24_2Raf_00/"
dirs = [
    "/home/alan/Documents/dev/skeletons/data/2021_08_24_2Raf_00/2021_08_24_2Raf_00"
    # "/home/alan/Documents/dev/libs/aliby/data/sofia/2021_08_24_2Raf_00/2021_08_24_2Raf_00",
    # "/home/alan/Documents/dev/libs/aliby/data/sofia/2021_09_15_1_Raf_06/2021_09_15_1_Raf_06",
    # "/home/alan/Documents/dev/libs/aliby/data/sofia/2021_09_17_0_1_Raf_00/2021_09_17_0_1_Raf_00",
]
import h5py

from abc import abstractclassmethod, abstractmethod
from typing import Iterable, Union, Dict, Tuple
from pathlib import PosixPath, Path
import warnings
from collections import Counter

import numpy as np
from numpy import ndarray
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from agora.abc import ProcessABC, ParametersABC
from postprocessor.grouper import NameGrouper


# group_pos_trap_ncells = (
#     concat.dropna().groupby(["group", "position", "trap"]).apply(len)
# )
# group_pos_trapswcell = (
#     group_pos_trap_ncells.dropna().groupby(["group", "position"]).apply(len)
# )


class Meta:
    """
    Convenience class to fetch data from hdf5 file
    """

    def __init__(self, filename):
        self.filename = filename

    @property
    def ntimepoints(self):
        with h5py.File(self.filename, "r") as f:
            return f.attrs["time_settings/ntimepoints"][0]


class Compiler(ProcessABC):
    def __init__(self, parameters):
        pass
        # super().__init__(parameters)

    @abstractmethod
    def load_data(self):
        """Abstract function that must be reimplemented"""
        pass

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
            for method in ("slice", "slices", "delta_traps", "pertrap_metric")
        }

    def load_data(self, path: PosixPath):
        self.grouper = NameGrouper(path)
        self.meta = Meta(self.grouper.files[0])

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

    def concat_signal(
        self, sigloc=None, mode="mothers", *args, **kwargs
    ) -> pd.DataFrame:

        if sigloc == None:
            sigloc = "extraction/general/None/volume"
        self.sigloc = sigloc

        if mode == None:
            mode = "mothers"

        if not hasattr(self, "_concat") or self.sigloc != sigloc:
            self._concat = self.grouper.concat_signal(
                self.sigloc, pool=7, mode="mothers"
            )

        return self._concat

    def get_tp(self, sigloc=None, tp=None, mode=None, *args, **kwargs) -> pd.Series:

        if tp is None:
            tp = 0

        if mode == None:
            mode = True

        return self.concat_signal(sigloc=sigloc, mode=mode, *args, **kwargs).iloc[:, tp]

    def compile_slices(self, nslices=2, *args, **kwargs):
        tps = [
            min(i * (self.meta.ntimepoints // nslices), self.meta.ntimepoints - 1)
            for i in range(nslices + 1)
        ]
        slices = [self.compile_slice(tp=tp) for tp in tps]
        slices_df = pd.concat(slices)

        slices_df["timepoint"] = np.concatenate(
            [np.repeat(tp, len(slice_df)) for tp, slice_df in zip(tps, slices)]
        )

        return slices_df

    def compile_slice_end(self, *args, **kwargs):
        return self.compile_slice(tp=-1, *args, **kwargs)

    def compile_slice(
        self, sigloc=None, tp=None, metrics=None, mode=None, *args, **kwargs
    ) -> pd.DataFrame:

        if sigloc == None:
            self.sigloc = "extraction/general/None/volume"

        if tp == None:
            tp = 0

        if metrics == None:
            metrics = ("max", "mean", "median", "count", "std", "sem")

        if mode == None:
            mode = True

        df = pd.concat(
            [
                getattr(
                    self.get_tp(sigloc=sigloc, tp=tp, mode=mode, *args, **kwargs)
                    .groupby(["group", "position", "trap"])
                    .max()
                    .groupby(["group", "position"]),
                    met,
                )()
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

        return df.reset_index()

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
        group_names = compiler.grouper.group_names
        tups = [
            (group_names[pos], pos, axis, val)
            for pos, coords in self.grouper.traplocs().items()
            for axis, vals in zip(("x", "y"), self.traploc_diffs(coords))
            for val in vals
        ]

        return pd.DataFrame(tups, columns=["group", "position", "axis", "value"])

    def compile_pertrap_metrics(self):
        pass

    def compile_pertrap_metric(
        self,
        ranges: Iterable[Iterable[int]] = [
            [0, -1],
        ],
        metric: str = "count",
    ):
        "Get the number of cells per trap present during the given ranges"
        sig = compiler.concat_signal()

        for i, rngs in enumerate(ranges):
            for j, edge in enumerate(rngs):
                if edge < 0:
                    ranges[i][j] = sig.shape[1] - i + 1

        return pd.concat(
            [
                self.get_filled_trapcounts(sig.loc(axis=1)[slice(*rng)], metric=metric)
                for rng in ranges
            ],
            axis=1,
        )

    def get_filled_trapcounts(self, signal: pd.DataFrame, metric: str) -> pd.Series:
        present = signal.apply(
            lambda x: (not x.first_valid_index())
            & (x.last_valid_index() == len(x) - 1),
            axis=1,
        )
        results = getattr(
            signal.loc[present].iloc[:, 0].groupby(["group", "position", "trap"]),
            metric,
        )()
        filled = self.fill_trapcount(results)
        return filled

    def fill_trapcount(
        self, srs: pd.Series, fill_value: Union[int, float] = 0
    ) -> pd.Series:
        """Fill the last level of a MultiIndex in a pd.Series

        Use compiler to get the max number of traps per position and use this
        information to add rows with empty values (with plottings of distributions
        in mind)

        Parameters
        ----------
        srs : pd.Series
            Series with a pd.MultiIndex index
        compiler : ExperimentCompiler
            class with 'ntraps' information that returns a dictionary with position
            -> ntraps.
        fill_value : Union[int, float]
            Value used to fill new rows.

        Returns
        -------
        pd.Series
            Series with no numbers skipped on the last level.

        Examples
        --------
        FIXME: Add docs.

        """

        all_sets = set(
            [(pos, i) for pos, ntraps in compiler.ntraps.items() for i in range(ntraps)]
        )
        dif = all_sets.difference(
            set(
                zip(*[srs.index.get_level_values(i) for i in ("position", "trap")])
            ).difference()
        )
        new_indices = pd.MultiIndex.from_tuples(
            [
                (compiler.grouper.group_names[idx[0]], idx[0], np.uint(idx[1]))
                for idx in dif
            ]
        )
        new_indices = new_indices.set_levels(
            new_indices.levels[-1].astype(np.uint), level=-1
        )
        empty = pd.Series(fill_value, index=new_indices, name="ncells")
        return pd.concat((srs, empty))


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
            grid_spec = (1, 1)
        title_fontsize = "x-small"
        # self.fig = plt.figure(dpi=300, tight_layout=True)
        self.fig = plt.figure(dpi=300)
        self.fig.set_size_inches(8.27, 11.69, forward=True)
        plt.figtext(0.02, 0.99, "", fontsize="small")
        self.gs = plt.GridSpec(*grid_spec, wspace=0.3, hspace=0.3)
        self.data = {
            k: df.reset_index().sort_values("group")
            if hasattr(df, "reset_index")
            else df.sort_values("group")
            for k, df in data.items()
        }

    def place_plot(self, func, xloc=None, yloc=None, *args, **kwargs):
        if xloc is None:
            xloc = slice(0, self.gs.ncols)
        if yloc is None:
            yloc = slice(0, self.gs.nrows)

        return func(
            *args,
            ax=self.fig.add_subplot(self.gs[xloc, yloc]),
            **kwargs,
        )

    def plot(self):
        instructions: Iterable[Dict[str, Union[str, Iterable]]] = (
            {
                "data": "slice",
                "func": "stripplot",
                "args": ("count", "group"),
                "kwargs": {
                    "hue": None,
                },
                "loc": (0, 0),
            },
            {
                "data": "delta_traps",
                "func": "barplot",
                "args": ("axis", "value"),
                "kwargs": {
                    "hue": "group",
                    # "hue": "group",
                },
                "loc": (0, 1),
            },
            {
                "data": "slices",
                "func": "violinplot",
                "args": ("group", "median"),
                "kwargs": {
                    "hue": "timepoint",
                },
                "loc": (1, 0),
            },
            {
                "data": "pertrap_metric",
                "func": "histplot",
                "args": (0, None),
                "kwargs": {
                    "hue": "group",
                    # "hue": "group",
                },
                "loc": (1, 1),
            },
        )

        self.plots = [
            self.place_plot(
                self.gen_sns_wrapper(how),
                *how["loc"],
            )
            for how in instructions
        ]

    def show(self):
        plt.show()

    def save(self, path: PosixPath):
        pp = PdfPages(path)
        pp.savefig(self.fig)
        pp.close()

    def gen_sns_wrapper(self, how):
        def sns_wrapper(ax):
            getattr(sns, how["func"])(
                data=self.data[how["data"]],
                x=how["args"][0],
                y=how["args"][1],
                **how["kwargs"],
                ax=ax,
            )

        return sns_wrapper


for dir in dirs:
    print(dir)
    compiler = ExperimentCompiler(None, dir)
    tmp = compiler.run()
    po = PageOrganiser(tmp, grid_spec=(3, 2))
    po.plot()
    po.save(Path(dir) / "report.pdf")

    #### Live editing

# plot.set_title("Trap identification robustness")
# plot.set_xlabel("Axis")
# plot.set_ylabel("Distance (pixels)")
# plt.show()


# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()


import numpy as np


# plt1 = dummyplot()
# plt2 = dummyplot()
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
