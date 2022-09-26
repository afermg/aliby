"""
Script in development
"""

# /usr/bin/env python3
import re
import warnings
from abc import abstractmethod
from collections import Counter
from pathlib import PosixPath
from typing import Dict, Iterable, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from agora.abc import ProcessABC
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ndarray
from scipy.signal import find_peaks

from postprocessor.grouper import NameGrouper

sns.set_style("darkgrid")


# Main dataframe structure

# | position | group | ntraps |robustness index | initial_ncells | final_ncells
# dir = "/home/alan/Documents/dev/skeletons/data/2021_06_15_pypipeline_unit_test_00/2021_06_15_pypipeline_unit_test_00/"
# dir = "/home/alan/Documents/dev/libs/aliby/data/2021_08_24_2Raf_00/2021_08_24_2Raf_00/"
# dirs = [
#     "16543_2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_01",
#     "16545_2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_secondRun_01",
#     "18069_2019_12_05_aggregates_updownshift_2_0_2_URA8_URA7H360A_URA7H360R_00",
#     "18616_2020_02_20_protAgg_downUpShift_2_0_2_Ura8_Ura8HA_Ura8HR_01",
#     "18617_2020_02_21_protAgg_downUpShift_2_0_2_pHluorin_Ura7HA_Ura7HR_00",
#     "19129_2020_09_06_DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin_00",
#     "19144_2020_09_07_DownUpshift_2_0_2_glu_ura_mig1msn2_phluorin_secondRound_00",
#     "19169_2020_09_09_downUpshift_2_0_2_glu_ura8_phl_mig1_phl_msn2_03",
#     "19199_2020_09_29_downUpshift_2_0_2_glu_ura8_ura8h360a_ura8h360r_00",
#     "19203_2020_09_30_downUpshift_twice_2_0_2_glu_ura8_ura8h360a_ura8h360r_00",
#     "19207_2020_10_01_exp_00",
#     "19232_2020_10_02_downUpshift_twice_2_0_2_glu_ura8_phluorinMsn2_phluorinMig1_01",
#     "19307_2020_10_22_downUpshift_2_01_2_glucose_dual_pH__dot6_nrg1_tod6__00",
#     "19310_2020_10_22_downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__thrice_00",
#     "19311_2020_10_23_downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__twice__04",
#     "19328_2020_10_31_downUpshift_four_2_0_2_glu_dual_phl__glt1_ura8_ura8__00",
#     "19329_2020_11_01_exp_00",
#     "19333_2020_11_02_downUpshift_2_0_2_glu_ura7_ura7ha_ura7hr_00",
#     "19334_2020_11_02_downUpshift_2_0_2_glu_ura8_ura8ha_ura8hr_00",
#     "19447_2020_11_18_downUpshift_2_0_2_glu_gcd2_gcd6_gcd7__02",
#     "19810_2021_02_21_ToxicityTest_00",
#     "19993_2021_06_15_pypipeline_unit_test_00",
#     "19996_2021_06_27_ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01",
#     "20419_2021_11_02_dose_response_raf_05_075_2_glu_005_2_constantMedia_00",
# ]
# outdir = "/home/alan/Documents/dev/skeletons/data"
# dirs = Path(outdir).glob("*ph*")


# from abc import abstractclassmethod, abstractmethod


# group_pos_trap_ncells = (
#     concat.dropna().groupby(["group", "position", "trap"]).apply(len)
# )
# group_pos_trapswcell = (
#     group_pos_trap_ncells.dropna().groupby(["group", "position"]).apply(len)
# )


class Meta:
    """Convenience class to fetch data from hdf5 file."""

    def __init__(self, filename):
        self.filename = filename

    @property
    def ntimepoints(self):
        with h5py.File(self.filename, "r") as f:
            return f.attrs["time_settings/ntimepoints"][0]


class Compiler(ProcessABC):
    # def __init__(self, parameters):
    #     super().__init__(parameters)

    @abstractmethod
    def load_data(self):
        """Abstract function that must be reimplemented."""
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
            for method in (
                "slice",
                "slices",
                "delta_traps",
                "pertrap_metric",
                "ncells",
                "last_valid_tp",
                "stages_dmetric",
                "fluorescence",
            )
        }

    def load_data(self, path: PosixPath):
        self.grouper = NameGrouper(path)
        self.meta = Meta(self.grouper.files[0])

    @property
    def ntraps(self) -> dict:
        """Get the number of traps in each position.

        Returns ------- dict str -> int  Examples -------- FIXME: Add
        docs.
        """
        return {
            pos: coords.shape[0]
            for pos, coords in self.grouper.traplocs().items()
        }

    def concat_signal(self, sigloc=None, mode=None, **kwargs) -> pd.DataFrame:
        if sigloc is None:
            sigloc = "extraction/general/None/volume"
        self.sigloc = sigloc

        if mode is None:
            mode = "retained"

        if not hasattr(self, "_concat") or self.sigloc != sigloc:
            self._concat = self.grouper.concat_signal(
                self.sigloc, mode=mode, **kwargs
            )

        return self._concat

    def get_tp(self, sigloc=None, tp=None, mode=None, **kwargs) -> pd.Series:
        if tp is None:
            tp = 0

        if mode is None:
            mode = True

        return self.concat_signal(sigloc=sigloc, mode=mode, **kwargs).iloc[
            :, tp
        ]

    def count_cells(
        self,
        signal="extraction/general/None/volume",
        mode="raw",
        **kwargs,
    ):
        df = self.grouper.concat_signal(signal, mode=mode, **kwargs)
        df = df.groupby(["group", "position", "trap"]).count()
        df[df == 0] = np.nan
        return df

    def compile_dmetrics(self, stages=None):
        """Generate dataframe with dVol metrics without major cell picking."""
        names_signals = {
            "dvol": "postprocessing/dsignal/postprocessing_savgol_extraction_general_None_volume",
            "bud_dvol": "postprocessing/bud_metric/postprocessing_dsignal_postprocessing_savgol_extraction_general_None_volume",
        }
        names_signals = {
            "dvol": "postprocessing/dsignal/postprocessing_savgol_extraction_general_None_volume",
            "bud_dvol": "postprocessing/bud_metric/postprocessing_dsignal_postprocessing_savgol_extraction_general_None_volume",
            "buddings": "postprocessing/buddings/extraction_general_None_volume",
        }
        operations = {
            "dvol": ("dvol", "max"),
            "bud_dvol": ("bud_dvol", "max"),
            "buddings": ("buddings", "sum"),
            "buddings_mean": ("buddings", "mean"),
        }

        input_signals = {
            k: self.grouper.concat_signal(v) for k, v in names_signals.items()
        }

        ids = input_signals["buddings"].index
        for v in input_signals.values():
            ids = ids.intersection(v.index)

        if stages:

            def process_dfs(dfs, rng):
                return pd.DataFrame(
                    {
                        k: getattr(dfs[sig].loc(axis=1)[rng].loc[ids], op)(
                            axis=1
                        )
                        if isinstance(op, str)
                        else dfs[sig].loc[ids].apply(op, axis=1)
                        for k, (sig, op) in operations.items()
                    }
                )

            # Note that all input_signals columns must be the same
            col_vals = list(input_signals.values())[0].columns
            stages_dfs = {"Full": process_dfs(input_signals, col_vals)}
            for k, rng in stages:
                stage_df = process_dfs(input_signals, col_vals[rng])
                stages_dfs[k] = stage_df

        concat = pd.concat([x.reset_index() for x in stages_dfs.values()])
        concat["stage"] = np.array(
            [
                np.repeat(x, len(concat) // len(stages_dfs))
                for x in stages_dfs.keys()
            ]
        ).flatten()

        return (
            concat.set_index(["group", "position", "trap", "cell_label"])
            .melt("stage", ignore_index=False, var_name="growth_metric")
            .reset_index()
        )

    def compile_stages_dmetric(self):
        stages = self.get_stages()
        return self.compile_dmetrics(stages=stages)

    def get_stages(self):
        """Use the metadata to give a prediction of the media being pumped at
        each time point. Works for traditional metadata (pre-fluigent).

        Returns: ------ A list of tuples where in each the first value
        is the active     pump's contents and the second its associated
        range of time points
        """
        fpath = list(self.grouper.signals.values())[0].filename
        with h5py.File(fpath, "r") as f:
            tinterval = f.attrs.get("time_settings/timeinterval", None)[0]
            tnorm = tinterval / 60
            switch_times = f.attrs.get("switchtimes", None) / tnorm
            last_tp = (
                f.attrs.get("time_settings/totaltime", None)[0] / tinterval
            )
            pump_contents = f.attrs.get("pumpinit/contents", None)
            init_frate = f.attrs.get("pumpinit/flowrate", None)
            prate = f.attrs.get("pumprate", None)
            main_pump = np.array((init_frate.argmax(), *prate.argmax(axis=0)))

            intervals = np.array((0, *switch_times, last_tp), dtype=int)

            extracted_tps = self.grouper.ntimepoints
            stages = [  # Only add intervals with length larger than zero
                (
                    ": ".join((str(i + 1), pump_contents[p_id])),
                    range(intervals[i], min(intervals[i + 1], extracted_tps)),
                )
                for i, p_id in enumerate(main_pump)
                if (intervals[i + 1] > intervals[i])
            ]
            return stages

    def compile_growth_metrics(
        self,
        min_nbuddings: int = 2,
    ):
        """Filter mothers with n number of buddings and get their metrics.

        Select cells with at least two recorded buddings
        """
        names_signals = {
            "dvol": "postprocessing/dsignal/postprocessing_savgol_extraction_general_None_volume",
            "bud_dvol": "postprocessing/bud_metric/postprocessing_dsignal_postprocessing_savgol_extraction_general_None_volume",
            "buddings": "postprocessing/buddings/extraction_general_None_volume",
        }
        operations = {
            "dvol": ("dvol", "max"),
            "bud_dvol": ("bud_dvol", "max"),
            "buddings": ("buddings", "sum"),
            "cycle_length_mean": (
                "buddings",
                lambda x: bn.nanmean(np.diff(np.where(x)[0])),
            ),
            "cycle_length_min": (
                "buddings",
                lambda x: bn.nanmin(np.diff(np.where(x)[0])),
            ),
            "cycle_length_median": (
                "buddings",
                lambda x: np.nanmedian(np.diff(np.where(x)[0])),
            ),
        }

        input_signals = {
            k: self.grouper.concat_signal(v) for k, v in names_signals.items()
        }
        ids = self.get_shared_ids(input_signals, min_nbuddings=min_nbuddings)

        compiled_df = pd.DataFrame(
            {
                k: getattr(input_signals[sig].loc[ids], op)(axis=1)
                if isinstance(op, str)
                else input_signals[sig].loc[ids].apply(op, axis=1)
                for k, (sig, op) in operations.items()
            }
        )
        return compiled_df

    def get_shared_ids(
        self, input_signals: Dict[str, pd.DataFrame], min_nbuddings: int = None
    ):
        """Get the intersection id of multiple signals.

        "buddings" must be one the keys in input_signals to use the
        argument min_nbuddings.
        """
        ids = list(input_signals.values())[0].index
        if min_nbuddings is not None:
            ids = (
                input_signals["buddings"]
                .loc[input_signals["buddings"].sum(axis=1) >= min_nbuddings]
                .index
            )
        for v in input_signals.values():
            ids = ids.intersection(v.index)

        return ids

    def compile_ncells(self):
        df = self.count_cells()
        df = df.melt(ignore_index=False)
        df.columns = ["timepoint", "ncells_pertrap"]

        return df

    def compile_last_valid_tp(self) -> pd.Series:
        """Last valid timepoint per position."""
        df = self.count_cells()
        df = df.apply(lambda x: x.last_valid_index(), axis=1)
        df = df.groupby(["group", "position"]).max()

        return df

    def compile_slices(self, nslices=2, **kwargs):
        tps = [
            min(
                i * (self.grouper.ntimepoints // nslices),
                self.grouper.ntimepoints - 1,
            )
            for i in range(nslices + 1)
        ]
        slices = [self.compile_slice(tp=tp, **kwargs) for tp in tps]
        slices_df = pd.concat(slices)

        slices_df["timepoint"] = np.concatenate(
            [np.repeat(tp, len(slice_df)) for tp, slice_df in zip(tps, slices)]
        )

        return slices_df

    def compile_slice_end(self, **kwargs):
        return self.compile_slice(tp=-1, **kwargs)

    def guess_metrics(self, metrics: Dict[str, Tuple[str]] = None):
        """First approach at autoselecting certain signals for automated
        analysis."""

        if metrics is None:
            metrics = {
                "GFP": ("median", "max5"),
                "mCherry": ("median", "max5"),
                # "general": ("eccentricity",),
                "Flavin": ("median",),
                "postprocessing/savgol": ("volume",),
                "dsignal/postprocessing_savgol": ("volume",),
                "bud_metric.*dsignal.*savgol": ("volume",),
                "ph_ratio": ("median",),
            }

        sigs = self.grouper.siglist
        selection = {
            ".".join((ch, metric)): sig
            for sig in sigs
            for ch, metric_set in metrics.items()
            for metric in metric_set
            if re.search("(?!.*bgsub).*".join((ch, metric)) + "$", sig)
        }
        return selection

    def compile_fluorescence(
        self,
        metrics: Dict[str, Tuple[str]] = None,
        norm: tuple = None,
        **kwargs,
    ):
        """Get a single signal per."""
        if norm is None:
            norm = (
                "GFP",
                "GFPFast",
                "ph_ratio",
                "Flavin",
                "Citrine",
                "mCherry",
            )

        selection = self.guess_metrics(metrics)

        input_signals = {
            k: self.grouper.concat_signal(v, **kwargs)
            for k, v in selection.items()
        }

        # ids = self.get_shared_ids(input_signals)

        to_concat = []

        def format_df(df):
            return df.melt(
                ignore_index=False, var_name="timepoint"
            ).reset_index()

        for k, v in input_signals.items():
            tmp_formatted = format_df(v)
            tmp_formatted["signal"] = k
            to_concat.append(tmp_formatted)
            if norm and k.split(".")[0] in norm:
                norm_v = v.subtract(v.min(axis=1), axis=0).div(
                    v.max(axis=1) - v.min(axis=1), axis=0
                )
                # norm_v = v.groupby(["position", "trap", "cell_label"]).transform(
                #     # lambda x: x - x.min() / (x.max() - x.min())
                #     lambda x: (x - x.min())
                #     / (x.max() - x.min())
                # )
                formatted = format_df(norm_v)
                formatted["signal"] = "norm_" + k
                to_concat.append(formatted)

        concated = pd.concat(to_concat, axis=0)

        return concated

    def compile_slice(
        self, sigloc=None, tp=None, metrics=None, mode=None, **kwargs
    ) -> pd.DataFrame:
        if sigloc is None:
            self.sigloc = "extraction/general/None/volume"

        if tp is None:
            tp = 0

        if metrics is None:
            metrics = ("max", "mean", "median", "count", "std", "sem")

        if mode is None:
            mode = True

        df = pd.concat(
            [
                getattr(
                    self.get_tp(sigloc=sigloc, tp=tp, mode=mode, **kwargs)
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
            warnings.warn(
                "ExpCompiler: Replacing existing column in compilation"
            )
        df[name] = [
            new_values_d[pos] for pos in df.index.get_level_values("position")
        ]

        return df

    @staticmethod
    def traploc_diffs(traplocs: ndarray) -> list:
        """Obtain metrics for trap localisation.

        Parameters ---------- traplocs : ndarray     (x,2) 2-dimensional
        array with the x,y coordinates of traps in each     column
        Examples -------- FIXME: Add docs.
        """
        signal = np.zeros((traplocs.max(), 2))
        for i in range(2):
            counts = Counter(traplocs[:, i])
            for j, v in counts.items():
                signal[j - 1, i] = v

        diffs = [
            np.diff(x)
            for x in np.apply_along_axis(find_peaks, 0, signal, distance=10)[0]
        ]
        return diffs

    def compile_delta_traps(self):
        group_names = self.grouper.group_names
        tups = [
            (group_names[pos], pos, axis, val)
            for pos, coords in self.grouper.traplocs().items()
            for axis, vals in zip(("x", "y"), self.traploc_diffs(coords))
            for val in vals
        ]

        return pd.DataFrame(
            tups, columns=["group", "position", "axis", "value"]
        )

    def compile_pertrap_metric(
        self,
        ranges: Iterable[Iterable[int]] = [
            [0, -1],
        ],
        metric: str = "count",
    ):
        """Get the number of cells per trap present during the given ranges."""
        sig = self.concat_signal()

        for i, rngs in enumerate(ranges):
            for j, edge in enumerate(rngs):
                if edge < 0:
                    ranges[i][j] = sig.shape[1] - i + 1
        df = pd.concat(
            [
                self.get_filled_trapcounts(
                    sig.loc(axis=1)[slice(*rng)], metric=metric
                )
                for rng in ranges
            ],
            axis=1,
        )
        return df.astype(str)

    def get_filled_trapcounts(
        self, signal: pd.DataFrame, metric: str
    ) -> pd.Series:
        present = signal.apply(
            lambda x: (not x.first_valid_index())
            & (x.last_valid_index() == len(x) - 1),
            axis=1,
        )
        results = getattr(
            signal.loc[present]
            .iloc[:, 0]
            .groupby(["group", "position", "trap"]),
            metric,
        )()
        filled = self.fill_trapcount(results)
        return filled

    def fill_trapcount(
        self, srs: pd.Series, fill_value: Union[int, float] = 0
    ) -> pd.Series:
        """Fill the last level of a MultiIndex in a pd.Series.

        Use self to get the max number of traps per position and use
        this information to add rows with empty values (with plottings
        of distributions in mind)  Parameters ---------- srs : pd.Series
        Series with a pd.MultiIndex index self : ExperimentSelf
        class with 'ntraps' information that returns a dictionary with
        position     -> ntraps. fill_value : Union[int, float]     Value
        used to fill new rows.  Returns ------- pd.Series     Series
        with no numbers skipped on the last level.  Examples --------
        FIXME: Add docs.
        """

        all_sets = set(
            [
                (pos, i)
                for pos, ntraps in self.ntraps.items()
                for i in range(ntraps)
            ]
        )
        dif = all_sets.difference(
            set(
                zip(
                    *[
                        srs.index.get_level_values(i)
                        for i in ("position", "trap")
                    ]
                )
            ).difference()
        )
        new_indices = pd.MultiIndex.from_tuples(
            [
                (self.grouper.group_names[idx[0]], idx[0], np.uint(idx[1]))
                for idx in dif
            ]
        )
        new_indices = new_indices.set_levels(
            new_indices.levels[-1].astype(np.uint), level=-1
        )
        empty = pd.Series(fill_value, index=new_indices, name="ncells")
        return pd.concat((srs, empty))


class Reporter(object):
    """Manages Multiple pages to generate a report."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        pages: dict = None,
        path: str = None,
    ):
        self.data = data

        if pages is None:
            pages = {
                "qa": self.gen_page_qa(),
                "growth": self.gen_page_growth(),
                "fluorescence": self.gen_page_fluorescence(),
            }
        self.pages = pages

        if path is not None:
            self.path = path

        self.porgs = {k: PageOrganiser(data, v) for k, v in pages.items()}

    @property
    def pdf(self):
        return self._pdf

    @pdf.setter
    def pdf(self, path: str):
        self._pdf = PdfPages(path)

    def plot_report(self, path: str = None):
        if path is None:
            path = self.path

        with PdfPages(path) as pdf:
            for page_org in list(self.porgs.values())[::-1]:
                page_org.plot_page()
                pdf.savefig(page_org.fig)
                # pdf.savefig()
                plt.close()

    @staticmethod
    def gen_page_qa():
        page_qc = (
            {
                "data": "slice",
                "func": "barplot",
                "args": ("ntraps", "position"),
                "kwargs": {"hue": "group", "palette": "muted"},
                "loc": (0, 0),
            },
            {
                "data": "delta_traps",
                "func": "barplot",
                "args": ("axis", "value"),
                "kwargs": {
                    "hue": "group",
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
                "loc": (2, 1),
            },
            {
                "data": "pertrap_metric",
                "func": "histplot",
                "args": (0, None),
                "kwargs": {
                    "hue": "group",
                    "multiple": "dodge",
                    "discrete": True,
                },
                "loc": (2, 0),
            },
            {
                "data": "ncells",
                "func": "lineplot",
                "args": ("timepoint", "ncells_pertrap"),
                "kwargs": {
                    "hue": "group",
                },
                "loc": (1, 1),
            },
            {
                "data": "last_valid_tp",
                "func": "stripplot",
                "args": (0, "position"),
                "kwargs": {
                    "hue": "group",
                },
                "loc": (1, 0),
            },
        )
        return page_qc

    @staticmethod
    def gen_page_fluorescence():
        return (
            {
                "data": "fluorescence",
                "func": "relplot",
                "args": ("timepoint", "value"),
                "kwargs": {
                    "col": "signal",
                    "col_wrap": 2,
                    "hue": "group",
                    "facet_kws": {"sharey": False, "sharex": True},
                    "kind": "line",
                },
            },
        )

    def gen_page_cell_cell_corr():
        pass

    @staticmethod
    def gen_page_growth():
        return (
            {
                "data": "stages_dmetric",
                "func": "catplot",
                "args": ("stage", "value"),
                "kwargs": {
                    "hue": "group",
                    "col": "growth_metric",
                    "col_wrap": 2,
                    "kind": "box",
                    "sharey": False,
                },
            },
        )

    def gen_all_instructions(self):
        qa = self.gen_page_qa()
        growth = self.gen_page_growth()

        return (qa, growth)


class PageOrganiser(object):
    """Add multiple plots to a single page, wither using seaborn multiplots or
    manual GridSpec."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        instruction_set: Iterable = None,
        grid_spec: tuple = None,
        fig_kws: dict = None,
    ):
        self.instruction_set = instruction_set
        self.data = {k: df for k, df in data.items()}

        self.single_fig = True
        if len(instruction_set) > 1:
            self.single_fig = False

        if not self.single_fig:  # Select grid_spec with location info
            if grid_spec is None:
                locs = np.array(
                    [x.get("loc", (0, 0)) for x in instruction_set]
                )
                grid_spec = locs.max(axis=0) + 1

            if fig_kws is None:
                self.fig = plt.figure(dpi=300)
                self.fig.set_size_inches(8.27, 11.69, forward=True)
                plt.figtext(0.02, 0.99, "", fontsize="small")
            self.gs = plt.GridSpec(*grid_spec, wspace=0.3, hspace=0.3)

            self.axes = {}
            reset_index = (
                lambda df: df.reset_index().sort_values("position")
                if isinstance(df.index, pd.core.indexes.multi.MultiIndex)
                else df.sort_values("position")
            )
            self.data = {k: reset_index(df) for k, df in self.data.items()}

    def place_plot(self, func, xloc=None, yloc=None, **kwargs):
        if xloc is None:
            xloc = 0
        if yloc is None:
            yloc = 0

        if (
            self.single_fig
        ):  # If plotting using a figure method using seaborn cols/rows
            self.g = func(**kwargs)
            self.axes = {
                ax.title.get_text().split("=")[-1][1:]: ax
                for ax in self.g.axes.flat
            }
            self.fig = self.g.fig
        else:
            self.axes[(xloc, yloc)] = self.fig.add_subplot(self.gs[xloc, yloc])
            func(
                ax=self.axes[(xloc, yloc)],
                **kwargs,
            )

        # Eye candy
        if np.any(  # If there is a long label, rotate them all
            [
                len(lbl.get_text()) > 8
                for ax in self.axes.values()
                for lbl in ax.get_xticklabels()
            ]
        ) and hasattr(self, "g"):
            for axes in self.g.axes.flat:
                _ = axes.set_xticklabels(
                    axes.get_xticklabels(),
                    rotation=15,
                    horizontalalignment="right",
                )

    def plot_page(
        self, instructions: Iterable[Dict[str, Union[str, Iterable]]] = None
    ):
        if instructions is None:
            instructions = self.instruction_set
        if isinstance(instructions, dict):
            how = (instructions,)

        for how in instructions:
            self.place_plot(
                self.gen_sns_wrapper(how),
                *how.get("loc", (None, None)),
            )

    def gen_sns_wrapper(self, how):
        def sns_wrapper(ax=None):
            kwargs = how.get("kwargs", {})
            if ax:
                kwargs["ax"] = ax
            elif "height" not in kwargs:
                ncols = kwargs.get("col_wrap", 1)
                if "col" in kwargs:
                    nrows = np.ceil(
                        len(np.unique(self.data[how["data"]][kwargs["col"]]))
                        / ncols
                    )
                else:
                    nrows = len(
                        np.unique(self.data[how["data"]][kwargs["row"]])
                    )

                kwargs["height"] = 11.7
                # kwargs["aspect"] = 8.27 / (11.7 / kwargs["col_wrap"])
                kwargs["aspect"] = (8.27 / ncols) / (kwargs["height"] / nrows)
            return getattr(sns, how["func"])(
                data=self.data[how["data"]],
                x=how["args"][0],
                y=how["args"][1],
                **kwargs,
            )

        return sns_wrapper


# fpath = "/home/alan/Documents/dev/skeletons/scripts/aggregates_exploration/18616_2020_02_20_protAgg_downUpShift_2_0_2_Ura8_Ura8HA_Ura8HR_01"
# # compiler = ExperimentCompiler(None, base_dir / dir)
# compiler = ExperimentCompiler(None, fpath)
# dfs = compiler.run()
# rep = Reporter(data=dfs, path=Path(fpath) / "report.pdf")
# rep.plot_report("./report.pdf")
# base_dir = Path("/home/alan/Documents/dev/skeletons/scripts/data/")
# for dir in dirs:
#     try:
#         compiler = ExperimentCompiler(None, base_dir / dir)
#         dfs = compiler.run()
#         rep = Reporter(data=dfs, path=base_dir / (dir + "/report.pdf"))
#         from time import time

#         rep.plot_report(base_dir / (dir + "/report.pdf"))
#     except Exception as e:
#         print("LOG:ERROR:", e)
# with open("errors.log", "a") as f:
#     f.write(e)
