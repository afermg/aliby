#!/usr/bin/env python3

import re
import typing as t
from abc import ABC, abstractproperty
from collections import Counter
from functools import cached_property as property
from pathlib import Path, PosixPath
from typing import Dict, List, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from agora.io.signal import Signal
from pathos.multiprocessing import Pool


class Grouper(ABC):
    """Base grouper class."""

    files = []

    def __init__(self, dir: Union[str, PosixPath]):
        path = Path(dir)
        self.name = path.name
        assert path.exists(), "Dir does not exist"
        self.files = list(path.glob("*.h5"))
        assert len(self.files), "No valid h5 files in dir"
        self.load_signals()

    def load_signals(self) -> None:
        # Sets self.signals
        self.signals = {f.name[:-3]: Signal(f) for f in self.files}

    @property
    def fsignal(self) -> Signal:
        # Returns first signal
        return list(self.signals.values())[0]

    @property
    def ntimepoints(self) -> int:
        return max([s.ntimepoints for s in self.signals.values()])

    @property
    def tintervals(self) -> float:
        tintervals = set([s.tinterval / 60 for s in self.signals.values()])
        assert (
            len(tintervals) == 1
        ), "Not all signals have the same time interval"

        return max(tintervals)

    @property
    def siglist(self) -> None:
        return self.fsignal.siglist

    @property
    def siglist_grouped(self) -> None:
        if not hasattr(self, "_siglist_grouped"):
            self._siglist_grouped = Counter(
                [x for s in self.signals.values() for x in s.siglist]
            )

        for s, n in self._siglist_grouped.items():
            print(f"{s} - {n}")

    @property
    def datasets(self) -> None:
        """Print available datasets in first Signal instance."""
        return self.fsignal.datasets

    @abstractproperty
    def positions_groups(self):
        pass

    def concat_signal(
        self,
        path: str,
        reduce_cols: t.Optional[bool] = None,
        axis: int = 0,
        mode: str = "retained",
        pool: t.Optional[int] = None,
        **kwargs,
    ):
        """Concatenate a single signal.

        Parameters
        ----------
        path : str
            signal address within h5py file.
        reduce_cols : bool
            Whether or not to collapse columns into a single one.
        axis : int
            Concatenation axis.
        pool : int
            Number of threads used. If 0 or None only one core is used.
        **kwargs : key, value pairings
            Named arguments to pass to concat_ind_function.

        Examples
        --------
        FIXME: Add docs.
        """
        if not path.startswith("/"):
            path = "/" + path

        # Check the path is in a given signal
        sitems = {k: v for k, v in self.signals.items() if path in v.siglist}
        nsignals_dif = len(self.signals) - len(sitems)
        if nsignals_dif:
            print(
                f"Grouper:Warning: {nsignals_dif} signals do not contain"
                f" channel {path}"
            )

        signals = self.pool_function(
            path=path, f=concat_signal_ind, mode=mode, pool=pool, **kwargs
        )

        errors = [k for s, k in zip(signals, self.signals.keys()) if s is None]
        signals = [s for s in signals if s is not None]
        if len(errors):
            print("Warning: Positions contain errors {errors}")
            assert len(signals), "All datasets contain errors"
        sorted = pd.concat(signals, axis=axis).sort_index()
        if reduce_cols:
            sorted = sorted.apply(np.nanmean, axis=1)
            spath = path.split("/")
            sorted.name = "_".join([spath[1], spath[-1]])

        return sorted

    @property
    def nmembers(self) -> t.Dict[str, int]:
        # Return the number of positions belonging to each group
        return Counter(self.positions_groups.values())

    @property
    def ntraps(self):
        for pos, s in self.signals.items():
            with h5py.File(s.filename, "r") as f:
                print(pos, f["/trap_info/trap_locations"].shape[0])

    @property
    def ntraps_by_pos(self) -> t.Dict[str, int]:
        # Return total number of traps grouped
        ntraps = {}
        for pos, s in self.signals.items():
            with h5py.File(s.filename, "r") as f:
                ntraps[pos] = f["/trap_info/trap_locations"].shape[0]

        ntraps_by_pos = {k: 0 for k in self.groups}
        for posname, vals in ntraps.items():
            ntraps_by_pos[self.positions_groups[posname]] += vals

        return ntraps_by_pos

    def traplocs(self):
        d = {}
        for pos, s in self.signals.items():
            with h5py.File(s.filename, "r") as f:
                d[pos] = f["/trap_info/trap_locations"][()]
        return d

    @property
    def groups(self) -> t.Tuple[str]:
        # Return groups sorted alphabetically
        return tuple(sorted(set(self.positions_groups.values())))

    @property
    def positions(self) -> t.Tuple[str]:
        # Return positions sorted alphabetically
        return tuple(sorted(set(self.positions_groups.keys())))

    def ncells(
        self, path="/extraction/general/None/area", mode="retained", **kwargs
    ) -> t.Dict[str, int]:
        """
        Returns number of cells retained per position in base channel
        """
        return (
            self.concat_signal(path=path, mode=mode, **kwargs)
            .groupby("group")
            .apply(len)
            .to_dict()
        )

    @property
    def nretained(self) -> t.Dict[str, int]:
        return self.ncells()

    def pool_function(
        self,
        path: str,
        f: t.Callable,
        pool: t.Optional[int] = None,
        **kwargs,
    ):
        """
        Wrapper to add support for threading to process independent signals.
        Particularly useful when aggregating multiple elements.
        """
        if pool or pool is None:
            if pool is None:
                pool = 8

            with Pool(pool) as p:
                kymographs = p.map(
                    lambda x: f(
                        path=path,
                        signal=x[1],
                        group=self.positions_groups[x[0]],
                        **kwargs,
                    ),
                    self.signals.items(),
                )
        else:
            kymographs = [
                f(
                    path=path,
                    signal=signal,
                    group=self.positions_groups[name],
                    **kwargs,
                )
                for name, signal in self.signals.items()
            ]

        return kymographs

    @property
    def stages_span(self):
        return self.fsignal.stages_span

    @property
    def max_span(self):
        return self.fsignal.max_span

    @property
    def tinterval(self):
        return self.fsignal.tinterval

    @property
    def stages(self):
        return self.fsignal.stages


class MetaGrouper(Grouper):
    """Group positions using metadata's 'group' number."""

    pass


class NameGrouper(Grouper):
    """Group a set of positions using a subsection of the name."""

    def __init__(self, dir, criteria=None):
        super().__init__(dir=dir)

        if criteria is None:
            criteria = (0, -4)
        self.criteria = criteria

    @property
    def positions_groups(self) -> t.Dict[str, str]:
        if not hasattr(self, "_positions_groups"):
            self._positions_groups = {}
            for name in self.signals.keys():
                self._positions_groups[name] = name[
                    self.criteria[0] : self.criteria[1]
                ]

        return self._positions_groups

    # def aggregate_multisignals(self, paths=None, **kwargs):
    #     aggregated = pd.concat(
    #         [
    #             self.concat_signal(path, reduce_cols=np.nanmean, **kwargs)
    #             for path in paths
    #         ],
    #         axis=1,
    #     )
    #     # ph = pd.Series(
    #     #     [
    #     #         self.ph_from_group(x[list(aggregated.index.names).index("group")])
    #     #         for x in aggregated.index
    #     #     ],
    #     #     index=aggregated.index,
    #     #     name="media_pH",
    #     # )
    #     # self.aggregated = pd.concat((aggregated, ph), axis=1)

    #     return aggregated


class phGrouper(NameGrouper):
    """Grouper for pH calibration experiments where all surveyed media pH
    values are within a single experiment."""

    def __init__(self, dir, criteria=(3, 7)):
        super().__init__(dir=dir, criteria=criteria)

    def get_ph(self) -> None:
        self.ph = {gn: self.ph_from_group(gn) for gn in self.positions_groups}

    @staticmethod
    def ph_from_group(group_name: int) -> float:
        if group_name.startswith("ph_"):
            group_name = group_name[3:]

        return float(group_name.replace("_", "."))

    def aggregate_multisignals(self, paths: list) -> pd.DataFrame:
        """Accumulate multiple signals."""

        aggregated = pd.concat(
            [
                self.concat_signal(path, reduce_cols=np.nanmean)
                for path in paths
            ],
            axis=1,
        )
        ph = pd.Series(
            [
                self.ph_from_group(
                    x[list(aggregated.index.names).index("group")]
                )
                for x in aggregated.index
            ],
            index=aggregated.index,
            name="media_pH",
        )
        aggregated = pd.concat((aggregated, ph), axis=1)

        return aggregated


def concat_signal_ind(
    path: str,
    signal: Signal,
    group: str,
    mode: str = "retained",
    position=None,
    **kwargs,
) -> pd.DataFrame:
    """Core function that handles retrieval of an individual signal, applies
    filtering if requested and adjusts indices."""
    if position is None:
        position = signal.stem
    if mode == "retained":
        combined = signal.retained(path, **kwargs)
    if mode == "mothers":
        raise (NotImplementedError)
    elif mode == "raw":
        combined = signal.get_raw(path, **kwargs)
    elif mode == "families":
        combined = signal[path]
    combined["position"] = position
    combined["group"] = group
    combined.set_index(["group", "position"], inplace=True, append=True)
    combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)

    return combined
    # except:
    #     return None


class MultiGrouper:
    """Wrap results from multiple experiments stored as folders inside a
    folder."""

    def __init__(self, source: Union[str, list]):
        if isinstance(source, str):
            source = Path(source)
            self.exp_dirs = list(source.glob("*"))
        else:
            self.exp_dirs = [Path(x) for x in source]
        self.groupers = [NameGrouper(d) for d in self.exp_dirs]
        for group in self.groupers:
            group.load_signals()

    @property
    def siglist(self) -> None:
        for gpr in self.groupers:
            print(gpr.siglist_grouped)

    @property
    def sigtable(self) -> pd.DataFrame:
        """Generate a matrix containing the number of datasets for each signal
        and experiment."""

        def regex_cleanup(x):
            x = re.sub(r"\/extraction\/", "", x)
            x = re.sub(r"\/postprocessing\/", "", x)
            x = re.sub(r"\/np_max", "", x)

            return x

        if not hasattr(self, "_sigtable"):
            raw_mat = [
                [s.siglist for s in gpr.signals.values()]
                for gpr in self.groupers
            ]
            siglist_grouped = [
                Counter([x for y in grp for x in y]) for grp in raw_mat
            ]

            nexps = len(siglist_grouped)
            sigs_idx = list(
                set([y for x in siglist_grouped for y in x.keys()])
            )
            sigs_idx = [regex_cleanup(x) for x in sigs_idx]

            nsigs = len(sigs_idx)

            sig_matrix = np.zeros((nsigs, nexps))
            for i, c in enumerate(siglist_grouped):
                for k, v in c.items():
                    sig_matrix[sigs_idx.index(regex_cleanup(k)), i] = v

            sig_matrix[sig_matrix == 0] = np.nan
            self._sigtable = pd.DataFrame(
                sig_matrix,
                index=sigs_idx,
                columns=[x.name for x in self.exp_dirs],
            )
        return self._sigtable

    def sigtable_plot(self) -> None:
        """Plot number of signals for all available experiments.

        Examples
        --------
        FIXME: Add docs.
        """
        ax = sns.heatmap(self.sigtable, cmap="viridis")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=10,
            ha="right",
            rotation_mode="anchor",
        )
        plt.show()

    def aggregate_signal(
        self,
        signals: Union[str, list],
        **kwargs,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Aggregate signals from multiple Groupers (and thus experiments)

        Parameters
        ----------
        signals : Union[str, list]
            string or list of strings indicating the signal(s) to fetch.
        **kwargs : keyword arguments to pass to Grouper.concat_signal
            Customise the filters and format to fetch signals.

        Returns
        -------
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            DataFrame or list of DataFrames

        Examples
        --------
        FIXME: Add docs.
        """
        if isinstance(signals, str):
            signals = [signals]

        sigs = {s: [] for s in signals}
        for s in signals:
            for grp in self.groupers:
                try:
                    sigset = grp.concat_signal(s, **kwargs)
                    new_idx = pd.MultiIndex.from_tuples(
                        [(grp.name, *x) for x in sigset.index],
                        names=("experiment", *sigset.index.names),
                    )

                    sigset.index = new_idx
                    sigs[s].append(sigset)
                except Exception as e:
                    print("Grouper {} failed: {}".format(grp.name, e))
                    # raise (e)

        concated = {
            name: pd.concat(multiexp_sig)
            for name, multiexp_sig in sigs.items()
        }
        if len(concated) == 1:
            concated = list(concated.values())[0]

        return concated
