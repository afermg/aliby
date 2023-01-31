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
from pathos.multiprocessing import Pool

from postprocessor.chainer import Chainer


class Grouper(ABC):
    """Base grouper class."""

    def __init__(self, dir: Union[str, PosixPath]):
        """Find h5 files and load a chain for each one."""
        path = Path(dir)
        assert path.exists(), "Dir does not exist"
        self.name = path.name
        self.files = list(path.glob("*.h5"))
        assert len(self.files), "No valid h5 files in dir"
        self.load_chains()

    def load_chains(self) -> None:
        """Load a chain for each position, or h5 file."""
        self.chainers = {f.name[:-3]: Chainer(f) for f in self.files}

    @property
    def fsignal(self) -> Chainer:
        """Get first chain."""
        return list(self.chainers.values())[0]

    @property
    def ntimepoints(self) -> int:
        """Find number of time points."""
        return max([s.ntimepoints for s in self.chainers.values()])

    @property
    def tintervals(self) -> float:
        """Find the maximum time interval for all chains."""
        tintervals = set([s.tinterval / 60 for s in self.chainers.values()])
        assert (
            len(tintervals) == 1
        ), "Not all chains have the same time interval"
        return max(tintervals)

    @property
    def available(self) -> t.Collection[str]:
        """Generate list of available signals in the first chain."""
        return self.fsignal.available

    @property
    def available_grouped(self) -> None:
        """Display available signals and the number of chains for each."""
        if not hasattr(self, "_available_grouped"):
            self._available_grouped = Counter(
                [x for s in self.chainers.values() for x in s.available]
            )
        for s, n in self._available_grouped.items():
            print(f"{s} - {n}")

    @property
    def datasets(self) -> None:
        """Print available data sets in the first chain."""
        return self.fsignal.datasets

    @abstractproperty
    def positions_groups(self):
        pass

    def concat_signal(
        self,
        path: str,
        pool: t.Optional[int] = None,
        mode: str = "retained",
        standard: t.Optional[bool] = False,
        **kwargs,
    ):
        """
        Concatenate data for one signal from different h5 files, with
        one h5 file per position, into a dataframe.

        Parameters
        ----------
        path : str
           Signal location within h5py file
        pool : int
           Number of threads used; if 0 or None only one core is used
        mode: str
        standard: boolean
        **kwargs : key, value pairings
           Named arguments for concat_ind_function

        Examples
        --------
        >>> record = grouper.concat_signal("extraction/GFP/max/median")
        """
        if path.startswith("/"):
            path = path.strip("/")
        good_chains = self.filter_chains(path)
        if standard:
            fn_pos = concat_standard
        else:
            fn_pos = concat_signal_ind
            kwargs["mode"] = mode
        records = self.pool_function(
            path=path,
            f=fn_pos,
            pool=pool,
            chainers=good_chains,
            **kwargs,
        )
        # check for errors
        errors = [
            k for kymo, k in zip(records, self.chainers.keys()) if kymo is None
        ]
        records = [record for record in records if record is not None]
        if len(errors):
            print("Warning: Positions contain errors {errors}")
        assert len(records), "All data sets contain errors"
        # combine into one dataframe
        concat = pd.concat(records, axis=0)
        if len(concat.index.names) > 4:
            # reorder levels in the multi-index dataframe when mother_label is present
            concat = concat.reorder_levels(
                ("group", "position", "trap", "cell_label", "mother_label")
            )
        concat_sorted = concat.sort_index()
        return concat_sorted

    def filter_chains(self, path: str) -> t.Dict[str, Chainer]:
        """Filter chains to those whose data is available in the h5 file."""
        good_chains = {
            k: v
            for k, v in self.chainers.items()
            if path in [*v.available, *v.common_chains]
        }
        nchains_dif = len(self.chainers) - len(good_chains)
        if nchains_dif:
            print(
                f"Grouper:Warning: {nchains_dif} chains do not contain"
                f" channel {path}"
            )
        assert len(
            good_chains
        ), f"No valid dataset to use. Valid datasets are {self.available}"
        return good_chains

    def pool_function(
        self,
        path: str,
        f: t.Callable,
        pool: t.Optional[int] = None,
        chainers: t.Dict[str, Chainer] = None,
        **kwargs,
    ):
        """Enable different threads for independent chains, particularly useful when aggregating multiple elements."""
        if pool is None:
            pass
        chainers = chainers or self.chainers
        if pool:
            with Pool(pool) as p:
                records = p.map(
                    lambda x: f(
                        path=path,
                        chainer=x[1],
                        group=self.positions_groups[x[0]],
                        position=x[0],
                        **kwargs,
                    ),
                    chainers.items(),
                )
        else:
            records = [
                f(
                    path=path,
                    chainer=chainer,
                    group=self.positions_groups[name],
                    position=name,
                    **kwargs,
                )
                for name, chainer in self.chainers.items()
            ]
        return records

    @property
    def nmembers(self) -> t.Dict[str, int]:
        """Get the number of positions belonging to each group."""
        return Counter(self.positions_groups.values())

    @property
    def ntiles(self):
        """Get total number of tiles per position (h5 file)."""
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                print(pos, f["/trap_info/trap_locations"].shape[0])

    @property
    def ntiles_by_group(self) -> t.Dict[str, int]:
        """Get total number of tiles per group."""
        ntiles = {}
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                ntiles[pos] = f["/trap_info/trap_locations"].shape[0]
        ntiles_by_group = {k: 0 for k in self.groups}
        for posname, vals in ntiles.items():
            ntiles_by_group[self.positions_groups[posname]] += vals
        return ntiles_by_group

    @property
    def tilelocs(self) -> t.Dict[str, np.ndarray]:
        """Get the locations of the tiles for each position as a dictionary."""
        d = {}
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                d[pos] = f["/trap_info/trap_locations"][()]
        return d

    @property
    def groups(self) -> t.Tuple[str]:
        """Get groups, sorted alphabetically."""
        return tuple(sorted(set(self.positions_groups.values())))

    @property
    def positions(self) -> t.Tuple[str]:
        """Get positions, sorted alphabetically."""
        return tuple(sorted(set(self.positions_groups.keys())))

    def ncells(
        self,
        path="extraction/general/None/area",
        mode="retained",
        **kwargs,
    ) -> t.Dict[str, int]:
        """Get number of cells retained per position in base channel as a dictionary."""
        return (
            self.concat_signal(path=path, mode=mode, **kwargs)
            .groupby("group")
            .apply(len)
            .to_dict()
        )

    @property
    def nretained(self) -> t.Dict[str, int]:
        """Get number of cells retained per position in base channel as a dictionary."""
        return self.ncells()

    @property
    def channels(self):
        """Get unique channels for all chains as a set."""
        return set(
            [
                channel
                for chainer in self.chainers.values()
                for channel in chainer.channels
            ]
        )

    @property
    def stages_span(self):
        # FAILS on my example
        return self.fsignal.stages_span

    @property
    def max_span(self):
        # FAILS on my example
        return self.fsignal.max_span

    @property
    def stages(self):
        # FAILS on my example
        return self.fsignal.stages

    @property
    def tinterval(self):
        """Get interval between time points."""
        return self.fsignal.tinterval


class MetaGrouper(Grouper):
    """Group positions using metadata's 'group' number."""

    pass


class NameGrouper(Grouper):
    """Group a set of positions with a shorter version of the group's name."""

    def __init__(self, dir, name_inds=(0, -4)):
        """Define the indices to slice names."""
        super().__init__(dir=dir)
        self.name_inds = name_inds

    @property
    def positions_groups(self) -> t.Dict[str, str]:
        """Get a dictionary with the positions as keys and groups as items."""
        if not hasattr(self, "_positions_groups"):
            self._positions_groups = {}
            for name in self.chainers.keys():
                self._positions_groups[name] = name[
                    self.name_inds[0] : self.name_inds[1]
                ]
        return self._positions_groups


class phGrouper(NameGrouper):
    """Grouper for pH calibration experiments where all surveyed media pH values are within a single experiment."""

    def __init__(self, dir, name_inds=(3, 7)):
        """Initialise via NameGrouper."""
        super().__init__(dir=dir, name_inds=name_inds)

    def get_ph(self) -> None:
        """Find the pH from the group names and store as a dictionary."""
        self.ph = {gn: self.ph_from_group(gn) for gn in self.positions_groups}

    @staticmethod
    def ph_from_group(group_name: str) -> float:
        """Find the pH from the name of a group."""
        if group_name.startswith("ph_") or group_name.startswith("pH_"):
            group_name = group_name[3:]
        return float(group_name.replace("_", "."))

    def aggregate_multichains(self, signals: list) -> pd.DataFrame:
        """Get data from a list of signals and combine into one multi-index dataframe with 'media-pH' included."""
        aggregated = pd.concat(
            [
                self.concat_signal(signal, reduce_cols=np.nanmean)
                for signal in signals
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


# Alan: why are these separate functions?
def concat_standard(
    path: str,
    chainer: Chainer,
    group: str,
    position: t.Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    combined = chainer.get(path, **kwargs).copy()
    combined["position"] = position
    combined["group"] = group
    combined.set_index(["group", "position"], inplace=True, append=True)
    combined.index = combined.index.reorder_levels(
        ("group", "position", "trap", "cell_label", "mother_label")
    )
    return combined


# why _ind ?
def concat_signal_ind(
    path: str,
    chainer: Chainer,
    group: str,
    mode: str = "retained",
    position=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve an individual signal.

    Applies filtering if requested and adjusts indices.
    """
    if position is None:
        # name of h5 file
        position = chainer.stem
    if mode == "retained":
        combined = chainer.retained(path, **kwargs)
    elif mode == "raw":
        combined = chainer.get_raw(path, **kwargs)
    elif mode == "daughters":
        combined = chainer.get_raw(path, **kwargs)
        combined = combined.loc[
            combined.index.get_level_values("mother_label") > 0
        ]
    elif mode == "families":
        combined = chainer[path]
    else:
        raise Exception(f"{mode} not recognised.")
    if combined is not None:
        # adjust indices
        combined["position"] = position
        combined["group"] = group
        combined.set_index(["group", "position"], inplace=True, append=True)
        combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)
    # should there be an error message if None is returned?
    return combined


class MultiGrouper:
    """Wrap results from multiple experiments stored as folders inside a
    folder."""

    def __init__(self, source: Union[str, list]):
        """
        Create NameGroupers for each experiment.

        Parameters
        ----------
        source: list of str
            List of folders, one per experiment, containing h5 files.
        """
        if isinstance(source, str):
            source = Path(source)
            self.exp_dirs = list(source.glob("*"))
        else:
            self.exp_dirs = [Path(x) for x in source]
        self.groupers = [NameGrouper(d) for d in self.exp_dirs]
        for group in self.groupers:
            group.load_chains()

    @property
    def available(self) -> None:
        """Print available signals and number of chains, one per position, for each Grouper."""
        for gpr in self.groupers:
            print(gpr.available_grouped)

    @property
    def sigtable(self) -> pd.DataFrame:
        """Generate a table showing the number of positions, or h5 files, available for each signal with one column per experiment."""

        def regex_cleanup(x):
            x = re.sub(r"extraction\/", "", x)
            x = re.sub(r"postprocessing\/", "", x)
            x = re.sub(r"\/max", "", x)
            return x

        if not hasattr(self, "_sigtable"):
            raw_mat = [
                [s.available for s in gpr.chainers.values()]
                for gpr in self.groupers
            ]
            available_grouped = [
                Counter([x for y in grp for x in y]) for grp in raw_mat
            ]
            nexps = len(available_grouped)
            sigs_idx = list(
                set([y for x in available_grouped for y in x.keys()])
            )
            sigs_idx = [regex_cleanup(x) for x in sigs_idx]
            nsigs = len(sigs_idx)
            sig_matrix = np.zeros((nsigs, nexps))
            for i, c in enumerate(available_grouped):
                for k, v in c.items():
                    sig_matrix[sigs_idx.index(regex_cleanup(k)), i] = v
            sig_matrix[sig_matrix == 0] = np.nan
            self._sigtable = pd.DataFrame(
                sig_matrix,
                index=sigs_idx,
                columns=[x.name for x in self.exp_dirs],
            )
        return self._sigtable

    # Alan: function seems out of place
    # seaborn is not in pyproject.toml
    def sigtable_plot(self) -> None:
        """
        Plot number of chains for all available experiments.

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
        path: Union[str, list],
        **kwargs,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Aggregate chains, one per position, from multiple Groupers, one per experiment.

        Parameters
        ----------
        path : Union[str, list]
            String or list of strings indicating the signal(s) to fetch.
        **kwargs :
            Passed to Grouper.concat_signal.

        Returns
        -------
        concatenated: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            A multi-index dataFrame or a dictionary of multi-index dataframes, one per signal

        Examples
        --------
        >>> mg = MultiGrouper(["pHCalibrate7_24", "pHCalibrate6_7"])
        >>> p405 = mg.aggregate_signal("extraction/pHluorin405_0_4/max/median")
        >>> p588 = mg.aggregate_signal("extraction/pHluorin488_0_4/max/median")
        >>> ratio = p405 / p488
        """
        if isinstance(path, str):
            path = [path]
        sigs = {s: [] for s in path}
        for s in path:
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
        concatenated = {
            name: pd.concat(multiexp_sig)
            for name, multiexp_sig in sigs.items()
        }
        if len(concatenated) == 1:
            concatenated = list(concatenated.values())[0]
        return concatenated
