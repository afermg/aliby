#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from pathos.multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

from aliby.io.signal import Signal


class Grouper(ABC):
    """
    Base grouper class
    """

    files = []

    def __init__(self, dir):
        self.files = list(Path(dir).glob("*.h5"))
        self.load_signals()

    def load_signals(self):
        self.signals = {f.name[:-3]: Signal(f) for f in self.files}

    @property
    def fsignal(self):
        return list(self.signals.values())[0]

    @property
    def siglist(self):
        return self.fsignal.datasets

    @abstractproperty
    def group_names():
        pass

    def concat_signal(self, path, reduce_cols=None, axis=0, pool=8):
        group_names = self.group_names
        sitems = self.signals.items()
        if pool:
            with Pool(pool) as p:
                signals = p.map(
                    lambda x: concat_signal_ind(path, group_names, x[0], x[1]),
                    sitems,
                )
        else:
            signals = [
                concat_signal_ind(path, group_names, name, signal)
                for name, signal in sitems
            ]

        signals = [s for s in signals if s is not None]
        sorted = pd.concat(signals, axis=axis).sort_index()
        if reduce_cols:
            sorted = sorted.apply(np.nanmean, axis=1)
            spath = path.split("/")
            sorted.name = "_".join([spath[1], spath[-1]])

        return sorted

    @property
    def ntraps(self):
        for pos, s in self.signals.items():
            with h5py.File(s.filename, "r") as f:
                print(pos, f["/trap_info/trap_locations"].shape[0])

    def traplocs(self):
        d = {}
        for pos, s in self.signals.items():
            with h5py.File(s.filename, "r") as f:
                d[pos] = f["/trap_info/trap_locations"][()]
        return d


class MetaGrouper(Grouper):
    """Group positions using metadata's 'group' number"""

    pass


class NameGrouper(Grouper):
    """
    Group a set of positions using a subsection of the name
    """

    def __init__(self, dir, by=None):
        super().__init__(dir=dir)

        if by is None:
            by = (0, -4)
        self.by = by

    @property
    def group_names(self):
        if not hasattr(self, "_group_names"):
            self._group_names = {}
            for name in self.signals.keys():
                self._group_names[name] = name[self.by[0] : self.by[1]]

        return self._group_names

    def aggregate_multisignals(self, paths=None, **kwargs):

        aggregated = pd.concat(
            [
                self.concat_signal(path, reduce_cols=np.nanmean, **kwargs)
                for path in paths
            ],
            axis=1,
        )
        # ph = pd.Series(
        #     [
        #         self.ph_from_group(x[list(aggregated.index.names).index("group")])
        #         for x in aggregated.index
        #     ],
        #     index=aggregated.index,
        #     name="media_pH",
        # )
        # self.aggregated = pd.concat((aggregated, ph), axis=1)

        return aggregated


class phGrouper(NameGrouper):
    """
    Grouper for pH calibration experiments where all surveyed media pH values
    are within a single experiment.
    """

    def __init__(self, dir, by=(3, 7)):
        super().__init__(dir=dir, by=by)

    def get_ph(self):
        self.ph = {gn: self.ph_from_group(gn) for gn in self.group_names}

    @staticmethod
    def ph_from_group(group_name):
        if group_name.startswith("ph_"):
            group_name = group_name[3:]

        return float(group_name.replace("_", "."))

    def aggregate_multisignals(self, paths):

        aggregated = pd.concat(
            [self.concat_signal(path, reduce_cols=np.nanmean) for path in paths], axis=1
        )
        ph = pd.Series(
            [
                self.ph_from_group(x[list(aggregated.index.names).index("group")])
                for x in aggregated.index
            ],
            index=aggregated.index,
            name="media_pH",
        )
        aggregated = pd.concat((aggregated, ph), axis=1)

        return aggregated


def concat_signal_ind(path, group_names, group, signal):
    print("Looking at ", group)
    # try:
    combined = signal[path]
    combined["position"] = group
    combined["group"] = group_names[group]
    combined.set_index(["group", "position"], inplace=True, append=True)
    combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)

    return combined
    # except:
    #     return None
