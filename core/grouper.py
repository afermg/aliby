#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from core.io.signal import Signal

# fname = "/shared_libs/pipeline-core/data/2021_04_19_pH_calibration_dual_phl__ura8__by4741_Alan4_00"
fname = "/shared_libs/pydask/pipeline-core/data/2021_08_21_KCl_pH_00/"


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

    def concat_signal(self, path, reduce_cols=None, axis=0):
        signals = []
        for group, signal in self.signals.items():
            print("looking at", signal.filename)
            combined = signal[path]
            combined["position"] = group
            combined["group"] = self.group_names[group]
            combined.set_index(["group", "position"], inplace=True, append=True)
            combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)
            signals.append(combined)

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

    def aggregate_multisignals(self, paths=None):

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
        self.aggregated = pd.concat((aggregated, ph), axis=1)

        return self.aggregated


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


# g = NameGrouper(fname)
# signame = "/extraction/em_ratio/np_max/mean"
# shortname = "_".join((signame.split("/")[2], signame.split("/")[4]))
# c = g.concat_signal(signame)
# d = c[c.notna().sum(axis=1) > c.shape[1] * 0.8]
# e = d.melt(var_name="tp", ignore_index=False, value_name=shortname).reset_index()
# e[shortname] = 1 / e[shortname]

# Plot comparable to Ivan's
# sns.lineplot(
#     data=e,
#     x="tp",
#     y=shortname,
#     hue="group",
#     palette=["blue", "orange", "yellow", "purple", "green"],
# )
# plt.title(signame)
# plt.ylabel(shortname)
# plt.show()

# Check if traplocs make sense
# for traplocs in tlocs.values():
#     x, y = list(zip(*traplocs))
#     plt.scatter(x, y)
#     plt.show()
