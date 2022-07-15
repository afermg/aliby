"""
Class to group multiple positions into one using one different available criteria.
"""

import re
from pathlib import Path

import h5py
import pandas as pd
from agora.io.bridge import groupsort
from agora.io.signal import Signal

from postprocessor.core.abc import ParametersABC, ProcessABC


class GroupParameters(ParametersABC):
    def __init__(self, by="name", processes=[], signals=[]):
        self.by = by
        self.signals = signals
        self.processes = processes

    @classmethod
    def default(cls):
        return cls.from_dict({"by": "name", "signals": [], "processes": []})


class Group(ProcessABC):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_position_filenames(self, exp_root, poses):
        """
        Get filenames as a dictionary where the key is the position and value the filename.
        """
        central_store = Path(exp_root) / "store.h5"
        if central_store.exists():
            hdf = h5py.File(central_store, "r")
            self.filenames = [
                pos.attrs["filename"] for pos in hdf["/positions/"]
            ]
            hdf.close()
        else:  # If no central store just list position files in expt root folder
            fullfiles = [x for x in Path(exp_root).glob("*store.h5")]
            files = [x.name for x in Path(exp_root).glob("*store.h5")]
            filenames = [False for _ in poses]
            for i, pos in enumerate(poses):
                matches = [
                    True if re.match(pos + ".*.h5", fname) else False
                    for fname in files
                ]
                if any(matches):
                    assert sum(matches) == 1, "More than one match"
                    filenames[i] = (pos, fullfiles[matches.index(True)])

            self.filenames = {
                fname[0]: fname[1] for fname in filenames if fname
            }

        self.positions = list(self.filenames.keys())
        return self.filenames

    def get_signals(self):
        # hdf = h5py.File(central_store, "r")
        # keys_d = groupsort(keys)
        self.signals = {pos: {} for pos in self.filenames.keys()}
        for pos, fname in self.filenames.items():
            for signal in self.parameters.signals:
                self.signals[pos][signal] = pd.read_hdf(fname, signal)

        return self.signals

    def gen_groups(self):
        if self.by == "group":  # Use group names in metadata
            pass
        elif self.by == "name":  # Infer groups from signal concatenation
            # Remove last four characters and find commonalities larger than 4
            # characters between posnames and group them that way.
            groupnames = list(set([x[:-3] for x in self.positions]))
            self.group_signal_tree = {group: [] for group in groupnames}
            self.poses_grouped = {group: [] for group in groupnames}
            for pos in self.positions:
                group = groupnames[groupnames.index(pos[:-3])]
                self.group_signal_tree[group].append(self.signals[pos])
                self.poses_grouped[group].append(pos)

        elif (
            type(self.by) == tuple
        ):  # Manually give groups as tuple or list of positions
            pass

    def concat_signals(self):
        self.concated_signals = {group: {} for group in self.group_signal_tree}
        for k, group in self.group_signal_tree.items():
            for signal in self.parameters.signals:
                self.concated_signals[k][signal] = pd.concat(
                    [g[signal] for g in group], keys=self.poses_grouped[k]
                )

        return self.concated_signals

    def process_signals(self, grouped_signals):
        pass

    def run(self, central_store, poses):

        self.get_position_filenames(central_store, poses)
        self.get_signals()
        self.gen_groups()
        self.concat_signals()
        # processed_signals = self.process_signals(grouped_signals)

        return self.concated_signals
        # return processed_signals


poses = [
    x.name.split("store")[0]
    for x in Path(
        "/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01"
    ).rglob("*")
    if x.name != "images.h5"
]
gr = Group(
    GroupParameters(
        signals=[
            "/extraction/general/None/area",
            "/extraction/mCherry/np_max/median",
        ]
    )
)
gr.run(
    central_store="/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01",
    poses=poses,
)
signal = Signal(
    "/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01/ph_5_04_001store.h5"
)
