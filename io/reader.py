#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np

from agora.io.bridge import groupsort
from agora.io.writer import load_attributes


class DynamicReader:
    group = ""

    def __init__(self, file: str):
        self.file = file
        self.metadata = load_attributes(file)


class StateReader(DynamicReader):
    """
    Analogous to StateWriter:


    Possible cases (and data shapes):
    - max_lbl  (ntraps, 1) -> One int per trap.
    - tp_back, trap, cell_label -> One int per cell_label-timepoint
    - prev_feats -> A fixed number of floats per cell_label-timepoint (default is 9)
    - lifetime, p_was_bud, p_is_mother -> (nTotalCells, 2)  A (Ncells, 2) matrix where the first column is the trap,
        and its index for such trap (+1) is its cell label.
    - ba_cum ->.  (2^n, 2^n, None) 3d array where the lineage score is contained for all traps - traps in the 3rd dimension 3d array where the lineage score is contained for all traps - traps in the 3rd dimension.
        2^n >= ncells, it is kept in powers of two for efficiency.

    """

    data_types = {}
    datatypes = {
        "max_lbl": ((None, 1), np.uint16),
        "tp_back": ((None, 1), np.uint16),
        "trap": ((None, 1), np.int16),
        "cell_lbls": ((None, 1), np.uint16),
        "prev_feats": ((None, None), np.float64),
        "lifetime": ((None, 2), np.uint16),
        "p_was_bud": ((None, 2), np.float64),
        "p_is_mother": ((None, 2), np.float64),
        "ba_cum": ((None, None), np.float64),
    }
    group = "last_state"

    def __init__(self, file: str):
        super().__init__(file)

    def format_tps(self):
        pass

    def format_traps(self):
        pass

    def format_bacum(self):
        pass

    def read_raw(self, key, dtype):
        with h5py.File(self.file, "r") as f:
            raw = f[self.group + "/" + key][()].astype(dtype)

        return raw

    def read_all(self):

        self.raw_data = {
            key: self.read_raw(key, dtype)
            for key, (_, dtype) in self.datatypes.items()
        }

        return self.raw_data

    def reconstruct_states(self, data: dict):
        ntps_back = max(data["tp_back"]) + 1

        from copy import copy

        tpback_as_idx = copy(data["tp_back"])
        trap_as_idx = copy(data["trap"])

        states = {k: {"max_lbl": v} for k, v in enumerate(data["max_lbl"])}
        for val_name in ("cell_lbls", "prev_feats"):
            for k in states.keys():
                if val_name == "cell_lbls":
                    states[k][val_name] = [[] for _ in range(ntps_back)]
                else:
                    states[k][val_name] = [
                        np.zeros(
                            (0, data[val_name].shape[1]), dtype=np.float64
                        )
                        for _ in range(ntps_back)
                    ]

            data[val_name] = list(
                zip(trap_as_idx, tpback_as_idx, data[val_name])
            )
            for k, v in groupsort(data[val_name]).items():
                states[k][val_name] = [
                    np.array([w[0] for w in val])
                    for val in groupsort(v).values()
                ]

        for val_name in ("lifetime", "p_was_bud", "p_is_mother"):
            for k in states.keys():
                states[k][val_name] = np.array([])
            # This contains no time points back
            for k, v in groupsort(data[val_name]).items():
                states[k][val_name] = np.array([val[0] for val in v])

        for trap_id, ba_matrix in enumerate(data["ba_cum"]):
            states[trap_id]["ba_cum"] = np.array(ba_matrix, dtype=np.float64)

        return [val for val in states.values()]

    def get_formatted_states(self):
        return self.reconstruct_states(self.read_all())
