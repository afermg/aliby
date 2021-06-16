from typing import Tuple, Union, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from core.cells import CellsHDF

from postprocessor.core.processes.base import ParametersABC, ProcessABC
from postprocessor.core.functions.tracks import max_ntps, max_nonstop_ntps


class PickerParameters(ParametersABC):
    def __init__(
        self,
        condition: Tuple[str, Union[float, int]] = None,
        lineage: str = None,
        sequence: List[str] = ["lineage", "condition"],
    ):
        self.condition = condition
        self.lineage = lineage
        self.sequence = sequence

    @classmethod
    def default(cls):
        return cls.from_dict(
            {
                "condition": ["present", 0.8],
                "lineage": None,
                "sequence": ["lineage", "condition"],
            }
        )


class Picker(ProcessABC):
    """
    :cells: Cell object passed to the constructor
    :condition: Tuple with condition and associated parameter(s), conditions can be
    "present", "nonstoply_present" or "quantile".
    Determines the thersholds or fractions of signals/signals to use.
    :lineage: str {"mothers", "daughters", "families" (mothers AND daughters), "orphans"}. Mothers/daughters picks cells with those tags, families pick the union of both and orphans the difference between the total and families.
    """

    def __init__(
        self,
        parameters: PickerParameters,
        cells: CellsHDF,
    ):
        super().__init__(parameters=parameters)

        self._cells = cells

    @staticmethod
    def mother_assign_to_mb_matrix(ma: List[np.array]):
        # Convert from list of lists to mother_bud sparse matrix
        ncells = sum([len(t) for t in ma])
        mb_matrix = np.zeros((ncells, ncells), dtype=bool)
        c = 0
        for cells in ma:
            for d, m in enumerate(cells):
                if m:
                    mb_matrix[c + d, c + m] = True

            c += len(cells)

        return mb_matrix

    def pick_by_lineage(self, signals):
        idx = signals.index

        if self.lineage:
            ma = self._cells["mother_assign"]
            mother_bud_mat = self.mother_assign_to_mb_matrix(ma)
            daughters, mothers = np.where(mother_bud_mat)
            if self.lineage == "mothers":
                idx = idx[mothers]
            elif self.lineage == "daughters":
                idx = idx[daughters]
            elif self.lineage == "families" or self.lineage == "orphans":
                families = list(set(np.append(daughters, mothers)))
                if self.lineage == "families":
                    idx = idx[families]
                else:  # orphans
                    idx = idx[list(set(range(len(idx))).difference(families))]

            idx = list(set(idx).intersection(signals.index))

        return signals.loc[idx]

    def pick_by_condition(self, signals):
        idx = self.switch_case(self.condition[0], signals, self.condition[1])
        return signals.loc[idx]

    def run(self, signals):
        for alg in self.sequence:
            self.signals = getattr(self, "pick_by_" + alg)(signals)
        return self.signals

    @staticmethod
    def switch_case(
        condition: str,
        signals: pd.DataFrame,
        threshold: Union[float, int],
    ):
        threshold_asint = _as_int(threshold, signals.shape[1])
        case_mgr = {
            "present": signals.apply(max_ntps, axis=1) > threshold_asint,
            "nonstoply_present": signals.apply(max_nonstop_ntps, axis=1)
            > threshold_asint,
            "quantile": [np.quantile(signals.values[signals.notna()], threshold)],
        }
        return case_mgr[condition]


def _as_int(threshold: Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = threshold / ntps
    return threshold
