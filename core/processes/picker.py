from typing import Tuple, Union, List
from abc import ABC, abstractmethod

from itertools import groupby
from utils_find_1st import find_1st, cmp_equal
import numpy as np
import pandas as pd

from core.cells import CellsHDF

from agora.base import ParametersABC, ProcessABC
from postprocessor.core.functions.tracks import max_ntps, max_nonstop_ntps


class pickerParameters(ParametersABC):
    def __init__(
        self,
        condition: Tuple[str, Union[float, int]] = None,
        lineage: str = None,
        lineage_conditional: str = None,
        sequence: List[str] = ["lineage", "condition"],
    ):
        self.condition = condition
        self.lineage = lineage
        self.lineage_conditional = lineage_conditional
        self.sequence = sequence

    @classmethod
    def default(cls):
        return cls.from_dict(
            {
                "condition": ["present", 0.8],
                "lineage": "families",
                "lineage_conditional": "include",
                "sequence": ["condition", "lineage"],
            }
        )


class picker(ProcessABC):
    """
    :cells: Cell object passed to the constructor
    :condition: Tuple with condition and associated parameter(s), conditions can be
    "present", "nonstoply_present" or "quantile".
    Determines the thersholds or fractions of signals/signals to use.
    :lineage: str {"mothers", "daughters", "families" (mothers AND daughters), "orphans"}. Mothers/daughters picks cells with those tags, families pick the union of both and orphans the difference between the total and families.
    """

    def __init__(
        self,
        parameters: pickerParameters,
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
                    mb_matrix[c + d, c + m - 1] = True

            c += len(cells)

        return mb_matrix

    @staticmethod
    def mother_assign_from_dynamic(ma, label, trap, ntraps: int):
        """
        Interpolate the list of lists containing the associated mothers from the mother_assign_dynamic feature
        """
        idlist = list(zip(trap, label))
        cell_gid = np.unique(idlist, axis=0)

        last_lin_preds = [
            find_1st(((label[::-1] == lbl) & (trap[::-1] == tr)), True, cmp_equal)
            for tr, lbl in cell_gid
        ]
        mother_assign_sorted = ma[last_lin_preds]

        traps = cell_gid[:, 0]
        iterator = groupby(zip(traps, mother_assign_sorted), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        nested_massign = [d.get(i, []) for i in range(ntraps)]

        return nested_massign

    def pick_by_lineage(self, signals):
        idx = signals.index

        if self.lineage:
            ma = self._cells["mother_assign_dynamic"]
            trap = self._cells["trap"]
            label = self._cells["cell_label"]
            nested_massign = self.mother_assign_from_dynamic(
                ma, label, trap, self._cells.ntraps
            )
            # mother_bud_mat = self.mother_assign_to_mb_matrix(nested_massign)

            idx = set(
                [
                    (tid, i + 1)
                    for tid, x in enumerate(nested_massign)
                    for i in range(len(x))
                ]
            )
            mothers, daughters = zip(
                *[
                    ((tid, m), (tid, d))
                    for tid, trapcells in enumerate(nested_massign)
                    for d, m in enumerate(trapcells, 1)
                    if m
                ]
            )
            self.mothers = mothers
            self.daughters = daughters

            mothers = set(mothers)
            daughters = set(daughters)
            # daughters, mothers = np.where(mother_bud_mat)
            if self.lineage == "mothers":
                idx = mothers
            elif self.lineage == "daughters":
                idx = daughters
            elif self.lineage == "families" or self.lineage == "orphans":
                families = mothers.union(daughters)
                if self.lineage == "families":
                    idx = families
                elif self.lineage == "orphans":  # orphans
                    idx = idx.diference(families)

            idx = idx.intersection(signals.index)

        return idx

    def pick_by_condition(self, signals):
        idx = self.switch_case(self.condition[0], signals, self.condition[1])
        return idx

    def run(self, signals):
        indices = set(signals.index)
        daughters, mothers = (None, None)
        for alg in self.sequence:
            indices = getattr(self, "pick_by_" + alg)(signals)

        daughters, mothers = self.daughters, self.mothers
        return np.array(daughters), np.array(mothers), np.array(list(indices))

    @staticmethod
    def switch_case(
        condition: str,
        signals: pd.DataFrame,
        threshold: Union[float, int],
    ):
        threshold_asint = _as_int(threshold, signals.shape[1])
        case_mgr = {
            "present": signals.notna().sum(axis=1) > threshold_asint,
            "nonstoply_present": signals.apply(max_nonstop_ntps, axis=1)
            > threshold_asint,
            "quantile": [np.quantile(signals.values[signals.notna()], threshold)],
        }
        return set(case_mgr[condition].index)


def _as_int(threshold: Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold
