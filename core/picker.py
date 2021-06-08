# from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from core.cells import CellsHDF
from postprocessor.core.functions.tracks import max_ntps, max_nonstop_ntps


# def BasePicker(ABC):
#     """
#     Base class to add mother-bud filtering support
#     """
#     def __init__(self, branch=None, lineage=None):
#         self.lineage = lineage


class Picker:
    """
    :tracks: pd.DataFrame
    :cells: Cell object passed to the constructor
    :condition: Tuple with condition and associated parameter(s), conditions can be
    "present", "nonstoply_present" or "quantile".
    Determines the thersholds or fractions of tracks/signals to use.
    :lineage: str {"mothers", "daughters", "families", "orphans"}. Mothers/daughters picks cells with those tags, families pick the union of both and orphans the difference between the total and families.
    """

    def __init__(
        self,
        tracks: pd.DataFrame,
        cells: CellsHDF,
        condition: Tuple[str, Union[float, int]] = None,
        lineage: str = None,
        sequence: List[str] = ["lineage", "condition"],
    ):
        self.tracks = tracks
        self._index = tracks.index
        self._cells = cells
        self.condition = condition
        self.lineage = lineage
        self.sequence = sequence

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

    def pick_by_lineage(self):
        idx = self.tracks.index

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

            idx = self._index[idx]
            idx = list(set(idx).intersection(self.tracks.index))

        return self.tracks.loc[idx]

    def pick_by_condition(self):
        idx = switch_case(self.condition[0], self.tracks, self.condition[1])
        return self.tracks.loc[idx]

    def run(self):
        for alg in self.sequence:
            self.tracks = getattr(self, "pick_by" + alg)()
        return self.tracks


def as_int(threshold: Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = threshold / ntps
    return threshold


def switch_case(
    condition: str,
    tracks: pd.DataFrame,
    threshold: Union[float, int],
):
    threshold_asint = as_int(threshold, tracks.shape[1])
    case_mgr = {
        "present": tracks.apply(max_ntps, axis=1) > threshold_asint,
        "nonstoply_present": tracks.apply(max_nonstop_ntps, axis=1) > threshold_asint,
        "quantile": [np.quantile(tracks.values[tracks.notna()], threshold)],
    }
    return case_mgr[condition]
