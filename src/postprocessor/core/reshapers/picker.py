from abc import ABC, abstractmethod

# from copy import copy
from itertools import groupby
from typing import List, Tuple, Union

import igraph as ig
import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from agora.io.cells import Cells
from utils_find_1st import cmp_equal, find_1st

from postprocessor.core.abc import PostProcessABC
from postprocessor.core.functions.tracks import max_nonstop_ntps, max_ntps


class pickerParameters(ParametersABC):
    _defaults = {
        "sequence": [
            ["lineage", "intersection", "families"],
            # ["condition", "intersection", "any_present", 0.7],
            # ["condition", "intersection", "growing", 80],
            ["condition", "intersection", "present", 7],
            # ["condition", "intersection", "mb_guess", 3, 0.7],
            # ("lineage", "intersection", "full_families"),
        ],
    }


class picker(PostProcessABC):
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
        cells: Cells,
    ):
        super().__init__(parameters=parameters)

        self._cells = cells

    def pick_by_lineage(self, signals, how):
        self.orig_signals = signals

        idx = signals.index

        if how:
            mothers = set(self.mothers)
            daughters = set(self.daughters)
            # daughters, mothers = np.where(mother_bud_mat)

            def search(a, b):
                return np.where(
                    np.in1d(
                        np.ravel_multi_index(
                            np.array(a).T, np.array(a).max(0) + 1
                        ),
                        np.ravel_multi_index(
                            np.array(b).T, np.array(a).max(0) + 1
                        ),
                    )
                )

            if how == "mothers":
                idx = mothers
            elif how == "daughters":
                idx = daughters
            elif how == "daughters_w_mothers":
                present_mothers = idx.intersection(mothers)
                idx = set(
                    [
                        tuple(x)
                        for m in present_mothers
                        for x in np.array(self.daughters)[
                            search(self.mothers, m)
                        ]
                    ]
                )

                print("associated daughters: ", idx)
            elif how == "mothers_w_daughters":
                present_daughters = idx.intersection(daughters)
                idx = set(
                    [
                        tuple(x)
                        for d in present_daughters
                        for x in np.array(self.mothers)[
                            search(self.daughters, d)
                        ]
                    ]
                )
            elif how == "full_families":
                present_mothers = idx.intersection(mothers)
                dwm_idx = set(
                    [
                        tuple(x)
                        for m in present_mothers
                        for x in np.array(self.daughters)[
                            search(np.array(self.mothers), m)
                        ]
                    ]
                )
                present_daughters = idx.intersection(daughters)
                mwd_idx = set(
                    [
                        tuple(x)
                        for d in present_daughters
                        for x in np.array(self.mothers)[
                            search(np.array(self.daughters), d)
                        ]
                    ]
                )
                idx = mwd_idx.union(dwm_idx)
            elif how == "families" or how == "orphans":
                families = mothers.union(daughters)
                if how == "families":
                    idx = families
                elif how == "orphans":
                    idx = idx.diference(families)

            idx = idx.intersection(signals.index)

        return idx

    def pick_by_condition(self, signals, condition, thresh):
        idx = self.switch_case(signals, condition, thresh)
        return idx

    def get_mothers_daughters(self):
        ma = self._cells["mother_assign_dynamic"]
        trap = self._cells["trap"]
        label = self._cells["cell_label"]
        nested_massign = self._cells.mother_assign_from_dynamic(
            ma, label, trap, self._cells.ntraps
        )
        # mother_bud_mat = self.mother_assign_to_mb_matrix(nested_massign)

        if sum([x for y in nested_massign for x in y]):

            mothers, daughters = zip(
                *[
                    ((tid, m), (tid, d))
                    for tid, trapcells in enumerate(nested_massign)
                    for d, m in enumerate(trapcells, 1)
                    if m
                ]
            )
        else:
            mothers, daughters = ([], [])
            print("Warning:Picker: No mother-daughters assigned")

        return mothers, daughters

    def run(self, signals):
        self.orig_signals = signals
        indices = set(signals.index)
        self.mothers, self.daughters = self.get_mothers_daughters()
        for alg, op, *params in self.sequence:
            new_indices = tuple()
            if indices:
                if alg == "lineage":
                    param1 = params[0]
                    new_indices = getattr(self, "pick_by_" + alg)(
                        signals.loc[list(indices)], param1
                    )
                else:
                    param1, *param2 = params
                    new_indices = getattr(self, "pick_by_" + alg)(
                        signals.loc[list(indices)], param1, param2
                    )

            if op == "union":
                # new_indices = new_indices.intersection(set(signals.index))
                new_indices = indices.union(new_indices)

            indices = indices.intersection(new_indices)

        return np.array(list(indices))

    def switch_case(
        self,
        signals: pd.DataFrame,
        condition: str,
        threshold: Union[float, int, list],
    ):
        if len(threshold) == 1:
            threshold = [_as_int(*threshold, signals.shape[1])]
        case_mgr = {
            "any_present": lambda s, thresh: any_present(s, thresh),
            "present": lambda s, thresh: s.notna().sum(axis=1) > thresh,
            "nonstoply_present": lambda s, thresh: s.apply(thresh, axis=1)
            > thresh,
            "growing": lambda s, thresh: s.diff(axis=1).sum(axis=1) > thresh,
        }
        return set(signals.index[case_mgr[condition](signals, *threshold)])


def _as_int(threshold: Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold


def any_present(signals, threshold):
    """
    Returns a mask for cells, True if there is a cell in that trap that was present for more than :threshold: timepoints.
    """
    any_present = pd.Series(
        np.sum(
            [
                np.isin([x[0] for x in signals.index], i) & v
                for i, v in (signals.notna().sum(axis=1) > threshold)
                .groupby("trap")
                .any()
                .items()
            ],
            axis=0,
        ).astype(bool),
        index=signals.index,
    )
    return any_present
