# from abc import ABC, abstractmethod

# from copy import copy
# from itertools import groupby
# from typing import List, Tuple, Union
import typing as t
from typing import Union

# import igraph as ig
import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from agora.io.cells import Cells

# from postprocessor.core.functions.tracks import max_nonstop_ntps, max_ntps
from agora.utils.association import validate_association
from postprocessor.core.lineageprocess import LineageProcess

# from utils_find_1st import cmp_equal, find_1st


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


class picker(LineageProcess):
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

        self.cells = cells

    def pick_by_lineage(
        self,
        signal: pd.DataFrame,
        how: str,
        mothers_daughters: t.Optional[np.ndarray] = None,
    ):
        self.orig_signals = signal

        idx = np.array(signal.index.to_list())

        if mothers_daughters is None:
            mothers_daughters = self.cells.mothers_daughters
        valid_indices, valid_lineage = [slice(None)] * 2

        if how == "mothers":
            valid_lineage, valid_indices = validate_association(
                mothers_daughters, idx, match_column=0
            )
        elif how == "daughters":
            valid_lineage, valid_indices = validate_association(
                mothers_daughters, idx, match_column=1
            )
        elif how == "families":  # Mothers and daughters that are still present
            valid_lineage, valid_indices = validate_association(
                mothers_daughters, idx
            )

        idx = idx[valid_indices]
        mothers_daughters = mothers_daughters[valid_lineage]

        return mothers_daughters, idx

    def loc_lineage(self, kymo: pd.DataFrame, how: str, lineage=None):
        _, valid_indices = self.pick_by_lineage(
            kymo, how, mothers_daughters=lineage
        )
        return kymo.loc[[tuple(x) for x in valid_indices]]

    def pick_by_condition(self, signals, condition, thresh):
        idx = self.switch_case(signals, condition, thresh)
        return idx

    def run(self, signals):
        self.orig_signals = signals
        indices = set(signals.index)
        self.mothers, self.daughters = self.cells.mothers_daughters
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
