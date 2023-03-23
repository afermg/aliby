import typing as t

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from agora.io.cells import Cells
from agora.utils.indexing import validate_association
from agora.utils.cast import _str_to_int
from agora.utils.kymograph import drop_mother_label
from postprocessor.core.lineageprocess import LineageProcess


class PickerParameters(ParametersABC):
    _defaults = {
        "sequence": [
            ["lineage", "families"],
            ["condition", "present", 7],
        ],
    }


class Picker(LineageProcess):
    """
    :cells: Cell object passed to the constructor
    :condition: Tuple with condition and associated parameter(s), conditions can be
    "present", "nonstoply_present" or "quantile".
    Determine the thresholds or fractions of signals to use.
    :lineage: str {"mothers", "daughters", "families" (mothers AND daughters), "orphans"}. Mothers/daughters picks cells with those tags, families pick the union of both and orphans the difference between the total and families.
    """

    def __init__(
        self,
        parameters: PickerParameters,
        cells: Cells or None = None,
    ):
        super().__init__(parameters=parameters)
        self.cells = cells

    def pick_by_lineage(
        self,
        signal: pd.DataFrame,
        how: str,
        mothers_daughters: t.Optional[np.ndarray] = None,
    ) -> pd.MultiIndex:
        cells_present = drop_mother_label(signal.index)
        mothers_daughters = self.get_lineage_information(signal)
        valid_indices = slice(None)
        if how == "mothers":
            _, valid_indices = validate_association(
                mothers_daughters, cells_present, match_column=0
            )
        elif how == "daughters":
            _, valid_indices = validate_association(
                mothers_daughters, cells_present, match_column=1
            )
        elif how == "families":  # Mothers and daughters that are still present
            _, valid_indices = validate_association(
                mothers_daughters, cells_present
            )
        return signal.index[valid_indices]

    def pick_by_condition(self, signal, condition, thresh):
        idx = self.switch_case(signal, condition, thresh)
        return idx

    def run(self, signal):
        self.orig_signal = signal
        indices = set(signal.index)
        lineage = self.get_lineage_information(signal)
        if len(lineage):
            self.mothers = lineage[:, :2]
            self.daughters = lineage[:, [0, 2]]
            for alg, *params in self.sequence:
                new_indices = tuple()
                if indices:
                    if alg == "lineage":
                        param1 = params[0]
                        new_indices = getattr(self, "pick_by_" + alg)(
                            signal.loc[list(indices)], param1
                        )
                    else:
                        param1, *param2 = params
                        new_indices = getattr(self, "pick_by_" + alg)(
                            signal.loc[list(indices)], param1, param2
                        )
                        new_indices = [tuple(x) for x in new_indices]
                indices = indices.intersection(new_indices)
        else:
            self._log(f"No lineage assignment")
            indices = np.array([])
        return np.array([tuple(map(_str_to_int, x)) for x in indices])

    def switch_case(
        self,
        signal: pd.DataFrame,
        condition: str,
        threshold: t.Union[float, int, list],
    ):
        if len(threshold) == 1:
            threshold = [_as_int(*threshold, signal.shape[1])]
        case_mgr = {
            "any_present": lambda s, thresh: any_present(s, thresh),
            "present": lambda s, thresh: s.notna().sum(axis=1) > thresh,
            "nonstoply_present": lambda s, thresh: s.apply(thresh, axis=1)
            > thresh,
            "growing": lambda s, thresh: s.diff(axis=1).sum(axis=1) > thresh,
        }
        return set(signal.index[case_mgr[condition](signal, *threshold)])


def _as_int(threshold: t.Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold


def any_present(signal, threshold):
    """
    Return a mask for cells, True if there is a cell in that trap that was present for more than :threshold: timepoints.
    """
    any_present = pd.Series(
        np.sum(
            [
                np.isin([x[0] for x in signal.index], i) & v
                for i, v in (signal.notna().sum(axis=1) > threshold)
                .groupby("trap")
                .any()
                .items()
            ],
            axis=0,
        ).astype(bool),
        index=signal.index,
    )
    return any_present
