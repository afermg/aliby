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
    """
    A dictionary specifying the sequence of picks in order.

    "lineage" is further specified by "mothers", "daughters", "families" (mother-bud pairs), and "orphans", where orphans picks cells that are not in families.

    "condition" is further specified by "present", "continuously_present", "any_present", or "growing" and a threshold, either a number of time points or a fraction of the total duration of the experiment.
    """

    _defaults = {
        "sequence": [
            ["lineage", "families"],
            ["condition", "present", 7],
        ],
    }


class Picker(LineageProcess):
    """
    Picker selects cells from a signal using lineage information and by how long and by how they are retained in the data set.
    """

    def __init__(
        self,
        parameters: PickerParameters,
        cells: Cells or None = None,
    ):
        """Initialise picker."""
        super().__init__(parameters=parameters)
        self.cells = cells

    def pick_by_lineage(
        self,
        signal: pd.DataFrame,
        how: str,
        mothers_daughters: t.Optional[np.ndarray] = None,
    ) -> pd.MultiIndex:
        """Return rows of a signal corresponding to either mothers, daughters, or mother-daughter pairs using lineage information."""
        cells_present = drop_mother_label(signal.index)
        mothers_daughters = self.get_lineage_information(signal)
        #: might be better if match_column defined as a string to make everything one line
        if how == "mothers":
            _, valid_indices = validate_association(
                mothers_daughters, cells_present, match_column=0
            )
        elif how == "daughters":
            _, valid_indices = validate_association(
                mothers_daughters, cells_present, match_column=1
            )
        elif how == "families":
            # mother-daughter pairs
            _, valid_indices = validate_association(
                mothers_daughters, cells_present
            )
        else:
            valid_indices = slice(None)
        return signal.index[valid_indices]

    def run(self, signal):
        """
        Pick indices from the index of a signal's dataframe and return as an array.

        Typically, we first pick by lineage, then by condition.
        """
        self.orig_signal = signal
        indices = set(signal.index)
        lineage = self.get_lineage_information(signal)
        if len(lineage):
            self.mothers = lineage[:, [0, 1]]
            self.daughters = lineage[:, [0, 2]]
            for alg, *params in self.sequence:
                if indices:
                    if alg == "lineage":
                        # pick mothers, buds, or mother-bud pairs
                        param1 = params[0]
                        new_indices = getattr(self, "pick_by_" + alg)(
                            signal.loc[list(indices)], param1
                        )
                    else:
                        # pick by condition
                        param1, *param2 = params
                        new_indices = getattr(self, "pick_by_" + alg)(
                            signal.loc[list(indices)], param1, param2
                        )
                else:
                    new_indices = tuple()
                # number of indices reduces for each iteration of the loop
                indices = indices.intersection(new_indices)
        else:
            self._log("No lineage assignment")
            indices = np.array([])
        # convert to array
        indices_arr = np.array([tuple(map(_str_to_int, x)) for x in indices])
        return indices_arr

    # def pick_by_condition(self, signal, condition, thresh):
    #     idx = self.switch_case(signal, condition, thresh)
    #     return idx

    def pick_by_condition(
        self,
        signal: pd.DataFrame,
        condition: str,
        threshold: t.Union[float, int, list],
    ):
        """Pick indices from signal by any_present, present, continuously_present, and growing."""
        if len(threshold) == 1:
            threshold = [_as_int(*threshold, signal.shape[1])]
            #: is this correct for "growing"?
        case_mgr = {
            "any_present": lambda s, thresh: any_present(s, thresh),
            "present": lambda s, thresh: s.notna().sum(axis=1) > thresh,
            #: continuously_present looks incorrect
            "continuously_present": lambda s, thresh: s.apply(thresh, axis=1)
            > thresh,
            "growing": lambda s, thresh: s.diff(axis=1).sum(axis=1) > thresh,
        }
        # apply condition
        idx = set(signal.index[case_mgr[condition](signal, *threshold)])
        new_indices = [tuple(x) for x in idx]
        return new_indices


def _as_int(threshold: t.Union[float, int], ntps: int):
    """Convert a fraction of the total experiment duration into a number of time points."""
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold


def any_present(signal, threshold):
    """Return pd.Series for cells where True indicates that cell was present for more than threshold time points."""
    #: isn't full_traps all we need?
    full_traps = (signal.notna().sum(axis=1) > threshold).groupby("trap")
    any_present = pd.Series(
        np.sum(
            [
                np.isin([x[0] for x in signal.index], i) & full
                for i, full in full_traps.any().items()
            ],
            axis=0,
        ).astype(bool),
        index=signal.index,
    )
    return any_present
