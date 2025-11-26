"""Define picker class to select cells, particularly those in lineages."""

import typing as t

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from agora.io.cells import Cells
from agora.utils.indexing import validate_lineage
from agora.utils.multiindex_utils import drop_mother_label
from postprocessor.core.lineageprocess import LineageProcess


class PickerParameters(ParametersABC):
    """
    A dictionary specifying the sequence of picks in order.

    "lineage" is further specified by "mothers", "daughters", and
    "families" (mother-bud pairs).

    "condition" is further specified by "present", "any_present", or
    "growing" and a threshold, either a number of time points or a
    fraction of the total duration of the experiment.

    Note that to pick most cells, particularly for short movies, use
        {"picker_sequence": [["condition", "present", 3]]}
    with no "lineage" condition specified. For short movies, lineage
    information is only available for a small fraction of the
    segmented cells.
    """

    _defaults = {
        "picker_sequence": [
            ["lineage", "families"],
            ["condition", "present", 3],
        ],
    }


class Picker(LineageProcess):
    """
    Picker selects cells.

    Cells are selected using lineage information and by how
    and for how long they are retained in the data set.
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
        """
        Return rows of a signal using lineage information.

        Rows correspond to either mothers, daughters, or mother-daughter
        pairs.
        """
        cells_present = drop_mother_label(signal.index)
        mothers_daughters = self.get_lineage_information(signal)
        _, valid_indices, mothers_daughters = validate_lineage(
            mothers_daughters, cells_present, how
        )
        return signal.index[valid_indices]

    def run(self, signal):
        """
        Pick indices from the index of a signal's dataframe.

        Typically, we first pick by lineage, then by condition.
        The indices are returned as a list for use in pd.loc.
        Converting indices that are a mixture of strings and integers
         into arrays changes integers into strings, breaking the
         index convention for Signals.
        """
        self.orig_signal = signal
        indices = set(signal.index)
        lineage = self.get_lineage_information(signal)
        if len(lineage):
            self.mothers = lineage[:, [0, 1]]
            self.daughters = lineage[:, [0, 2]]
            for method, *params in self.picker_sequence:
                if indices:
                    if method == "lineage":
                        # pick by lineage
                        param1 = params[0]
                        new_indices = self.pick_by_lineage(
                            signal.loc[list(indices)], param1
                        )
                    else:
                        # pick by condition
                        param1, *param2 = params
                        new_indices = self.pick_by_condition(
                            signal.loc[list(indices)], param1, param2
                        )
                else:
                    new_indices = tuple()
                # number of indices reduces for each iteration of the loop
                indices = indices.intersection(new_indices)
        else:
            self.log("No lineage assignment")
            indices = np.array([])
        # return as list
        indices_arr = [tuple(x) for x in indices]
        return indices_arr

    def pick_by_condition(
        self,
        signal: pd.DataFrame,
        condition: str,
        threshold: t.Union[float, int, list],
    ):
        """Pick indices from signal by any_present, present, and growing."""
        if len(threshold) == 1:
            threshold = [as_int(*threshold, signal.shape[1])]
            #: is this correct for "growing"?
        case_mgr = {
            "any_present": lambda s, threshold: any_present(s, threshold),
            "present": lambda s, threshold: s.notna().sum(axis=1) > threshold,
            "growing": lambda s, threshold: s.diff(axis=1).sum(axis=1)
            > threshold,
        }
        # apply condition
        idx = set(signal.index[case_mgr[condition](signal, *threshold)])
        new_indices = [tuple(x) for x in idx]
        return new_indices


def as_int(threshold: t.Union[float, int], ntps: int):
    """Convert a fraction of the total experiment duration into a number of time points."""
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold


def any_present(signal, threshold):
    """Find traps where at least one cell stays for more than threshold time points."""
    # all_traps contains repeated traps, which have more than one cell
    all_traps = [x[0] for x in signal.index]
    # full_traps contains only traps that have at least one cell
    full_traps = (signal.notna().sum(axis=1) > threshold).groupby("trap")
    # expand full_traps to size of signal.index
    # rows highlight traps in signal_index for each full trap
    trap_array = np.array(
        [
            np.isin(all_traps, trap_id) & full
            for trap_id, full in full_traps.any().items()
        ]
    )
    # convert to pd.Series
    any_present = pd.Series(
        np.sum(trap_array, axis=0).astype(bool), index=signal.index
    )
    return any_present


def str_to_int(x: str or None):
    """Cast string as int if possible."""
    if x is None:
        return x
    else:
        try:
            return int(x)
        except ValueError:
            return x
