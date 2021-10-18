import seaborn as sns
from matplotlib import pyplot as plt  # TODO DELETE THIS

from typing import Tuple, Union, List
from abc import ABC, abstractmethod

from itertools import groupby

from utils_find_1st import find_1st, cmp_equal
import numpy as np
import pandas as pd

from pcore.cells import CellsHDF

from agora.base import ParametersABC, ProcessABC
from postprocessor.core.functions.tracks import max_ntps, max_nonstop_ntps


class pickerParameters(ParametersABC):
    def __init__(
        self,
        sequence: List[str] = ["lineage", "condition"],
    ):
        self.sequence = sequence

    @classmethod
    def default(cls):
        return cls.from_dict(
            {
                "sequence": [
                    # ("lineage", "intersection", "families"),
                    ("condition", "intersection", "any_present", 0.8),
                    ("condition", "intersection", "growing", 40),
                    ("condition", "intersection", "present", 8),
                    ("condition", "intersection", "mother_buds", 5, 0.8),
                    # ("lineage", "full_families", "intersection"),
                ],
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

    def pick_by_lineage(self, signals, how):

        idx = signals.index

        if how:
            mothers = set(self.mothers)
            daughters = set(self.daughters)
            # daughters, mothers = np.where(mother_bud_mat)

            search = lambda a, b: np.where(
                np.in1d(
                    np.ravel_multi_index(np.array(a).T, np.array(a).max(0) + 1),
                    np.ravel_multi_index(np.array(b).T, np.array(a).max(0) + 1),
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
                        for x in np.array(self.daughters)[search(self.mothers, m)]
                    ]
                )

                print("associated daughters: ", idx)
            elif how == "mothers_w_daughters":
                present_daughters = idx.intersection(daughters)
                idx = set(
                    [
                        tuple(x)
                        for d in present_daughters
                        for x in np.array(self.mothers)[search(self.daughters, d)]
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
        return mothers, daughters

    def run(self, signals):
        indices = set(signals.index)
        self.mothers, self.daughters = self.get_mothers_daughters()
        for alg, op, *params in self.sequence:
            if alg is "lineage":
                param1 = params[0]
                new_indices = getattr(self, "pick_by_" + alg)(
                    signals.loc[list(indices)], param1
                )
            else:
                param1, *param2 = params
                new_indices = getattr(self, "pick_by_" + alg)(
                    signals.loc[list(indices)], param1, param2
                )

            if op is "union":
                # new_indices = new_indices.intersection(set(signals.index))
                new_indices = indices.union(new_indices)

            indices = indices.intersection(new_indices)

        return np.array(list(indices))

    @staticmethod
    def switch_case(
        signals: pd.DataFrame,
        condition: str,
        threshold: Union[float, int, list],
    ):
        if len(threshold) == 1:
            threshold = [_as_int(*threshold, signals.shape[1])]
        case_mgr = {
            "any_present": lambda s, thresh: any_present(s, thresh),
            "present": lambda s, thresh: s.notna().sum(axis=1) > thresh,
            "nonstoply_present": lambda s, thresh: s.apply(thresh, axis=1) > thresh,
            "growing": lambda s, thresh: s.diff(axis=1).sum(axis=1) > thresh,
            "mother_buds": lambda s, p1, p2: mother_buds_wrap(s, p1, p2)
            # "quantile": [np.quantile(signals.values[signals.notna()], threshold)],
        }
        return set(signals.index[case_mgr[condition](signals, *threshold)])


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


from copy import copy


def mother_buds(df, min_budgrowth_t, min_mobud_ratio):
    """
    Parameters
    ----------
    signals : pd.DataFrame
    min_budgrowth_t: Minimal number of timepoints we lock reassignment after assigning bud
    min_initial_size: Minimal mother-bud ratio at the assignment
    #TODO incorporate bud-assignment data?

    # If more than one bud start in the same time point pick the smallest one
    """

    ntps = df.notna().sum(axis=1)
    mother_id = df.index[ntps.argmax()]
    nomother = df.drop(mother_id)
    if not len(nomother):
        return []
    nomother = (  # Clean short-lived cells outside our mother cell's timepoints
        nomother.loc[
            nomother.apply(
                lambda x: x.first_valid_index() >= df.loc[mother_id].first_valid_index()
                and x.last_valid_index() <= df.loc[mother_id].last_valid_index(),
                axis=1,
            )
        ]
    )

    start = nomother.apply(pd.Series.first_valid_index, axis=1)

    # clean duplicates
    duplicates = start.duplicated(False)
    if duplicates.any():
        dup_tps = np.unique(start[duplicates])
        idx, tps = zip(
            *[(nomother.loc[start == tp, tp].idxmin(), tp) for tp in dup_tps]
        )
        start = start[~duplicates]
        start = pd.concat(
            (start, pd.Series(tps, index=idx, dtype="int", name="cell_label"))
        )
        nomother = nomother.loc[start.index]
        nomother.index = nomother.index.astype("int")

    d_to_mother = nomother[start] - df.loc[mother_id, start] * min_mobud_ratio
    size_filter = d_to_mother[
        d_to_mother.apply(lambda x: x.dropna().iloc[0], axis=1) < 0
    ]
    cols_sorted = (
        size_filter.sort_index(axis=1)
        .apply(pd.Series.first_valid_index, axis=1)
        .sort_values()
    )
    if not len(cols_sorted):
        return []
    bud_candidates = cols_sorted.loc[
        [True, *(np.diff(cols_sorted.values) > min_budgrowth_t)]
    ]

    return [mother_id] + [int(i) for i in bud_candidates.index.tolist()]


def mother_buds_wrap(signals, *args):
    ids = []
    for trap in signals.index.unique(level="trap"):
        df = signals.loc[trap]
        selected_ids = mother_buds(df, *args)
        ids += [(trap, i) for i in selected_ids]

    idx_srs = pd.Series(False, signals.index).astype(bool)
    idx_srs.loc[ids] = True
    return idx_srs


def _as_int(threshold: Union[float, int], ntps: int):
    if type(threshold) is float:
        threshold = ntps * threshold
    return threshold
