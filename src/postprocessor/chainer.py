#!/usr/bin/env jupyter

import typing as t
from copy import copy

import numpy as np
import pandas as pd

from agora.io.signal import Signal
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import LineageProcessParameters
from agora.utils.association import validate_association


class Chainer(Signal):
    """
    Class that extends signal by applying postprocesess.
    Instead of reading processes previously applied, it executes
    them when called.
    """

    process_types = ("multisignal", "processes", "reshapers")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(
        self,
        dataset: str,
        chain: t.Collection[str] = ("standard", "interpolate", "savgol"),
        in_minutes: bool = True,
        **kwargs,
    ):
        data = self.get_raw(dataset, in_minutes=in_minutes)
        if chain:
            data = self.apply_chain(data, chain, **kwargs)
        return data

    def apply_chain(
        self, input_data: pd.DataFrame, chain: t.Tuple[str, ...], **kwargs
    ):
        """Apply a series of processes to a dataset.

        In a similar fashion to how postprocessing works, Chainer allows the
        consecutive application of processes to a dataset. Parameters can be
        passed as kwargs. It does not support the same process multiple times
        with different parameters.

        Parameters
        ----------
        input_data : pd.DataFrame
            Input data to iteratively process.
        chain : t.Tuple[str, ...]
            Tuple of strings with the name of processes.
        **kwargs : kwargs
            Arguments passed on to Process.as_function() method to modify the parameters.

        Examples
        --------
        FIXME: Add docs.


        """
        result = copy(input_data)
        self._intermediate_steps = []
        for process in chain:
            if process == "standard":
                result = standard(result, self.lineage())
            else:
                params = kwargs.get(process, {})
                process_cls = get_process(process)
                result = process_cls.as_function(result, **params)
                process_type = process_cls.__module__.split(".")[-2]
                if process_type == "reshapers":
                    if process == "merger":
                        merges = process.as_function(result, **params)
                        result = self.apply_merges(result, merges)

            self._intermediate_steps.append(result)
        return result


def standard(
    raw: pd.DataFrame,
    lin: np.ndarray,
    presence_filter_min: int = 7,
    presence_filter_mothers: float = 0.8,
):
    """
    This requires a double-check that mothers-that-are-daughters still are accounted for after
    filtering daughters by the minimal threshold.
    """
    # Get all mothers
    raw = raw.loc[raw.notna().sum(axis=1) > presence_filter_min].sort_index()
    indices = np.array(raw.index.to_list())
    valid_lineages, valid_indices = validate_association(lin, indices)

    daughters = lin[valid_lineages][:, [0, 2]]
    mothers = lin[valid_lineages][:, :2]
    in_lineage = raw.loc[valid_indices].copy()
    mother_label = np.repeat(0, in_lineage.shape[0])

    daughter_ids = (
        (
            np.array(in_lineage.index.to_list())
            == np.unique(daughters, axis=0)[:, None]
        )
        .all(axis=2)
        .any(axis=0)
    )
    mother_label[daughter_ids] = mothers[:, 1]
    # Filter mothers by presence
    in_lineage["mother_label"] = mother_label
    present = in_lineage.loc[
        (
            in_lineage.iloc[:, :-2].notna().sum(axis=1)
            > (in_lineage.shape[1] * presence_filter_mothers)
        )
        | mother_label
    ]
    present.set_index("mother_label", append=True, inplace=True)

    # In the end, we get the mothers present for more than {presence_lineage1}% of the experiment
    # and their tracklets present for more than {presence_lineage2} time-points
    return present
