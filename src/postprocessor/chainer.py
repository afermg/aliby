#!/usr/bin/env jupyter

import typing as t
from copy import copy

import numpy as np
import pandas as pd

from agora.io.signal import Signal
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import LineageProcessParameters
from agora.utils.association import validate_association

import re


class Chainer(Signal):
    """
    Class that extends signal by applying postprocesess.
    Instead of reading processes previously applied, it executes
    them when called.
    """

    process_types = ("multisignal", "processes", "reshapers")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channel = [ch for ch in self.channels if re.match("GFP", ch)][0]
        if (
            channel == "GFPFast" and "mCherry" in self.channels
        ):  # Use mCherry for Batman if available
            channel = "mCherry"

        equivalences = {
            "m5m": (
                f"extraction/{channel}/max/max5px",
                f"extraction/{channel}/max/median",
            )
        }

        def replace_url(url: str, bgsub: str = ""):
            # return pattern with bgsub
            channel = url.split("/")[1]
            if "bgsub" in bgsub:
                url = re.sub(channel, f"{channel}_bgsub", url)
            return url

        self.common_chains = {
            alias
            + bgsub: lambda **kwargs: self.get(
                replace_url(denominator, alias + bgsub), **kwargs
            )
            / self.get(replace_url(numerator, alias + bgsub), **kwargs)
            for alias, (denominator, numerator) in equivalences.items()
            for bgsub in ("", "_bgsub")
        }

    def get(
        self,
        dataset: str,
        chain: t.Collection[str] = ("standard", "interpolate", "savgol"),
        in_minutes: bool = True,
        stages: bool = True,
        retain: t.Optional[float] = None,
        **kwargs,
    ):
        if dataset in self.common_chains:  # Produce dataset on the fly
            data = self.common_chains[dataset](**kwargs)
        else:
            data = self.get_raw(dataset, in_minutes=in_minutes)
            if chain:
                data = self.apply_chain(data, chain, **kwargs)

        if retain:
            data = data.loc[data.notna().sum(axis=1) > data.shape[1] * retain]

        if (
            stages and "stage" not in data.columns.names
        ):  # Return stages as additional column level

            stages_index = [
                x
                for i, (name, span) in enumerate(self.stages_span)
                for x in (f"{i} { name }",) * span
            ]
            data.columns = pd.MultiIndex.from_tuples(
                zip(stages_index, data.columns),
                names=("stage", "time"),
            )

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
                result = standard_filtering(result, self.lineage())
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


# def standard(
#     raw: pd.DataFrame,
#     lin: np.ndarray,
#     presence_filter_min: int = 7,
#     presence_filter_mothers: float = 0.8,
# ):
#     """
#     This requires a double-check that mothers-that-are-daughters still are accounted for after
#     filtering daughters by the minimal threshold.
#     """
#     raw = raw.loc[raw.notna().sum(axis=1) > presence_filter_min].sort_index()
#     indices = np.array(raw.index.to_list())
#     # Get remaining full families
#     valid_lineages, valid_indices = validate_association(lin, indices)

#     daughters = lin[valid_lineages][:, [0, 2]]
#     mothers = lin[valid_lineages][:, :2]
#     in_lineage = raw.loc[valid_indices].copy()
#     mother_label = np.repeat(0, in_lineage.shape[0])

#     daughter_ids = (
#         (
#             np.array(in_lineage.index.to_list())
#             == np.unique(daughters, axis=0)[:, None]
#         )
#         .all(axis=2)
#         .any(axis=0)
#     )
#     mother_label[daughter_ids] = mothers[:, 1]
#     # Filter mothers by presence
#     in_lineage["mother_label"] = mother_label
#     present = in_lineage.loc[
#         (
#             in_lineage.iloc[:, :-1].notna().sum(axis=1)
#             > ((in_lineage.shape[1] - 1) * presence_filter_mothers)
#         )
#         | mother_label
#     ]
#     present.set_index("mother_label", append=True, inplace=True)

#     # Finally, check full families again
#     final_indices = np.array(present.index.to_list())
#     _, final_mask = validate_association(
#         np.array([tuple(x) for x in present.index.swaplevel(1, 2)]),
#         final_indices[:, :2],
#     )
#     return present.loc[final_mask]

#     # In the end, we get the mothers present for more than {presence_lineage1}% of the experiment
#     # and their tracklets present for more than {presence_lineage2} time-points
#     return present


def standard_filtering(
    raw: pd.DataFrame,
    lin: np.ndarray,
    presence_high: float = 0.8,
    presence_low: int = 7,
):
    # Get all mothers
    _, valid_indices = validate_association(
        lin, np.array(raw.index.to_list()), match_column=0
    )
    in_lineage = raw.loc[valid_indices]

    # Filter mothers by presence
    present = in_lineage.loc[
        in_lineage.notna().sum(axis=1) > (in_lineage.shape[1] * presence_high)
    ]

    # Get indices
    indices = np.array(present.index.to_list())
    to_cast = np.stack((lin[:, :2], lin[:, [0, 2]]), axis=1)
    ndin = to_cast[..., None] == indices.T[None, ...]

    # use indices to fetch all daughters
    valid_association = ndin.all(axis=2)[:, 0].any(axis=-1)

    # Remove repeats
    mothers, daughters = np.split(to_cast[valid_association], 2, axis=1)
    mothers = mothers[:, 0]
    daughters = daughters[:, 0]
    d_m_dict = {tuple(d): m[-1] for m, d in zip(mothers, daughters)}

    # assuming unique sorts
    raw_mothers = raw.loc[_as_tuples(mothers)]
    raw_mothers["mother_label"] = 0
    raw_daughters = raw.loc[_as_tuples(daughters)]
    raw_daughters["mother_label"] = d_m_dict.values()
    concat = pd.concat((raw_mothers, raw_daughters)).sort_index()
    concat.set_index("mother_label", append=True, inplace=True)

    # Last filter to remove tracklets that are too short
    removed_buds = concat.notna().sum(axis=1) <= presence_low
    filt = concat.loc[~removed_buds]

    # We check that no mothers are left child-less
    m_d_dict = {tuple(m): [] for m in mothers}
    for (trap, d), m in d_m_dict.items():
        m_d_dict[(trap, m)].append(d)

    for trap, daughter, mother in concat.index[removed_buds]:
        idx_to_delete = m_d_dict[(trap, mother)].index(daughter)
        del m_d_dict[(trap, mother)][idx_to_delete]

    bud_free = []
    for m, d in m_d_dict.items():
        if not d:
            bud_free.append(m)

    final_result = filt.drop(bud_free)

    # In the end, we get the mothers present for more than {presence_lineage1}% of the experiment
    # and their tracklets present for more than {presence_lineage2} time-points
    return final_result


def _as_tuples(array: t.Collection) -> t.List[t.Tuple[int, int]]:
    return [tuple(x) for x in np.unique(array, axis=0)]
