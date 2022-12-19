#!/usr/bin/env jupyter

import re
import typing as t
from copy import copy

import numpy as np
import pandas as pd

from agora.io.signal import Signal
from agora.utils.association import validate_association
from agora.utils.kymograph import bidirectional_retainment_filter
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import LineageProcessParameters


class Chainer(Signal):
    """
    Class that extends signal by applying postprocesess.
    Instead of reading processes previously applied, it executes
    them when called.
    """

    process_types = ("multisignal", "processes", "reshapers")
    common_chains = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for channel in self.candidate_channels:
            try:
                channel = [
                    ch for ch in self.channels if re.match("channel", ch)
                ][0]
                break
            except:
                pass

        try:
            equivalences = {
                "m5m": (
                    f"extraction/{channel}/max/max5px",
                    f"extraction/{channel}/max/median",
                ),
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

        except:
            pass

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
                for i, (name, span) in enumerate(self.stages_span_tp)
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
                result = bidirectional_retainment_filter(result)
            else:
                params = kwargs.get(process, {})
                process_cls = get_process(process)
                result = process_cls.as_function(result, **params)
                process_type = process_cls.__module__.split(".")[-2]
                if process_type == "reshapers":
                    if process == "merger":
                        raise (NotImplementedError)
                        merges = process.as_function(result, **params)
                        result = self.apply_merges(result, merges)

            self._intermediate_steps.append(result)
        return result
