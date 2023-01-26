#!/usr/bin/env jupyter

import re
import typing as t
from copy import copy

import pandas as pd

from agora.io.signal import Signal
from agora.utils.association import validate_association
from agora.utils.kymograph import bidirectional_retainment_filter
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import LineageProcessParameters


class Chainer(Signal):
    """
    Extend Signal by applying post-processes and allowing composite signals that combine basic signals.
    It "chains" multiple processes upon fetching a dataset to produce the desired datasets.

    Instead of reading processes previously applied, it executes
    them when called.
    """

    _synonyms = {
        "m5m": ("extraction/GFP/max/max5px", "extraction/GFP/max/median")
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def replace_path(path: str, bgsub: bool = ""):
            # function to add bgsub to paths
            channel = path.split("/")[1]
            suffix = "_bgsub" if bgsub else ""
            path = re.sub(channel, f"{channel}{suffix}", path)
            return path

        # Add chain with and without bgsub for composite statistics
        self.common_chains = {
            alias
            + bgsub: lambda **kwargs: self.get(
                replace_path(denominator, alias + bgsub), **kwargs
            )
            / self.get(replace_path(numerator, alias + bgsub), **kwargs)
            for alias, (denominator, numerator) in self._synonyms.items()
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
        """Load data from an h5 file."""
        if dataset in self.common_chains:
            # get dataset for composite chains
            data = self.common_chains[dataset](**kwargs)
        else:
            # use Signal's get_raw
            data = self.get_raw(dataset, in_minutes=in_minutes)
            if chain:
                data = self.apply_chain(data, chain, **kwargs)
        if retain:
            # keep data only from early time points
            data = self.get_retained(data, retain)
            # data = data.loc[data.notna().sum(axis=1) > data.shape[1] * retain]
        if stages and "stage" not in data.columns.names:
            # return stages as additional column level
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
        """
        Apply a series of processes to a data set.

        Like postprocessing, Chainer consecutively applies processes.

        Parameters can be passed as kwargs.

        Chainer does not support applying the same process multiple times with different parameters.

        Parameters
        ----------
        input_data : pd.DataFrame
            Input data to process.
        chain : t.Tuple[str, ...]
            Tuple of strings with the names of the processes
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
            self._intermediate_steps.append(result)
        return result
