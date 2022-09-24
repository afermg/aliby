from itertools import cycle

import numpy as np
import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class aggregateParameters(ParametersABC):
    """
    Parameters
    reduction: str to be passed to a dataframe for collapsing across columns
    """

    _defaults = {
        "reductions": ["mean", "median", "max"],
        "axis": 1,
        "ranges": None,
    }


class aggregate(PostProcessABC):
    """
    Aggregate multiple datasets for cell-to-cell feature comparison.
    """

    def __init__(self, parameters: aggregateParameters):
        super().__init__(parameters)

    def run(self, signals):
        index = signals[0].index
        for s in signals[0:]:
            index = index.intersection(s.index)

        tmp_signals = [s.loc[index] for s in signals]
        for i, s in enumerate(signals):
            tmp_signals[i].name = s.name
        signals = tmp_signals

        assert len(signals), "Signals is empty"

        bad_words = {
            "postprocessing",
            "extraction",
            "None",
            "np_max",
            "",
        }

        def get_keywords(df):
            return [
                ind
                for item in df.name.split("/")
                for ind in item.split("/")
                if ind not in bad_words
            ]

        colnames = [
            "_".join(get_keywords(s) + [red])
            for s in signals
            for red in self.parameters.reductions
        ]
        concat = pd.concat(
            [
                getattr(signal, red)(axis=self.parameters.axis)
                for signal in signals
                for red in self.parameters.reductions
            ],
            names=signals[0].index.names,
            axis=self.parameters.axis,
        )
        if self.parameters.axis:
            concat.columns = colnames
        else:
            concat.columns = pd.MultiIndex.from_product(
                (
                    colnames,
                    [
                        "_".join((str(start), str(stop)))
                        for x in self.parameters.ranges
                        for start, stop in x
                    ],
                )
            )

        return concat
