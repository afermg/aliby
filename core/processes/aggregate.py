from itertools import cycle

import numpy as np
import pandas as pd

from agora.base import ParametersABC, ProcessABC


class aggregateParameters(ParametersABC):
    """
    Parameters
    reduction: str to be passed to a dataframe for collapsing across columns
    """

    def __init__(self, reductions):
        super().__init__()
        self.reductions = reductions

    @classmethod
    def default(cls):
        return cls.from_dict({"reductions": ["mean", "median", "max"]})


class aggregate(ProcessABC):
    """
    aggregate multiple datasets
    """

    def __init__(self, parameters: aggregateParameters):
        super().__init__(parameters)

    def run(self, signals):
        names = np.array([signal.index.names for signal in signals])
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
        get_keywords = lambda df: [
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
                getattr(signal, red)(axis=1)
                for signal in signals
                for red in self.parameters.reductions
            ],
            names=signals[0].index.names,
            axis=1,
        )
        concat.columns = colnames

        return concat
