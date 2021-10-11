from itertools import cycle

import numpy as np
import pandas as pd

from agora.base import ParametersABC, ProcessABC


class aggregateParameters(ParametersABC):
    """
    Parameters
    reduction: str to be passed to a dataframe for collapsing across columns
    """

    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    @classmethod
    def default(cls):
        return cls.from_dict({"reduction": "median"})


class aggregate(ProcessABC):
    """
    aggregate multiple datasets
    """

    def __init__(self, parameters: aggregateParameters):
        super().__init__(parameters)

    def run(self, signals):
        names = np.array([signal.index.names for signal in signals])
        if not np.all(names == names[0]):
            "Not all indices are the same, selecting smallest set"
            index = signals[0].index
            for s in signals[0:]:
                index = index.intersection(s.index)

            signals = [s.loc[index] for s in signals]

        assert len(signals), "Signals is empty"

        bad_words = {
            "postprocessing",
            "extraction",
            "None",
            "np",
            "general",
        }
        get_keywords = lambda df: [
            ind
            for item in df.name.split("/")
            for ind in item.split("/")
            if ind not in bad_words
        ]
        colnames = ["_".join(get_keywords(s)) for s in signals]
        concat = pd.concat(
            [getattr(signal, self.parameters.reduction)(axis=1) for signal in signals],
            names=signals[0].index.names,
            axis=1,
        )
        concat.columns = colnames

        return concat
