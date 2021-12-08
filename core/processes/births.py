#!/usr/bin/env python3

import numpy as np
import pandas as pd

from agora.base import ParametersABC, ProcessABC


class birthsParameters(ParametersABC):
    """
    :window: Number of timepoints to consider for signal.
    """

    def __init__(self):
        pass

    @classmethod
    def default(cls):
        return cls.from_dict({})


class births(ProcessABC):
    """
    Calculate births in a trap assuming one mother per trap

    returns a pandas series with the births
    """

    def __init__(self, parameters: birthsParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        # vals = signal.groupby("trap").apply(lambda x: len(x) - 1)
        birth_events = signal.groupby("trap").apply(lambda x: x.first_valid_index())
        fvi = signal.apply(lambda x: x.first_valid_index(), axis=1)
        fvi_list = fvi.groupby("trap").apply(list)

        mothers = signal.groupby("trap").apply(
            lambda x: x.index[x.notna().sum(axis=1).argmax()]
        )

        births = pd.DataFrame(
            np.zeros((mothers.shape[0], signal.shape[1])).astype(bool),
            index=mothers.index,
            columns=signal.columns,
        )
        for i, v in fvi_list.items():
            births.loc[i, v] = True

        return births
