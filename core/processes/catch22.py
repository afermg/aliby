#!/usr/bin/env python3


import numpy as np
import pandas as pd
from sklearn import decomposition
from catch22 import catch22_all

from agora.abc import ParametersABC, ProcessABC


class catch22Parameters(ParametersABC):
    """
    Parameters
        :min_len: Prefilter to account only for long-signal cells
    """

    def __init__(self, min_len, n_components):
        self.min_len = min_len
        self.n_components = n_components

    @classmethod
    def default(cls):
        return cls.from_dict(
            {
                "min_len": 0.8,
                "n_components": None,
            }
        )


class catch22(ProcessABC):
    """
    catch22 class. It produces 22 normalised features for each time lapse in the signal (using the catch22 library.)
    """

    def __init__(self, parameters: catch22Parameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        thresh = (
            self.min_len
            if isinstance(self.min_len, int)
            else signal.shape[1] * self.min_len
        )
        adf = signal.loc[signal.notna().sum(axis=1) > thresh]
        catches = [catch22_all(adf.iloc[i, :].dropna().values) for i in range(len(adf))]

        norm = pd.DataFrame(
            [catches[j]["values"] for j in range(len(catches))],
            index=adf.index,
            columns=catches[0]["names"],
        )

        return norm
