#!/usr/bin/env python3

import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from pycatch22 import catch22_all
from sklearn import decomposition

from postprocessor.core.abc import PostProcessABC


class catch22Parameters(ParametersABC):
    """
    Parameters
        :min_len: Prefilter to account only for long-signal cells
    """

    _defaults = {
        "min_len": 0.8,
        "n_components": None,
    }


class catch22(PostProcessABC):
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
        catches = [
            catch22_all(adf.iloc[i, :].dropna().values)
            for i in range(len(adf))
        ]

        norm = pd.DataFrame(
            [catches[j]["values"] for j in range(len(catches))],
            index=adf.index,
            columns=catches[0]["names"],
        )

        return norm
