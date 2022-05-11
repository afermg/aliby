#!/usr/bin/env python3

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC

# TODO: Add documentation


class crosscorrParameters(ParametersABC):
    """
    Parameters for the 'align' process.

    Attributes
    ----------
    """

    _defaults = {
        "t": None,
    }


class crosscorr(PostProcessABC):
    """ """

    def __init__(self, parameters: alignParameters):
        super().__init__(parameters)

    # TODO: make variable names more informative
    def run(self, yA: pd.DataFrame, yB: pd.DataFrame = None):
        exampledf = yA.copy() if type(yA) == pd.core.frame.DataFrame else None
        # convert from aliby dataframe to arrays
        yA = yA.to_numpy()
        # number of time points
        n = yA.shape[1]
        # deviation from mean at each time point
        dyA = yA - np.nanmean(yA, axis=0).reshape((1, n))
        # standard deviation at each time point
        stdA = np.sqrt(np.nanmean(dyA ** 2, axis=0).reshape((1, n)))
        if yB is not None:
            yB = yB.to_numpy()
            # cross correlation
            dyB = yB - np.nanmean(yB, axis=0).reshape((1, n))
            stdB = np.sqrt(np.nanmean(dyB ** 2, axis=0).reshape((1, n)))
        else:
            # auto correlation
            dyB = dyA
            stdB = stdA
        # calculate correlation
        corr = np.nan * np.ones(yA.shape)
        # lag r runs over time points
        for r in np.arange(0, n):
            prods = [dyA[:, self.t] * dyB[:, self.t + r] for t in range(n - r)]
            corr[:, r] = np.nansum(prods, axis=0) / (n - r)
        norm_corr = np.array(corr) / stdA / stdB
        # return as a df if yA is a df else as an array
        return pd.DataFrame(norm_corr, index=exampledf.index, columns=exampledf.columns)
