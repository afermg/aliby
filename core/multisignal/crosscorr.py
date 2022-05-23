#!/usr/bin/env python3

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class crosscorrParameters(ParametersABC):
    """
    Parameters for the 'align' process.

    Attributes
    ----------
    t: int
        Lag, in time points
        FIXME: clarify with Peter.
    """

    _defaults = {
        "lagtime": None,
    }


class crosscorr(PostProcessABC):
    """ """

    def __init__(self, parameters: crosscorrParameters):
        super().__init__(parameters)

    def run(self, trace_dfA: pd.DataFrame, trace_dfB: pd.DataFrame = None):
        """Calculates normalised cross-correlations as a function of time.

        Calculates normalised auto- or cross-correlations as a function of time.
        Normalisation is by the product of the standard deviation for each
        variable calculated across replicates at each time point.
        With zero lag, the normalised correlation should be one.

        Parameters
        ----------
        trace_dfA: dataframe
            An array of signal values, with each row a replicate measurement
            and each column a time point.
        trace_dfB: dataframe (required for cross-correlation only)
            An array of signal values, with each row a replicate measurement
            and each column a time point.

        Returns
        -------
        norm_corr: array or aliby dataframe
            An array of the correlations with each row the result for the
            corresponding replicate and each column a time point
        """

        df = trace_dfA.copy() if type(trace_dfA) == pd.core.frame.DataFrame else None
        # convert from aliby dataframe to arrays
        trace_A = trace_dfA.to_numpy()
        # number of time points
        n_tps = trace_A.shape[1]
        # number of replicates
        n_replicates = trace_A.shape[0]
        # deviation from mean at each time point
        dmean_A = trace_A - np.nanmean(trace_A, axis=0).reshape((1, n_tps))
        # standard deviation over time for each replicate
        stdA = np.sqrt(np.nanmean(dmean_A ** 2, axis=1).reshape((n_replicates, 1)))
        if trace_dfB is not None:
            trace_B = trace_dfB.to_numpy()
            # cross correlation
            dmean_B = trace_B - np.nanmean(trace_B, axis=0).reshape((1, n_tps))
            stdB = np.sqrt(np.nanmean(dmean_B ** 2, axis=1).reshape((n_replicates, 1)))
        else:
            # auto correlation
            dmean_B = dmean_A
            stdB = stdA
        # calculate correlation
        corr = np.nan * np.ones(trace_A.shape)
        # lag r runs over time points
        for r in np.arange(0, n_tps):
            prods = [
                dmean_A[:, self.lagtime] * dmean_B[:, self.lagtime + r]
                for self.lagtime in range(n_tps - r)
            ]
            corr[:, r] = np.nansum(prods, axis=0) / (n_tps - r)
        norm_corr = np.array(corr) / stdA / stdB
        # return as a df if trace_dfA is a df else as an array
        return pd.DataFrame(norm_corr, index=df.index, columns=df.columns)
