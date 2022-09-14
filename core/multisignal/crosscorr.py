#!/usr/bin/env python3

import numpy as np
import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class crosscorrParameters(ParametersABC):
    """
    Parameters for the 'crosscorr' process.

    Attributes
    ----------
    normalised: boolean (optional)
        If True, normalise the result for each replicate by the standard
        deviation over time for that replicate.
    only_pos: boolean (optional)
        If True, return results only for positive lags.
    """

    _defaults = {"normalised": True, "only_pos": False}


class crosscorr(PostProcessABC):
    """ """

    def __init__(self, parameters: crosscorrParameters):
        super().__init__(parameters)

    def run(self, trace_dfA: pd.DataFrame, trace_dfB: pd.DataFrame = None):
        """Calculates normalised cross-correlations as a function of lag.

        Calculates normalised auto- or cross-correlations as a function of lag.
        Lag is given in multiples of the unknown time interval between data points.

        Normalisation is by the product of the standard deviation over time for
        each replicate for each variable.

        For the cross-correlation between sA and sB, the closest peak to zero lag
        should in the positive lags if sA is delayed compared to signal B and in the
        negative lags if sA is advanced compared to signal B.

        Parameters
        ----------
        trace_dfA: dataframe
            An array of signal values, with each row a replicate measurement
            and each column a time point.
        trace_dfB: dataframe (required for cross-correlation only)
            An array of signal values, with each row a replicate measurement
            and each column a time point.
        normalised: boolean (optional)
            If True, normalise the result for each replicate by the standard
            deviation over time for that replicate.
        only_pos: boolean (optional)
            If True, return results only for positive lags.

        Returns
        -------
        corr: dataframe
            An array of the correlations with each row the result for the
            corresponding replicate and each column a time point
        lags: array
            A 1D array of the lags in multiples of the unknown time interval

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import pandas as pd
        >>> from postprocessor.core.multisignal.crosscorr import crosscorr

        Define a sine signal with 200 time points and 333 replicates

        >>> t = np.linspace(0, 4, 200)
        >>> ts = np.tile(t, 333).reshape((333, 200))
        >>> s = 3*np.sin(2*np.pi*ts + 2*np.pi*np.random.rand(333, 1))
        >>> s_df = pd.DataFrame(s)

        Find and plot the autocorrelaton

        >>> ac = crosscorr.as_function(s_df)
        >>> plt.figure()
        >>> plt.plot(ac.columns, ac.mean(axis=0, skipna=True))
        >>> plt.show()
        """

        df = (
            trace_dfA.copy()
            if type(trace_dfA) == pd.core.frame.DataFrame
            else None
        )
        # convert from aliby dataframe to arrays
        trace_A = trace_dfA.to_numpy()
        # number of replicates
        n_replicates = trace_A.shape[0]
        # number of time points
        n_tps = trace_A.shape[1]
        # deviation from the mean, where the mean is calculated over replicates
        # at each time point, which allows for non-stationary behaviour
        dmean_A = trace_A - np.nanmean(trace_A, axis=0).reshape((1, n_tps))
        # standard deviation over time for each replicate
        stdA = np.sqrt(
            np.nanmean(dmean_A**2, axis=1).reshape((n_replicates, 1))
        )
        if trace_dfB is not None:
            trace_B = trace_dfB.to_numpy()
            # cross correlation
            dmean_B = trace_B - np.nanmean(trace_B, axis=0).reshape((1, n_tps))
            stdB = np.sqrt(
                np.nanmean(dmean_B**2, axis=1).reshape((n_replicates, 1))
            )
        else:
            # auto correlation
            dmean_B = dmean_A
            stdB = stdA
        # calculate correlation
        # lag r runs over positive lags
        pos_corr = np.nan * np.ones(trace_A.shape)
        for r in range(n_tps):
            prods = [
                dmean_A[:, lagtime] * dmean_B[:, lagtime + r]
                for lagtime in range(n_tps - r)
            ]
            pos_corr[:, r] = np.nanmean(prods, axis=0)
        # lag r runs over negative lags
        # use corr_AB(-k) = corr_BA(k)
        neg_corr = np.nan * np.ones(trace_A.shape)
        for r in range(n_tps):
            prods = [
                dmean_B[:, lagtime] * dmean_A[:, lagtime + r]
                for lagtime in range(n_tps - r)
            ]
            neg_corr[:, r] = np.nanmean(prods, axis=0)
        if self.normalised:
            # normalise by standard deviation
            pos_corr = pos_corr / stdA / stdB
            neg_corr = neg_corr / stdA / stdB
        # combine lags
        lags = np.arange(-n_tps + 1, n_tps)
        corr = np.hstack((np.flip(neg_corr[:, 1:], axis=1), pos_corr))
        if self.only_pos:
            return pd.DataFrame(
                corr[:, int(lags.size / 2) :],
                index=df.index,
                columns=lags[int(lags.size / 2) :],
            )
        else:
            return pd.DataFrame(corr, index=df.index, columns=lags)
