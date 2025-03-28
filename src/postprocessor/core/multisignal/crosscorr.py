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
    stationary: boolean
        If True, the underlying dynamic process is assumed to be
        stationary with the mean a constant, estimated from all
        data points.
    normalised: boolean (optional)
        If True, normalise the result for each replicate by the standard
        deviation over time for that replicate.
    only_pos: boolean (optional)
        If True, return results only for positive lags.
    """

    _defaults = {"stationary": False, "normalised": True, "only_pos": False}


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
        stationary: boolean
            If True, the underlying dynamic process is assumed to be
            stationary with the mean a constant, estimated from all
            data points.
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

        Reference
        ---------
        Dunlop MJ, Cox RS, Levine JH, Murray RM, Elowitz MB (2008). Regulatory
        activity revealed by dynamic correlations in gene expression noise.
        Nat Genet, 40, 1493-1498.
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
        # autocorrelation if 2nd dataframe is not supplied
        if trace_dfB is None:
            trace_dfB = trace_dfA
            trace_B = trace_A
        else:
            trace_B = trace_dfB.to_numpy()
        # find deviation from the mean
        dmean_A, stdA = _dev(trace_A, n_replicates, n_tps, self.stationary)
        dmean_B, stdB = _dev(trace_B, n_replicates, n_tps, self.stationary)
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


def _dev(y, nr, nt, stationary=False):
    # calculate deviation from the mean
    if stationary:
        # mean calculated over time and over replicates
        dy = y - np.nanmean(y)
    else:
        # mean calculated over replicates at each time point
        dy = y - np.nanmean(y, axis=0).reshape((1, nt))
    # standard deviation calculated for each replicate
    stdy = np.sqrt(np.nanmean(dy**2, axis=1).reshape((nr, 1)))
    return dy, stdy
