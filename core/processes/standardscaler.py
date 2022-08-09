#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import StandardScaler

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class standardscalerParameters(ParametersABC):
    """
    Parameters for the 'scale' process.
    """

    _defaults = {}


class standardscaler(PostProcessABC):
    """
    Process to scale a DataFrame of a signal using the standard scaler.

    Methods
    -------
    run(signal: pd.DataFrame)
        Scale values in a dataframe of time series.
    """

    def __init__(self, parameters: standardscalerParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        """Scale values in a dataframe of time series.

        Scale values in a dataframe of time series.  This function is effectively a
        wrapper for sklearn.preprocessing.StandardScaler.

        Parameters
        ----------
        signal : pd.DataFrame
            Time series, with rows indicating individual time series (e.g. from
            each cell), and columns indicating time points.
        """
        signal_array = signal.to_numpy()
        scaler = StandardScaler().fit(signal_array.transpose())
        signal_scaled_array = scaler.transform(signal_array.transpose())
        signal_scaled_array = signal_scaled_array.transpose()
        signal_scaled = pd.DataFrame(
            signal_scaled_array, columns=signal.columns, index=signal.index
        )
        return signal_scaled
