import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class detrendParameters(ParametersABC):
    """Parameters for the 'detrend' process.

    Parameters for the 'detrend' process.

    Attributes
    ----------
    window : int
        Size of sliding window.
    """

    _defaults = {"window": 45}


class detrend(PostProcessABC):
    """Process to detrend using sliding window

    Methods
    -------
    run(signal: pd.DataFrame)
        Detrend each time series in a dataframe using a specified sliding window
    """

    def __init__(self, parameters: detrendParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        """Detrend using sliding window

        Detrend each time series in a dataframe using a specified sliding window

        Parameters
        ----------
        signal : pd.DataFrame
            Time series, with rows indicating individual time series (e.g. from
            each cell), and columns indicating time points.

        Returns
        -------
        signal_norm : pd.DataFrame
            Detrended time series.

        """
        # Normalise time series: divide by the mean time series
        signal_norm = signal.div(signal.mean(axis=1), axis=0)
        # Compute moving average
        signal_movavg = signal_norm.rolling(
            window=self.window, center=True, axis=1
        ).mean()
        # Detrend: divide normalised time series by moving average
        signal_detrend = signal_norm.div(signal_movavg)
        return signal_detrend.dropna(axis=1)  # Remove columns with NaNs
