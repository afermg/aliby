import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


def moving_average(input_timeseries, window):
    """Compute moving average of time series

    Compute moving average of time series

    Parameters
    ----------
    input_timeseries : array_like
        Input time series.
    window : int
        Size of sliding window to compute the moving average over.
    """
    processed_timeseries = np.cumsum(input_timeseries, dtype=float)
    processed_timeseries[window:] = (
        processed_timeseries[window:] - processed_timeseries[:-window]
    )
    return processed_timeseries[window - 1 :] / window


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
        signal = signal.div(signal.mean(axis=1), axis=0)
        signal_movavg = signal.apply(
            lambda x: pd.Series(moving_average(x.values, self.window)), axis=1
        )
        signal_norm = (
            signal.iloc(axis=1)[self.window // 2 : -self.window // 2]
            / signal_movavg.iloc[:, 0 : signal_movavg.shape[1] - 1].values
        )
        return signal_norm
