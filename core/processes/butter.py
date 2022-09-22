import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from scipy import signal

from postprocessor.core.abc import PostProcessABC


class butterParameters(ParametersABC):
    """Parameters for the 'butter' process.

    Parameters for the 'butter' process.

    Attributes
    ----------
    order : int
        The order of the filter.
    critical_freqs : array_like
        The critical frequency or frequencies.  For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a
        length-2 sequence.  For a Butterworth filter, this is the point at which
        the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
        For digital filters, if fs is not specified, Wn units are normalized
        from 0 to 1, where 1 is the Nyquist frequency (Wn is thus in
        half cycles / sample and defined as 2*critical frequencies / fs).
        If fs is specified, Wn is in the same units as fs.  For analog filters,
        Wn is an angular frequency (e.g. rad/s).
    filter_type : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        The type of filter. Default is ‘lowpass’.
    sampling_freq : float
        The sampling frequency of the digital system.
    """

    _defaults = {
        "order": 2,
        "critical_freqs": 1 / 350,
        "filter_type": "highpass",
        "sampling_freq": 1 / 5,
    }


class butter(PostProcessABC):
    """Process to apply Butterworth filter
    based on scipy.signal.butter

    Methods
    -------
    run(signal: pd.DataFrame)
        Apply Butterworth filter constructed according to user parameters
        to each time series in a DataFrame
    """

    def __init__(self, parameters: butterParameters):
        super().__init__(parameters)

    def butterfilter(self, timeseries):
        """Apply Butterworth filter to one time series"""
        # second-order-sections output
        # by default, using a digital filter
        sos = signal.butter(
            N=self.order,
            Wn=self.critical_freqs,
            btype=self.filter_type,
            fs=self.sampling_freq,
            output="sos",
        )
        # subtract time series by mean
        # otherwise the first couple time series will look like the acf,
        # which is not what we want
        # filter time series
        timeseries_norm = timeseries - np.mean(timeseries)
        return signal.sosfiltfilt(sos, timeseries_norm)

    def run(self, signal_df: pd.DataFrame):
        """Apply Butterworth filter

        Parameters
        ----------
        signal : pd.DataFrame
            Time series, with rows indicating individual time series (e.g. from
            each cell), and columns indicating time points.

        Returns
        -------
        signal_filtered : pd.DataFrame
            Filtered time series.

        """
        signal_filtered = signal_df.apply(
            self.butterfilter, axis=1, result_type="expand"
        )
        signal_filtered.columns = signal_df.columns
        return signal_filtered
