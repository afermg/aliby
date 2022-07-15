from collections import namedtuple

import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from scipy.signal import periodogram

from postprocessor.core.abc import PostProcessABC


class fftParameters(ParametersABC):
    """
    Parameters for the 'fft' process.

    Attributes
    ----------
    sampling_period: float
        Sampling period of measurement values, in unit time.

    oversampling_factor: float
        Oversampling factor, which indicates how many times a signal is
        sampled over the Nyquist rate.  For example, if the oversampling
        factor is 2, the signal is sampled at 2 times the Nyquist rate.
    """

    _defaults = {
        "sampling_period": 5,
        "oversampling_factor": 1,
    }


class fft(PostProcessABC):
    """
    Process to estimate power spectral density (classical/Fourier).

    Methods
    -------
    classical_periodogram(timeseries: array_like, sampling_period: float,
        oversampling_factor: float)
        Estimates the power spectral density using a periodogram.
    run(signal: pd.DataFrame)
        Estimates the power spectral density of a dataframe of time series.
    """

    def __init__(self, parameters: fftParameters):
        super().__init__(parameters)

    def classical_periodogram(
        self,
        timeseries,
        sampling_period,
        oversampling_factor,
    ):
        """
        Estimates the power spectral density using a periodogram.

        This is based on a fast Fourier transform.

        The power spectrum is normalised so that the area under the power
        spectrum is constant across different time series, thus allowing users
        to easily compare spectra across time series.  See:
        * Scargle (1982). Astrophysical Journal 263 p. 835-853
        * Glynn et al (2006). Bioinformatics 22(3) 310-316

        Parameters
        ---------
        timeseries: array_like
            Time series of measurement values.

        sampling_period: float
            Sampling period of measurement values, in unit time.

        oversampling_factor: float
            Oversampling factor, which indicates how many times a signal is
            sampled over the Nyquist rate.  For example, if the oversampling
            factor is 2, the signal is sampled at 2 times the Nyquist rate.

        Returns
        -------
        freqs: ndarray
            Array of sample frequencies, unit time-1.

        power: ndarray
            Power spectral density or power spectrum of the time series,
            arbitrary units.
        """
        freqs, power = periodogram(
            timeseries,
            fs=1 / (oversampling_factor * sampling_period),
            nfft=len(timeseries) * oversampling_factor,
            detrend="constant",
            return_onesided=True,
            scaling="spectrum",
        )
        # Required to correct units; units will be expressed in min-1 (or any other
        # unit)
        freqs = oversampling_factor * freqs
        # Normalisation (Scargle, 1982; Glynn et al., 2006)
        power = power * (0.5 * len(timeseries))
        # Normalisation by variance to allow comparing different time series
        # (Scargle, 1982)
        power = power / np.var(timeseries, ddof=1)
        return freqs, power

    def run(self, signal: pd.DataFrame):
        """
        Estimates the power spectral density of a dataframe of time series.

        This uses the classical periodogram.

        All NaNs are dropped as the Fourier transform used does not afford
        missing time points or time points with uneven spacing in the time
        dimension.  For time series with missing values, the Lomb-Scargle
        periodogram is suggested (processes/lsp.py)

        Parameters
        ----------
        signal: pandas.DataFrame
            Time series, with rows indicating individiual time series (e.g. from
            each cell), and columns indicating time points.

        Returns
        -------
        freqs_df: pandas.DataFrame
            Sample frequencies from each time series, with labels preserved from
            'signal'.

        power_df: pandas.DataFrame
            Power spectrum from each time series, with labels preserved from
            'signal'.
        """
        FftAxes = namedtuple("FftAxes", ["freqs", "power"])
        # Each element in this list is a named tuple: (freqs, power)
        axes = [
            FftAxes(
                *self.classical_periodogram(
                    timeseries=signal.iloc[row_index, :].dropna().values,
                    sampling_period=self.sampling_period,
                    oversampling_factor=self.oversampling_factor,
                )
            )
            for row_index in range(len(signal))
        ]

        freqs_df = pd.DataFrame(
            [element.freqs for element in axes], index=signal.index
        )

        power_df = pd.DataFrame(
            [element.power for element in axes], index=signal.index
        )

        return freqs_df, power_df
