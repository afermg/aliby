from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.linalg as linalg
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC

# TODO: Provide the option of whether to optimise AR order -- see issue #1
# TODO: Provide the functionality of 'smoothing' a time series with AR -- see
# issue #1


class autoregParameters(ParametersABC):
    """Parameters for the 'autoreg' process

    Parameters for the 'autoreg' process.

    Attributes
    ----------
    sampling_period : float
        Sampling period of measurement values, in unit time.

    freq_npoints : int
        Number of points for the frequency axis of the closed-form solution of
        the estimated periodogram.  Defines the resolution for use in, for
        example, plots.

        # Sampling period should be listed in the HDF5 metadata (i.e. the
        # 'time_settings/timeinterval' attribute, unit seconds).  When using
        # this processor in a typical routine, use the sampling period from
        # there rather than relying on this default value of 5 minutes.
    """

    _defaults = {
        "sampling_period": 5,
        "freq_npoints": 100,
    }


class autoreg(PostProcessABC):
    """
    Process to estimate power spectral density (autoregressive model).

    Methods
    -------
    fit_autoreg(timeseries: array_like, ar_order: int)
        Computes linear algebra parameters used to fit an autoregressive model
        to a time series
    optimise_ar_order(timeseries: array_like, ar_order_upper_limit: int)
        Optimise autoregressive model order to fit a time series
    autoreg_periodogram(timeseries: array_like, sampling_period: float,
        freq_npoints: int, ar_order: int)
        Estimates the closed-form solution of the sample power spectrum of a
        timeseries, based on fitting an autoregressive model.
    run(signal: pd.DataFrame)
        Estimates the power spectral densities of a dataframe of time series,
        using autoregressive model
    """

    def __init__(
        self,
        parameters: autoregParameters,
    ):
        super().__init__(parameters)

    # Should this be a static method?
    def fit_autoreg(self, timeseries, ar_order) -> dict:
        """Computes linear algebra parameters to fit an AR model to a timeseries

        Computes linear algebra parameters used to fit an autoregressive model
        of a specified order to a time series.  Ultimately computes the Akaike
        information criterion of the model with the specified order.

        This model-fitting implementation is based on the supplementary
        information of:
        * Jia & Grima (2020). BioRxiv [https://doi.org/10.1101/2020.09.23.309724]

        Parameters
        ----------
        timeseries : array_like
            Time series of measurement values.
        ar_order : int
            Order of the autoregressive model. The order defines the complexity
            of the model; if the order is P, then each time point is modelled as
            a linear combination of P time points preceding it.

        Returns
        -------
        This function returns a dictionary, whose values are defined by these
        keys:

        sample_acfs : 1d array of floats
            Sample autocorrelation function; element indices go from 0 to
            ar_order.

        ar_coeffs : 1d array of floats
            Coefficients for the autoregressive model; defines the linear
            combination that defines the model.

        noise_param : float
            Noise parameter.

        aic : float
            Akaike information criterion of the autoregressive model.

        """
        # Estimates sample autocorrelation function (R).
        # sample_acfs: 1D array of R values
        sample_acfs = np.zeros(ar_order + 1)
        for ii in range(ar_order + 1):
            sample_acfs[ii] = (1 / len(timeseries)) * np.sum(
                [
                    (timeseries[k] - np.mean(timeseries))
                    * (timeseries[k + ii] - np.mean(timeseries))
                    for k in range(len(timeseries) - ii)
                ]
            )

        # Estimates AR coefficients (phi) by solving Yule-Walker equation.
        # ar_coeffs: 1D array of coefficients (i.e. phi values)
        sample_acfs_toeplitz = linalg.toeplitz(sample_acfs[0:ar_order])
        # phi vector goes from 1 to P in Jia & Grima (2020)
        ar_coeffs = linalg.inv(sample_acfs_toeplitz).dot(
            sample_acfs[1 : ar_order + 1]
        )
        # defines a dummy phi_0 as 1.  This is so that the indices I use in
        # get_noise_param are consistent with Jia & Grima (2020)
        ar_coeffs = np.insert(ar_coeffs, 0, 1.0, axis=0)

        # Estimates noise parameter (noise_param)
        noise_param = sample_acfs[0] - np.sum(
            [ar_coeffs[k] * sample_acfs[k] for k in range(1, ar_order + 1)]
        )

        # Calculates AIC (aic)
        aic = np.log(noise_param) + (ar_order) / len(timeseries)

        return {
            "sample_acfs": sample_acfs,
            "ar_coeffs": ar_coeffs,
            "noise_param": noise_param,
            "aic": aic,
        }

    # Should this be a static method?
    def optimise_ar_order(
        self,
        timeseries,
        ar_order_upper_limit,
    ) -> int:
        """Optimise autoregressive model order to fit a time series

        Optimise the autoregressive model order to fit a time series. Model
        selection relies on minimising the Akaike information criterion,
        sweeping over a range of possible orders.

        This implementation is based on the supplementary
        information of:
        * Jia & Grima (2020). BioRxiv [https://doi.org/10.1101/2020.09.23.309724]

        Parameters
        ----------
        timeseries : array_like
            Time series of measurement values.
        ar_order_upper_limit : int
            Upper bound for autoregressive model order; recommended to be the
            square root of the length of the time series.

        Returns
        -------
        int
            Optimal order for an autoregressive model that fits the time series.

        """
        # Bug: artificial dip at order 1 if time series is a smooth sinusoid.
        # Will probably need to fix it so that it checks if the minimum also
        # corresponds to a zero derivative.
        ar_orders = np.arange(1, ar_order_upper_limit)
        aics = np.zeros(len(ar_orders))
        for ii, ar_order in enumerate(ar_orders):
            model = self.fit_autoreg(timeseries, ar_order)
            aics[ii] = model["aic"]
        return ar_orders[np.argmin(aics)]

    def autoreg_periodogram(
        self,
        timeseries,
        sampling_period,
        freq_npoints,
        ar_order,
    ):
        """Estimates the power spectrum of a timeseries, based on an AR model

        Estimates the closed-form solution of the sample power spectrum of a
        time series. This estimation is based on fitting an autoregressive model
        of an optimised order to the time series, and using computed linear
        algebra parameters to compute the closed-form solution.

        This implementation is based on the supplementary
        information of:
        * Jia & Grima (2020). BioRxiv [https://doi.org/10.1101/2020.09.23.309724]

        Parameters
        ---------
        timeseries : array_like
            Time series of measurement values.
        sampling_period : float
            Sampling period of measurement values, in unit time.
        freq_npoints : int
            Number of points for the frequency axis of the closed-form solution of
            the estimated periodogram.  Defines the resolution for use in, for
            example, plots.
        ar_order : int
            Order of the autoregressive model. The order defines the complexity
            of the model; if the order is P, then each time point is modelled as
            a linear combination of P time points preceding it.

        Returns
        -------
        freqs: ndarray
            Array of sample frequencies, unit time-1.

        power: ndarray
            Power spectral density or power spectrum of the time series,
            arbitrary units.

        Examples
        --------
        FIXME: Add docs.

        """
        ar_model = self.fit_autoreg(timeseries, ar_order)
        freqs = np.linspace(0, 1 / (2 * sampling_period), freq_npoints)
        power = np.zeros(len(freqs))
        # Sweeps the frequencies; corresponds to the 'xi' variable in
        # Jia & Grima (2020)
        for ii, freq in enumerate(freqs):
            # multiplied 2pi into the exponential to get the frequency rather
            # than angular frequency
            summation = [
                ar_model["ar_coeffs"][k] * np.exp(-1j * k * (2 * np.pi) * freq)
                for k in range(ar_order + 1)
            ]
            summation[0] = 1  # minus sign error???
            divisor = np.sum(summation)
            power[ii] = (ar_model["noise_param"] / (2 * np.pi)) / np.power(
                np.abs(divisor), 2
            )
        # Normalise by first element of power axis.  This is consistent with
        # the MATLAB code from Ramon Grima.
        power = power / power[0]
        return freqs, power

    def run(self, signal: pd.DataFrame):
        # TODO: Return an additional DataFrame that contains the orders,
        # optimised or not
        """Estimates the power spectra of a dataframe of time series, using AR

        Estimates the power spectral densities of a dataframe of time series.
        This estimation is based on fitting an autoregressive model
        of an optimised order to the time series, and using computed linear
        algebra parameters to compute the closed-form solution.

        This implementation is based on the supplementary
        information of:
        * Jia & Grima (2020). BioRxiv [https://doi.org/10.1101/2020.09.23.309724]

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

        order_df: pandas.DataFrame
            Optimised order of the autoregressive model fitted to each time
            series, with labels preserved from 'signal'.

        Examples
        --------
        FIXME: Add docs.

        """
        AutoregAxes = namedtuple("AutoregAxes", ["freqs", "power", "order"])
        # Each element in this list is a named tuple: (freqs, power, order)
        axes = []
        # Use enumerate instead?
        for row_index in range(len(signal)):
            timeseries = signal.iloc[row_index, :].dropna().values
            optimal_ar_order = self.optimise_ar_order(
                timeseries, int(3 * np.sqrt(len(timeseries)))
            )
            axes.insert(
                row_index,
                AutoregAxes(
                    *self.autoreg_periodogram(
                        timeseries=timeseries,
                        sampling_period=self.sampling_period,
                        freq_npoints=self.freq_npoints,
                        ar_order=optimal_ar_order,
                    ),
                    optimal_ar_order,
                ),
            )

        freqs_df = pd.DataFrame(
            [element.freqs for element in axes], index=signal.index
        )
        power_df = pd.DataFrame(
            [element.power for element in axes], index=signal.index
        )
        order_df = pd.DataFrame(
            [element.order for element in axes], index=signal.index
        )

        return freqs_df, power_df, order_df
