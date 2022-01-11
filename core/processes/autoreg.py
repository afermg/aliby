import numpy as np
import pandas as pd
from collections import namedtuple
import scipy.linalg as linalg

from agora.base import ParametersABC, ProcessABC

# TODO: Provide the option of whether to optimise AR order
# TODO: Provide the functionality of 'smoothing' a time series with AR


class autoregParameters(ParametersABC):
    """
    TODO: EDIT THIS AND OTHER DOCSTRINGS, USING numpydoc.el, AND UPDATE COMMENTS
    Parameters for the 'fft' process.

    """

    def __init__(
        self,
        sampling_period,
        freq_npoints,
    ):
        # super().__init__()
        self.sampling_period = sampling_period
        self.freq_npoints = freq_npoints

    @classmethod
    def default(cls):
        # Sampling period should be listed in the HDF5 metadata (i.e. the
        # 'time_settings/timeinterval' attribute, unit seconds).  When using
        # this processor in a typical routine, use the sampling period from
        # there rather than relying on this default value of 5 minutes.
        return cls.from_dict(
            {
                "sampling_period": 5,
                # "oversampling_factor": 1,
                "freq_npoints": 100,
            }
        )


class autoreg(ProcessABC):
    """ """

    def __init__(
        self,
        parameters: autoregParameters,
    ):
        super().__init__(parameters)

    # AR_Fit class now becomes a function that gives dictionary output;
    # because outside of this context there hardly is any reason to access
    # the intermediate variables
    # BTW should this be a static method?
    def fit_autoreg(self, timeseries, ar_order):
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
        # phi vector goes from 1 to P in the publication...
        ar_coeffs = linalg.inv(sample_acfs_toeplitz).dot(sample_acfs[1 : ar_order + 1])
        # defines a dummy phi_0 as 1.  This is so that the indices I use in
        # get_noise_param are consistent with the publication.
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
    ):
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
        ar_model = self.fit_autoreg(timeseries, ar_order)
        freqs = np.linspace(0, 1 / (2 * sampling_period), freq_npoints)
        power = np.zeros(len(freqs))
        for ii, freq in enumerate(freqs):  # xi
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
        # normalise
        power = power / power[0]
        return freqs, power

    def run(self, signal: pd.DataFrame):
        """ """
        AutoregAxes = namedtuple("AutoregAxes", ["freqs", "power"])
        # Each element in this list is a named tuple: (freqs, power)
        axes = [
            AutoregAxes(
                *self.autoreg_periodogram(
                    timeseries=signal.iloc[row_index, :].dropna().values,
                    sampling_period=self.sampling_period,
                    freq_npoints=self.freq_npoints,
                    # Make this bit more readable
                    ar_order=self.optimise_ar_order(
                        signal.iloc[row_index, :].dropna().values,
                        int(
                            3 * np.sqrt(len(signal.iloc[row_index, :].dropna().values))
                        ),
                    ),
                )
            )
            for row_index in range(len(signal))
        ]

        freqs_df = pd.DataFrame([element.freqs for element in axes], index=signal.index)

        power_df = pd.DataFrame([element.power for element in axes], index=signal.index)

        # order_df = pd.DataFrame()

        return freqs_df, power_df  # , order_df
