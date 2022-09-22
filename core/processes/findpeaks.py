#!/usr/bin/env python3

import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from scipy.signal import find_peaks

from postprocessor.core.abc import PostProcessABC


class findpeaksParameters(ParametersABC):
    """
    Parameters for the 'findpeaks' process.

    Attributes
    ----------
    distance: number
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.

    prominence: number or ndarray or sequence
        Required prominence of peaks. Either a number, None, an array matching x
        or a 2-element sequence of the former. The first element is always
        interpreted as the minimal and the second, if supplied, as the maximal
        required prominence.
    """

    _defaults = {
        "height": None,
        "threshold": None,
        "distance": 10,
        "prominence": 0.035,
        "width": None,
        "wlen": None,
        "rel_height": 0.5,
        "plateau_size": None,
    }


class findpeaks(PostProcessABC):
    """
    Process to find peaks inside a signal.

    Methods
    -------
    _find_peaks_mask(timeseries: sequence, distance: number, prominence: number
        or ndarray or sequence)
        Find peaks of a time series and returns a binary mask locating these peaks
    run(signal: pd.DataFrame)
        Find peaks in a dataframe of time series.
    """

    def __init__(self, parameters: findpeaksParameters):
        super().__init__(parameters)

    def _find_peaks_mask(
        self,
        timeseries,
        height,
        threshold,
        distance,
        prominence,
        width,
        wlen,
        rel_height,
        plateau_size,
    ):
        """Find peaks of a time series and returns a binary mask locating these peaks

        Parameters
        ----------
        timeseries : sequence
            Time series with peaks.
        distance : number
            Required minimal horizontal distance in samples between neighbouring
            peaks.
        prominence : number or ndarray or sequence
            Required prominence of peaks.
        """
        peak_indices = find_peaks(
            timeseries,
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            rel_height=rel_height,
            plateau_size=plateau_size,
        )[0]
        mask = np.zeros(len(timeseries), dtype=int)
        mask[peak_indices] = 1
        return mask

    def run(self, signal: pd.DataFrame):
        """Find peaks in a dataframe of time series.

        Find peaks of a dataframe of time series. This function is effectively a
        wrapper for scipy.signal.find_peaks.

        Parameters
        ----------
        signal : pd.DataFrame
            Time series, with rows indicating individual time series (e.g. from
            each cell), and columns indicating time points.
        """
        mask_df = signal.apply(
            lambda x: self._find_peaks_mask(
                x,
                height=self.height,
                threshold=self.threshold,
                distance=self.distance,
                prominence=self.prominence,
                width=self.width,
                wlen=self.wlen,
                rel_height=self.rel_height,
                plateau_size=self.plateau_size,
            ),
            axis=1,
            result_type="expand",
        )
        mask_df.columns = signal.columns
        return mask_df
