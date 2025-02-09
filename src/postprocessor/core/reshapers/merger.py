"""Class for merging single-cell tracks."""

import numpy as np

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC
from postprocessor.core.reshapers.tracks import get_merges


class MergerParameters(ParametersABC):
    """
    Define the parameters for merger from a dict.

    There are five parameters expected in the dict:

    smooth: boolean
        Whether or not to smooth with a savgol_filter.
    tol: float or int
        The threshold of average prediction error/std necessary to
        consider two tracks to be the same.
        If float, the threshold is the fraction of the first track;
        if int, the threshold is in absolute units.
    window: int
        The size of the window of the savgol_filter.
    degree: int v
        The order of the polynomial used by the savgol_filter
    """

    _defaults = {
        "smooth": False,
        "tolerance": 0.2,
        "window": 5,
        "degree": 3,
        "min_avg_delta": 0.5,
    }


class Merger(PostProcessABC):
    """Find array of pairs of (trap, cell) indices to be merged."""

    def __init__(self, parameters):
        """Initialise with PostProcessABC."""
        super().__init__(parameters)

    def run(self, signal):
        """Merge."""
        if signal.shape[1] > 4:
            merges = get_merges(
                signal,
                smooth=self.parameters.smooth,
                tol=self.parameters.tolerance,
                window=self.parameters.window,
                degree=self.parameters.degree,
            )
        else:
            merges = np.array([])
        return merges
