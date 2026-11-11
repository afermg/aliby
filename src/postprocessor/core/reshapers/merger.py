"""Class for merging single-cell tracks."""

import numpy as np

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC
from postprocessor.core.reshapers.tracks import get_merges


class MergerParameters(ParametersABC):
    """
    Define the parameters for merger from a dict.

    There are four parameters expected in the dict:

    smooth: boolean
        Whether or not to smooth with a savgol_filter.
    tolerance: float or int
        The greatest difference between two tracks' values where they
        meet for the tracks to be joined. If less than one, a fraction
        of the left track's last value; otherwise, in the units of the
        Signal.
    window: int
        The number of values used to predict where a track is going,
        and the size of the window of the savgol_filter, which must be
        odd.
    degree: int
        The order of the polynomial used by the savgol_filter.
    """

    _defaults = {
        "smooth": False,
        "tolerance": 0.2,
        "window": 5,
        "degree": 3,
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
