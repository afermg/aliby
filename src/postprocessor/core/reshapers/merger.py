from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC
from postprocessor.core.functions.tracks import get_joinable


class MergerParameters(ParametersABC):
    """
    Define the parameters for merger from a dict.

    There are five parameters expected in the dict:

    smooth, boolean
        Whether or not to smooth with a savgol_filter.
    tol: float or  int
        The threshold of average prediction error/std necessary to
        consider two tracks the same.
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
    """Combine rows of tracklet that are likely to be the same."""

    def __init__(self, parameters):
        super().__init__(parameters)

    def run(self, signal):
        joinable = []
        if signal.shape[1] > 4:
            joinable = get_joinable(
                signal,
                smooth=self.parameters.smooth,
                tol=self.parameters.tolerance,
                window=self.parameters.window,
                degree=self.parameters.degree,
            )
        return joinable
