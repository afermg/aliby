from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC
from postprocessor.core.functions.tracks import get_joinable


class mergerParameters(ParametersABC):
    """
    :param tol: float or int threshold of average (prediction error/std) necessary
        to consider two tracks the same. If float is fraction of first track,
        if int it is absolute units.
    :param window: int value of window used for savgol_filter
    :param degree: int value of polynomial degree passed to savgol_filter
    """

    _defaults = {
        "smooth": False,
        "tolerance": 0.2,
        "window": 5,
        "degree": 3,
        "min_avg_delta": 0.5,
    }


class merger(PostProcessABC):
    """
    Combines rows of tracklet that are likely to be the same.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def run(self, signal):
        joinable = []
        if signal.shape[1] > 4:
            joinable = get_joinable(signal, tol=self.parameters.tolerance)
        # merged, _ = merge_tracks(signal)  # , min_len=self.window + 1)
        # indices = (*zip(*merged.index.tolist()),)
        # names = merged.index.names
        # return {name: ids for name, ids in zip(names, indices)}
        return joinable
