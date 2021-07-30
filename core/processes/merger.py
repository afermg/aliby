from postprocessor.core.processes.base import ParametersABC, ProcessABC
from postprocessor.core.functions.tracks import clean_tracks, merge_tracks, join_tracks


class MergerParameters(ParametersABC):
    """
    :param tol: float or int threshold of average (prediction error/std) necessary
        to consider two tracks the same. If float is fraction of first track,
        if int it is absolute units.
    :param window: int value of window used for savgol_filter
    :param degree: int value of polynomial degree passed to savgol_filter
    """

    def __init__(
        self,
        smooth: bool = False,
        tolerance: float = 0.1,
        window: int = 5,
        degree: int = 3,
        min_avg_delta: float = 0.9,
    ):

        self.smooth = smooth

        self.tolerance = tolerance

        self.window = window

        self.degree = degree

        self.min_avg_delta = min_avg_delta

    @classmethod
    def default(cls):
        return cls.from_dict(
            {
                "smooth": False,
                "tolerance": 0.1,
                "window": 5,
                "degree": 3,
                "min_avg_delta": 0.9,
            }
        )


class Merger(ProcessABC):
    """
    TODO Integrate functions/tracks.py inside this class?
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def run(self, signal):
        merged, _ = merge_tracks(signal)  # , min_len=self.window + 1)
        indices = (*zip(*merged.index.tolist()),)
        names = merged.index.names
        return {name: ids for name, ids in zip(names, indices)}
