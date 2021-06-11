class Parameters:
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


class Merger:
    # Class in charge of merging tracks
    def __init__(self, parameters):
        self.parameters = parameters

    def run(self, signal):
        merged, joint_pairs = merge_tracks(signal, min_len=self.window + 1)
