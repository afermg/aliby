from scipy.signal import argrelmax, argrelmin
from postprocessor.core.processes.base import ParametersABC, ProcessABC


class PeaksParameters(ParametersABC):
    """
    Parameters
        type : str {minima,  maxima, "all"}. Determines which type of peaks to identify
        order : int Parameter to pass to scipy.signal.argrelextrema indicating
            how many points to use for comparison.
    """

    def __init__(self, type):
        self.type = type

    @classmethod
    def default(cls):
        return cls.from_dict({"type": "minima", "order": 3})


class Peaks(ProcessABC):
    """
    Identifies a signal sharply dropping.
    """

    def __init__(self, parameters: PeaksParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        """
        Returns a boolean dataframe with the same shape as the
        original signal but with peaks as true values.
        """
        peaks_mat = np.zeros_like(signal, dtype=bool)

        comparator = np.less if self.parameters.type is "minima" else np.greater
        peaks_ids = argrelextrema(new_df, comparator=comparator, order=order)
        peaks_mat[peak_ids] = True

        return pd.DataFrame(
            peaks_mat,
            index=signal.index,
            columns=signal.columns,
        )
