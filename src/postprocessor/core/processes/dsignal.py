import bottleneck as bn
import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class dsignalParameters(ParametersABC):
    """
    :window: Number of timepoints to consider for signal.
    """

    _defaults = {"window": 10, "min_count": 5}


class dsignal(PostProcessABC):
    """
    Calculate the change in a signal using the mean of a moving window.
    """

    def __init__(self, parameters: dsignalParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        matrix = np.diff(
            bn.move_mean(
                signal,
                window=self.parameters.window,
                min_count=self.parameters.min_count,
                axis=1,
            ),
            axis=1,
        )

        # Pad values to keep the same signal shape
        matrix = np.pad(matrix, ((0, 0), (0, 1)), constant_values=np.nan)
        return pd.DataFrame(matrix, index=signal.index, columns=signal.columns)
