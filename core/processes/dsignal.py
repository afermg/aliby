import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class dsignalParameters(ParametersABC):
    """
    :window: Number of timepoints to consider for signal.
    """

    _defaults = {"window": 3}


class dsignal(PostProcessABC):
    """
    Calculate the change in a signal depending on a window
    """

    def __init__(self, parameters: dsignalParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        return signal.rolling(window=self.parameters.window, axis=1).mean().diff(axis=1)
