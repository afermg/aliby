import pandas as pd

from agora.abc import ParametersABC, ProcessABC


class dsignalParameters(ParametersABC):
    """
    :window: Number of timepoints to consider for signal.
    """

    def __init__(self, window: int):
        self.window = window

    @classmethod
    def default(cls):
        return cls.from_dict({"window": 3})


class dsignal(ProcessABC):
    """
    Calculate the change in a signal depending on a window
    """

    def __init__(self, parameters: dsignalParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        return signal.rolling(window=self.parameters.window, axis=1).mean().diff(axis=1)
