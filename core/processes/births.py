#!/usr/bin/env python3

import pandas as pd

from agora.io.base import ParametersABC, ProcessABC


class birthsParameters(ParametersABC):
    """
    :window: Number of timepoints to consider for signal.
    """

    def __init__(self):
        pass

    @classmethod
    def default(cls):
        return cls.from_dict({})


class births(ProcessABC):
    """
    Calculate the change in a signal depending on a window
    """

    def __init__(self, parameters: birthsParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        pass
