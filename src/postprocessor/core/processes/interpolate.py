#!/usr/bin/env jupyter

import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class interpolateParameters(ParametersABC):
    """
    Parameters
    """

    _defaults = {"limit_area": "inside"}


class interpolate(PostProcessABC):
    """
    Interpolate process.
    """

    def __init__(self, parameters: interpolateParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        if len(signal):
            signal = signal.interpolate(limit_area="inside", axis=1)
        return signal
