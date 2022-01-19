import numpy as np
import pandas as pd

from agora.abc import ParametersABC, ProcessABC


class LineageProcessParameters(ParametersABC):
    """
    Parameters
    """

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def default(cls):
        return cls.from_dict({})


class LineageProcess(ProcessABC):
    """
    Lineage process that must be passed a (N,3) lineage matrix (where the coliumns are trap, mother, daughter respectively)
    """

    def __init__(self, parameters: LineageProcessParameters):
        super().__init__(parameters)

    def run(
        self,
    ):
        pass
