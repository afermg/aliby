import numpy as np
import pandas as pd

from agora.abc import ParametersABC, ProcessABC


class TemplateParameters(ParametersABC):
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


class Template(ProcessABC):
    """
    Template for process class.
    """

    def __init__(self, parameters: TemplateParameters):
        super().__init__(parameters)

    def run(self):
        pass
