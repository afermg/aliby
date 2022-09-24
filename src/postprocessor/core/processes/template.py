import numpy as np
import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class TemplateParameters(ParametersABC):
    """
    Parameters
    """

    _defaults = {}


class Template(PostProcessABC):
    """
    Template for process class.
    """

    def __init__(self, parameters: TemplateParameters):
        super().__init__(parameters)

    def run(self):
        pass
