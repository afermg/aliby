# TODO Module docstring
import typing as t
from abc import abstractmethod

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class LineageProcessParameters(ParametersABC):
    """
    Parameters
    """

    _defaults = {}


class LineageProcess(PostProcessABC):
    """
    Lineage process that must be passed a (N,3) lineage matrix (where the coliumns are trap, mother, daughter respectively)
    """

    def __init__(self, parameters: LineageProcessParameters):
        super().__init__(parameters)

    @abstractmethod
    def run(
        self,
        signal: pd.DataFrame,
        lineage: np.ndarray,
        *args,
    ):
        pass

    @classmethod
    def as_function(
        cls,
        data: pd.DataFrame,
        lineage: t.Union[t.Dict[t.Tuple[int], t.List[int]]],
        *extra_data,
        **kwargs,
    ):
        """
        Overrides PostProcess.as_function classmethod.
        Lineage functions require lineage information to be passed if run as function.
        """
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(
            data, lineage=lineage, *extra_data
        )

    def load_lineage(self, lineage):
        """
        Reshape the lineage information if needed
        """
        # TODO does this need to be a function?
        self.lineage = lineage
