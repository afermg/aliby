import typing as t
from abc import abstractmethod

import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from agora.utils.lineage import group_matrix
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

    def filter_signal_cells(self, signal: pd.DataFrame):
        """
        Use casting to filter cell ids in signal and lineage
        """

        sig_ind = np.array(list(signal.index)).T[:, None, :]
        mo_av = (
            (self.lineage[:, :2].T[:, :, None] == sig_ind)
            .all(axis=0)
            .any(axis=1)
        )
        da_av = (
            (self.lineage[:, [0, 2]].T[:, :, None] == sig_ind)
            .all(axis=0)
            .any(axis=1)
        )

        return self.lineage[mo_av & da_av]

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        mother_bud_ids: t.Dict[t.Tuple[int], t.Collection[int]],
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
        if isinstance(lineage, np.ndarray):
            lineage = group_matrix(lineage, n_keys=2)

        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(
            data, mother_bud_ids=lineage, *extra_data
        )
        # super().as_function(data, *extra_data, lineage=lineage, **kwargs)
