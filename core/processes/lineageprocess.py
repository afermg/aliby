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

    def run(
        self,
    ):
        pass

    def filter_signal_cells(self, signal: pd.DataFrame):
        """
        Use casting to filter cell ids in signal and lineage
        """

        sig_ind = np.array(list(signal.index)).T[:, None, :]
        mo_av = (self.lineage[:, :2].T[:, :, None] == sig_ind).all(axis=0).any(axis=1)
        da_av = (
            (self.lineage[:, [0, 2]].T[:, :, None] == sig_ind).all(axis=0).any(axis=1)
        )

        return self.lineage[mo_av & da_av]
