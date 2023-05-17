# TODO Module docstring
import typing as t
from abc import abstractmethod

import h5py
import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from agora.utils.kymograph import get_index_as_np
from postprocessor.core.abc import PostProcessABC


class LineageProcessParameters(ParametersABC):
    """Parameters."""

    _defaults = {}


class LineageProcess(PostProcessABC):
    """
    To analyse lineage data.

    Currently bare bones, but extracts lineage information from a signal or a Cells object.
    """

    def __init__(self, parameters: LineageProcessParameters):
        """Initialise using PostProcessABC."""
        super().__init__(parameters)

    @abstractmethod
    def run(
        self,
        signal: pd.DataFrame,
        lineage: np.ndarray,
        *args,
    ):
        """Implement method required by PostProcessABC."""
        pass

    @classmethod
    def as_function(
        cls,
        data: pd.DataFrame,
        lineage: t.Union[t.Dict[t.Tuple[int], t.List[int]]] = None,
        *extra_data,
        **kwargs,
    ):
        """
        Override PostProcess.as_function classmethod.

        Lineage functions require lineage information to be passed if run as functions.
        """
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(
            data, lineage=lineage, *extra_data
        )

    def get_lineage_information(self, signal=None, merged=True):
        """Get lineage as an array with tile IDs, mother labels, and corresponding bud labels."""
        if signal is not None and "mother_label" in signal.index.names:
            # from kymograph
            lineage = get_index_as_np(signal)
        elif hasattr(self, "lineage"):
            lineage = self.lineage
        elif hasattr(self, "cells"):
            with h5py.File(self.cells.filename, "r") as f:
                if (lineage_loc := "modifiers/lineage_merged") in f and merged:
                    lineage = f.get(lineage_loc)[()]
                elif (lineage_loc := "modifiers/lineage)") in f:
                    lineage = f.get(lineage_loc)[()]
                elif self.cells is not None:
                    lineage = self.cells.mothers_daughters
        else:
            raise Exception("No lineage information found")
        return lineage
