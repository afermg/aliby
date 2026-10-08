"""LineageProcess subclass of PostProcessABC that supplies lineage information."""

import typing as t
from abc import abstractmethod

import h5py
import numpy as np
import pandas as pd

from agora.abc import ParametersABC
from postprocessor.core.abc import PostProcessABC


class LineageProcessParameters(ParametersABC):
    """No parameters required."""

    _defaults = {}


class LineageProcess(PostProcessABC):
    """
    To analyse lineage data.

    Extracts lineage information from a Signal or Cells object.
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
        """Implement method required by PostProcessABC - undefined."""
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
        Override PostProcesABC.as_function method.

        Lineage functions require lineage information to be run as functions.
        """
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(
            data, lineage=lineage, *extra_data
        )

    def get_lineage_information(self, signal=None, merged=True):
        """
        Get lineage as an array with tile IDs, mother and bud labels.

        The lineage is taken from the first of these that there is: a
        Signal with mother_label in its index, a lineage attribute, the
        h5 file of a Cells attribute, and that Cells' mother_assign.

        Parameters
        ----------
        signal: pd.DataFrame (optional)
            A Signal, whose index gives the lineage if it includes
            mother_label.
        merged: boolean
            If True, read the lineage after merging from an h5 file if
            there is one.

        Returns
        -------
        lineage: np.ndarray
            An array with columns (trap, mother_label, daughter_label).
        """
        if signal is not None and "mother_label" in signal.index.names:
            lineage = lineage_from_index(signal.index)
        elif hasattr(self, "lineage"):
            lineage = self.lineage
        elif getattr(self, "cells", None) is not None:
            lineage = None
            with h5py.File(self.cells.filename, "r") as f:
                if (lineage_loc := "modifiers/lineage_merged") in f and merged:
                    lineage = f.get(lineage_loc)[()]
                elif (lineage_loc := "modifiers/lineage") in f:
                    lineage = f.get(lineage_loc)[()]
            if lineage is None:
                lineage = self.cells.mothers_daughters
        else:
            raise AttributeError("No lineage information found")
        return lineage


def lineage_from_index(index: pd.MultiIndex) -> np.ndarray:
    """
    Get lineage from the index of a Signal that includes mother_label.

    A row of the index is a cell and names its mother, with zero for a
    cell that has none. A row of the lineage is a mother and one of her
    daughters, so the cell is the daughter and comes last.

    Parameters
    ----------
    index: pd.MultiIndex
        An index with levels trap, cell_label and mother_label.

    Returns
    -------
    lineage: np.ndarray
        An array with columns (trap, mother_label, daughter_label), with
        a row for each cell that has a mother.
    """
    lineage = np.array(
        [
            index.get_level_values(name)
            for name in ("trap", "mother_label", "cell_label")
        ],
        dtype=int,
    ).T
    # only cells with mothers
    lineage = lineage[lineage[:, 1] > 0]
    return np.unique(lineage, axis=0)
