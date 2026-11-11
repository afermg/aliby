#!/usr/bin/env python3

import numpy as np
import pandas as pd

from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class buddingsParameters(LineageProcessParameters):
    """No parameters required: the lineage is given to the process."""

    _defaults = {}


class buddings(LineageProcess):
    """
    Generate a dataframe of budding events.

    We assume one mother per trap.

    A bud may not be considered a bud until later in the experiment.
    """

    def __init__(self, parameters: buddingsParameters):
        """Initialise buddings."""
        super().__init__(parameters)

    def run(
        self, signal: pd.DataFrame, lineage: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Generate dataframe of budding events.

        Find daughters for those mothers in a Signal with lineage data.
        Create a dataframe indicating the time each daughter first appears.

        We use the data from Signal only to find when the daughters appear, by
        their first non-NaN value.
        """
        # lineage is (trap, mother, daughter)
        lineage = self.lineage if lineage is None else lineage
        lineage = np.asarray(lineage, dtype=int).reshape(-1, 3).tolist()
        # the row of each cell in the signal
        row_of = {index: row for row, index in enumerate(signal.index)}
        # the column of the first non-NaN value of the signal for every
        # cell; a cell with no value at any time point has none
        seen = signal.notna().to_numpy()
        first = np.where(seen.any(axis=1), seen.argmax(axis=1), -1)
        # select the mothers in the signal that have lineage data, in
        # the signal's order
        mother_rows = sorted(
            {
                row_of[(trap, mother)]
                for trap, mother, _daughter in lineage
                if (trap, mother) in row_of
            }
        )
        budding_row = {row: i for i, row in enumerate(mother_rows)}
        budded = np.zeros((len(mother_rows), signal.shape[1]), dtype=bool)
        # fill the budding events
        for trap, mother, daughter in lineage:
            if (trap, mother) in row_of and (trap, daughter) in row_of:
                appeared = first[row_of[(trap, daughter)]]
                # buds in the first image are not budding events; the
                # first image is the first column, which is not time
                # point zero if processing began later
                if appeared > 0:
                    budded[budding_row[row_of[(trap, mother)]], appeared] = (
                        True
                    )
        # a new dataframe with dimensions (n_mother_cells * n_tps)
        buddings = pd.DataFrame(
            budded,
            index=signal.index[mother_rows],
            columns=signal.columns,
        )
        buddings.columns.names = ["timepoint"]
        return buddings
