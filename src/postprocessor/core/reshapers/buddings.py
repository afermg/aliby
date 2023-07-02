#!/usr/bin/env python3

import typing as t
from itertools import product

import numpy as np
import pandas as pd

from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class buddingsParameters(LineageProcessParameters):
    """Give the location of lineage information in the h5 file."""

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


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

        Find daughters for mothers in a Signal for which we have lineage data.
        Create a dataframe indicating the time each daughter first appears.

        We use the data from Signal only to find when the daughters appear, by
        their first non-NaN value.
        """
        # lineage is (trap, mother, daughter)
        lineage = lineage or self.lineage
        # select traps and mothers in the signal that have lineage data
        traps_mothers: t.Dict[tuple, list] = {
            tuple(trap_mo): []
            for trap_mo in lineage[:, :2]
            if tuple(trap_mo) in signal.index
        }
        # find daughters for these traps and mothers
        for trap, mother, daughter in lineage:
            if (trap, mother) in traps_mothers.keys():
                traps_mothers[(trap, mother)].append(daughter)
        # sub dataframe of signal for the selected mothers
        mothers = signal.loc[
            set(signal.index).intersection(traps_mothers.keys())
        ]
        # a new dataframe with dimensions (n_mother_cells * n_tps)
        buddings = pd.DataFrame(
            np.zeros((mothers.shape[0], signal.shape[1])).astype(bool),
            index=mothers.index,
            columns=signal.columns,
        )
        buddings.columns.names = ["timepoint"]
        # get time of first non-NaN value of signal for every mother using Pandas
        fvi = signal.apply(lambda x: x.first_valid_index(), axis=1)
        # fill the budding events
        for trap_mother_id, daughters in traps_mothers.items():
            times_of_bud_appearance = fvi.loc[
                fvi.index.intersection(
                    list(product((trap_mother_id[0],), daughters))
                )
            ].values
            # ignore zeros - ignore buds in first image
            daughters_idx = set(times_of_bud_appearance).difference({0})
            buddings.loc[trap_mother_id, daughters_idx] = True
        return buddings
