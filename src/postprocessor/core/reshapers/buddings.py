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
    """Parameter class to obtain budding events.

    Parameters
    ----------
    LineageProcessParameters : lineage_location
        Location of lineage matrix to be used for calculations.

    Examples
    --------
    FIXME: Add docs.

    """

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class buddings(LineageProcess):
    """
    Calculate buddings in a trap assuming one mother per trap
    returns a pandas series with the buddings.

    We define a budding event as the moment in which a bud was identified for
    the first time, even if the bud is not considered one until later
    in the experiment.
    """

    def __init__(self, parameters: buddingsParameters):
        super().__init__(parameters)

    def run(
        self, signal: pd.DataFrame, lineage: np.ndarray = None
    ) -> pd.DataFrame:
        lineage = lineage or self.lineage

        # Get time of first appearance for all cells
        fvi = signal.apply(lambda x: x.first_valid_index(), axis=1)

        # Select mother cells in a given dataset
        traps_mothers: t.Dict[tuple, list] = {
            tuple(mo): [] for mo in lineage[:, :2] if tuple(mo) in signal.index
        }
        for trap, mother, daughter in lineage:
            if (trap, mother) in traps_mothers.keys():
                traps_mothers[(trap, mother)].append(daughter)

        mothers = signal.loc[
            set(signal.index).intersection(traps_mothers.keys())
        ]
        # Create a new dataframe with dimensions (n_mother_cells * n_timepoints)
        buddings = pd.DataFrame(
            np.zeros((mothers.shape[0], signal.shape[1])).astype(bool),
            index=mothers.index,
            columns=signal.columns,
        )
        buddings.columns.names = ["timepoint"]

        # Fill the budding events
        for mother_id, daughters in traps_mothers.items():
            daughters_idx = set(
                fvi.loc[
                    fvi.index.intersection(
                        list(product((mother_id[0],), daughters))
                    )
                ].values
            ).difference({0})
            buddings.loc[
                mother_id,
                daughters_idx,
            ] = True

        return buddings
