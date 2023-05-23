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
    """
    Parameter class to obtain budding events.

    Define the location of lineage information in the h5 file.

    """

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class buddings(LineageProcess):
    """
    Calculate buddings in a trap assuming one mother per trap.

    Return a pandas series with the buddings.

    We define a budding event as when a bud is first identified.

    This bud may not be considered a bud until later in the experiment.
    """

    def __init__(self, parameters: buddingsParameters):
        """Initialise buddings."""
        super().__init__(parameters)

    def run(
        self, signal: pd.DataFrame, lineage: np.ndarray = None
    ) -> pd.DataFrame:
        """TODO."""
        lineage = lineage or self.lineage
        # select traps and mother cells in a given signal
        traps_mothers: t.Dict[tuple, list] = {
            tuple(mo): [] for mo in lineage[:, :2] if tuple(mo) in signal.index
        }
        for trap, mother, daughter in lineage:
            if (trap, mother) in traps_mothers.keys():
                traps_mothers[(trap, mother)].append(daughter)
        mothers = signal.loc[
            set(signal.index).intersection(traps_mothers.keys())
        ]
        # create a new dataframe with dimensions (n_mother_cells * n_timepoints)
        buddings = pd.DataFrame(
            np.zeros((mothers.shape[0], signal.shape[1])).astype(bool),
            index=mothers.index,
            columns=signal.columns,
        )
        buddings.columns.names = ["timepoint"]
        # get time of first appearance for every cell using Pandas
        fvi = signal.apply(lambda x: x.first_valid_index(), axis=1)
        # fill the budding events
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
