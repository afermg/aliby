#!/usr/bin/env python3

from itertools import product

import numpy as np
import pandas as pd

from postprocessor.core.processes.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class birthsParameters(LineageProcessParameters):
    """Parameter class to obtain birth events.

    Parameters
    ----------
    LineageProcessParameters : lineage_location
        Location of lineage matrix to be used for calculations.

    Examples
    --------
    FIXME: Add docs.

    """

    def __init__(self, lineage_location: str):
        self.lineage_location = lineage_location

    @classmethod
    def default(cls):
        return cls.from_dict({"lineage_location": "postprocessing/lineage_merged"})


class births(LineageProcess):
    """
    Calculate births in a trap assuming one mother per trap

    returns a pandas series with the births
    """

    def __init__(self, parameters: birthsParameters):
        super().__init__(parameters)

    def load_lineage(self, lineage):
        self.lineage = lineage

    def run(self, signal: pd.DataFrame, lineage: np.ndarray = None) -> pd.DataFrame:
        if lineage is None:
            lineage = self.lineage

        get_mothers = lambda trap: lineage[:, 1][lineage[:, 0] == trap]
        # get_daughters = lambda trap: lineage[:, 2][lineage[:, 0] == trap]

        # birth_events = signal.groupby("trap").apply(lambda x: x.first_valid_index())
        fvi = signal.apply(lambda x: x.first_valid_index(), axis=1)

        traps_mothers = {
            tuple(mo): [] for mo in lineage[:, :2] if tuple(mo) in signal.index
        }
        for trap, mother, daughter in lineage:
            if (trap, mother) in traps_mothers.keys():
                traps_mothers[(trap, mother)].append(daughter)

        mothers = signal.loc[set(signal.index).intersection(traps_mothers.keys())]
        births = pd.DataFrame(
            np.zeros((mothers.shape[0], signal.shape[1])).astype(bool),
            index=mothers.index,
            columns=signal.columns,
        )
        for mother_id, daughters in traps_mothers.items():
            daughters_idx = set(
                fvi.loc[
                    fvi.index.intersection(list(product((mother_id[0],), daughters)))
                ].values
            ).difference({0})
            births.loc[
                mother_id,
                daughters_idx,
            ] = True

        return births
