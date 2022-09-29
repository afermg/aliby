from typing import Dict, Tuple

import numpy as np
import pandas as pd
from agora.utils.lineage import mb_array_to_dict

from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class bud_metricParameters(LineageProcessParameters):
    """
    Parameters
    """

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class bud_metric(LineageProcess):
    """
    Requires mother-bud information to create a new dataframe where the indices are mother ids and
    values are the daughters' values for a given signal.
    """

    def __init__(self, parameters: bud_metricParameters):
        super().__init__(parameters)

    def run(
        self,
        signal: pd.DataFrame,
        mother_bud_ids: Dict[pd.Index, Tuple[pd.Index]] = None,
    ):
        if mother_bud_ids is None:
            filtered_lineage = self.filter_signal_cells(signal)
            mother_bud_ids = mb_array_to_dict(filtered_lineage)

        return self.get_bud_metric(signal, mother_bud_ids)

    @staticmethod
    def get_bud_metric(signal: pd.DataFrame, md: Dict[Tuple, Tuple[Tuple]]):
        """

        signal: Daughter-inclusive dataframe
        md: Mother-daughters dictionary where key is mother's index and value a list of daugher indices

        Get fvi (First Valid Index) for all cells
        Create empty matrix
        for every mother:
         - Get daughters' subdataframe
         - sort  daughters by cell label
         - get series of fvis
         - concatenate the values of these ranges from the dataframe
        Fill the empty matrix
        Convert matrix into dataframe using mother indices

        """
        mothers_mat = np.zeros((len(md), signal.shape[1]))
        for i, daughters in enumerate(md.values()):
            dau_vals = signal.loc[set(daughters)].droplevel("trap")
            sorted_da_ids = dau_vals.sort_index(level="cell_label")
            tp_fvt = sorted_da_ids.apply(
                lambda x: x.last_valid_index(), axis=0
            )

            tp_fvt = sorted_da_ids.index.get_indexer(tp_fvt)
            tp_fvt[tp_fvt < 0] = sorted_da_ids.shape[0] - 1

            buds_metric = np.choose(tp_fvt, sorted_da_ids.values)
            # mothers_mat[i, tp_fvt[0] : tp_fvt[0] + len(buds_metric)] = buds_metric
            mothers_mat[i] = buds_metric

        df = pd.DataFrame(mothers_mat, index=md.keys(), columns=signal.columns)
        df.index.names = signal.index.names
        return df
