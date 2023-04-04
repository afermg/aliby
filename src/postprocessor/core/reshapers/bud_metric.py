import typing as t
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from agora.utils.lineage import mb_array_to_dict
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class BudMetricParameters(LineageProcessParameters):
    """
    Parameters
    """

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class BudMetric(LineageProcess):
    """
    Requires mother-bud information to create a new dataframe where the indices are mother ids and
    values are the daughters' values for a given signal.
    """

    def __init__(self, parameters: BudMetricParameters):
        super().__init__(parameters)

    def run(
        self,
        signal: pd.DataFrame,
        lineage: Dict[pd.Index, Tuple[pd.Index]] = None,
    ):
        if lineage is None:
            if hasattr(self, "lineage"):
                lineage = self.lineage
            else:
                assert "mother_label" in signal.index.names
                lineage = signal.index.to_list()

        return self.get_bud_metric(signal, mb_array_to_dict(lineage))

    @staticmethod
    def get_bud_metric(
        signal: pd.DataFrame, md: Dict[Tuple, Tuple[Tuple]] = None
    ):
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
        cells_were_dropped = 0  # Flag determines if mothers (1), daughters (2) or both were missing (3)

        md_index = signal.index
        if (
            "mother_label" not in md_index.names
        ):  # Generate mother label from md dict if unavailable
            d = {v: k for k, values in md.items() for v in values}
            signal["mother_label"] = list(
                map(lambda x: d.get(x, [0])[-1], signal.index)
            )
            signal.set_index("mother_label", append=True, inplace=True)
            related_items = set(
                [*md.keys(), *[y for x in md.values() for y in x]]
            )
            md_index = md_index.intersection(related_items)
        elif "mother_label" in md_index.names:
            md_index = md_index.droplevel("mother_label")
        else:
            raise ("Unavailable relationship information")

        if len(md_index) < len(signal):
            print("Dropped cells before bud_metric")  # TODO log

        signal = (
            signal.reset_index("mother_label")
            .loc(axis=0)[md_index]
            .set_index("mother_label", append=True)
        )

        names = list(signal.index.names)
        del names[-2]

        output_df = (
            signal.loc[signal.index.get_level_values("mother_label") > 0]
            .groupby(names)
            .apply(lambda x: _combine_daughter_tracks(x))
        )
        output_df.columns = signal.columns
        output_df["padding_level"] = 0
        output_df.set_index("padding_level", append=True, inplace=True)

        if len(output_df):
            output_df.index.names = signal.index.names
        return output_df


def _combine_daughter_tracks(tracks: t.Collection[pd.Series]):
    """
    Combine multiple time series of cells into one, overwriting values
    prioritising the most recent entity.
    """
    sorted_da_ids = tracks.sort_index(level="cell_label")
    sorted_da_ids.index = range(len(sorted_da_ids))
    tp_fvt = sorted_da_ids.apply(lambda x: x.first_valid_index(), axis=0)
    tp_fvt = sorted_da_ids.columns.get_indexer(tp_fvt)
    tp_fvt[tp_fvt < 0] = len(sorted_da_ids) - 1

    _metric = np.choose(tp_fvt, sorted_da_ids.values)
    return pd.Series(_metric, index=tracks.columns)
