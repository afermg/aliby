import typing as t

import numpy as np
import pandas as pd

from agora.io.bridge import groupsort
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)
import logging


def mother_bud_array_to_dict(mb_array: np.ndarray):
    """
    Convert a lineage into a dict of lists.

    A lineage is an array (trap, mother_id, daughter_id) and
    becomes a dictionary of lists (mother_id->[daughters_ids])
    """
    return {
        (trap, mo): [(trap, d[0]) for d in daughters]
        for trap, mo_da in groupsort(mb_array).items()
        for mo, daughters in groupsort(mo_da).items()
    }


class BudMetricParameters(LineageProcessParameters):
    """Give default location of lineage information."""

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class BudMetric(LineageProcess):
    """
    Create a dataframe with indices mother IDs and values from buds.

    Requires mother-bud information.
    """

    def __init__(self, parameters: BudMetricParameters):
        """Initialise using LineageProcess."""
        super().__init__(parameters)

    def run(
        self,
        signal: pd.DataFrame,
        lineage: t.Dict[pd.Index, t.Tuple[pd.Index]] = None,
    ):
        """Calculate a metric for all buds."""
        if lineage is None:
            # define lineage
            if hasattr(self, "lineage"):
                lineage = self.lineage
            else:
                # lineage information in the Signal data frame
                assert "mother_label" in signal.index.names
                lineage = signal.index.to_list()
        return self.get_bud_metric(signal, mother_bud_array_to_dict(lineage))

    @staticmethod
    def get_bud_metric(
        signal: pd.DataFrame,
        lineage_dict: t.Dict[t.Tuple, t.Tuple[t.Tuple]] = None,
    ):
        """
        Generate a dataframe of a Signal for buds.

        The data frame is indexed by the buds' mothers and concatenates
        data from all the buds for each mother.

        Parameters
        ---------
        signal: pd.Dataframe
            A dataframe that includes data for both mothers and daughters.
        md: dict
            A dict of lineage information with each key a mother's index,
            defined as (trap, cell_label), and the corresponding values are a
            list of daughter indices, also defined as (trap, cell_label).
        """
        md_index = signal.index
        # md_index should only comprise (trap, cell_label)
        if "mother_label" not in md_index.names:
            # dict with daughter indices as keys and mother indices as values
            bud_dict = {
                bud: mother
                for mother, buds in lineage_dict.items()
                for bud in buds
            }
            # generate mother_label in Signal using the mother's cell_label
            # cells with no mothers have a mother_label of 0
            signal["mother_label"] = list(
                map(lambda x: bud_dict.get(x, [0])[-1], signal.index)
            )
            signal.set_index("mother_label", append=True, inplace=True)
            # combine mothers and daughter indices
            mothers_index = lineage_dict.keys()
            daughters_index = [
                bud for buds in lineage_dict.values() for bud in buds
            ]
            relations = set([*mothers_index, *daughters_index])
            # keep only cells that are mother or daughters
            md_index = md_index.intersection(relations)
        else:
            md_index = md_index.droplevel("mother_label")
        if len(md_index) < len(signal):
            logging.getLogger("aliby").log(
                logging.WARNING,
                f"Dropped {len(signal) - len(md_index)} cells before "
                "applying bud_metric.",
            )
        # restrict signal to the cells in md_index moving mother_label to do so
        signal = (
            signal.reset_index("mother_label")
            .loc(axis=0)[md_index]
            .set_index("mother_label", append=True)
        )
        # restrict to daughters: cells with a mother
        mother_labels = signal.index.get_level_values("mother_label")
        daughter_df = signal.loc[mother_labels > 0]
        # join data for daughters with the same mother
        output_df = daughter_df.groupby(["trap", "mother_label"]).apply(
            combine_daughter_tracks
        )
        output_df.columns = signal.columns
        # daughter data is indexed by mothers, which themselves have no mothers
        output_df["temp_mother_label"] = 0
        output_df.set_index("temp_mother_label", append=True, inplace=True)
        if len(output_df):
            output_df.index.names = signal.index.names
        return output_df


def combine_daughter_tracks(tracks: pd.DataFrame):
    """
    Combine multiple time series of daughter cells into one time series.

    Concatenate daughter values into one time series starting with the first
    daughter and replacing later values with the values from the next daughter,
    and so on.

    Parameters
    ----------
    tracks: a Signal
        Data for all daughters, which are distinguished by different cell_labels,
        for a particular trap and mother_label.
    """
    # sort by daughter IDs
    bud_df = tracks.sort_index(level="cell_label")
    # remove multi-index
    no_rows = len(bud_df)
    bud_df.index = range(no_rows)
    # find time point of first non-NaN data point of each row
    init_tps = [
        bud_df.iloc[irow].first_valid_index() for irow in range(no_rows)
    ]
    # sort so that earliest daughter is first
    sorted_rows = np.argsort(init_tps)
    init_tps = np.sort(init_tps)
    # combine data for all daughters
    combined_tracks = np.nan * np.ones(tracks.columns.size)
    for j, jrow in enumerate(sorted_rows):
        # over-write with next earliest daughter
        combined_tracks[bud_df.columns.get_loc(init_tps[j]) :] = (
            bud_df.iloc[jrow].loc[init_tps[j] :].values
        )
    return pd.Series(combined_tracks, index=tracks.columns)
