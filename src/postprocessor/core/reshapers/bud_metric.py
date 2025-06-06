import typing as t

import numpy as np
import pandas as pd

from agora.utils.lineage import mb_array_to_dict
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class BudMetricParameters(LineageProcessParameters):
    """Give default location of lineage information."""

    _defaults = {"lineage_location": "postprocessing/lineage_merged"}


class BudMetric(LineageProcess):
    """
    Requires mother-bud information to create a new dataframe where the
    indices are mother ids and values are the daughters' values for a
    given signal.
    """

    def __init__(self, parameters: BudMetricParameters):
        super().__init__(parameters)

    def run(
        self,
        signal: pd.DataFrame,
        lineage: t.Dict[pd.Index, t.Tuple[pd.Index]] = None,
    ):
        if lineage is None:
            # define lineage
            if hasattr(self, "lineage"):
                lineage = self.lineage
            else:
                # lineage information in the Signal dataframe
                assert "mother_label" in signal.index.names
                lineage = signal.index.to_list()
        return self.get_bud_metric(signal, mb_array_to_dict(lineage))

    @staticmethod
    def get_bud_metric(
        signal: pd.DataFrame, md: t.Dict[t.Tuple, t.Tuple[t.Tuple]] = None
    ):
        """
        Generate a dataframe of a Signal for buds indexed by their mothers,
        concatenating data from all the buds for each mother.

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
            bud_dict = {v: k for k, values in md.items() for v in values}
            # generate mother_label in Signal using the mother's cell_label
            # cells with no mothers have a mother_label of 0
            signal["mother_label"] = list(
                map(lambda x: bud_dict.get(x, [0])[-1], signal.index)
            )
            signal.set_index("mother_label", append=True, inplace=True)
            # combine mothers and daughter indices
            mothers_index = md.keys()
            daughters_index = [y for x in md.values() for y in x]
            relations = set([*mothers_index, *daughters_index])
            # keep from md_index only cells that are mother or daughters
            md_index = md_index.intersection(relations)
        else:
            md_index = md_index.droplevel("mother_label")
        if len(md_index) < len(signal):
            print(
                f"Dropped {len(signal) - len(md_index)} cells before applying bud_metric"
            )  # TODO log
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


def _combine_daughter_tracks_original(tracks: pd.DataFrame):
    """
    Combine multiple time series of daughter cells into one time series.

    At any one time, a mother cell should have only one daughter.

    Two daughters are still sometimes present at the same time point, and we
    then choose the daughter that appears first.

    TODO We need to fix examples with more than one daughter at a time point.

    Parameters
    ----------
    tracks: a Signal
        Data for all daughters, which are distinguished by different cell_labels,
        for a particular trap and mother_label.
    """
    # sort by daughter IDs
    bud_df = tracks.sort_index(level="cell_label")
    # remove multi-index
    bud_df.index = range(len(bud_df))
    # find which row of sorted_df has the daughter for each time point
    tp_fvt: pd.Series = bud_df.apply(lambda x: x.first_valid_index(), axis=0)
    # combine data for all daughters
    combined_tracks = np.nan * np.ones(tracks.columns.size)
    for bud_row in np.unique(tp_fvt.dropna().values).astype(int):
        ilocs = np.where(tp_fvt.values == bud_row)[0]
        combined_tracks[ilocs] = bud_df.values[bud_row, ilocs]
    # TODO delete old version
    tp_fvt = bud_df.columns.get_indexer(tp_fvt)
    tp_fvt[tp_fvt == -1] = len(bud_df) - 1
    old = np.choose(tp_fvt, bud_df.values)
    assert (
        (combined_tracks == old) | (np.isnan(combined_tracks) & np.isnan(old))
    ).all(), "yikes"
    return pd.Series(combined_tracks, index=tracks.columns)
