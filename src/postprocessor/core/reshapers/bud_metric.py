import typing as t

import numpy as np
import pandas as pd

from agora.io.bridge import groupsort
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)


class BudMetricParameters(LineageProcessParameters):
    """No parameters required: the lineage is given to the process."""

    _defaults = {}


class BudMetric(LineageProcess):
    """
    Create a data frame with indices mother IDs and values from buds.

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
                if "mother_label" not in signal.index.names:
                    raise ValueError("Missing 'mother_label'.")
                lineage = signal.index.to_list()
        result = get_bud_metric(signal, mother_bud_array_to_dict(lineage))
        return result


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
    # md_index should comprise only (trap, cell_label)
    if "mother_label" not in md_index.names:
        # dict with daughter indices as keys and mother indices as values
        bud_dict = {
            bud: mother
            for mother, buds in lineage_dict.items()
            for bud in buds
        }
        # generate mother_label in Signal using the mother's cell_label
        # cells with no mothers have a mother_label of 0
        # in a new data frame, so the Signal passed keeps its index
        mother_labels = pd.Index(
            [bud_dict.get(index, [0])[-1] for index in signal.index],
            name="mother_label",
        )
        signal = signal.set_index(mother_labels, append=True)
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
    # restrict signal to the cells in md_index moving mother_label
    md_signal = (
        signal.reset_index("mother_label")
        .loc(axis=0)[md_index]
        .set_index("mother_label", append=True)
    )
    # restrict to daughters: cells with a mother
    mother_labels = md_signal.index.get_level_values("mother_label")
    daughter_df = md_signal.loc[mother_labels > 0]
    if not len(daughter_df):
        # no buds: a Signal with no cells, not None
        return pd.DataFrame(
            columns=signal.columns,
            index=pd.MultiIndex.from_arrays(
                [[], []], names=("trap", "cell_label")
            ),
            dtype=float,
        )
    # join data for daughters with the same mother
    bud_signal = daughter_df.groupby(["trap", "mother_label"]).apply(
        combine_daughter_tracks
    )
    bud_signal.columns = md_signal.columns
    # daughter data is indexed by mothers, which themselves have no mothers
    bud_signal["temp_mother_label"] = 0
    bud_signal.set_index("temp_mother_label", append=True, inplace=True)
    bud_signal.index.names = md_signal.index.names
    bud_signal = bud_signal.droplevel("mother_label")
    return bud_signal


def combine_daughter_tracks(tracks: pd.DataFrame):
    """
    Combine multiple time series of daughter cells into one time series.

    Concatenate daughter values into one time series starting with the
    first daughter and replacing later values with the values from the
    next daughter, and so on.

    Parameters
    ----------
    tracks: a Signal
        Data for all daughters, which are distinguished by different
        cell_labels, for a particular trap and mother_label.
    """
    # sort by daughter IDs
    values = tracks.sort_index(level="cell_label").to_numpy(dtype=float)
    seen = ~np.isnan(values)
    # find the column of the first non-NaN data point of each row
    first = seen.argmax(axis=1)
    # a daughter with no value at any time point has nothing to add: it
    # had no first time point, which could not be sorted
    with_values = np.flatnonzero(seen.any(axis=1))
    # sort so that earliest daughter is first, and of two that start
    # together the one with the higher ID is last
    sorted_rows = with_values[np.argsort(first[with_values], kind="stable")]
    # combine data for all daughters
    combined_tracks = np.full(tracks.columns.size, np.nan)
    for row in sorted_rows:
        # over-write with next earliest daughter
        combined_tracks[first[row] :] = values[row, first[row] :]
    return pd.Series(combined_tracks, index=tracks.columns)
