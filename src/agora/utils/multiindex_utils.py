import typing as t
from copy import copy

import numpy as np
import pandas as pd


def drop_mother_label(index: pd.MultiIndex) -> np.ndarray:
    """Remove mother_label level from a MultiIndex."""
    no_mother_label = index
    if "mother_label" in index.names:
        no_mother_label = index.droplevel("mother_label")
    return np.array(no_mother_label.tolist())


def add_index_levels(
    df: pd.DataFrame, additional_ids: t.Dict[str, pd.Series] = {}
) -> pd.DataFrame:
    """Add index to multiindex dataframe, such as 'mother_label'."""
    new_df = copy(df)
    for k, srs in additional_ids.items():
        assert len(srs) == len(
            new_df
        ), f"Series and new_df must match; sizes {len(srs)} and {len(new_df)}"
        new_df[k] = srs
        new_df.set_index(k, inplace=True, append=True)
    return new_df
