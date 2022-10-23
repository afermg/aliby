#!/usr/bin/env jupyter
import typing as t
from copy import copy

import pandas as pd


def add_index_levels(
    df: pd.DataFrame, additional_ids: t.Dict[str, pd.Series] = {}
) -> pd.DataFrame:
    new_df = copy(df)
    for k, srs in additional_ids.items():
        assert len(srs) == len(
            new_df
        ), f"Series and new_df must match; sizes {len(srs)} and {len(new_df)}"
        new_df[k] = srs
        new_df.set_index(k, inplace=True, append=True)
    return new_df
