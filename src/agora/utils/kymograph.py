#!/usr/bin/env jupyter
import typing as t
from copy import copy

import numpy as np
import pandas as pd

index_row = t.Tuple[str, str, int, int]


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


def drop_level(
    df: pd.DataFrame, name: str = "mother_label", as_list: bool = True
) -> t.Union[t.List[index_row], pd.Index]:
    """Drop index level

    Parameters
    ----------
    df : pd.DataFrame
        dataframe whose multiindex we will drop
    name : str
        name of index level to drop
    as_list : bool
        Whether to return as a list instead of an index

    Examples
    --------
    FIXME: Add docs.

    """
    short_index = df.index.droplevel(name)
    if as_list:
        short_index = short_index.to_list()
    return short_index


def intersection_matrix(
    index1: pd.MultiIndex, index2: pd.MultiIndex
) -> np.ndarray:
    """
    Use casting to obtain the boolean mask of the intersection of two multiindices
    """
    if not isinstance(index1, np.ndarray):
        index1 = np.array(index1.to_list())
    if not isinstance(index2, np.ndarray):
        index2 = np.array(index2.to_list())

    return (index1[..., None] == index2.T).all(axis=1)
