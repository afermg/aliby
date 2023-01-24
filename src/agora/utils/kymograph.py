#!/usr/bin/env jupyter
import typing as t
from copy import copy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
    df: pd.DataFrame,
    name: str = "mother_label",
    as_list: bool = True,
) -> t.Union[t.List[index_row], pd.Index]:
    """
    Drop index level.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose multiindex we will drop
    name : str
        Mame of index level to drop
    as_list : bool
        Whether to return as a list instead of an index
    """
    short_index = df.index.droplevel(name)
    if as_list:
        short_index = short_index.to_list()
    return short_index


def intersection_matrix(
    index1: pd.MultiIndex, index2: pd.MultiIndex
) -> np.ndarray:
    """Use casting to obtain the boolean mask of the intersection of two multi-indices."""
    indices = [index1, index2]
    for i in range(2):
        if hasattr(indices[i], "to_list"):
            indices[i]: t.List = indices[i].to_list()
        indices[i]: np.ndarray = np.array(indices[i])
    return (indices[0][..., None] == indices[1].T).all(axis=1)


def get_mother_ilocs_from_daughters(df: pd.DataFrame) -> np.ndarray:
    """Fetch mother locations in the index of df for all daughters in df."""
    daughter_ids = df.index[df.index.get_level_values("mother_label") > 0]
    mother_ilocs = intersection_matrix(
        daughter_ids.droplevel("cell_label"),
        drop_level(df, "mother_label", as_list=False),
    ).any(axis=0)
    return mother_ilocs


def get_mothers_from_another_df(whole_df: pd.DataFrame, da_df: pd.DataFrame):
    daughter_ids = da_df.index[
        da_df.index.get_level_values("mother_label") > 0
    ]
    mother_ilocs = intersection_matrix(
        daughter_ids.droplevel("cell_label"),
        drop_level(whole_df, "mother_label", as_list=False),
    ).any(axis=0)
    return mother_ilocs


def bidirectional_retainment_filter(
    df: pd.DataFrame,
    mothers_thresh: float = 0.8,
    daughters_thresh: int = 7,
) -> pd.DataFrame:
    """
    Retrieve families where mothers are present for more than a fraction of the experiment, and daughters for longer than some number of time-points.

    Parameters
    ----------
    df: pd.DataFrame
        Data
    mothers_thresh: float
        Minimum fraction of experiment's total duration for which mothers must be present.
    daughters_thresh: int
        Minimum number of time points for which daughters must be observed
    """
    # daughters
    all_daughters = df.loc[df.index.get_level_values("mother_label") > 0]
    # keep daughters observed sufficiently often
    retained_daughters = all_daughters.loc[
        all_daughters.notna().sum(axis=1) > daughters_thresh
    ]
    # fetch mother using existing daughters
    mothers = df.loc[get_mothers_from_another_df(df, retained_daughters)]
    # keep mothers present for at least a fraction of the experiment's duration
    retained_mothers = mothers.loc[
        mothers.notna().sum(axis=1) > mothers.shape[1] * mothers_thresh
    ]
    # drop daughters with no valid mothers
    final_da_mask = intersection_matrix(
        drop_level(retained_daughters, "cell_label", as_list=False),
        drop_level(retained_mothers, "mother_label", as_list=False),
    )
    final_daughters = retained_daughters.loc[final_da_mask.any(axis=1)]
    # join mothers and daughters and sort index
    return pd.concat((final_daughters, retained_mothers), axis=0).sort_index()


def melt_reset(df: pd.DataFrame, additional_ids: t.Dict[str, pd.Series] = {}):
    new_df = add_index_levels(df, additional_ids)

    return new_df.melt(ignore_index=False).reset_index()


# Drop cells that if used would reduce info the most
def filt_cluster(
    kymograph: pd.DataFrame,
    n: int = 2,
):
    mask = ~kymograph.iloc[:, kymograph.shape[1] // 2 :].isna().any(axis=1)
    informative = kymograph.loc[mask]

    clusters = cluster_kymograph(informative, n)

    return informative, clusters


def cluster_kymograph(kymograph: pd.DataFrame, n: int = 2):
    import bottleneck as bn
    from sklearn.cluster import KMeans

    # Normalise according to mean value of signal
    X = (
        kymograph.divide(bn.nanmean(kymograph, axis=1), axis=0)
        .dropna(axis=1)
        .values
    )
    km = KMeans(n, random_state=42).fit(X)
    clusters = km.predict(X)
    return clusters


def split_df(df, slices):
    return [df.iloc(axis=1)[slc] for slc in slices]


def slices_from_spans(spans: t.Tuple[int], df: pd.DataFrame) -> t.List[slice]:
    cumsum = np.cumsum(spans)

    slices = [
        slice(start, min(end, df.columns.get_level_values("time")[-1]))
        for start, end in zip(cumsum[:-1], cumsum[1:])
    ]
    return slices
