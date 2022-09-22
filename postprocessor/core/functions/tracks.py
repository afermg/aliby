"""
Functions to process, filter and merge tracks.
"""

# from collections import Counter

import typing as t
from copy import copy
from typing import List, Union

import more_itertools as mit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils_find_1st import cmp_larger, find_1st

from postprocessor.core.processes.savgol import non_uniform_savgol


def load_test_dset():
    # Load development dataset to test functions
    return pd.DataFrame(
        {
            ("a", 1, 1): [2, 5, np.nan, 6, 8] + [np.nan] * 5,
            ("a", 1, 2): list(range(2, 12)),
            ("a", 1, 3): [np.nan] * 8 + [6, 7],
            ("a", 1, 4): [np.nan] * 5 + [9, 12, 10, 14, 18],
        },
        index=range(1, 11),
    ).T


def max_ntps(track: pd.Series) -> int:
    # Get number of timepoints
    indices = np.where(track.notna())
    return np.max(indices) - np.min(indices)


def max_nonstop_ntps(track: pd.Series) -> int:
    nona_tracks = track.notna()
    consecutive_nonas_grouped = [
        len(list(x))
        for x in mit.consecutive_groups(np.flatnonzero(nona_tracks))
    ]
    return max(consecutive_nonas_grouped)


def get_tracks_ntps(tracks: pd.DataFrame) -> pd.Series:
    return tracks.apply(max_ntps, axis=1)


def get_avg_gr(track: pd.Series) -> int:
    """
    Get average growth rate for a track.

    :param tracks: Series with volume and timepoints as indices
    """
    ntps = max_ntps(track)
    vals = track.dropna().values
    gr = (vals[-1] - vals[0]) / ntps
    return gr


def get_avg_grs(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Get average growth rate for a group of tracks

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    """
    return tracks.apply(get_avg_gr, axis=1)


def clean_tracks(
    tracks, min_len: int = 15, min_gr: float = 1.0
) -> pd.DataFrame:
    """
    Clean small non-growing tracks and return the reduced dataframe

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param min_len: int number of timepoints cells must have not to be removed
    :param min_gr: float Minimum mean growth rate to assume an outline is growing
    """
    ntps = get_tracks_ntps(tracks)
    grs = get_avg_grs(tracks)

    growing_long_tracks = tracks.loc[(ntps >= min_len) & (grs > min_gr)]

    return growing_long_tracks


def merge_tracks(
    tracks, drop=False, **kwargs
) -> t.Tuple[pd.DataFrame, t.Collection]:
    """
    Join tracks that are contiguous and within a volume threshold of each other

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param kwargs: args passed to get_joinable

    returns

    :joint_tracks: (m x n) Dataframe where rows are cell tracks and
        columns are timepoints. Merged tracks are still present but filled
        with np.nans.
    """

    # calculate tracks that can be merged until no more traps can be merged
    joinable_pairs = get_joinable(tracks, **kwargs)
    if joinable_pairs:
        tracks = join_tracks(tracks, joinable_pairs, drop=drop)

    return (tracks, joinable_pairs)


def get_joint_ids(merging_seqs) -> dict:
    """
    Convert a series of merges into a dictionary where
    the key is the cell_id of destination and the value a list
    of the other track ids that were merged into the key

    :param merging_seqs: list of tuples of indices indicating the
    sequence of merging events. It is important for this to be in sequential order

    How it works:

    The order of merging matters for naming, always the leftmost track will keep the id

    For example, having tracks (a, b, c, d) and the iterations of merge events:

    0 a b c d
    1 a b cd
    2 ab cd
    3 abcd

    We shold get:

    output {a:a, b:a, c:a, d:a}

    """
    if not merging_seqs:
        return {}

    targets, origins = list(zip(*merging_seqs))
    static_tracks = set(targets).difference(origins)

    joint = {track_id: track_id for track_id in static_tracks}
    for target, origin in merging_seqs:
        joint[origin] = target

    moved_target = [
        k for k, v in joint.items() if joint[v] != v and v in joint.values()
    ]

    for orig in moved_target:
        joint[orig] = rec_bottom(joint, orig)

    return {
        k: v for k, v in joint.items() if k != v
    }  # remove ids that point to themselves


def rec_bottom(d, k):
    if d[k] == k:
        return k
    else:
        return rec_bottom(d, d[k])


def join_tracks(tracks, joinable_pairs, drop=True) -> pd.DataFrame:
    """
    Join pairs of tracks from later tps towards the start.

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints

    returns (copy)

    :param joint_tracks: (m x n) Dataframe where rows are cell tracks and
        columns are timepoints. Merged tracks are still present but filled
        with np.nans.
    :param drop: bool indicating whether or not to drop moved rows

    """

    tmp = copy(tracks)
    for target, source in joinable_pairs:
        tmp.loc[target] = join_track_pair(tmp.loc[target], tmp.loc[source])

        if drop:
            tmp = tmp.drop(source)

    return tmp


def join_track_pair(target, source):
    tgt_copy = copy(target)
    end = find_1st(target.values[::-1], 0, cmp_larger)
    tgt_copy.iloc[-end:] = source.iloc[-end:].values
    return tgt_copy


def get_joinable(tracks, smooth=False, tol=0.1, window=5, degree=3) -> dict:
    """
    Get the pair of track (without repeats) that have a smaller error than the
    tolerance. If there is a track that can be assigned to two or more other
    ones, it chooses the one with a lowest error.

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param tol: float or int threshold of average (prediction error/std) necessary
        to consider two tracks the same. If float is fraction of first track,
        if int it is absolute units.
    :param window: int value of window used for savgol_filter
    :param degree: int value of polynomial degree passed to savgol_filter

    """
    tracks = tracks.loc[tracks.notna().sum(axis=1) > 2]

    # Commented because we are not smoothing in this step yet
    # candict = {k:v for d in contig.values for k,v in d.items()}

    # smooth all relevant tracks

    if smooth:  # Apply savgol filter TODO fix nans affecting edge placing
        clean = clean_tracks(
            tracks, min_len=window + 1, min_gr=0.9
        )  # get useful tracks

        def savgol_on_srs(x):
            return non_uniform_savgol(x.index, x.values, window, degree)

        contig = clean.groupby(["trap"]).apply(get_contiguous_pairs)
        contig = contig.loc[contig.apply(len) > 0]
        flat = set([k for v in contig.values for i in v for j in i for k in j])
        smoothed_tracks = clean.loc[flat].apply(savgol_on_srs, 1)
    else:
        contig = tracks.groupby(["trap"]).apply(get_contiguous_pairs)
        contig = contig.loc[contig.apply(len) > 0]
        flat = set([k for v in contig.values for i in v for j in i for k in j])
        smoothed_tracks = tracks.loc[flat].apply(
            lambda x: np.array(x.values), axis=1
        )

    # fetch edges from ids TODO (IF necessary, here we can compare growth rates)
    def idx_to_edge(preposts):
        return [
            (
                [get_val(smoothed_tracks.loc[pre], -1) for pre in pres],
                [get_val(smoothed_tracks.loc[post], 0) for post in posts],
            )
            for pres, posts in preposts
        ]

    # idx_to_means = lambda preposts: [
    #     (
    #         [get_means(smoothed_tracks.loc[pre], -window) for pre in pres],
    #         [get_means(smoothed_tracks.loc[post], window) for post in posts],
    #     )
    #     for pres, posts in preposts
    # ]

    def idx_to_pred(preposts):
        result = []
        for pres, posts in preposts:
            pre_res = []
            for pre in pres:
                y = get_last_i(smoothed_tracks.loc[pre], -window)
                pre_res.append(
                    np.poly1d(np.polyfit(range(len(y)), y, 1))(len(y) + 1),
                )
            pos_res = [
                get_means(smoothed_tracks.loc[post], window) for post in posts
            ]
            result.append([pre_res, pos_res])

        return result

    edges = contig.apply(idx_to_edge)  # Raw edges
    # edges_mean = contig.apply(idx_to_means)  # Mean of both
    pre_pred = contig.apply(idx_to_pred)  # Prediction of pre and mean of post

    # edges_dMetric = edges.apply(get_dMetric_wrap, tol=tol)
    # edges_dMetric_mean = edges_mean.apply(get_dMetric_wrap, tol=tol)
    edges_dMetric_pred = pre_pred.apply(get_dMetric_wrap, tol=tol)

    # combined_dMetric = pd.Series(
    #     [
    #         [np.nanmin((a, b), axis=0) for a, b in zip(x, y)]
    #         for x, y in zip(edges_dMetric, edges_dMetric_mean)
    #     ],
    #     index=edges_dMetric.index,
    # )
    # closest_pairs = combined_dMetric.apply(get_vec_closest_pairs, tol=tol)
    solutions = []
    # for (i, dMetrics), edgeset in zip(combined_dMetric.items(), edges):
    for (i, dMetrics), edgeset in zip(edges_dMetric_pred.items(), edges):
        solutions.append(solve_matrices_wrap(dMetrics, edgeset, tol=tol))

    closest_pairs = pd.Series(
        solutions,
        index=edges_dMetric_pred.index,
    )

    # match local with global ids
    joinable_ids = [
        localid_to_idx(closest_pairs.loc[i], contig.loc[i])
        for i in closest_pairs.index
    ]

    return [pair for pairset in joinable_ids for pair in pairset]


def get_val(x, n):
    return x[~np.isnan(x)][n] if len(x[~np.isnan(x)]) else np.nan


def get_means(x, i):
    if not len(x[~np.isnan(x)]):
        return np.nan
    if i > 0:
        v = x[~np.isnan(x)][:i]
    else:
        v = x[~np.isnan(x)][i:]

    return np.nanmean(v)


def get_last_i(x, i):
    if not len(x[~np.isnan(x)]):
        return np.nan
    if i > 0:
        v = x[~np.isnan(x)][:i]
    else:
        v = x[~np.isnan(x)][i:]

    return v


def localid_to_idx(local_ids, contig_trap):
    """Fetch then original ids from a nested list with joinable local_ids

    input
    :param local_ids: list of list of pairs with cell ids to be joint
    :param local_ids: list of list of pairs with corresponding cell ids

    return
    list of pairs with (experiment-level) ids to be joint
    """
    lin_pairs = []
    for i, pairs in enumerate(local_ids):
        if len(pairs):
            for left, right in pairs:
                lin_pairs.append(
                    (contig_trap[i][0][left], contig_trap[i][1][right])
                )
    return lin_pairs


def get_vec_closest_pairs(lst: List, **kwargs):
    return [get_closest_pairs(*sublist, **kwargs) for sublist in lst]


def get_dMetric_wrap(lst: List, **kwargs):
    return [get_dMetric(*sublist, **kwargs) for sublist in lst]


def solve_matrices_wrap(dMetric: List, edges: List, **kwargs):
    return [
        solve_matrices(mat, edgeset, **kwargs)
        for mat, edgeset in zip(dMetric, edges)
    ]


def get_dMetric(
    pre: List[float], post: List[float], tol: Union[float, int] = 1
):
    """Calculate a cost matrix

    input
    :param pre: list of floats with edges on left
    :param post: list of floats with edges on right
    :param tol: int or float if int metrics of tolerance, if float fraction

    returns
    :: list of indices corresponding to the best solutions for matrices

    """
    if len(pre) > len(post):
        dMetric = np.abs(np.subtract.outer(post, pre))
    else:
        dMetric = np.abs(np.subtract.outer(pre, post))

    dMetric[np.isnan(dMetric)] = (
        tol + 1 + np.nanmax(dMetric)
    )  # nans will be filtered
    return dMetric


def solve_matrices(
    dMetric: np.ndarray, prepost: List, tol: Union[float, int] = 1
):
    """
    Solve the distance matrices obtained in get_dMetric and/or merged from
    independent dMetric matrices
    """

    ids = solve_matrix(dMetric)
    if not len(ids[0]):
        return []
    pre, post = prepost

    norm = (
        np.array(pre)[ids[len(pre) > len(post)]] if tol < 1 else 1
    )  # relative or absolute tol
    result = dMetric[ids] / norm
    ids = ids if len(pre) < len(post) else ids[::-1]

    return [idx for idx, res in zip(zip(*ids), result) if res <= tol]


def get_closest_pairs(
    pre: List[float], post: List[float], tol: Union[float, int] = 1
):
    """Calculate a cost matrix the Hungarian algorithm to pick the best set of
    options

    input
    :param pre: list of floats with edges on left
    :param post: list of floats with edges on right
    :param tol: int or float if int metrics of tolerance, if float fraction

    returns
    :: list of indices corresponding to the best solutions for matrices

    """
    dMetric = get_dMetric(pre, post, tol)

    return solve_matrices(dMetric, pre, post, tol)


def solve_matrix(dMetric):
    """
    Solve cost matrix focusing on getting the smallest cost at each iteration.

    input
    :param dMetric: np.array cost matrix

    returns
    tuple of np.arrays indicating picks with lowest individual value
    """
    glob_is = []
    glob_js = []
    if (~np.isnan(dMetric)).any():
        tmp = copy(dMetric)
        std = sorted(tmp[~np.isnan(tmp)])
        while (~np.isnan(std)).any():
            v = std[0]
            i_s, j_s = np.where(tmp == v)
            i = i_s[0]
            j = j_s[0]
            tmp[i, :] += np.nan
            tmp[:, j] += np.nan
            glob_is.append(i)
            glob_js.append(j)

            std = sorted(tmp[~np.isnan(tmp)])

    return (np.array(glob_is), np.array(glob_js))


def plot_joinable(tracks, joinable_pairs):
    """
    Convenience plotting function for debugging and data vis
    """

    nx = 8
    ny = 8
    _, axes = plt.subplots(nx, ny)
    for i in range(nx):
        for j in range(ny):
            if i * ny + j < len(joinable_pairs):
                ax = axes[i, j]
                pre, post = joinable_pairs[i * ny + j]
                pre_srs = tracks.loc[pre].dropna()
                post_srs = tracks.loc[post].dropna()
                ax.plot(pre_srs.index, pre_srs.values, "b")
                # try:
                #     totrange = np.arange(pre_srs.index[0],post_srs.index[-1])
                #     ax.plot(totrange, interpolate(pre_srs, totrange), 'r-')
                # except:
                #     pass
                ax.plot(post_srs.index, post_srs.values, "g")

    plt.show()


def get_contiguous_pairs(tracks: pd.DataFrame) -> list:
    """
    Get all pair of contiguous track ids from a tracks dataframe.

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param min_dgr: float minimum difference in growth rate from
        the interpolation
    """

    mins, maxes = [
        tracks.notna().apply(np.where, axis=1).apply(fn)
        for fn in (np.min, np.max)
    ]

    mins_d = mins.groupby(mins).apply(lambda x: x.index.tolist())
    mins_d.index = mins_d.index - 1  # make indices equal
    # TODO add support for skipping time points
    maxes_d = maxes.groupby(maxes).apply(lambda x: x.index.tolist())

    common = sorted(
        set(mins_d.index).intersection(maxes_d.index), reverse=True
    )

    return [(maxes_d[t], mins_d[t]) for t in common]


# def fit_track(track: pd.Series, obj=None):
#     if obj is None:
#         obj = objective

#     x = track.dropna().index
#     y = track.dropna().values
#     popt, _ = curve_fit(obj, x, y)

#     return popt

# def interpolate(track, xs) -> list:
#     '''
#     Interpolate next timepoint from a track

#     :param track: pd.Series of volume growth over a time period
#     :param t: int timepoint to interpolate
#     '''
#     popt = fit_track(track)
#     # perr = np.sqrt(np.diag(pcov))
#     return objective(np.array(xs), *popt)


# def objective(x,a,b,c,d) -> float:
#     # return (a)/(1+b*np.exp(c*x))+d
#     return (((x+d)*a)/((x+d)+b))+c

# def cand_pairs_to_dict(candidates):
#     d={x:[] for x,_ in candidates}
#     for x,y in candidates:
#         d[x].append(y)
#     return d
