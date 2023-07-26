"""
Functions to process, filter, and merge tracks.

We call two tracks contiguous if they are adjacent in time: the
maximal time point of one is one time point less than the
minimal time point of the other.

A right track can have multiple potential left tracks. We must
pick the best.
"""

import typing as t
from copy import copy
from typing import List, Union

import more_itertools as mit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils_find_1st import cmp_larger, find_1st

from postprocessor.core.processes.savgol import non_uniform_savgol


def get_joinable(tracks, smooth=False, tol=0.1, window=5, degree=3) -> dict:
    """
    Get the pair of track (without repeats) that have a smaller error than the
    tolerance. If there is a track that can be assigned to two or more other
    ones, choose the one with lowest error.

    Parameters
    ----------
    tracks: (m x n) Signal
        A Signal, usually area, dataframe where rows are cell tracks and
        columns are time points.
    tol: float or int
        threshold of average (prediction error/std) necessary
        to consider two tracks the same. If float is fraction of first track,
        if int it is absolute units.
    window: int
        value of window used for savgol_filter
    degree: int
        value of polynomial degree passed to savgol_filter
    """
    # only consider time series with more than two non-NaN data points
    tracks = tracks.loc[tracks.notna().sum(axis=1) > 2]
    # get contiguous tracks
    if smooth:
        # specialise to tracks with growing cells and of long duration
        clean = clean_tracks(tracks, min_duration=window + 1, min_gr=0.9)
        contigs = clean.groupby(["trap"]).apply(get_contiguous_pairs)
    else:
        contigs = tracks.groupby(["trap"]).apply(get_contiguous_pairs)
    # remove traps with no contiguous tracks
    contigs = contigs.loc[contigs.apply(len) > 0]
    # flatten to (trap, cell_id) pairs
    flat = set([k for v in contigs.values for i in v for j in i for k in j])
    # make a data frame of contiguous tracks with the tracks as arrays
    if smooth:
        smoothed_tracks = clean.loc[flat].apply(
            lambda x: non_uniform_savgol(x.index, x.values, window, degree),
            axis=1,
        )
    else:
        smoothed_tracks = tracks.loc[flat].apply(
            lambda x: np.array(x.values), axis=1
        )
    # get the Signal values for neighbouring end points of contiguous tracks
    actual_edges = contigs.apply(lambda x: get_edge_values(x, smoothed_tracks))
    # get the predicted values
    predicted_edges = contigs.apply(
        lambda x: get_predicted_edge_values(x, smoothed_tracks, window)
    )
    # Prediction of pre and mean of post
    prediction_costs = predicted_edges.apply(get_dMetric_wrap, tol=tol)
    solutions = [
        solve_matrices_wrap(cost, edges, tol=tol)
        for (trap_id, cost), edges in zip(
            prediction_costs.items(), actual_edges
        )
    ]
    breakpoint()
    closest_pairs = pd.Series(
        solutions,
        index=edges_dMetric_pred.index,
    )
    # match local with global ids
    joinable_ids = [
        localid_to_idx(closest_pairs.loc[i], contigs.loc[i])
        for i in closest_pairs.index
    ]
    return [pair for pairset in joinable_ids for pair in pairset]


def load_test_dset():
    """Load development dataset to test functions."""
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
    """Get number of time points."""
    indices = np.where(track.notna())
    return np.max(indices) - np.min(indices)


def max_nonstop_ntps(track: pd.Series) -> int:
    nona_tracks = track.notna()
    consecutive_nonas_grouped = [
        len(list(x))
        for x in mit.consecutive_groups(np.flatnonzero(nona_tracks))
    ]
    return max(consecutive_nonas_grouped)


def get_avg_gr(track: pd.Series) -> float:
    """Get average growth rate for a track."""
    ntps = max_ntps(track)
    vals = track.dropna().values
    gr = (vals[-1] - vals[0]) / ntps
    return gr


def clean_tracks(
    tracks, min_duration: int = 15, min_gr: float = 1.0
) -> pd.DataFrame:
    """Remove small non-growing tracks and return the reduced data frame."""
    ntps = tracks.apply(max_ntps, axis=1)
    grs = tracks.apply(get_avg_gr, axis=1)
    growing_long_tracks = tracks.loc[(ntps >= min_duration) & (grs > min_gr)]
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

    We should get:

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


def get_edge_values(contigs_ids, smoothed_tracks):
    """Get Signal values for adjacent end points for each contiguous track."""
    values = [
        (
            [get_value(smoothed_tracks.loc[pre_id], -1) for pre_id in pre_ids],
            [
                get_value(smoothed_tracks.loc[post_id], 0)
                for post_id in post_ids
            ],
        )
        for pre_ids, post_ids in contigs_ids
    ]
    return values


def get_predicted_edge_values(contigs_ids, smoothed_tracks, window):
    """
    Find neighbouring values of two contiguous tracks.

    Predict the next value for the leftmost track using window values
    and find the mean of the initial window values of the rightmost
    track.
    """
    result = []
    for pre_ids, post_ids in contigs_ids:
        pre_res = []
        # left contiguous tracks
        for pre_id in pre_ids:
            # get last window values of a track
            y = get_values_i(smoothed_tracks.loc[pre_id], -window)
            # predict next value
            pre_res.append(
                np.poly1d(np.polyfit(range(len(y)), y, 1))(len(y) + 1),
            )
        # right contiguous tracks
        pos_res = [
            # mean value of initial window values of a track
            get_mean_value_i(smoothed_tracks.loc[post_id], window)
            for post_id in post_ids
        ]
        result.append([pre_res, pos_res])
    return result


def get_value(x, n):
    """Get value from an array ignoring NaN."""
    val = x[~np.isnan(x)][n] if len(x[~np.isnan(x)]) else np.nan
    return val


def get_mean_value_i(x, i):
    """Get track's mean Signal value from values either from or up to an index."""
    if not len(x[~np.isnan(x)]):
        return np.nan
    else:
        if i > 0:
            v = x[~np.isnan(x)][:i]
        else:
            v = x[~np.isnan(x)][i:]
        return np.nanmean(v)


def get_values_i(x, i):
    """Get track's Signal values either from or up to an index."""
    if not len(x[~np.isnan(x)]):
        return np.nan
    else:
        if i > 0:
            v = x[~np.isnan(x)][:i]
        else:
            v = x[~np.isnan(x)][i:]
        return v


def localid_to_idx(local_ids, contig_trap):
    """
    Fetch the original ids from a nested list with joinable local_ids.

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
    """Calculate dMetric on a list."""
    return [get_dMetric(*sublist, **kwargs) for sublist in lst]


def solve_matrices_wrap(dMetric: List, edges: List, **kwargs):
    """Calculate solve_matrices on a list."""
    return [
        solve_matrices(mat, edgeset, **kwargs)
        for mat, edgeset in zip(dMetric, edges)
    ]


def get_dMetric(pre: List[float], post: List[float], tol):
    """
    Calculate a cost matrix based on the difference between two Signal
    values.

    Parameters
    ----------
    pre: list of floats
        Values of the Signal for left contiguous tracks.
    post: list of floats
        Values of the Signal for right contiguous tracks.
    """
    if len(pre) > len(post):
        dMetric = np.abs(np.subtract.outer(post, pre))
    else:
        dMetric = np.abs(np.subtract.outer(pre, post))
    # replace NaNs with maximal cost values
    dMetric[np.isnan(dMetric)] = tol + 1 + np.nanmax(dMetric)
    return dMetric


def solve_matrices(cost: np.ndarray, edges: List, tol: Union[float, int] = 1):
    """
    Solve the distance matrices obtained in get_dMetric and/or merged from
    independent dMetric matrices.
    """
    ids = solve_matrix(cost)
    if len(ids[0]):
        pre, post = edges
        norm = (
            np.array(pre)[ids[len(pre) > len(post)]] if tol < 1 else 1
        )  # relative or absolute tol
        result = dMetric[ids] / norm
        ids = ids if len(pre) < len(post) else ids[::-1]
        return [idx for idx, res in zip(zip(*ids), result) if res <= tol]
    else:
        return []


def get_closest_pairs(
    pre: List[float], post: List[float], tol: Union[float, int] = 1
):
    """
    Calculate a cost matrix for the Hungarian algorithm to pick the best set of
    options.

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
    """Arrange indices to the cost matrix in order of increasing cost."""
    glob_is = []
    glob_js = []
    if (~np.isnan(dMetric)).any():
        lMetric = copy(dMetric)
        sortedMetric = sorted(lMetric[~np.isnan(lMetric)])
        while (~np.isnan(sortedMetric)).any():
            # indices of point with minimal cost
            i_s, j_s = np.where(lMetric == sortedMetric[0])
            i = i_s[0]
            j = j_s[0]
            # store this point
            glob_is.append(i)
            glob_js.append(j)
            # remove from lMetric
            lMetric[i, :] += np.nan
            lMetric[:, j] += np.nan
            sortedMetric = sorted(lMetric[~np.isnan(lMetric)])
    indices = (np.array(glob_is), np.array(glob_js))
    breakpoint()
    return indices


def plot_joinable(tracks, joinable_pairs):
    """Convenience plotting function for debugging."""
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
    Get all pair of contiguous track ids from a tracks data frame.

    For two tracks to be contiguous, they must be exactly adjacent.

    Parameters
    ----------
    tracks:  pd.Dataframe
        A dataframe where rows are cell tracks and columns are time
        points.
    """
    # TODO add support for skipping time points
    # find time points bounding tracks of non-NaN values
    mins, maxs = [
        tracks.notna().apply(np.where, axis=1).apply(fn)
        for fn in (np.min, np.max)
    ]
    # mins.name = "min_tpt"
    # maxs.name = "max_tpt"
    # df = pd.merge(mins, maxs, right_index=True, left_index=True)
    # df["duration"] = df.max_tpt - df.min_tpt
    #
    # flip so that time points become the index
    mins_d = mins.groupby(mins).apply(lambda x: x.index.tolist())
    maxs_d = maxs.groupby(maxs).apply(lambda x: x.index.tolist())
    # reduce minimal time point to make a right track overlap with a left track
    mins_d.index = mins_d.index - 1
    # find common end points
    common = sorted(set(mins_d.index).intersection(maxs_d.index), reverse=True)
    contigs = [(maxs_d[t], mins_d[t]) for t in common]
    return contigs
