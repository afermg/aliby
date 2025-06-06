"""
Functions to process, filter, and merge tracks.

We call two tracks contiguous if they are adjacent in time: the
maximal time point of one is one time point less than the
minimal time point of the other.

A right track can have multiple potential left tracks. We pick the best.
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


def get_merges(tracks, smooth=False, tol=0.2, window=5, degree=3) -> dict:
    """
    Find all pairs of tracks that should be joined.

    Each track is defined by (trap_id, cell_id).

    If there are multiple choices of which, say, left tracks to join to a
    right track, pick the best using the Signal values to do so.

    To score two tracks, we predict the future value of a left track and
    compare with the mean initial values of a right track.

    Parameters
    ----------
    tracks: pd.DataFrame
        A Signal, usually area, where rows are cell tracks and columns are
        time points.
    smooth: boolean
        If True, smooth tracks with a savgol_filter.
    tol: float < 1 or int
        If int, compare the absolute distance between predicted values
        for the left and right end points of two contiguous tracks.
        If float, compare the distance relative to the magnitude of the
        end point of the left track.
    window: int
        Length of window used for predictions and for any savgol_filter.
    degree: int
        The degree of the polynomial used by the savgol_filter.
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
    actual_edge_values = contigs.apply(
        lambda x: get_edge_values(x, smoothed_tracks)
    )
    # get the predicted values
    predicted_edge_values = contigs.apply(
        lambda x: get_predicted_edge_values(x, smoothed_tracks, window)
    )
    # score predicted edge values: low values are best
    prediction_scores = predicted_edge_values.apply(get_dMetric_wrap)
    # find contiguous tracks to join for each trap
    trap_contigs_to_join = []
    for idx in contigs.index:
        local_contigs = contigs.loc[idx]
        # find indices of best left and right tracks to join
        best_indices = find_best_from_scores_wrap(
            prediction_scores.loc[idx], actual_edge_values.loc[idx], tol=tol
        )
        # find tracks from the indices
        trap_contigs_to_join.append(
            [
                (contig[0][left], contig[1][right])
                for best_index, contig in zip(best_indices, local_contigs)
                for (left, right) in best_index
                if best_index
            ]
        )
    # return only the pairs of contiguous tracks
    contigs_to_join = [
        contigs
        for trap_tracks in trap_contigs_to_join
        for contigs in trap_tracks
    ]
    merges = np.array(contigs_to_join, dtype=int)
    return merges


def clean_tracks(
    tracks, min_duration: int = 15, min_gr: float = 1.0
) -> pd.DataFrame:
    """Remove small non-growing tracks and return the reduced data frame."""
    ntps = tracks.apply(max_ntps, axis=1)
    grs = tracks.apply(get_avg_gr, axis=1)
    growing_long_tracks = tracks.loc[(ntps >= min_duration) & (grs > min_gr)]
    return growing_long_tracks


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
    # flip so that time points become the index
    mins_d = mins.groupby(mins).apply(lambda x: x.index.tolist())
    maxs_d = maxs.groupby(maxs).apply(lambda x: x.index.tolist())
    # reduce minimal time point to make a right track overlap with a left track
    mins_d.index = mins_d.index - 1
    # find common end points
    common = sorted(set(mins_d.index).intersection(maxs_d.index), reverse=True)
    contigs = [(maxs_d[t], mins_d[t]) for t in common]
    return contigs


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


def get_dMetric_wrap(lst: List, **kwargs):
    """Calculate dMetric on a list."""
    return [get_dMetric(*sublist, **kwargs) for sublist in lst]


def get_dMetric(pre_values: List[float], post_values: List[float]):
    """
    Calculate a scoring matrix based on the difference between two Signal
    values.

    We generate one score per pair of contiguous tracks.

    Lower scores are better.

    Parameters
    ----------
    pre_values: list of floats
        Values of the Signal for left contiguous tracks.
    post_values: list of floats
        Values of the Signal for right contiguous tracks.
    """
    if len(pre_values) > len(post_values):
        dMetric = np.abs(np.subtract.outer(post_values, pre_values))
    else:
        dMetric = np.abs(np.subtract.outer(pre_values, post_values))
    # replace NaNs with maximal values
    dMetric[np.isnan(dMetric)] = 1 + np.nanmax(dMetric)
    return dMetric


def find_best_from_scores_wrap(dMetric: List, edges: List, **kwargs):
    """Calculate solve_matrices on a list."""
    return [
        find_best_from_scores(mat, edgeset, **kwargs)
        for mat, edgeset in zip(dMetric, edges)
    ]


def find_best_from_scores(
    scores: np.ndarray, actual_edge_values: List, tol: Union[float, int] = 1
):
    """Find indices for left and right contiguous tracks with scores below a tolerance."""
    ids = find_best_indices(scores)
    if len(ids[0]):
        pre_value, post_value = actual_edge_values
        # score with relative or absolute distance
        norm = (
            np.array(pre_value)[ids[len(pre_value) > len(post_value)]]
            if tol < 1
            else 1
        )
        best_scores = scores[ids] / norm
        ids = ids if len(pre_value) < len(post_value) else ids[::-1]
        # keep only indices with best_score less than the tolerance
        indices = [
            idx for idx, score in zip(zip(*ids), best_scores) if score <= tol
        ]
        return indices
    else:
        return []


def find_best_indices(dMetric):
    """Find indices for left and right contiguous tracks with minimal scores."""
    glob_is = []
    glob_js = []
    if (~np.isnan(dMetric)).any():
        lMetric = copy(dMetric)
        sortedMetric = sorted(lMetric[~np.isnan(lMetric)])
        while (~np.isnan(sortedMetric)).any():
            # indices of point with the lowest score
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
    return indices


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


def get_avg_gr(track: pd.Series) -> float:
    """Get average growth rate for a track."""
    ntps = max_ntps(track)
    vals = track.dropna().values
    gr = (vals[-1] - vals[0]) / ntps
    return gr


######################################################################


def get_joinable_original(
    tracks, smooth=False, tol=0.1, window=5, degree=3
) -> dict:
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
    actual_edge_values = contigs.apply(
        lambda x: get_edge_values(x, smoothed_tracks)
    )
    # get the predicted values
    predicted_edge_values = contigs.apply(
        lambda x: get_predicted_edge_values(x, smoothed_tracks, window)
    )
    # score predicted edge values
    prediction_scores = predicted_edge_values.apply(get_dMetric_wrap, tol=tol)

    solutions = [
        # for all sets of contigs at a trap
        find_best_from_scores_wrap(cost, edge_values, tol=tol)
        for (trap_id, cost), edge_values in zip(
            prediction_scores.items(), actual_edge_values
        )
    ]
    closest_pairs = pd.Series(
        solutions,
        index=prediction_scores.index,
    )
    # match local with global ids
    joinable_ids = [
        localid_to_idx(closest_pairs.loc[i], contigs.loc[i])
        for i in closest_pairs.index
    ]

    contigs_to_join = [
        contigs for trap_tracks in joinable_ids for contigs in trap_tracks
    ]
    return contigs_to_join


def load_test_dset():
    """Load test data set."""
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
    return find_best_from_scores(dMetric, pre, post, tol)


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
