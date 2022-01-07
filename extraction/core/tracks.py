'''
Functions to process, filter and merge tracks.
'''

# from collections import Counter

from copy import copy
from typing import Union, List

import numpy as np
import pandas as pd

from scipy.signal   import savgol_filter
# from scipy.optimize import linear_sum_assignment
# from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

def load_test_dset():
    # Load development dataset to test functions
    return pd.DataFrame({('a',1,1):[2, 5, np.nan, 6,8] + [np.nan] * 5,
                         ('a',1,2):list(range(2,12)),
                         ('a',1,3):[np.nan] * 8 + [6,7],
                         ('a',1,4):[np.nan] * 5 + [9,12,10,14,18]},
                        index=range(1,11)).T

def get_ntps(track:pd.Series) -> int:
    # Get number of timepoints
    indices = np.where(track.notna())
    return np.max(indices) - np.min(indices)


def get_tracks_ntps(tracks:pd.DataFrame) -> pd.Series:
    return tracks.apply(get_ntps, axis=1)

def get_avg_gr(track:pd.Series) -> int:
    '''
    Get average growth rate for a track.

    :param tracks: Series with volume and timepoints as indices
    '''
    ntps = get_ntps(track)
    vals = track.dropna().values
    gr = (vals[-1] - vals[0] )/ ntps
    return gr


def get_avg_grs(tracks:pd.DataFrame) -> pd.DataFrame:
    '''
    Get average growth rate for a group of tracks

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    '''
    return tracks.apply(get_avg_gr, axis=1)


def clean_tracks(tracks, min_len:int=6, min_gr:float=0.5) -> pd.DataFrame:
    '''
    Clean small non-growing tracks and return the reduced dataframe

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param min_len: int number of timepoints cells must have not to be removed
    :param min_gr: float Minimum mean growth rate to assume an outline is growing
    '''
    ntps = get_tracks_ntps(tracks)
    grs = get_avg_grs(tracks)

    growing_long_tracks = tracks.loc[(ntps >= min_len) & (grs > min_gr)]

    return (growing_long_tracks)

def merge_tracks(tracks, drop=False, **kwargs) -> pd.DataFrame:
    '''
    Join tracks that are contiguous and within a volume threshold of each other

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param kwargs: args passed to get_joinable

    returns

    :joint_tracks: (m x n) Dataframe where rows are cell tracks and
        columns are timepoints. Merged tracks are still present but filled
        with np.nans.
    '''

    # calculate tracks that can be merged until no more traps can be merged
    joinable_pairs = get_joinable(tracks, **kwargs)
    if joinable_pairs:
        tracks = join_tracks(tracks, joinable_pairs, drop=drop)
    joint_ids = get_joint_ids(joinable_pairs)

    return (tracks, joint_ids)


def get_joint_ids(merging_seqs) -> dict:
    '''
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

    '''
    targets, origins = list(zip(*merging_seqs))
    static_tracks = set(targets).difference(origins)

    joint = {track_id: track_id for track_id in static_tracks}
    for target, origin in merging_seqs:
        joint[origin] = target

    moved_target = [k for k,v in joint.items() \
                      if joint[v]!=v and v in joint.values()]

    for orig in moved_target:
        joint[orig] = rec_bottom(joint, orig)

    return {k:v for k,v in joint.items() if k!=v} # remove ids that point to themselves

def rec_bottom(d, k):
    if d[k] == k:
        return k
    else:
        return rec_bottom(d, d[k])

def join_tracks(tracks, joinable_pairs, drop=False) -> pd.DataFrame:
    '''
    Join pairs of tracks from later tps towards the start.

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints

    returns (copy)

    :param joint_tracks: (m x n) Dataframe where rows are cell tracks and
        columns are timepoints. Merged tracks are still present but filled
        with np.nans.
    :param drop: bool indicating whether or not to drop moved rows

    '''


    tmp = copy(tracks)
    for target, source in joinable_pairs:
        tmp.loc[target] = join_track_pairs(tmp.loc[target], tmp.loc[source])

        if drop:
            tmp = tmp.drop(source)

    return (tmp)

def join_track_pairs(track1, track2):
    tmp = copy(track1)
    tmp.loc[track2.dropna().index] = track2.dropna().values

    return tmp

def get_joinable(tracks, smooth=False, tol=0.1, window=5, degree=3) -> dict:
    '''
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

    '''

    tracks.index.names = ['pos', 'trap', 'cell'] #TODO remove this once it is integrated in the tracker
    # contig=tracks.groupby(['pos','trap']).apply(tracks2contig)
    clean = clean_tracks(tracks, min_len=window+1, min_gr = 0.9) # get useful tracks
    contig=clean.groupby(['pos','trap']).apply(get_contiguous_pairs)
    contig = contig.loc[contig.apply(len) > 0]
    # candict = {k:v for d in contig.values for k,v in d.items()}

    # smooth all relevant tracks

    linear=set([k for v in contig.values for i in v for j in i for k in j])
    if smooth: # Apply savgol filter TODO fix nans affecting edge placing
        savgol_on_srs = lambda x: non_uniform_savgol(x.index, x.values,
                                            window, degree)
        smoothed_tracks = clean.loc[linear].apply(savgol_on_srs,1)
    else:
        smoothed_tracks = clean.loc[linear].apply(lambda x: np.array(x.values), axis=1)

    # fetch edges from ids TODO (IF necessary, here we can compare growth rates)
    idx_to_edge = lambda preposts: [([get_val(smoothed_tracks.loc[pre],-1) for pre in pres],
                            [get_val(smoothed_tracks.loc[post],0) for post in posts])
                            for pres, posts in preposts]
    edges = contig.apply(idx_to_edge)

    closest_pairs = edges.apply(get_vec_closest_pairs, tol=tol)

    #match local with global ids
    joinable_ids = [localid_to_idx(closest_pairs.loc[i], contig.loc[i])\
                    for i in closest_pairs.index]

    return [pair for pairset in joinable_ids for pair in pairset]

get_val = lambda x, n: x[~np.isnan(x)][n] if len(x[~np.isnan(x)]) else np.nan


def localid_to_idx(local_ids, contig_trap):
    """Fetch then original ids from a nested list with joinable local_ids

    input
    :param local_ids: list of list of pairs with cell ids to be joint
    :param local_ids: list of list of pairs with corresponding cell ids

    return
    list of pairs with (experiment-level) ids to be joint
    """
    lin_pairs = []
    for i,pairs in enumerate(local_ids):
        if len(pairs):
            for left,right in pairs:
                lin_pairs.append((contig_trap[i][0][left],
                                  contig_trap[i][1][right]))
    return lin_pairs

def get_vec_closest_pairs(lst:List, **kwags):
    return [get_closest_pairs(*l, **kwags) for l in lst]

def get_closest_pairs(pre:List[float], post:List[float], tol:Union[float,int]=1):
    """Calculate a cost matrix the Hungarian algorithm to pick the best set of
    options

    input
    :param pre: list of floats with edges on left
    :param post: list of floats with edges on right
    :param tol: int or float if int metrics of tolerance, if float fraction

    returns
    :: list of indices corresponding to the best solutions for matrices

    """
    if len(pre) > len(post):
        dMetric = np.abs(np.subtract.outer(post,pre))
    else:
        dMetric = np.abs(np.subtract.outer(pre,post))
    # dMetric[np.isnan(dMetric)] = tol + 1 + np.nanmax(dMetric) # nans will be filtered
    # ids = linear_sum_assignment(dMetric)
    dMetric[np.isnan(dMetric)] = tol + 1 + np.nanmax(dMetric) # nans will be filtered

    ids = solve_matrix(dMetric)
    if not len(ids[0]):
        return []

    norm = np.array(pre)[ids[len(pre)>len(post)]] if tol<1 else 1 # relative or absolute tol
    result = dMetric[ids]/norm
    ids = ids if len(pre)<len(post) else ids[::-1]

    return [idx for idx,res in zip(zip(*ids), result) if res < tol]

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
        tmp = copy(dMetric )
        std = sorted(tmp[~np.isnan(tmp)])
        while (~np.isnan(std)).any():
            v = std[0]
            i_s,j_s = np.where(tmp==v)
            i = i_s[0]
            j = j_s[0]
            tmp[i,:]+= np.nan
            tmp[:,j]+= np.nan
            glob_is.append(i)
            glob_js.append(j)

            std = sorted(tmp[~np.isnan(tmp)])


    return (np.array( glob_is ), np.array( glob_js ))

def plot_joinable(tracks, joinable_pairs, max=64):
    """
    Convenience plotting function for debugging and data vis
    """

    nx=8
    ny=8
    _, axes = plt.subplots(nx,ny)
    for i in range(nx):
        for j in range(ny):
            if i*ny+j < len(joinable_pairs):
                ax = axes[i, j]
                pre, post = joinable_pairs[i * ny + j]
                pre_srs = tracks.loc[pre].dropna()
                post_srs = tracks.loc[post].dropna()
                ax.plot(pre_srs.index, pre_srs.values , 'b')
                # try:
                #     totrange = np.arange(pre_srs.index[0],post_srs.index[-1])
                #     ax.plot(totrange, interpolate(pre_srs, totrange), 'r-')
                # except:
                #     pass
                ax.plot(post_srs.index, post_srs.values, 'g')

    plt.show()

def get_contiguous_pairs(tracks: pd.DataFrame) -> list:
    '''
    Get all pair of contiguous track ids from a tracks dataframe.

    :param tracks: (m x n) dataframe where rows are cell tracks and
        columns are timepoints
    :param min_dgr: float minimum difference in growth rate from the interpolation
    '''
    # indices = np.where(tracks.notna())


    mins, maxes = [tracks.notna().apply(np.where, axis=1).apply(fn)
                   for fn in (np.min, np.max)]

    mins_d = mins.groupby(mins).apply(lambda x: x.index.tolist())
    mins_d.index = mins_d.index - 1 # make indices equal
    maxes_d = maxes.groupby(maxes).apply(lambda x: x.index.tolist())

    common = sorted(set(mins_d.index).intersection(maxes_d.index), reverse=True)

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


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    source: https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed
