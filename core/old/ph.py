from typing import Dict, List, Union
import re

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns

import compress_pickle

from postprocessor.core.postprocessor import PostProcessor
from postprocessor.core.tracks import get_avg_grs, non_uniform_savgol
from postprocessor.core.ph import *


def filter_by_gfp(dfs):
    gfps = pd.concat([t[("GFPFast_bgsub", np.maximum, "median")] for t in dfs])
    avgs_gfp = gfps.mean(axis=1)
    high_gfp = get_high_k2(avgs_gfp)
    # high_gfp = avgs_gfp[avgs_gfp > 200]

    return high_gfp


def filter_by_area(dfs, min=50):
    areas = pd.concat([t[("general", None, "area")] for t in dfs])
    avgs_areas = areas[(areas.notna().sum(axis=1) > areas.shape[1] // (1.25))].mean(
        axis=1
    )
    avgs_areas = avgs_areas[(avgs_areas > min)]

    return avgs_areas


def get_high_k2(df):
    kmeans = KMeans(n_clusters=2)
    vals = df.values.reshape(-1, 1)
    kmeans.fit(vals)
    high_clust_id = kmeans.cluster_centers_.argmax()

    return df.loc[kmeans.predict(vals) == high_clust_id]


def get_concats(dfs, keys):
    return pd.concat([t.get((keys), pd.DataFrame()) for t in dfs])


def get_dfs(pp):
    dfs = [pp.extraction[pp.expt.positions[i]] for i in range(len(pp.expt.positions))]
    return dfs


combine_dfs = lambda dfs: {k: get_concats(dfs, k) for k in dfs[0].keys()}


def merge_channels(pp, min_area=50):
    dfs = get_dfs(pp)
    # rats = get_concats(dfs, ("em_ratio_bgsub", np.maximum, "median"))
    # gfps = filter_by_gfp(dfs)

    avgs_area = filter_by_area(dfs, min=50)
    # ids = [x for x in set(gfps.index).intersection(avgs_area.index)]
    ids = avgs_area.index

    new_dfs = combine_dfs(dfs)

    h = pd.DataFrame(
        {
            k[0] + "_" + k[2]: v.loc[ids].mean(axis=1)
            for k, v in new_dfs.items()
            if k[-1] != "imBackground"
        }
    )
    return h


def process_phs(pp, min_area=200):
    h = merge_channels(pp, min_area)
    h.index.names = ["pos", "trap", "cell"]
    ids = h.index
    h = h.reset_index()

    h["ph"] = h["pos"].apply(lambda x: float(x[3:7].replace("_", ".")))
    h["max5_d_med"] = h["mCherry_max2p5pc"] / h["mCherry_median"]

    h = h.set_index(ids)
    h = h.drop(["pos", "trap", "cell"], axis=1)
    return h


def growth_rate(
    data: Series, alg=None, filt={"kind": "savgol", "window": 5, "degree": 3}
):
    window = filt["window"]
    degree = filt["degree"]
    if alg is None:
        alg = "standard"

    if filt:  # TODO add support for multiple algorithms
        data = Series(
            non_uniform_savgol(
                data.dropna().index, data.dropna().values, window, degree
            ),
            index=data.dropna().index,
        )

    return Series(np.convolve(data, diff_kernel, "same"), index=data.dropna().index)


# import numpy as np

# diff_kernel = np.array([1, -1])
# gr = clean.apply(growth_rate, axis=1)
# from postprocessor.core.tracks import non_uniform_savgol, clean_tracks


def sort_df(df, by="first", rev=True):
    nona = df.notna()
    if by == "len":
        idx = nona.sum(axis=1)
    elif by == "first":
        idx = nona.idxmax(axis=1)
    idx = idx.sort_values().index

    if rev:
        idx = idx[::-1]

    return df.loc[idx]


# test = tmp[("GFPFast", np.maximum, "median")]
# test2 = tmp[("pHluorin405", np.maximum, "median")]
# ph = test / test2
# ph = ph.stack().reset_index(1)
# ph.columns = ["tp", "fl"]


def m2p5_med(ext, ch, red=np.maximum):
    m2p5pc = ext[(ch, red, "max2p5pc")]
    med = ext[(ch, red, "median")]

    result = m2p5pc / med

    return result


def plot_avg(df):
    df = df.stack().reset_index(1)
    df.columns = ["tp", "val"]

    sns.relplot(x=df["tp"], y=df["val"], kind="line")
    plt.show()


def split_data(df: DataFrame, splits: List[int]):
    dfs = [df.iloc[:, i:j] for i, j in zip((0,) + splits, splits + (df.shape[1],))]
    return dfs


def growth_rate(
    data: Series, alg=None, filt={"kind": "savgol", "window": 7, "degree": 3}
):
    if alg is None:
        alg = "standard"

    if filt:  # TODO add support for multiple algorithms
        window = filt["window"]
        degree = filt["degree"]
        data = Series(
            non_uniform_savgol(
                data.dropna().index, data.dropna().values, window, degree
            ),
            index=data.dropna().index,
        )

    diff_kernel = np.array([1, -1])

    return Series(np.convolve(data, diff_kernel, "same"), index=data.dropna().index)


# pp = PostProcessor(source=19831)
# pp.load_tiler_cells()
# f = "/home/alan/Documents/sync_docs/libs/postproc/gluStarv_2_0_x2_dual_phl_ura8_00/extraction"
# pp.load_extraction(
#     "/home/alan/Documents/sync_docs/libs/postproc/postprocessor/"
#     + pp.expt.name
#     + "/extraction/"
# )
# tmp = pp.extraction["phl_ura8_002"]


def _check_bg(data):
    for k in list(pp.extraction.values())[0].keys():
        for p in pp.expt.positions:
            if k not in pp.extraction[p]:
                print(p, k)


# data = {
#     k: pd.concat([pp.extraction[pos][k] for pos in pp.expt.positions[:-3]])
#     for k in list(pp.extraction.values())[0].keys()
# }


def hmap(df, **kwargs):
    g = sns.heatmap(sort_df(df), robust=True, cmap="mako_r", **kwargs)
    plt.xlabel("")
    return g


# from random import randint
# x = randint(0, len(smooth))
# plt.plot(clean.iloc[x], 'b')
# plt.plot(smooth.iloc[x], 'r')
# plt.show()


# data = tmp
# df = data[("general", None, "area")]
# clean = clean_tracks(df, min_len=160)
# clean = clean.loc[clean.notna().sum(axis=1) > 9]
# gr = clean.apply(growth_rate, axis=1)
# splits = (72, 108, 180)
# gr_sp = split_data(gr, splits)

# idx = gr.index

# bg = get_bg(data)
# test = data[("GFPFast", np.maximum, "median")]
# test2 = data[("pHluorin405", np.maximum, "median")]
# ph = (test / test2).loc[idx]
# c = pd.concat((ph.mean(1), gr.max(1)), axis=1)
# c.columns = ["ph", "gr_max"]
# # ph = ph.stack().reset_index(1)
# # ph.columns = ['tp', 'fl']

# ph_sp = split_data(gr, splits)


def get_bg(data):
    bg = {}
    fl_subkeys = [
        x
        for x in data.keys()
        if x[0] in ["GFP", "GFPFast", "mCherry", "pHluorin405"]
        and x[-1] != "imBackground"
    ]
    for k in fl_subkeys:
        nk = list(k)
        bk = tuple(nk[:-1] + ["imBackground"])
        nk = tuple(nk[:-1] + [nk[-1] + "_BgSub"])
        tmp = []
        for i, v in data[bk].iterrows():
            if i in data[k].index:
                newdf = data[k].loc[i] / v
                newdf.index = pd.MultiIndex.from_tuples([(*i, c) for c in newdf.index])
            tmp.append(newdf)
        bg[nk] = pd.concat(tmp)

    return bg


def calc_ph(bg):
    fl_subkeys = [x for x in bg.keys() if x[0] in ["GFP", "GFPFast", "pHluorin405"]]
    chs = list(set([x[0] for x in fl_subkeys]))
    assert len(chs) == 2, "Too many channels"
    ch1 = [x[1:] for x in fl_subkeys if x[0] == chs[0]]
    ch2 = [x[1:] for x in fl_subkeys if x[0] == chs[1]]
    inter = list(set(ch1).intersection(ch2))
    ph = {}
    for red_fld in inter:
        ph[tuple(("ph",) + red_fld)] = (
            bg[tuple((chs[0],) + red_fld)] / bg[tuple((chs[1],) + red_fld)]
        )


def get_traps(pp):
    t0 = {}
    for pos in pp.tiler.positions:
        pp.tiler.current_position = pos
        t0[pos] = pp.tiler.get_traps_timepoint(
            0, channels=[0, pp.tiler.channels.index("mCherry")], z=[0, 1, 2, 3, 4]
        )

    return t0


def get_pos_ph(pp):
    pat = re.compile(r"ph_([0-9]_[0-9][0-9])")
    return {
        pos: float(pat.findall(pos)[0].replace("_", ".")) for pos in pp.tiler.positions
    }


def plot_sample_bf_mch(pp):
    bf_mch = get_traps(pp)
    ts = [{i: v[:, j, ...] for i, v in bf_mch.items()} for j in [0, 1]]
    tsbf = {i: v[:, 0, ...] for i, v in bf_mch.items()}

    posdict = {k: v for k, v in get_pos_ph(pp).items()}
    posdict = {v: k for k, v in posdict.items()}
    posdict = {v: k for k, v in posdict.items()}
    ph = np.unique(list(posdict.values())).tolist()
    counters = {ph: 0 for ph in ph}
    n = [np.random.randint(ts[0][k].shape[0]) for k in posdict.keys()]

    fig, axes = plt.subplots(2, 5)
    for k, (t, name) in enumerate(zip(ts, ["Bright field", "mCherry"])):
        for i, (pos, ph) in enumerate(posdict.items()):
            # i = ph.index(posdict[pos])
            axes[k, i].grid(False)
            axes[k, i].set(
                xticklabels=[],
                yticklabels=[],
            )
            axes[k, i].set_xlabel(posdict[pos] if k else None, fontsize=28)
            axes[k, i].imshow(
                np.maximum.reduce(t[pos][n[i], 0], axis=2),
                cmap="gist_gray" if not k else None,
            )
            # counters[posdict[pos]] += 1
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        axes[k, 0].set_ylabel(name, fontsize=28)
    plt.tight_layout()
    plt.show()


# Plotting calibration curve
from scipy.optimize import curve_fit


def fit_calibration(h):
    ycols = [x for x in h.columns if "em_ratio" in x]
    xcol = "ph"

    def objective(x, a, b):
        return a * x + b

    # fig, axes = plt.subplots(1, len(ycols))
    # for i, ycol in enumerate(ycols):
    #     d = h[[xcol, ycol]]
    #     params, _ = curve_fit(objective, *[d[col].values for col in d.columns])
    #     sns.lineplot(x=xcol, y=ycol, data=h, alpha=0.5, err_style="bars", ax=axes[i])
    #     # sns.lineplot(d[xcol], objective(d[xcol].values, *params), ax=axes[i])
    # plt.show()

    ycol = "em_ratio_mean"
    d = h[[xcol, *ycols]]
    tmp = d.groupby("ph").mean()
    calibs = {ycol: curve_fit(objective, tmp.index, tmp[ycol])[0] for ycol in ycols}
    # sns.lineplot(x=xcol, y=ycol, data=d, alpha=0.5, err_style="bars")
    # plt.xlabel("pH")
    # plt.ylabel("pHluorin emission ratio")
    # sns.lineplot(d[xcol], objective(d[xcol], *params))

    return calibs
