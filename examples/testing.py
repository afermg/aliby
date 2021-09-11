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
from postprocessor.core.tracks import clean_tracks, merge_tracks, join_tracks
from postprocessor.core.ph import *

sns.set_context("talk", font_scale=1.8)
sns.set_theme(style="whitegrid")

# pp_c = PostProcessor(source=19920)  # 19916
# pp_c.load_tiler_cells()
# # f = "/home/alan/Documents/libs/extraction/extraction/examples/gluStarv_2_0_x2_dual_phl_ura8_00/extraction"
# # f = "/home/alan/Documents/tmp/pH_calibration_dual_phl__ura8__by4741__01/extraction/"
# f = "/home/alan/Documents/tmp/pH_calibration_dual_phl__ura8__by4741_Alan4_00/extraction/"
# pp_c.load_extraction(f)

# calib = process_phs(pp_c)
# # c = calib.loc[(5.4 < calib["ph"]) & (calib["ph"] < 8)].groupby("ph").mean()
# sns.violinplot(x="ph", y="em_ratio_mean", data=calib)
# plt.show()


# bring timelapse data and convert it to pH

pp = PostProcessor(source=19831)  # 19831
pp.load_tiler_cells()
f = "/home/alan/Documents/tmp/gluStarv_2_0_x2_dual_phl_ura8_00/extraction/"
# f = "/home/alan/Documents/tmp/downUpshift_2_0_2_glu_dual_phluorin__glt1_psa1_ura7__thrice_00/extraction/"
pp.load_extraction(f)

import compress_pickle

compress_pickle.dump(pp.extraction, "/home/alan/extraction_example.pkl")

if True:  # Load extracted data or pkld version
    new_dfs = compress_pickle.load("/home/alan/Documents/tmp/new_dfs.gz")
# Combine dataframes
else:
    new_dfs = combine_dfs(get_dfs(pp))
    # t = [x.index for x in new_dfs.values()]
    # i = set(t[0])
    # for j in t:
    #     i = i.intersection(j)
    new_dfs = {
        k: v
        for k, v in new_dfs.items()
        if k[2] != "imBackground"
        and k[2] != "median"
        and ~(((k[0] == "GFPFast") | (k[0] == "pHluorin405")) and k[2] == "max2p5pc")
    }

del pp
compress_pickle.dump(new_dfs, "/home/alan/Documents/tmp/new_dfs.gz")


def get_clean_dfs(dfs=None):
    if dfs is None:
        clean_dfs = compress_pickle.load("/home/alan/Documents/tmp/clean_dfs.gz")
    else:

        # Clean timelapse
        clean = clean_tracks(new_dfs[("general", None, "area")])
        tra, joint = merge_tracks(clean)
        clean_dfs = new_dfs
        i_ids = set(clean.index).intersection(
            clean_dfs[("general", None, "area")].index
        )
        clean_dfs = {k: v.loc[i_ids] for k, v in clean_dfs.items()}
        clean_dfs = {k: join_tracks(v, joint, drop=True) for k, v in clean_dfs.items()}

        del new_dfs
        compress_pickle.dump(clean_dfs, "/home/alan/Documents/tmp/clean_dfs.gz")


def plot_ph_hmap(clean_dfs):
    GFPFast = clean_dfs[("GFPFast", np.maximum, "mean")]
    phluorin = clean_dfs[("pHluorin405", np.maximum, "mean")]
    ph = GFPFast / phluorin
    ph = ph.loc[ph.notna().sum(axis=1) > 0.7 * ph.shape[1]]
    ph = 1 / ph

    fig, ax = plt.subplots()
    hmap(ph, cbar_kws={"label": r"emission ratio $\propto$ pH"})
    plt.xlabel("Time (hours)")
    plt.ylabel("Cells")
    xticks = plt.xticks(fontsize=15)[0]
    ax.set(yticklabels=[], xticklabels=[str(round(i * 5 / 60, 1)) for i in xticks])
    # plt.setp(ax.get_xticklabels(), Rotation=90)
    plt.show()


def fit_calibs(c, h):
    h = process_phs(pp_c)
    h["ratio"] = h["GFPFast_bgsub_median"] / h["pHluorin405_bgsub_median"]
    sns.lineplot(x="ph", y="ratio", data=h, err_style="bars")
    plt.show()

    calibs = fit_calibration(c)
    for k, params in calibs.items():
        i, j = ("_".join(k.split("_")[:-1]), k.split("_")[-1])
        if j == "mean" and "k2" not in k:
            clean_dfs[k] = objective(clean_dfs[i, np.maximum, j], *params)


# max 2.5% / med
def plot_ratio_vs_max2p5(h):
    fig, ax = plt.subplots()
    sns.regplot(
        x="em_ratio_median",
        y="mCherry_max2p5pc",
        data=h,
        scatter=False,
        ax=ax,
        color="teal",
    )
    sns.scatterplot(x="em_ratio_median", y="max5_d_med", data=h, hue="ph", ax=ax)
    plt.xlabel(r"Fluorescence ratio $R \propto (1/pH)$")
    plt.ylabel("Max 2.5% px / median")
    plt.show()


em = clean_dfs[("em_ratio", np.maximum, "mean")]
area = clean_dfs[("general", None, "area")]


def get_grs(clean_dfs):
    area = clean_dfs[("general", None, "area")]
    area = area.loc[area.notna().sum(axis=1) > 10]
    return area.apply(growth_rate, axis=1)


def get_agg(dfs, rng):
    # df dict of DataFrames containing an area/vol one TODO generalise this beyond area
    # rng tuple of section to use
    grs = get_grs(dfs)
    smooth = grs.loc(axis=1)[list(range(rng[0], rng[1]))].dropna(how="all")

    aggregate_mean = lambda dfs, rng: pd.concat(
        {
            k[0] + "_" + k[2]: dfs[k].loc[smooth.index, rng[0] : rng[1]].mean(axis=1)
            for k in clean_dfs.keys()
        },
        axis=1,
    )
    # f_comp_df = comp_df.loc[(comp_df["gr"] > 0) & (area.notna().sum(axis=1) > 50)]

    agg = aggregate_mean(dfs, rng)
    agg["max2_med"] = agg["mCherry_max2p5pc"] / agg["mCherry_mean"]

    for c in agg.columns:
        agg[c + "_log"] = np.log(agg[c])

    agg["gr_mean"] = smooth.loc[set(agg.index).intersection(smooth.index)].mean(axis=1)
    agg["gr_max"] = smooth.loc[set(agg.index).intersection(smooth.index)].max(axis=1)

    return agg


def plot_scatter_fit(x, y, data, hue=None, xlabel=None, ylabel=None, ylim=None):
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, data=data, scatter=False, ax=ax)
    sns.scatterplot(x=x, y=y, data=data, ax=ax, alpha=0.1, hue=hue)
    # plt.show()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)

    fig.savefig(
        "/home/alan/Documents/sync_docs/drafts/third_year_pres/figs/"
        + str(len(data))
        + "_"
        + x
        + "_vs_"
        + y
        + ".png",
        dpi=200,
    )


from extraction.core.argo import Argo, annot_from_dset


def additional_feats(aggs):
    aggs["gr_mean_norm"] = aggs["gr_mean"] * 12
    aggs["log_ratio_r"] = np.log(1 / aggs["em_ratio_mean"])
    return aggs


def compare_methods_ph_calculation(dfs):
    GFPFast = dfs[("GFPFast", np.maximum, "mean")]
    phluorin = dfs[("pHluorin405", np.maximum, "mean")]
    ph = GFPFast / phluorin

    sns.scatterplot(
        dfs["em_ratio", np.maximum, "mean"].values.flatten(),
        ph.values.flatten(),
        alpha=0.1,
    )
    plt.xlabel("ratio_median")
    plt.ylabel("median_ratio")
    plt.title("Comparison of ph calculation")
    plt.show()


# get the number of changes in a bool list
nchanges = lambda x: sum([i for i, j in zip(x[:-2], x[1:]) if operator.xor(i, j)])
