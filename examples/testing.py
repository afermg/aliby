#!/usr/bin/env python3


from matplotlib import pyplot as plt
from pandas import Series

from postprocessor.core.postprocessor import PostProcessor
from postprocessor.core.tracks import non_uniform_savgol

pp = PostProcessor(source=19916)  # 19916
pp.load_tiler_cells()
# f = '/home/alan/Documents/libs/extraction/extraction/examples/gluStarv_2_0_x2_dual_phl_ura8_00/extraction'
f = "/home/alan/Documents/libs/extraction/extraction/examples/pH_calibration_dual_phl__ura8__by4741__01"
pp.load_extraction(
    "/home/alan/Documents/libs/extraction/extraction/examples/"
    + pp.expt.name
    + "/extraction/"
)

tmp = pp.extraction[pp.expt.positions[0]]

# prepare data
test = tmp[("GFPFast", np.maximum, "mean")]
clean = test.loc[test.notna().sum(axis=1) > 30]

window = 9
degree = 3
savgol_on_srs = lambda x: Series(
    non_uniform_savgol(x.dropna().index, x.dropna().values, window, degree),
    index=x.dropna().index,
)

smooth = clean.apply(savgol_on_srs, axis=1)

from random import randint

x = randint(0, len(smooth))
plt.plot(clean.iloc[x], "b")
plt.plot(smooth.iloc[x], "r")
plt.show()


def growth_rate(
    data: Series, alg=None, filt={"kind": "savgol", "window": 9, "degree": 3}
):
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


import numpy as np

diff_kernel = np.array([1, -1])
gr = clean.apply(growth_rate, axis=1)


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


test = tmp[("GFPFast", np.maximum, "median")]
test2 = tmp[("pHluorin405", np.maximum, "median")]
ph = test / test2
ph = ph.stack().reset_index(1)
ph.columns = ["tp", "fl"]


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
