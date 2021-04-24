from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt
import seaborn as sns

from postprocessor.core.postprocessor import PostProcessor
<<<<<<< HEAD
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
=======
from postprocessor.core.tracks import non_uniform_savgol, clean_tracks
>>>>>>> 96f513af38080e6ebb6d301159ca973b5d90ce81


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
<<<<<<< HEAD


test = tmp[("GFPFast", np.maximum, "median")]
test2 = tmp[("pHluorin405", np.maximum, "median")]
ph = test / test2
ph = ph.stack().reset_index(1)
ph.columns = ["tp", "fl"]


=======
>>>>>>> 96f513af38080e6ebb6d301159ca973b5d90ce81
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

def split_data(df:DataFrame, splits:List[int]):
    dfs = [df.iloc[:,i:j] for i,j in zip( (0,) + splits,
                                                splits + (df.shape[1],))]
    return dfs

def growth_rate(data:Series, alg=None, filt = {'kind':'savgol','window':7, 'degree':3}):
    if alg is None:
        alg='standard'

    if filt: #TODO add support for multiple algorithms
        window = filt['window']
        degree = filt['degree']
        data = Series(non_uniform_savgol(data.dropna().index, data.dropna().values,
                                         window, degree), index = data.dropna().index)

    diff_kernel = np.array([1,-1])


    return Series(np.convolve(data,diff_kernel ,'same'), index=data.dropna().index)

pp = PostProcessor(source=19831)
pp.load_tiler_cells()
f = '/home/alan/Documents/sync_docs/libs/postproc/gluStarv_2_0_x2_dual_phl_ura8_00/extraction'
pp.load_extraction('/home/alan/Documents/sync_docs/libs/postproc/postprocessor/' + pp.expt.name + '/extraction/')
tmp=pp.extraction['phl_ura8_002']

def _check_bg(data):
    for k in list(pp.extraction.values())[0].keys():
        for p in pp.expt.positions:
            if k not in pp.extraction[p]:
                print(p, k)
data = {k:pd.concat([pp.extraction[pos][k] for pos in \
                     pp.expt.positions[:-3]]) for k in list(pp.extraction.values())[0].keys()}

hmap = lambda df: sns.heatmap(sort_df(df), robust=True);
# from random import randint
# x = randint(0, len(smooth))
# plt.plot(clean.iloc[x], 'b')
# plt.plot(smooth.iloc[x], 'r')
# plt.show()


# data = tmp
df= data[('general',None,'area')]
clean = clean_tracks(df, min_len=160)
clean = clean.loc[clean.notna().sum(axis=1) > 9]
gr = clean.apply(growth_rate, axis=1)
splits = (72,108,180)
gr_sp = split_data(gr, splits)

idx = gr.index

bg = get_bg(data)
test = data[('GFPFast', np.maximum, 'median')]
test2 = data[('pHluorin405', np.maximum, 'median')]
ph = (test/test2).loc[idx]
c=pd.concat((ph.mean(1), gr.max(1)), axis=1); c.columns = ['ph', 'gr_max']
# ph = ph.stack().reset_index(1)
# ph.columns = ['tp', 'fl']

ph_sp=split_data(gr, splits)

def get_bg(data):
    bg = {}
    fl_subkeys = [x for x in data.keys() if x[0] in \
                  ['GFP', 'GFPFast', 'mCherry', 'pHluorin405'] and x[-1]!='imBackground']
    for k in fl_subkeys:
            nk = list(k)
            bk = tuple(nk[:-1] + ['imBackground'])
            nk = tuple(nk[:-1] +  [nk[-1] + '_BgSub'])
            tmp = []
            for i,v in data[bk].iterrows():
                if i in data[k].index:
                    newdf = data[k].loc[i] / v
                    newdf.index = pd.MultiIndex.from_tuples([(*i, c) for c in \
                                                          newdf.index])
                tmp.append(newdf)
            bg[nk] = pd.concat(tmp)

    return bg

def calc_ph(bg):
    fl_subkeys = [x for x in bg.keys() if x[0] in \
                  ['GFP', 'GFPFast', 'pHluorin405']]
    chs = list(set([x[0] for x in fl_subkeys]))
    assert len(chs)==2, 'Too many channels'
    ch1 = [x[1:] for x in fl_subkeys if x[0]==chs[0]]
    ch2 = [x[1:] for x in fl_subkeys if x[0]==chs[1]]
    inter = list(set(ch1).intersection(ch2))
    ph = {}
    for red_fld in inter:
        ph[tuple(('ph',) + red_fld)] = bg[tuple((chs[0],) + red_fld)] / bg[tuple((chs[1],) + red_fld)]

# sns.heatmap(sort_df(data[('mCherry', np.maximum, 'max2p5pc_BgSub')] / data[('mCherry', np.maximum, 'median_BgSub')]), robust=True)

# from postprocessor.core.tracks import clean_tracks
