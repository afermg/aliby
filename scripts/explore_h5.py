#!/usr/bin/env python3
from pcore.io.signal import Signal

fpath = "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/raf_15_s17.h5"

s = Signal(fpath)

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# df = s["/extraction/general/None/area"]
df = 1 / s["extraction/GFPFast/np_max/median"]
df = df.div(df.mean(axis=1), axis=0)

vol = s["postprocessing/bud_metric/extraction_general_None_volume"]
vol = vol.div(vol.mean(axis=1), axis=0)
# sns.heatmap(df.sort_index(), robust=True)
# plt.show()


import pandas as pd

n = 3
fig, axes = plt.subplots(n, 1)
trap_ids = df.index.get_level_values("trap")
trapn = trap_ids[np.random.randint(len(trap_ids), size=n)]
for i, t in enumerate(trapn):
    subdf = df.loc[t].melt(ignore_index=False).reset_index()
    subdf["ch"] = "ratio"
    voldf = vol.loc[t].melt(ignore_index=False).reset_index()
    voldf["ch"] = "vol"
    combined = pd.concat((voldf, subdf), axis=0)
    sns.scatterplot(
        data=combined,
        x="variable",
        y="value",
        ax=axes[i],
        hue="ch",
        palette="Set2",
        style="cell_label",
    )
plt.show()
