#!/usr/bin/env python3
# Post process a single file
from postprocessor.core.processor import PostProcessorParameters, PostProcessor
from core.io.signal import Signal

# filepath = "/shared_libs/pydask/pipeline-core/data/2021_08_21_KCl_pH_00/YST_1511_005.h5"
filepath = "/shared_libs/pydask/pipeline-core/data/bak_kcl/YST_1511_005.h5"
import h5py

with h5py.File(filepath, "a") as f:
    if "postprocessing" in f:
        del f["/postprocessing"]
    if "modifiers" in f:
        del f["/modifiers"]

params = PostProcessorParameters.default()
pp = PostProcessor(filepath, params)
pp.run()
s = Signal(filepath)
s.datasets
df = s["/extraction/general/None/area"]

import seaborn as sns
from matplotlib import pyplot as plt

# with h5py.File(filepath, "r") as f:
#     print(f["postprocessing/lineage"].keys())
# sns.heatmap(df.sort_index())
# plt.show()


def budMetrics(signal, mother, daughters):
    """
    Parameters
    ----------
    signal : pd.DataFrame
        Description of parameter Symbol’s value as variable is void: x.
    """
    bud_mets = [signal.loc[d] for d in daughters]


def budMetric(signal, daughters):
    first_na = signal.apply(pd.Series.first_valid_index, axis=1).sort_values()
    # signal = signal.loc[first_na.index]
    bud_met = pd.concat(
        [
            signal.columns[start:end]
            for start, end in zip(first_na.values[:-2], first_na.values[1:])
        ],
        axis=1,
    )
    return bud_met


# Get data
with h5py.File(filepath, "a") as f:
    traps = f["postprocessing/lineage/trap"][()]
    mothers = f["postprocessing/lineage/mother_label"][()]
    daughters = f["postprocessing/lineage/daughter_label"][()]
    mothers = [(t, l) for t, l in zip(traps, mothers)]
    daughters = [(t, d) for t, d in zip(traps, daughters)]

    mtraps = f["postprocessing/lineage_merged/trap"][()]
    mmothers = f["postprocessing/lineage_merged/mother_label"][()]
    mdaughters = f["postprocessing/lineage_merged/daughter_label"][()]
    mmothers = [(t, l) for t, l in zip(mtraps, mmothers)]
    mdaughters = [(t, d) for t, d in zip(mtraps, mdaughters)]

import numpy as np
from itertools import groupby


search = lambda a, b: np.where(
    np.in1d(
        np.ravel_multi_index(a.T, a.max(0) + 1),
        np.ravel_multi_index(b.T, a.max(0) + 1),
    )
)
iterator = groupby(zip(mothers, daughters), lambda x: x[0])
dic = {key: [x[1] for x in group] for key, group in iterator}
# for m, d in dic.items():
#     # if set(d).intersection(df.index) and m in df.index:
#     print(m, len(d))
#     # search(np.array(df.index.tolist()), np.array(d))
#     # if m in df.index and

raw = s.get_raw("extraction/general/None/volume").sort_index()
df = s["extraction/general/None/volume"].sort_index()

sns.set_theme(style="darkgrid")
trap_ids = df.index.get_level_values("trap").unique()

plot = sns.lineplot
if plot is not sns.heatmap:
    kwargs = dict(
        x="tp",
        y="volume",
        hue="cell_label",
        style="cell_label",
        palette="Accent"
        # estimator=None,
        # markers="True",
    )


import pandas as pd

ix = np.array([0, 4])

# TODO check why some good-looking tracks are lost. But first write process to estimate
# growth rate and births
ix += 3
fig, axes = plt.subplots(abs(np.diff(ix)[0]), 2, sharex=True)
for i, trap_id in enumerate(trap_ids[ix[0] : ix[1]]):
    # kwargs["ax"] = axes[i]
    raw_data = (
        pd.concat(
            (
                raw.loc[df.index].loc[trap_id],
                raw.loc[trap_id].loc[
                    raw.loc[trap_id].index.difference(df.loc[trap_id].index)
                ],
            )
        )
        .melt(var_name="tp", value_name="volume", ignore_index=False)
        .reset_index()
    )
    raw_data["cell_label"] = raw_data["cell_label"].astype(str)
    df_data = (
        df.loc[trap_id]
        .melt(var_name="tp", value_name="volume", ignore_index=False)
        .reset_index()
    )
    df_data["cell_label"] = df_data["cell_label"].astype(str)
    plot(data=raw_data, ax=axes[i, 0], **kwargs)
    plot(data=df_data, ax=axes[i, 1], **kwargs)
    # plot(raw.loc[trap_id], ax=axes[i, 0], **kwargs)
    # plot(df.loc[trap_id], ax=axes[i, 1], **kwargs)
    for ax in axes[i]:
        ax.get_legend().remove()
plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
