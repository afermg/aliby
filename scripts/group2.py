#!/usr/bin/env python3


from pcore.grouper import NameGrouper

folder = Path(
    "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/"
)
g = NameGrouper(folder)
# g.signals = {k: v for k, v in g.signals.items() if str(v) != filename}
# signame = "postprocessing/experiment_wide/aggregated"
# signame = "extraction/general/None/volume"


def get_df(signame):
    c = g.concat_signal(signame)
    d = c.loc[c.notna().sum(axis=1) > 60]
    # for i in c.columns:
    #     if "ratio" in i:
    #         c[i] = 1 / c[i]

    # splits = [x.split("_") for x in c.columns]
    # new = [
    #     [
    #         x
    #         for x in s
    #         if x
    #         not in [
    #             "general",
    #             "metric",
    #             "postprocessing",
    #             "extraction",
    #             "None",
    #             "np",
    #             "max",
    #             "em",
    #         ]
    #     ]
    #     for s in splits
    # ]
    # joint = ["_".join(n) for n in new]
    # c.columns = joint

    return d


signame = (
    "postprocessing/dsignal/postprocessing_bud_metric_extraction_general_None_volume"
)
signame = "extraction/mCherry/np_max/max2p5pc"
window = 15
whi5 = get_df(signame)
whi5 = whi5.div(whi5.mean(axis=1), axis=0)
whi5_movavg = whi5.apply(lambda x: pd.Series(moving_average(x.values, window)), axis=1)
ph = 1 / get_df("extraction/em_ratio/np_max/median")
ph = ph.div(ph.mean(axis=1), axis=0)
ph_movavg = ph.apply(lambda x: pd.Series(moving_average(x.values, window)), axis=1)
ph_norm = ph.iloc(axis=1)[window // 2 : -window // 2] / ph_movavg
whi5_norm = whi5.iloc(axis=1)[window // 2 : -window // 2] / whi5_movavg
rand = np.random.randint(len(whi5), size=4)

melted = ph_norm.iloc[rand].melt(ignore_index=False).reset_index()
melted["signal"] = "ph"
combined = whi5_norm.iloc[rand].melt(ignore_index=False).reset_index()
combined["signal"] = "whi5"
combined = pd.concat((melted, combined))
combined["t_id"] = [str(x) + y for x, y in zip(combined["trap"], combined["position"])]
h = sns.FacetGrid(combined, col="t_id", col_wrap=2)
h.map_dataframe(sns.scatterplot, x="variable", y="value", hue="signal")
h.add_legend()
plt.show()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("darkgrid")
sns.lineplot(
    data=d.melt(ignore_index=False).reset_index("group"),
    x="variable",
    y="value",
    # y="gsum_median_mean",
    # y="dsignal_postprocessing_bud_metric_extraction_general_None_volume_max",
    # y="dsignal_extraction_general_None_volume_max",
    # hue="position",
    hue="group",
    # size="general_volume",
    # alpha=0.5,
    palette="muted",
    # ci=None,
)
# plt.xlim((0, 5))
plt.show()
