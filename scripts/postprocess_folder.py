#!/usr/bin/env python3
# Post process a single file
from pathlib import Path
import h5py
from postprocessor.core.processor import PostProcessorParameters, PostProcessor
from core.io.signal import Signal

from pathos.multiprocessing import Pool

folder = "/shared_libs/pydask/pipeline-core/data/bak_kcl/"


def process_file(filepath):
    try:
        # for filepath in Path(folder).rglob("*.h5"):
        with h5py.File(filepath, "a") as f:
            if "postprocessing" in f:
                del f["/postprocessing"]
            if "modifiers" in f:
                del f["/modifiers"]

            params = PostProcessorParameters.default()
            pp = PostProcessor(filepath, params)
            pp.run()
            s = Signal(filepath)
            # s.datasets
        # df = s["/extraction/general/None/area"]
    except Exception as e:
        print(filepath, " failed")


with Pool(10) as p:
    results = p.map(lambda x: process_file(x), Path(folder).rglob("*.h5"))


from core.grouper import NameGrouper

g = NameGrouper(folder)
# g.signals = {k: v for k, v in g.signals.items() if str(v) != filename}
signame = "postprocessing/experiment_wide/aggregated"
shortname = "aggregated"
c = g.concat_signal(signame)
for i in c.columns:
    if "ratio" in i:
        c[i] = 1 / c[i]

splits = [x.split("_") for x in c.columns]
new = [
    [
        x
        for x in s
        if x
        not in [
            "general",
            "metric",
            "postprocessing",
            "extraction",
            "None",
            "np",
            "max",
            "em",
        ]
    ]
    for s in splits
]
joint = ["_".join(n) for n in new]
c.columns = joint
# cf = c.loc[
#     c["dsignal_postprocessing_bud_metric_extraction_general_None_volume"].notna()
# ]
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("darkgrid")
sns.lmplot(
    data=c.reset_index(),
    x="em_ratio_median_mean",
    # y="gsum_median_mean",
    # y="dsignal_postprocessing_bud_metric_extraction_general_None_volume_max",
    # y="dsignal_extraction_general_None_volume_max",
    # hue="position",
    hue="group",
    # size="general_volume",
    # alpha=0.5,
    palette="muted",
    ci=None,
)
# plt.xlim((0, 5))
plt.show()

grid = sns.pairplot(c[[i for i in c.columns if i.endswith("max")]])
for ax in grid.axes.flat[:2]:
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=90)

plt.savefig("pairplot.png", dpi=400)
plt.show()

for s in g.signals.values():
    with h5py.File(s.filename, "r") as f:
        if "postprocessing/experiment_wide" not in f:
            print(s.filename)

filepath = "/shared_libs/pydask/pipeline-core/data/bak_kcl/YST_1510_009.h5"
with h5py.File(filepath, "a") as f:
    print(f["postprocessing"].keys())
    # if "postprocessing" in f:
    #     del f["/postprocessing"]
    # if "modifiers" in f:
    #     del f["/modifiers"]

    params = PostProcessorParameters.default()
    pp = PostProcessor(filepath, params)
    pp.run()
    s = Signal(filepath)
    s.datasets
