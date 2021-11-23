#!/usr/bin/env python3


folder = Path(
    "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/"
)
g = NameGrouper(folder)
# g.signals = {k: v for k, v in g.signals.items() if str(v) != filename}
# signame = "extraction/general/None/volume"
# signame = "extraction/em_ratio_bgsub/np_max/median"
# signame = (
#     "postprocessing/dsignal/postprocessing_bud_metric_extraction_general_None_volume"
# )

c = g.concat_signal(signame)
d = c.loc[c.notna().sum(axis=1) > 50]
sns.lineplot(
    data=d.melt(ignore_index=False).reset_index(), x="variable", y="value", hue="group"
)
plt.show()

# Aggregated metrics
signame = "postprocessing/experiment_wide/aggregated"
grid = sns.pairplot(
    c[[i for i in c.columns if i.endswith("mean")]].reset_index(level="group"),
    plot_kws=dict(alpha=0.5),
    hue="group",
)
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
