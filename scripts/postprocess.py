#!/usr/bin/env python3

import h5py
from postprocessor.core.processor import PostProcessorParameters, PostProcessor

from pathlib import Path

fpath = Path("/home/alan/Downloads/Vph1_005.h5")

with h5py.File(fpath, "a") as f:
    if "postprocessing" in f:
        del f["/postprocessing"]
    if "modifiers" in f:
        del f["/modifiers"]

params = PostProcessorParameters.default().to_dict()
pp = PostProcessor(fpath, params)
pp.run()

from aliby.io.signal import Signal

s = Signal(fpath)
vol = s["extraction/general/None/volume"]
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(vol.sort_index(), robust=True)
plt.show()
