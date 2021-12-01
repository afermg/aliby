#!/usr/bin/env python3

import h5py
from postprocessor.core.processor import PostProcessorParameters, PostProcessor

from pathlib import Path

fpath = Path(
    "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/glu_01_016.h5"
)

with h5py.File(fpath, "a") as f:
    if "postprocessing" in f:
        del f["/postprocessing"]
    if "modifiers" in f:
        del f["/modifiers"]

params = PostProcessorParameters.default().to_dict()
pp = PostProcessor(fpath, params)
pp.run()
