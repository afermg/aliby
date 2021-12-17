#!/usr/bin/env python3
# Post process a single file
from pathlib import Path
import h5py
from postprocessor.core.processor import PostProcessorParameters, PostProcessor
from pcore.io.signal import Signal

from pathos.multiprocessing import Pool

folder = Path(
    "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/"
)


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
        print(e)


with Pool(10) as p:
    results = p.map(lambda x: process_file(x), Path(folder).rglob("*.h5"))
