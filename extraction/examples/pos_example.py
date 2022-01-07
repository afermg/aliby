from extraction.core.parameters import Parameters
from extraction.core.extractor import Extractor
import numpy as np

params = Parameters(
    tree={
        "general": {"None": ["area"]},
        "GFPFast": {"np_max": ["mean", "median", "imBackground"]},
        "pHluorin405": {"np_max": ["mean", "median", "imBackground"]},
        "mCherry": {
            "np_max": ["mean", "median", "imBackground", "max5px", "max2p5pc"]
        },
    }
)


ext = Extractor(params, omero_id=19310)
# ext.extract_exp(tile_size=117)
d=ext.extract_tp(tp=1,tile_size=117)

