"""Script to apply a non-default function to the images."""

from aliby.global_settings import global_settings
from aliby.pipeline import Pipeline, PipelineParameters

# add max5px_median
# all functions are in extraction/core/functions/cell_functions.py
global_settings.fluorescence_functions.append("max5px_median")

# add calculation of a pixel-by-pixel ratio of two fluorescence channels
# a function of two channels is given the channels, how to reduce their
# z-sections, and its name; each is stored under its key, here as
# extraction/GFP_over_mCherry/max/ratio_1_over_2 and, with the background
# subtracted, extraction/GFP_over_mCherry/max/ratio_1_over_2_bgsub
# the channels must be named as the experiment names them
multichannel_funs = {
    "GFP_over_mCherry": [["GFP", "mCherry"], "max", "ratio_1_over_2"],
    "mCherry_over_GFP": [["GFP", "mCherry"], "max", "ratio_2_over_1"],
}


# guard the entry point so that, under the spawn start method on macOS,
# worker processes re-importing this module do not re-run the pipeline
if __name__ == "__main__":
    params = PipelineParameters.default(
        general={
            "expt_id": 2172,
            "distributed": 0,
            "directory": ".",
            "host": "staffa.bio.ed.ac.uk",
            "username": "pass",
            "password": "pass",
        },
        extraction={"multichannel_funs": multichannel_funs},
    )
    p = Pipeline(params)
    p.run()
