"""Script to apply a non-default function to the images."""

from aliby.global_settings import global_settings
from aliby.pipeline import Pipeline, PipelineParameters

# add max5px_median
# all functions are in extraction/core/functions/cell_functions.py
global_settings.fluorescence_functions.append("max5px_median")


params = PipelineParameters.default(
    general={
        "expt_id": 2172,
        "distributed": 0,
        "directory": ".",
        "host": "staffa.bio.ed.ac.uk",
        "username": "pass",
        "password": "pass",
    }
)
p = Pipeline(params)

p.run()
