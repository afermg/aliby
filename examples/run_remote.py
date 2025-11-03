"""Script to run aliby using OMERO."""

from aliby.pipeline import PipelineParameters, Pipeline

params = PipelineParameters.default(
    general={
        "expt_id": 2172,
        "distributed": 0,
        "directory": ".",
        "host": "staffa.bio.ed.ac.uk",
        "username": pass,
        "password": pass,
    }
)
# specify OMERO_channels if the channels on OMERO have a different
# order from the logfiles
p = Pipeline(params)

p.run()
