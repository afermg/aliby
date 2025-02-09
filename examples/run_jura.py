from aliby.pipeline import Pipeline, PipelineParameters

params = PipelineParameters.default(
    general={
        "expt_id": 2172,
        "distributed": 0,
        "directory": ".",
        "host": "staffa.bio.ed.ac.uk",
        "username": "XXXXX",
        "password": "XXXXX",
    }
)
# specify OMERO_channels if the channels on OMERO have a different order from the logfiles
p = Pipeline(params)

p.run()
