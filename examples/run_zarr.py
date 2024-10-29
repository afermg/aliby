from aliby.pipeline import Pipeline, PipelineParameters

omero_dir = "/Swainlab/omero_backup/"
omid = "19477_2020_11_27_steadystate_glucose_898_exposure_901_2w0p01_00"
h5dir = "~/aliby_output/"

# setup and run pipeline
params = PipelineParameters.default(
    general={
        "expt_id": omero_dir + omid,
        "distributed": 4,
        "directory": h5dir,
    },
    # some old movies have three not five z stacks
    # you may need to uncomment
    # baby={"n_stacks": "3z"},
)

# initialise and run pipeline
p = Pipeline(params)
p.run()
