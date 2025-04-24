from aliby.pipeline import Pipeline, PipelineParameters

omid = "25681_2022_04_30_flavin_htb2_glucose_10mgpL_01_00"

h5dir = "/Users/pswain/wip/aliby_output/"
omero_dir = "/Users/pswain/wip/aliby_input/"

# setup and run pipeline
params = PipelineParameters.default(
    general={
        "expt_id": omero_dir + omid,
        "distributed": 2,
        "directory": h5dir,
        # specify position to segment
        "filter": "fy4_007",
        # specify final time point
        # "tps": 4,
    },
)


# initialise and run pipeline
p = Pipeline(params, OMERO_channels=["Brightfield", "Flavin"])
p.run()
