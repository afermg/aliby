"""Script to run aliby to process a directory of tiff files."""

from pathlib import Path

from aliby.pipeline import Pipeline, PipelineParameters
from postprocessor.grouper import Grouper

omid = "25681_2022_04_30_flavin_htb2_glucose_10mgpL_01_00"

h5dir = "/Users/pswain/wip/aliby_output/"
omero_dir = "/Users/pswain/wip/aliby_input/"

# setup and run pipeline
params = PipelineParameters.default(
    general={
        "expt_id": omero_dir + omid,
        "distributed": 2,
        "directory": h5dir,
        # optional: specify position to segment
        "filter": "fy4_007",
        # optional: specify final time point
        # "tps": 4,
    },
)


# initialise and run pipeline
# optional: specify OMERO channels if the order on OMERO breaks
# convention, with Brightfield not first
p = Pipeline(params, OMERO_channels=["Brightfield", "Flavin"])
p.run()
