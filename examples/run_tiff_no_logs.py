"""
Example script for analysing tiffs with specified metadata.

The tiff files must be stored in a separate directory per position
and use a particular naming structure.
E.g., for position htb2mCherry_001, a tiff file should be in a
directory called htb2mCherry_001 and follow the convention
exemplified by

 htb2mCherry_001_t0100_mCherry_z05.tiff

for timepoint 100 (four digits required), for imaging channel mCherry,
and for z-slice number 5 (two digits required).
"""

from aliby.pipeline import Pipeline, PipelineParameters
from wela.dataloader import DataLoader
from wela.plotting import kymograph


h5dir = "/Users/pswain/wip/aliby_output/"
omero_dir = "/Users/pswain/wip/aliby_input/"

# master directory for an experiment
# with one subdirectory per position
omid = "test26643_tiff"

params = PipelineParameters.default(
    general={
        "expt_id": omero_dir + omid,
        "distributed": 0,
        "directory": h5dir,
        "filter": [0],
        "tps": 8,
    },
    # optional if log files exist in the master directory
    metadata={
        "channels": ["Brightfield", "GFP_Z", "mCherry_Z"],
        "time_settings/ntimepoints": 240,
        "time_settings/timeinterval": 300,
    },
)


p = Pipeline(params)
p.run()


# run dataloader
dl = DataLoader(h5dir, ".")
g = dl.load(omid, key_index="mean_GFP_Z", cutoff=0.8)

kymograph(dl.df, hue="volume")
# Dataloader automatically drops _Z from any fluorescence names
kymograph(dl.df, hue="mean_GFP")
