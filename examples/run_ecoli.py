"""
Run aliby on an external dataset. In this case it is
- Name: Analysis of division and replication cycles in E. coli using time-lapse microscopy, microfluidics and the MoMA software
- URL: https://zenodo.org/records/3149097

Every file in the dataset is a 5-dimensional tiff with uint16 data
Apparently with dims 1TCYZ
- Channel 0 is just one-dimensional
"""

from pathlib import Path

from pooch import Unzip, retrieve

data_path = {
    "ecoli": (
        "https://zenodo.org/api/records/3149097/files/PreProcessed.zip/content",
        "f1ae1db09732ea3cd463f1ed5d5b2a846fcdea7992798d77301dfda872a4c416",
    )
}

retrieved = retrieve(
    *data_path["ecoli"],
    processor=Unzip(),
)
d = {}
for f in retrieved:
    if f.endswith("tif"):
        fpath = Path(f)
        dirname = fpath.parents[1].name
        if dirname in d:
            d[dirname].append(f)
        else:
            d[dirname] = [f]

ntps = 30  # Number of time points to process
seg_channel = 0  # Channel to use for segmentation
nchannels = 1  # Channels to extract
threaded = False
