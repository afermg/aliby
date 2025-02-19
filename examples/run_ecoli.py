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

ntps = 10  # Number of time points to process
seg_channel = 1  # Channel to use for segmentation
nchannels = 2  # Channels to extract
threaded = False
capture_order = "0TCYX"

base_pipeline = dict(
    steps=dict(
        tile=dict(
            image_kwargs=dict(
                capture_order=capture_order,
            ),
            tile_size=None,
            ref_channel=0,
        ),
        segment=dict(
            segmenter_kwargs=dict(
                kind="cyto3",
            ),
            img_channel=0,
        ),
        track=dict(kind="stitch"),
        extract=dict(
            channels=(0, 1),
            tree={
                "general": {
                    "None": [
                        "area",
                        "area_bbox",
                        "area_convex",
                        "area_filled",
                        "axis_major_length",
                        "axis_minor_length",
                        "eccentricity",
                        "equivalent_diameter_area",
                        "euler_number",
                        "extent",
                        "feret_diameter_max",
                        "orientation",
                        "perimeter",
                        "perimeter_crofton",
                        "solidity",
                    ],
                },
                **{
                    i: {
                        "max": [
                            "radial_distribution",
                            "radial_zernikes",
                            "intensity",
                            "sizeshape",
                            "zernike",
                            "ferret",
                            "granularity",
                            "texture",
                        ]
                    }
                    for i in range(nchannels)
                },
            },
            multichannel_ops={},
        ),
    ),
    passed_data=dict(  # A=-> [(B,C,D)] where A receives variable B (or field D) from C.
        track=[("masks", "segment"), ("track_info", "track")],
        extract=[("cell_labels", "track"), ("masks", "segment")],
    ),
    # key -> (step, method, parameter (from key))
    passed_methods=dict(
        segment=("tile", "get_tp_data", "img_channel"),
    ),
)

from aliby.pipe import run_pipeline

sample_file = list(d.values())[0][0]
pipeline = base_pipeline
# pipeline["steps"]["tiler"]["img_source"] = sample_file
results = run_pipeline(pipeline=pipeline, img_source=sample_file, ntps=ntps)
