#!/usr/bin/env jupyter
"""
Run aliby with baby pipeline.
"""

from functools import partial
from pathlib import Path
from time import perf_counter

from pathos.multiprocessing import Pool

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_save

# Variables
paths = [
    "/datastore/alan/aliby/flavin_htb2_pyruvate_20gpL_01_00/",
    "/datastore/alan/aliby/PDR5_GFP_100ugml_flc_25hr_00/",
]
path = Path(paths[1])
out_dir = Path(f"/datastore/alan/swainlab/results/{path.stem}")
ntps = 30  # Number of time points to process
seg_channel = 0  # Channel to use for segmentation
nchannels = 1  # Channels to extract
regex = ".+\/(.+)\/.*([0-9]{6})_(\S+)_([0-9]{3}).tif"
capture_order = "FTCZ"
dimorder = "TCZYX"  # Order of dimensions to feed into the pipeline, do not change
threaded = False

ntps = 3
assert Path(path).exists(), "Folder does not exist"

# Load dataset from a regular expression
base_pipeline = dict(
    steps=dict(
        tile=dict(
            image_kwargs=dict(
                regex=regex,
                capture_order=capture_order,
            ),
            tile_size=117,
            ref_channel=0,
            ref_z=0,
        ),
        segment=dict(
            segmenter_kwargs=dict(
                kind="baby",
                camera="prime95b",
                channel="brightfield",
                zoom="60x",
                n_stacks="5z",
                # modelset_filter=None,
            ),
            img_channel=0,
        ),
        track=dict(kind="baby"),
        extract=dict(
            channels=(1, 2),
            tree={
                "general": {
                    "None": [
                        "area",
                        "area_bbox",
                        "area_convex",
                        "area_filled",
                        "axis_major_length",
                        "axis_minor_length",
                        # "bbox", # Requires skimage >=0.24
                        # "centroid_local",
                        # "centroid_weighted", # requires img
                        # "centroid_weighted_local",
                        # "coords_scaled", # newer
                        # "coords",
                        "eccentricity",
                        "equivalent_diameter_area",
                        "euler_number",
                        "extent",
                        "feret_diameter_max",
                        # Requires skimage >=0.24
                        # "inertia_tensor",
                        # "inertia_tensor_eigvals",
                        # "intensity_max",
                        # "intensity_mean",
                        # "intensity_min",
                        # "moments",
                        # "moments_central",
                        # "moments_hu",
                        # "moments_normalized",
                        # requires img
                        # "moments_weighted",
                        # "moments_weighted_central",
                        # "moments_weighted_hu",
                        # "moments_weighted_normalized",
                        # "num_pixels", # skimage > 0.24
                        "orientation",
                        "perimeter",
                        "perimeter_crofton",
                        # "slice", # Gives a really weird error, probably the string is interpolated somewhere TODO FIXME
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
        # track=[("masks", "segment"), ("track_info", "track")],
        extract=[("cell_labels", "track"), ("masks", "segment")],
    ),
    # key -> (step, method, parameter (from key))
    passed_methods=dict(),
    # passed_methods=dict(
    #     segment=("tile", "get_tp_data", "img_channel"),
    # ),
)

# Load dataset
dif = DatasetDir(
    path,
    regex=regex,
    capture_order=capture_order,
)

# Pathos seems to result in a highetr cpu usage, (which is good)
fov_to_files = dif.get_position_ids(regex, capture_order)
run_pipeline_save_curried = partial(
    run_pipeline_save, pipeline=base_pipeline, ntps=ntps, overwrite=True
)

# Run pipelines
t0 = perf_counter()
if not threaded:  # Threaded or not, non-threaded is for easy debug
    results = []
    for ws, files in fov_to_files.items():
        result = run_pipeline_save_curried(
            img_source=files, out_file=out_dir / f"{'_'.join(ws)}.parquet"
        )
        results.append(result)
else:
    with Pool() as p:
        result = p.map(
            lambda ws, files: run_pipeline_save_curried(
                img_source=files, out_file=out_dir / f"{'_'.join(ws)}.parquet"
            ),
            fov_to_files.items(),
        )
print(f"Analysis took {perf_counter() - t0} seconds")
