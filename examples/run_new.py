"""
CURRENT TASK 2025/02: Run the whole examples. Double-check that the cellpose pipeline is still working.

Run Swain Lab experiments with new pipeline.

Process:
1. List dataset
2. Select and instance Image
3. Process in a similar way to
/home/amunoz/projects/gsk-ia/scripts/2024_08_21_get_fully_morphological_profiles.py

4. Test in multiple cases
- Ura9 (GFP only)
- pHluorin (pH)

5. Extend to new data types (/datastore/alan/aliby/)
"""

#!/usr/bin/env jupyter
from pathlib import Path
from time import perf_counter

from pathos.multiprocessing import Pool

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_save

# Variables
# path = Path("/datastore/alan/aliby/flavin_htb2_pyruvate_20gpL_01_00/")
path = Path("/datastore/alan/aliby/PDR5_GFP_100ugml_flc_25hr_00/")
out_dir = Path(f"/datastore/alan/swainlab/results/{path.stem}")
regex = ".+\/(.+)\/.*([0-9]{6})_(\S+)_([0-9]{3}).tif"
capture_order = "FTCZ"

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
                1: {
                    "max": [
                        "total",
                        "max5px_median",
                        # "volume",
                        # "centroid_x",
                        # "centroid_y",
                    ]
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


t0 = perf_counter()
if False:  # Threaded or not, non-threaded is for easy debug
    results = []
    for well_site, wc in list(fov_to_files.items())[:2]:
        result = run_pipeline_save(
            base_pipeline=base_pipeline,
            wc=wc,
            out_file=out_dir / f"{'_'.join(well_site)}.parquet",
            ntps=ntps,
        )
        print(f"Result: {result}")
        results.append(result)
else:
    with Pool() as p:
        p.map(
            lambda ws_wc: run_pipeline_save(
                base_pipeline,
                ws_wc[1],
                out_file=out_dir / f"{'_'.join(ws_wc[0])}.parquet",
                ntps=1,
            ),
            fov_to_files.items(),
        )
print(f"Analysis took {perf_counter() - t0} seconds")
