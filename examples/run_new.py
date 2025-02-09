"""
Run Swain Lab experiments with new pipeline.

Process:
1. List dataset
2. Select and instance Image
3. Process in a similar way to
/home/amunoz/projects/gsk-ia/scripts/2024_08_21_get_full_morphological_profiles.py

4. Test in multiple cases
- Ura9 (GFP only)
- pHluorin (pH)

5. Extend to new data types (/datastore/alan/aliby/)
"""

from pathlib import Path

datasets_path = Path("/home/amunoz/projects/microscopy_backup/data")
datasets = {
    int(x.stem.split("_")[0]): str(x) for x in datasets_path.glob("*") if x.is_dir()
}

#!/usr/bin/env jupyter
from pathlib import Path
from time import perf_counter

from pathos.multiprocessing import Pool

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline

# Variables
path = Path("/datastore/alan/aliby/flavin_htb2_pyruvate_20gpL_01_00/")
out_dir = Path(f"/datastore/alan/swainlab/results/{path.stem}")
regex = ".+\/(.+)\/.*([0-9]{6})_(\S+)_([0-9]{3}).tif"
capture_order = "FTCZ"
assert Path(path).exists(), "Folder does not exist"


# Load dataset from a regular expression

base_pipeline = dict(
    steps=dict(
        tile=dict(
            image_kwargs=dict(
                regex=regex,
                # dimorder="CWTFZ",
                capture_order=capture_order,
            ),
            tile_size=None,
            ref_channel=0,
            ref_z=0,
        ),
        segment=dict(
            segmenter_kwargs=dict(
                kind="nuclei",
                diameter=None,
                channels=[0, 0],
            ),
            img_channel=1,
        ),
        track=dict(kind="stitch"),
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
                        "radial_distribution",
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "zernike",
                        "ferret",
                        "granularity",
                        "texture",
                    ]
                },
                0: {
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

# Load dataset
dif = DatasetDir(
    path,
    regex=regex,
    capture_order=capture_order,
)

# Pathos seems to result in a highetr cpu usage, (which is good)
fov_to_files = dif.get_position_ids(regex, capture_order)


def run_pipeline_save(base_pipeline: dict, wc: str, out_file: str | Path, ntps=1):
    print(f"Running {out_file}")

    result = None
    if not Path(out_file).exists():
        # try:
        result = run_pipeline(base_pipeline, wc, ntps=20)
        out_dir = Path(out_file).parent
        if not out_dir.exists():  # Only create a dir after we have files to save
            out_dir.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_file)
        # except Exception as e:
        #     print(e)
        #     with open("logfile.txt", "a") as f:
        #         f.write(f"{out_file} failed:{e}\n")
    return result


t0 = perf_counter()
if True:  # Threaded or not, non-threaded is for easy debug
    results = []
    for well_site, wc in list(fov_to_files.items())[:2]:
        result = run_pipeline_save(
            base_pipeline=base_pipeline,
            wc=wc,
            out_file=out_dir / f"{'_'.join(well_site)}.parquet",
            ntps=1,
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
            wellsite_to_wildcard.items(),
        )
print(f"Analysis took {perf_counter() - t0} seconds")
