#!/usr/bin/env jupyter
"""Test a basic pipeline with a unique zarr directory as input."""

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post


def _create_extract_multich_tree(channels: list[int]) -> dict:
    """Generate the extract_multich_tree dictionary for colocalization."""
    return {
        pair: {
            "None": {
                "max": ["pearson", "costes", "manders_fold", "rwc"],
            },
        }
        for pair in combinations(channels, r=2)
    }


# fpath = "/work/datasets/jump_toy_compress/zstd.zarr/"
fpath = "/work/datasets/jump_target2_subset_BR00121438/zstd.zarr/"
dset = dispatch_dataset(fpath, is_monozarr=True)
image_paths = dset.get_position_ids()

# %%
input_paths = list(image_paths.values())


def process_input_path(input_path: str):
    fluo_base_config = {
        "input_path": input_path,
        "capture_order": "CYX",
        "ntps": 1,
        "segmentation_channel": {"nuclei": 1, "cell": 2},
        # "gstep_params": {},
    }
    fl_channels = range(5)

    segmentation_channel: dict[str, int] = fluo_base_config["segmentation_channel"]
    seg_params = {
        f"segment_{obj}": dict(
            segmenter_kwargs=dict(
                kind="cellpose",
            ),
            img_channel=ch_id,
        )
        for i, (obj, ch_id) in enumerate(
            fluo_base_config["segmentation_channel"].items()
        )
    }

    extract_base = dict(
        channels=fl_channels,
        tree={
            **{
                i: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "ferret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                        # "granularity", # Too time-consuming, deactivated for now
                    ]
                }
                for i in fl_channels
            },
        },
    )

    ext_params = {
        f"extract{name}_{obj}": var
        for (name, var), obj in product(
            (
                ("", extract_base),
                # ("multi", extract_multich_tree),
            ),
            segmentation_channel,
        )
        if len(var)
    }

    base_pipeline = {
        "io": {**fluo_base_config},
        "nchannels": 5,
        "fl_channels": fl_channels,
        "extract_multich_tree": _create_extract_multich_tree(fl_channels),
        "steps": dict(
            tile=dict(
                image_kwargs=dict(
                    source=input_path,
                    # regex=regex,
                    capture_order=fluo_base_config["capture_order"],
                    # dimorder=fluo_base_config["dimorder"],
                ),
                tile_size=None,
                ref_channel=0,
                ref_z=0,
                calculate_drift=False,
            ),
            **seg_params,
            **ext_params,
        ),
        "passed_data": dict(
            **{
                f"extract_{obj}": [
                    ("masks", f"segment_{obj}"),
                    ("pixels", "tile"),
                ]
                for obj in fluo_base_config["segmentation_channel"]
            },
        ),
        "passed_methods": {
            f"segment_{obj}": ("tile", "get_tp_data", "img_channel")
            for obj in segmentation_channel
        },
        "save": (
            # "tile",
            *seg_params.keys(),
        ),
        "save_interval": 1,
    }

    expt_name = str(input_path).split("/")[5]
    output_path = Path("/work/datasets/aliby_output") / expt_name

    # t0 = perf_counter()
    result, _ = run_pipeline_and_post(
        pipeline=base_pipeline,
        img_source=input_path,
        output_path=output_path,
        fov=input_path.path,
        overwrite=True,
    )
    # print(f"Data {i} took {perf_counter() - t0} seconds")


if True:
    with ProcessPoolExecutor() as p:
        result = list(
            p.map(
                process_input_path,
                input_paths,
            )
        )
else:
    process_input_path(input_paths[0])
