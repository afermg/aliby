#!/usr/bin/env jupyter
"""Test a basic pipeline with a unique zarr directory as input."""

from itertools import combinations, product
from pathlib import Path

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


fpath = "/work/datasets/jump_toy/zstd.zarr"
dset = dispatch_dataset(fpath, is_monozarr=True)
image_paths = dset.get_position_ids()

input_path = list(image_paths.values())[0]
fluo_base_config = {
    "input_path": input_path,
    "capture_order": "CYX",
    "ntps": 1,
    "segmentation_channel": {"nuclei": 1, "cell": 4},
    # "gstep_params": {},
}
fl_channels = range(1, 3)

segmentation_channel: dict[str, int] = fluo_base_config["segmentation_channel"]
seg_params = {
    f"segment_{obj}": dict(
        segmenter_kwargs=dict(
            kind="cellpose",
        ),
        img_channel=ch_id,
    )
    for i, (obj, ch_id) in enumerate(fluo_base_config["segmentation_channel"].items())
}

extract_base = dict(
    channels=fl_channels,
    tree={
        **{
            i: {
                "max": [
                    # "radial_zernikes",
                    "intensity",
                    "sizeshape",
                    # "ferret",
                    # "texture",
                    # "radial_distribution",
                    # "zernike",
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
    "nchannels": 2,
    "fl_channels": fl_channels,
    "extract_multich_tree": _create_extract_multich_tree(range(1, 3)),
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
}

output_path = Path("output")
capture_order: str = base_pipeline["io"]["capture_order"]
result, _ = run_pipeline_and_post(
    pipeline=base_pipeline,
    img_source=input_path,
    output_path=output_path,
    fov=input_path.name[1:],
    overwrite=True,
)
