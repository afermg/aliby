#!/usr/bin/env jupyter
"""Test a basic pipeline with a unique zarr directory as input."""

from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

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


fpath = "../../../../../work/datasets/jump_toy_compress/zstd.zarr/"
dset = dispatch_dataset(fpath, is_monozarr=True)
image_paths = dset.get_position_ids()
# t2_plate = [v for k, v in image_paths.items() if "BR00121438" in k]
t2_plate = [v for k, v in image_paths.items() if "BR00125163" in k]

# input_path = t2_plate[0]
input_path = list(image_paths.values())[0]

for i, v in enumerate(np.array(input_path)):
    plt.imshow(v)
    plt.savefig(f"{i}_new.png")
    plt.close()

# %%
fluo_base_config = {
    "input_path": input_path,
    "capture_order": "CYX",
    "ntps": 1,
    # AGP, DNA, ER, Mito, RNA
    "segmentation_channel": {"nuclei": 1, "cell": 2},
}
fl_channels = range(0, 5)

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
                    "intensity",
                    "sizeshape",
                    "ferret",
                    "texture",
                    "radial_distribution",
                    "zernike",
                    "radial_zernikes",
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
        (("", extract_base),),
        segmentation_channel,
    )
    if len(var)
}


base_pipeline = {
    "io": {**fluo_base_config},
    "nchannels": len(fl_channels),
    "fl_channels": fl_channels,
    # "extract_multich_tree": _create_extract_multich_tree(fl_channels),
    "steps": dict(
        tile=dict(
            image_kwargs=dict(
                source=input_path,
                capture_order=fluo_base_config["capture_order"],
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

# %%
output_path = Path("output")
capture_order: str = base_pipeline["io"]["capture_order"]
t0 = perf_counter()
result, _ = run_pipeline_and_post(
    pipeline=base_pipeline,
    img_source=input_path,
    output_path=output_path,
    fov=input_path.name[1:],
    overwrite=True,
)
print(f"It took {perf_counter() - t0} seconds")
