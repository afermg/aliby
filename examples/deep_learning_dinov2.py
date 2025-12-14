#!/usr/bin/env jupyter
"""Run a deep learning embedding via Nahual."""

import os
import shutil
from functools import partial
from itertools import combinations
from pathlib import Path
from time import strftime

from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post

dataset = "araceli"
datasets_path = Path(
    f"/datastore/shared/araceli_phenix/BR00141077_split_paired/model_148/"
)
base_output_path = Path("/datastore/shared/aliby_output/")
dataset_paths = [x for x in datasets_path.glob("*/") if x.name.endswith("wholecrop")]
dataset_paths = [dataset_paths[1]]
regex = ".*(r.+c.+)f([0-9][0-9])p01-rgb.tiff"
capture_order = "WF"
input_dimensions = "YXC"
nchannels = 3


def process_input_path(input_path: str):
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "capture_order": "WF",
            "regex": ".*(r.+c.+)f([0-9][0-9])p01-rgb.tiff",
            "input_dimensions": "YXC",
        },
        "ntps": 1,
        "tile": {
            "kind": "crop",
            "tile_size": 420,
            "calculate_drift": False,
        },
    }
    embed_params = dict(
        address="ipc:///tmp/dinov2.ipc",
        setup_params=dict(
            repo_or_dir="facebookresearch/dinov2",
            model_name="dinov2_vits14_lc",
            execution_params=dict(channels=(0, 1, 2)),
        ),
    )

    base_pipeline = {
        "io": {**fluo_base_config},
        "steps": dict(
            tile=dict(
                **fluo_base_config["tile"],
                **dict(
                    image_kwargs=dict(
                        source=input_path,
                        **fluo_base_config["image_kwargs"],
                    )
                ),
            ),
            nahual_embed_dinov2=embed_params,
        ),
        "passed_data": dict(nahual_embed_dinov2=[("pixels", "tile", "data")]),
        "save": (),
        "save_interval": 1,
    }

    result, _ = run_pipeline_and_post(
        pipeline=base_pipeline,
        img_source=input_path,
        output_path=output_path,
        fov=Path(input_path[0]).name,
        overwrite=True,
    )
    return result


dsets = list(
    map(
        partial(dispatch_dataset, regex=regex, capture_order=capture_order),
        dataset_paths,
    )
)
# %%
for dataset_dir, dset in tqdm(zip(dataset_paths, dsets), total=len(dsets)):
    input_paths = list(dset.get_position_ids().values())

    output_path = base_output_path / dataset_dir.name
    if __name__ == "__main__":  # Add logging
        timestamp = strftime("%s%m%d%H%M")

        output_path = base_output_path / "tmp_araceli" / dataset_dir.name

        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset}.log")
        # shutil.copy(__file__, output_path / f"{timestamp}_script.py")

    if False:
        result = Parallel(10)(delayed(process_input_path)(x) for x in input_paths)
    else:
        result = []
        for input_path in input_paths:
            new_result = process_input_path(input_path)
            breakpoint()
            result.append(new_result)
