"""
This test checks the integration of the imaging pipeline for nuclear segmentation
and measurement using Cellpose. It verifies that:
1. The DatasetDir correctly identifies positions from TIFF files using a regex.
2. The Tiler correctly processes the full image when tile_size is None.
3. The Cellpose segmenter correctly generates masks for the specified channel.
4. The extractor successfully computes intensity features for the segmented nuclei.
5. The pipeline correctly handles data passing (masks and pixels) between steps.


The `pipeline` dictionary can be built using helper function in `src/aliby/pipe_builder.py`.
"""

import pytest
from pathlib import Path

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps

input_path = Path("/datastore/alan/aliby/test_dataset/data/crop_cellpainting_256")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


@pytest.mark.skipif(not input_path.exists(), reason="Test dataset not found")
def test_cellpose_minimal(tmp_path):
    dif = DatasetDir(
        input_path,
        regex=regex,
        capture_order=capture_order,
    )

    positions = (
        dif.get_position_ids()
    )  # This asserts that at least one set of images is found.

    key = positions[0]["key"]
    path = positions[0]["path"]

    pipeline = {
        "io": {
            "input_path": {
                "key": key,
                "path": path,
            },
            "capture_order": "CYX",
            "segmentation_channel": {"nuclei": 1},
        },
        "steps": {
            "tile": {
                "image_kwargs": {
                    "source": {
                        "key": key,
                        "path": path,
                    },
                    "regex": regex,
                    "capture_order": capture_order,
                },
                "tile_size": None,
            },
            "segment_nuclei": {
                "segmenter_kwargs": {
                    "kind": "cellpose",
                },
                "channel_to_segment": 1,
            },
            "extract_nuclei": {
                "tree": {
                    "None": {
                        "None": [
                            "sizeshape",
                        ]
                    },
                    0: {
                        "max": [
                            "intensity",
                        ]
                    },
                },
            },
        },
        "passed_data": {
            "extract_nuclei": [("masks", "segment_nuclei"), ("pixels", "tile")],
        },
        "passed_methods": {
            "segment_nuclei": ("tile", "get_fczyx"),
        },
        "save": ("segment_nuclei",),  # Which steps to write to disk
    }
    run_pipeline_and_post(
        pipeline=pipeline,
        pipeline_name=key,
        output_path=tmp_path,
    )


DATASETS = [
    {
        "name": "crop_cellpainting_256",
        "path": "/datastore/alan/aliby/test_dataset/data/crop_cellpainting_256",
        "regex": ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif",
        "capture_order": "WFC",
        "channels_to_segment": {"nuclei": 1, "cell": 0},
        "channels_to_extract": [0, 1],
    },
    {
        "name": "crop_timeseries_alcatras_square_same_channels_293",
        "path": "/datastore/alan/aliby/test_dataset/data/crop_timeseries_alcatras_square_same_channels_293",
        "regex": ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "capture_order": "FTCZ",
        "channels_to_segment": {"cells": 0},
        "channels_to_extract": [0, 1],
    },
    {
        "name": "crop_timeseries_alcatras_round_diff_dims_293",
        "path": "/datastore/alan/aliby/test_dataset/data/crop_timeseries_alcatras_round_diff_dims_293",
        "regex": ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "capture_order": "FTCZ",
        "channels_to_segment": {"cells": 0},
        "channels_to_extract": [0, 1],
    },
]


@pytest.mark.parametrize("ds_info", DATASETS, ids=lambda x: x["name"])
def test_pipeline_builder(ds_info, tmp_path):
    input_path = Path(ds_info["path"])
    if not input_path.exists():
        pytest.skip(f"Test dataset {input_path} not found")

    dif = DatasetDir(
        input_path,
        regex=ds_info["regex"],
        capture_order=ds_info["capture_order"],
    )

    positions = dif.get_position_ids()
    assert len(positions) > 0

    key = positions[0]["key"]
    path = positions[0]["path"]

    pipeline = build_pipeline_steps(
        channels_to_segment=ds_info["channels_to_segment"],
        channels_to_extract=ds_info["channels_to_extract"],
        features_to_extract=["intensity", "sizeshape"],
    )

    pipeline["io"] = {
        "input_path": {"key": key, "path": path},
        "capture_order": ds_info["capture_order"],
    }

    pipeline["steps"]["tile"]["image_kwargs"] = {
        "source": {"key": key, "path": path},
        "regex": ds_info["regex"],
        "capture_order": ds_info["capture_order"],
    }

    if "T" in ds_info["capture_order"]:
        pipeline["ntps"] = 2

    run_pipeline_and_post(
        pipeline=pipeline,
        pipeline_name=key,
        output_path=tmp_path,
    )
