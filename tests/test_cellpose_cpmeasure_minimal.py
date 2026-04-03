"""
This test checks the integration of the imaging pipeline for nuclear segmentation
and measurement using Cellpose. It verifies that:
1. The DatasetDir correctly identifies positions from TIFF files using a regex.
2. The Tiler correctly processes the full image when tile_size is None.
3. The Cellpose segmenter correctly generates masks for the specified channel.
4. The extractor successfully computes intensity features for the segmented nuclei.
5. The pipeline correctly handles data passing (masks and pixels) between steps.


The `pipeline` dictionary can be built using helper function in `src/aliby/config.py`.
"""

import pytest
from pathlib import Path

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post

input_path = Path("/datastore/alan/aliby/test_dataset/256")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


@pytest.mark.skipif(True, reason="Test dataset not found")
def test_cellpose_minimal():
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
                    "channel_to_segment": 1,
                },
            },
            "extract_nuclei": {
                "tree": {
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
        img_source=path,
        pipeline=pipeline,
        output_path="test_delme/",
        fov=key,
    )
