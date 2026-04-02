from pathlib import Path
import pytest

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post

input_path = Path("/datastore/alan/aliby/test_dataset/")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci


@pytest.mark.skipif(True, reason="Test dataset not found")
def test_imageset():
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
            "ntps": 1,
            "segmentation_channel": {"nuclei": 1},
        },
        "nchannels": 5,
        "fl_channels": range(0, 5),
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
                "img_channel": 1,
            },
            "extract_nuclei": {
                "channels": range(0, 5),
                "tree": {
                    1: {
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
            "segment_nuclei": ("tile", "get_tp_data", "img_channel"),
        },
        "save": ("segment_nuclei",),
        "save_interval": 1,
    }
    result = run_pipeline_and_post(
        img_source=path,
        pipeline=pipeline,
        output_path="./",
        fov=key,
    )
