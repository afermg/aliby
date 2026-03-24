from pathlib import Path

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post

# import pooch
# dirpath = pooch.retrieve("/datastore/alan/aliby/test_dataset/", known_hash="")
input_path = Path("/datastore/alan/aliby/test_dataset/")
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "WFC"  # Plate, Well, Channel Foci
# input_dimensions = "YX"
# nchannels = 5

dif = DatasetDir(
    input_path,
    regex=regex,
    capture_order=capture_order,
)

positions = (
    dif.get_position_ids()
)  # This asserts that at least one set of images is found.

"""Build a pipeline
from aliby.config import build_pipeline
addresses = [f"ipc:///tmp/cellpose_{i}.ipc" for i in range(1, 4)]

build_pipeline(positions[0], n_devices=1, addresses=addresses, extract_ncores=None)

"""
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
        "segmentation_channel": {"nuclei": 1, "cell": 2},
    },
    "nchannels": 5,
    "fl_channels": range(0, 5),
    "extract_multich_tree": {
        "tree": {
            (0, 1): {"None": {"max": ["pearson", "costes", "manders_fold", "rwc"]}},
            (0, 4): {"None": {"max": ["pearson", "costes", "manders_fold", "rwc"]}},
            (2, 3): {"None": {"max": ["pearson", "costes", "manders_fold", "rwc"]}},
            (3, 4): {"None": {"max": ["pearson", "costes", "manders_fold", "rwc"]}},
        },
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
            "ref_channel": 0,
            "ref_z": 0,
            "calculate_drift": False,
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
                0: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "feret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                    ]
                },
                1: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "feret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                    ]
                },
                2: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "feret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                    ]
                },
                3: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "feret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                    ]
                },
                4: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "feret",
                        "texture",
                        "radial_distribution",
                        "zernike",
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
run_pipeline_and_post(
    img_source=path,
    pipeline=pipeline,
    output_path="./",
    fov=key,
)
