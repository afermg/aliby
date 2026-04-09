"""
Evaluate the different ways supported to load images.
"""

import pytest

from aliby.io.dataset import DatasetDir, DatasetZarr
from aliby.io.image import (
    dispatch_image,
    ImageList,
    ImageZarr,
    ImageDir,
    ImageMultiTiff,
)
# from aliby.pipe import run_pipeline_and_post

from common import REGEX_PARAMETERS

# --- Dispatcher test ---


def test_dispatch_image_types():
    assert dispatch_image({"path": ["a.tif", "b.tif"]}) is ImageList
    assert dispatch_image(["a.tif", "b.tif"]) is ImageList
    assert dispatch_image({"path": "/path.zarr", "key": "1"}) is ImageZarr
    assert dispatch_image("*.tif") is ImageList
    assert dispatch_image("img.tif") is ImageMultiTiff
    assert dispatch_image("/tmp") is ImageDir


# --- ImageList tests ---


@pytest.mark.parametrize("dataset, regex, capture_order", REGEX_PARAMETERS)
def test_image_list(data_dir, dataset, regex, capture_order):
    dataset_obj = DatasetDir(
        data_dir / dataset,
        regex=regex,
        capture_order=capture_order,
    )
    positions = dataset_obj.get_position_ids()

    img = ImageList(
        source=positions[0]["path"],
        regex=regex,
        capture_order=capture_order,
    )

    assert img.name is not None
    data = img.get_data_lazy()
    assert data is not None
    assert len(data.shape) == 5
    assert img.dimorder is not None


# --- ImageZarr tests ---


@pytest.mark.parametrize("dataset", [f"{x[0]}.zarr" for x in REGEX_PARAMETERS])
def test_image_zarr(data_dir, dataset):
    dataset_obj = DatasetZarr(data_dir / dataset)
    positions = dataset_obj.get_position_ids()

    img = ImageZarr(source=positions[0])

    data = img.get_data_lazy()
    assert data is not None
    assert len(data.shape) == 5
    assert img.dimorder is not None
    assert img.name is not None


# # --- Original test kept skipped ---
# @pytest.mark.skipif(True, reason="Test dataset not found")
# def test_imageset_pipeline(data_dir):
#     dataset = REGEX_PARAMETERS[0][0]
#     regex = REGEX_PARAMETERS[0][1]
#     capture_order = REGEX_PARAMETERS[0][2]

#     dif = DatasetDir(
#         data_dir / dataset,
#         regex=regex,
#         capture_order=capture_order,
#     )

#     positions = (
#         dif.get_position_ids()
#     )  # This asserts that at least one set of images is found.

#     key = positions[0]["key"]
#     path = positions[0]["path"]

#     pipeline = {
#         "io": {
#             "input_path": {
#                 "key": key,
#                 "path": path,
#             },
#             "capture_order": "CYX",
#             "segmentation_channel": {"nuclei": 1},
#         },
#         "nchannels": 5,
#         "fl_channels": range(0, 5),
#         "steps": {
#             "tile": {
#                 "image_kwargs": {
#                     "source": {
#                         "key": key,
#                         "path": path,
#                     },
#                     "regex": regex,
#                     "capture_order": capture_order,
#                 },
#                 "tile_size": None,
#             },
#             "segment_nuclei": {
#                 "img_channel": 1,
#                 "segmenter_kwargs": {
#                     "kind": "cellpose",
#                     "channel_to_segment": 1,
#                 },
#             },
#             "extract_nuclei": {
#                 "channels": range(0, 5),
#                 "tree": {
#                     1: {
#                         "max": [
#                             "intensity",
#                         ]
#                     },
#                 },
#             },
#         },
#         "passed_data": {
#             "extract_nuclei": [("masks", "segment_nuclei"), ("pixels", "tile")],
#         },
#         "passed_methods": {
#             "segment_nuclei": ("tile", "get_tp_data", "img_channel"),
#         },
#         "save": ("segment_nuclei",),
#         "save_interval": 1,
#     }
#     run_pipeline_and_post(
#         img_source=path,
#         pipeline=pipeline,
#         output_path="./",
#         fov=key,
#     )
