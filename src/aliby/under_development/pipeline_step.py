#!/usr/bin/env jupyter

"""
Isolated pipeline step
"""

from copy import copy
from itertools import cycle

from aliby.io.dataset import DatasetDir
from aliby.io.image import dispatch_image
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import Tiler, TilerParameters
from aliby.track.dispatch import dispatch_tracker
from extraction.core.extractor import Extractor, ExtractorParameters

base_pipeline = dict(
    steps=dict(
        tile=dict(
            image_kwargs=dict(
                path="/home/amunoz/gsk/batches/ELN201687_subset/ELN201687_subset/H00DJKJread1BF48hrs_20230926_095825/",
                regex=".+\/(.+)\/_.+([A-P][0-9]{2}).*_T([0-9]{4})F([0-9]{3}).*Z([0-9]{2}).*[0-9].tif",
                dimorder="CTZ",
            ),
            tile_size=None,
            ref_channel=0,
            ref_z=0,
        ),
        segment=dict(channel=0),
        track=dict(kind="stitch"),
        extract=dict(
            channels=(1, 2),
            tree={
                "general": {
                    "None": [
                        "area",
                        "eccentricity",
                        "centroid_x",
                        "centroid_y",
                    ],
                }
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
        segment=("tile", "get_tp_data", "channel"),
    ),
)


def init_step(
    step_name: str,
    parameters: dict[str, str or callable or int or dict],
    other_steps: callable,
) -> callable:
    match step_name:
        case "tile":
            image_kwargs = parameters["image_kwargs"]
            tiler_kwargs = {k: v for k, v in parameters.items() if k != "image_kwargs"}
            image = dispatch_image(source=image_kwargs["path"])(**image_kwargs)
            step = Tiler.from_image(image, TilerParameters(**tiler_kwargs))
        case "segment":
            step = dispatch_segmenter("nuclei", diameter=None, channels=[0, 0])
        case "track":
            step = dispatch_tracker(**parameters)
        case "extract":
            tiler = other_steps["tile"]
            step = Extractor(ExtractorParameters(parameters["tree"]), tiler=tiler)
        case _:
            raise ("Invalid step name")

    return step


def run_step(step, *args, **kwargs):
    if hasattr(step, "run_tp"):  # in case of older OO-style
        result = step.run_tp(*args, **kwargs)
    else:  # Functional version, all relevant kwargs are provided but no more
        if "tp" in kwargs:
            del kwargs["tp"]
        result = step(*args, **kwargs)

    return result


def pipeline_step(
    pipeline: dict,
    state: dict = {},
) -> dict:
    """ """
    steps = pipeline["steps"]
    passed_data = pipeline["passed_data"]
    passed_methods = pipeline["passed_methods"]
    tp = list(state.get("tps", {None: 0}).values())[0]
    if not tp:  # Initialise steps
        state = {"tps": dict(zip(steps, cycle([0]))), "data": {}, "fn": {}}

    for step_name, parameters in steps.items():
        # Get or initialise step
        if step_name not in state["data"]:
            state["data"][step_name] = []
        step = state["fn"].get(step_name, init_step(step_name, parameters, state["fn"]))

        # Pass input data if available
        this_step_receives = pipeline["passed_data"].get(step_name, {})
        passed_data = {
            kwd: state["data"].get(from_step) for kwd, from_step in this_step_receives
        }

        # Run step
        args = []
        if step_name == "segment":  # Pass correct images from tiler
            source_step, method, param_name = passed_methods["segment"]
            args = getattr(state["fn"][source_step], method)(tp, parameters[param_name])

        state["data"][step_name].append(run_step(step, *args, tp=tp, **passed_data))

        # Update state
        state["fn"][step_name] = step
        state["tps"][step_name] = tp + 1

    return state


# Load dataset from a regular expression

path = "/home/amunoz/gsk/batches/ELN201687_subset/ELN201687_subset/H00DJKJread1BF48hrs_20230926_095825/"
dif = DatasetDir(
    path,
    regex=".+\/(.+)\/_.+([A-P][0-9]{2}).*_T([0-9]{4})F([0-9]{3}).*Z([0-9]{2}).*[0-9].tif",
    dimorder="CFTZ",
)


# create pipeline for ImageDir
# for wildcard, image_filenames in dif.get_position_ids().values():
wildcard, image_filenames = list(dif.get_position_ids().values())[0]
pipeline = copy(base_pipeline)
pipeline["steps"]["tile"]["image_kwargs"]["wildcard"] = wildcard
data = []
state = {}

for i in range(10):
    state = pipeline_step(pipeline, state)
    data.append(copy(state["data"]["extract"][-1]))

# pixels = tiler.get_tp_data(0, 0)
# masks = segment(pixels)
# tracked_mask = track(masks, masks)
# extracted_tp = extractor.run_tp([0], tree=tree, masks=[masks], save=False)
