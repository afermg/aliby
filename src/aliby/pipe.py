#!/usr/bin/env jupyter

"""
New and simpler pipeline that uses dictionaries as parameters and to define variable and between-step method execution.
"""

from copy import copy
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aliby.io.image import dispatch_image
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import Tiler, TilerParameters
from aliby.track.dispatch import dispatch_tracker
from extraction.extract import (
    extract_tree,
    extract_tree_multi,
    format_extraction,
    process_tree_masks,
)


def init_step(
    step_name: str,
    parameters: dict[str, str or callable or int or dict],
    other_steps: callable,
) -> callable:
    """
    Set up the parameters for any step. This mostly includes dispatching a specific step subtype.
    """
    match step_name:
        case "tile":
            image_kwargs = parameters["image_kwargs"]
            tiler_kwargs = {k: v for k, v in parameters.items() if k != "image_kwargs"}
            image_type = dispatch_image(source=image_kwargs["source"])
            image = image_type(**image_kwargs)

            step = Tiler.from_image(image, TilerParameters(**tiler_kwargs))
        case s if s.startswith("segment"):
            if (
                parameters["segmenter_kwargs"]["kind"] == "baby"
            ):  # Baby needs a tiler inside
                parameters["segmenter_kwargs"]["tiler"] = other_steps["tile"]
            step = dispatch_segmenter(**{
                **parameters["segmenter_kwargs"],
            })
        case "track":
            if (
                parameters["kind"] == "baby"
            ):  # Tracker needs to pull info from baby crawler
                parameters["crawler"] = other_steps["segment"].crawler
            step = dispatch_tracker(**parameters)
        case s if s.startswith("extract_"):
            step = partial(
                process_tree_masks, measure_fn=extract_tree, tree=parameters["tree"]
            )
        case s if s.startswith("extractmulti_"):
            step = partial(
                process_tree_masks, measure_fn=extract_tree_multi, tree=parameters
            )
        case _:
            raise Exception("Invalid step name")

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
    state: dict = None,
    steps_dir: str = None,
) -> dict:
    """Run one step of the pipeline."""
    if state is None:
        state = {}

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

        passed_data = {}
        for kwd, from_step in this_step_receives:
            passed_value = state["data"].get(from_step, [])

            if len(passed_value):
                if (
                    step_name == "track" and kwd == "masks"
                ):  # Only tracking segmentation masks require multiple time points
                    # Convert tp,tile,y,x to tile,tp,y,x for stitch tracking
                    passed_data[kwd] = [
                        [tp_tiles[tile] for tp_tiles in passed_value[-2:]]
                        for tile in range(len(passed_value[-1]))
                    ]
                else:  # We only care about the last time point
                    last_value = passed_value[-1]
                    if isinstance(last_value, dict):  # Select a subfield of the data
                        last_value = last_value[kwd]
                    passed_data[kwd] = last_value

                    # passed_data[kwd] = passed_value

        # Run step
        args = []
        if (
            step_name.startswith("segment")
            and parameters["segmenter_kwargs"]["kind"] != "baby"
        ):  # Pass correct images from tiler
            source_step, method, param_name = passed_methods.get(step_name)
            args = getattr(state["fn"][source_step], method)(tp, parameters[param_name])

        step_result = run_step(step, *args, tp=tp, **passed_data)

        # Save state
        steps_to_write = pipeline.get("save")
        save_interval = pipeline.get("save_interval", 0)
        if len(steps_to_write) and not (tp % save_interval):
            if step_name in steps_to_write:
                write_fn = dispatch_write_fn(step_name)
                write_fn(
                    step_result,
                    steps_dir=steps_dir,
                    step_identifier=step_name,
                    tp=tp,
                )

        # Update state
        # TODO replace this with a variable to adjust ntps in memory
        state["data"][step_name].append(step_result)
        # We carry on all the states, as they are necessary for processing later
        # state["data"][step_name] = state["data"][step_name][-5:]

        # Update or initialise objects
        if step_name not in state["fn"]:
            state["fn"][step_name] = step
        state["tps"][step_name] = tp + 1

    return state


def _validate_pipeline(pipeline: dict):
    """
    Sanity checks before computationally expensive operations.
    """
    steps_to_write = pipeline.get("save")
    assert not steps_to_write or set(steps_to_write).intersection(pipeline["steps"])


# TODO pass sorted images instead of wildcard
def run_pipeline(
    pipeline: dict, img_source: str or list[str], ntps: int, steps_dir: str = None
) -> pa.lib.Table:
    _validate_pipeline(pipeline)

    pipeline = copy(pipeline)
    pipeline["steps"]["tile"]["image_kwargs"]["source"] = img_source
    data = []
    state = {}

    for tp in range(ntps):
        state = pipeline_step(pipeline, state, steps_dir=steps_dir)
        for step_name in pipeline["steps"]:
            if step_name.startswith("ext"):
                table = format_extraction(state["data"][step_name][-1])
                if len(table):  # Cover case whence measurements are empty
                    table = table.append_column(
                        "object",
                        pa.array([step_name.split("_")[-1]] * len(table), pa.string()),
                    )
                    table = table.append_column(
                        "tp",
                        pa.array([tp] * len(table), pa.uint8()),
                    )
                    data.append(table)

    extracted_fov = pa.concat_tables(data)
    return extracted_fov


def run_pipeline_return_state(
    pipeline: dict, img_source: str or list[str], ntps: int, steps_dir: str = None
) -> pa.lib.Table:
    _validate_pipeline(pipeline)

    pipeline = copy(pipeline)
    pipeline["steps"]["tile"]["image_kwargs"]["source"] = img_source
    state = {}

    for _ in range(ntps):
        state = pipeline_step(pipeline, state, steps_dir=steps_dir)

    return state


def run_pipeline_save(out_file: Path, overwrite: bool = False, **kwargs) -> None:
    """
    Runs a pipeline and saves the result to a parquet file.

    Parameters
    ----------
    base_pipeline : dict
        The base pipeline configuration.
    img_source : str or list[str]
        Input files for the pipeline. It can be a list of files
    or an expression with a wildcard.
    out_file : str or Path
        Output file path for the result.

    Returns
    -------
    result
        The result of running the pipeline.
    """
    print(f"Running {out_file}")
    result = None
    if overwrite or not Path(out_file).exists():
        result = run_pipeline(**kwargs)
        out_dir = Path(out_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(result, out_file)

    return result


def dispatch_write_fn(
    step_name: str,
):
    match step_name:
        case s if s.startswith("segment"):
            return write_ndarray

        case s if s.startswith("tile"):
            return write_ndarray

        case _:
            raise Exception(f"Writing {step_name} is not supported yet")


def write_ndarray(result, steps_dir: Path, step_identifier: str or int, tp: int):
    this_step_path = Path(steps_dir) / step_identifier
    this_step_path.mkdir(exist_ok=True, parents=True)
    if step_identifier == "tile":
        step_identifier = "pixels"
        result = result["pixels"]

    out_file = this_step_path / f"{tp:04d}.npz"
    assert isinstance(result, np.ndarray), (
        f"Output is {type(result)} instead of ndarray"
    )
    np.savez(out_file, np.array(result))
