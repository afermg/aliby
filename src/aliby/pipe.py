#!/usr/bin/env jupyter

"""
New and simpler pipeline that uses dictionaries as parameters and to define variable and between-step method execution.
"""

from copy import deepcopy
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Callable

import numpy
import pyarrow

from aliby.global_steps import dispatch_global_step
from aliby.io.image import dispatch_image
from aliby.io.write import dispatch_write_fn
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
    other_steps: callable = None,
) -> callable:
    """
    Set up the parameters for any step. This mostly includes dispatching a specific step subtype.

    If we need to perform some validation before commiting to a pipeline (e.g., checking the
    servers for a nahual process), this is the place to do it.
    """
    match step_name:
        case "tile":
            image_kwargs = parameters[
                "image_kwargs"
            ]  # TODO replace with pop() and simplify next line
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
            # Nahual steps (running server in a different process)
        case s if s.startswith("nahual"):
            # Validate that we can contact the server-side
            # Setup also helps check that the remote server exists
            setup_fn, process_fn = dispatch_global_step(step_name)
            # print(f"NAHUAL: Setting up remote process on {parameters['address']}.")
            setup_output = setup_fn(**parameters)
            print(f"NAHUAL: Remote process set up, returned {setup_output}.")

            # For the final step we provide the address used for setting the remote up
            step = partial(process_fn, address=parameters["address"])
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
                    subpath=step_name,
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


def run_pipeline(
    pipeline: dict, img_source: str or list[str], ntps: int, steps_dir: str = None
) -> pyarrow.Table:
    _validate_pipeline(pipeline)

    pipeline = deepcopy(pipeline)
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
                        pyarrow.array(
                            [step_name.split("_")[-1]] * len(table), pyarrow.string()
                        ),
                    )
                    table = table.append_column(
                        "tp",
                        pyarrow.array([tp] * len(table), pyarrow.uint8()),
                    )
                    data.append(table)

    extracted_fov = pyarrow.concat_tables(data)

    return extracted_fov


def run_pipeline_return_state(
    pipeline: dict, img_source: str or list[str], ntps: int, steps_dir: str = None
) -> dict:
    _validate_pipeline(pipeline)

    pipeline = deepcopy(pipeline)
    pipeline["steps"]["tile"]["image_kwargs"]["source"] = img_source
    state = {}

    for _ in range(ntps):
        # print(f"Processing frame {frame}")
        state = pipeline_step(pipeline, state, steps_dir=steps_dir)
        # print(f"Finished frame {frame}")

    return state


def run_pipeline_and_post(
    pipeline: dict,
    img_source: str or list[str],
    ntps: int,
    out_dir: Path = None,
    fov: str = None,
) -> tuple[pyarrow.Table, pyarrow.Table]:
    """
    Run a step-based pipeline and at the end run a series of post-processiong steps,
    such as tracking of the whole time series.

    Parameters
    -----

    Notes
    -----
    It returns the profiles in a tidy format
    (i.e., the frame number or time point is its own column).

    Assumptions:
    - extraction fields start with 'ext', and then are followed by the object name (e.g., cyto, nuclei)
    - The pipeline's output is nested in the following order: step -> time point -> tile.
    """
    steps_dir = out_dir / "steps" / fov

    # Main processing loop
    state = run_pipeline_return_state(pipeline, img_source, ntps, steps_dir=steps_dir)

    # Aggregate profiles from the state output
    profiles = get_profiles_from_state(state, pipeline)

    # Run global processing steps (post-processing)
    post_results = {}

    if ntps == 1:  # Temporarily do not perform global operations on non-timeseries
        return profiles, post_results

    for step_name, parameters in pipeline["global_steps"].items():
        associated_data = [
            x for x in pipeline["global_passed_data"] if x.startswith(step_name)
        ]
        assert len(associated_data), (
            f"Incorrect pipeline: Missing information of which data to ingest for step {step_name}"
        )
        for output_name in associated_data:
            state["fn"] = init_step(step_name, parameters)

            input_data = get_step_output(
                state["data"], pipeline["global_passed_data"][output_name]
            )
            post_results[output_name] = state["fn"](input_data=input_data)

        # Save global steps into files (steps are saved as they go, not at the end)
        if step_name in pipeline["save"]:
            write_fn = dispatch_write_fn(step_name)
            for out_dirname in post_results:
                if out_dirname.startswith(step_name):
                    write_fn(
                        post_results[out_dirname],
                        out_dir / out_dirname,
                        subpath=step_name,
                        filename=fov,
                    )

    return profiles, post_results


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
        pyarrow.parquet.write_table(result, out_file)

    return result


def get_profiles_from_state(state: dict, pipeline: dict) -> pyarrow.Table:
    # Isolate features for every time point
    # We assume that extraction steps start with "ext"
    profiles = pyarrow.Table.from_pylist([])
    extraction_steps = [
        step_name for step_name in pipeline["steps"] if step_name.startswith("ext")
    ]
    data = []
    for ext_step in extraction_steps:
        # Data is stored in the following order: step -> time points -> tiles
        for tp, ext_output in enumerate(state["data"][ext_step]):
            table: pyarrow.PyTable = format_extraction(ext_output)
            if len(table):  # Cover case whence measurements are empty
                table = table.append_column(
                    "object",
                    pyarrow.array(
                        [ext_step.split("_")[-1]] * len(table), pyarrow.string()
                    ),
                )
                table = table.append_column(
                    "tp",
                    pyarrow.array([tp] * len(table), pyarrow.uint8()),
                )
                data.append(table)

    if len(data):
        profiles = pyarrow.concat_tables(data)

    return profiles


def get_step_output(state_data: dict, fetchers: tuple[Callable | str]) -> numpy.ndarray:
    """Dynamic fetcher for other outputs. It aggregates data over time points.

    fetchers: if a string, just fetch the entire output, if a function (callable) apply this to the output of every time point

    Returns a
    """
    combined_outputs = []
    for fetcher in fetchers:
        if isinstance(fetcher, str):
            # HACK: This is assuming a monotile, we should instead output always assuming tiles
            aggregated_output = [x[0] for x in state_data[fetcher]]
        elif isinstance(fetcher, Callable):
            aggregated_output = fetcher(state_data)
        else:
            raise Exception(
                f"Invalid type, expected Callable or string, got {type(fetcher)}"
            )
        combined_outputs.append(aggregated_output)

    return numpy.asarray(combined_outputs)
