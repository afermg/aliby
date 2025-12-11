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
import pyarrow as pa

from aliby.global_steps import dispatch_global_step
from aliby.io.image import dispatch_image
from aliby.io.write import dispatch_write_fn
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import Tiler, TilerParameters, dispatch_tiler
from aliby.track.dispatch import dispatch_tracker
from extraction.extract import (
    extract_tree,
    extract_tree_multi,
    format_extraction,
    process_tree_masks,
    process_tree_masks_overlap,
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
            image_kwargs = parameters.pop("image_kwargs")
            tiler_constructor = dispatch_tiler(parameters.pop("kind", None), parameters)
            image_type = dispatch_image(source=image_kwargs["source"])
            image = image_type(**image_kwargs)
            step = tiler_constructor(image)
        case s if s.startswith("segment"):
            if parameters["segmenter_kwargs"]["kind"].endswith(
                "baby"
            ):  # Baby needs a tiler inside
                parameters["segmenter_kwargs"]["tiler"] = other_steps["tile"]
            step = dispatch_segmenter(**{**parameters["segmenter_kwargs"]})
        case "track":
            if parameters["kind"].endswith(
                "baby"
            ):  # tracker needs to pull info from baby crawler
                parameters["crawler"] = other_steps["segment"].crawler
            step = dispatch_tracker(**parameters)
        case s if s.startswith("extract_"):
            # Whether to use overlapping masks or not
            process = process_tree_masks
            measure_fn = extract_tree
            if parameters.get("overlap"):
                process = process_tree_masks_overlap
                measure_fn = partial(extract_tree, overlap=True)
            step = partial(
                process,
                measure_fn=measure_fn,
                tree=parameters["tree"],
                **parameters.get("kwargs", {}),
            )
        case s if s.startswith("extractmulti_"):
            step = partial(
                process_tree_masks,
                measure_fn=extract_tree_multi,
                tree=parameters["tree"],
                **parameters.get("kwargs", {}),
            )
            # Nahual steps (running server in a different process)
        case s if s.startswith("nahual"):
            # Validate that we can contact the server-side
            # Setup also helps check that the remote server exists

            address = parameters["address"]  # Must have!
            setup, process = dispatch_global_step(step_name)
            # print(f"NAHUAL: Setting up remote process on {parameters['address']}.")
            setup_output = setup(parameters["parameters"], address=address)
            print(f"NAHUAL: Remote process set up, returned {setup_output}.")

            # For the final step we provide the address used for setting the remote up
            step = partial(process, address=address)
        case _:
            raise Exception(f"Invalid step name {step_name=}")

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

        # Specifies method calls between steps to get data.
        # Format: {consumer_step: (producer_step, method_name, parameter_key)}
        # parameter_key is pulled from the "parameters" subdict
        passed_data = {}
        for kwd, from_step in this_step_receives:
            passed_value = state["data"].get(from_step, [])

            if len(passed_value):
                if (
                    step_name == "track" and kwd == "masks"
                ):  # Only tracking  masks require multiple time points
                    # Convert tp,tile,y,x -> tile,tp,y,x for stitch tracking
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

        # HACK Adjust some edge cases to match BABY & Cellpose segmentation and downstream
        # args = []
        # if step_name.startswith("segment") or (
        #     step_name.startswith("extract")
        #     and any(x for x in steps if x.endswith("baby"))
        # ):
        #     # Pass correct images from tiler, except for "baby" which does everything by itself
        #     segment_kind = parameters["segmenter_kwargs"]["kind"]
        #     # if segment_kind != "nahual_baby":
        #     source_step, method, param_name = passed_methods.get(step_name)

        #     # Use tiler to pull data
        #     args = getattr(state["fn"][source_step], method)(tp, parameters[param_name])
        #     if segment_kind.endswith(
        #         "baby"
        #     ):  # We will expand the args so this is necessary TODO Homogenize
        #         args = (args,)

        # if step_name.startswith("extract") and any(x for x in steps if "baby" in x):
        #     # source_step, method, param_name = passed_methods.get(step_name)
        #     # args = getattr(state["fn"][source_step], method)(tp, parameters[param_name])
        #     passed_data["pixels"] = args
        #     args = []

        args = []
        step_result = run_step(step, *args, tp=tp, **passed_data)

        # Save state
        steps_to_write = pipeline.get("save", [])
        save_interval = pipeline.get("save_interval", 0)
        if len(steps_to_write) and (tp == 0 or (tp % save_interval) == 0):
            if step_name in steps_to_write:
                print(f"Saving {step_name} to {steps_dir}")
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
    assert "ntps" in pipeline.get("io", {}), (
        "You must pass the number of time points to analyse."
    )
    for k in pipeline["steps"]:
        if k.startswith("nahual"):
            assert "address" in pipeline["steps"][k], (
                "If using a nahual-deployed model you must provide an address."
            )


def run_pipeline_return_state(
    pipeline: dict, img_source: str or list[str], steps_dir: str = None
) -> dict:
    _validate_pipeline(pipeline)

    pipeline = deepcopy(pipeline)
    pipeline["steps"]["tile"]["image_kwargs"]["source"] = img_source
    state = {}

    ntps = pipeline.get("io", {"ntps": 20})["ntps"]
    for _ in range(ntps):
        # print(f"Processing frame {frame}")
        state = pipeline_step(pipeline, state, steps_dir=steps_dir)
        # print(f"Finished frame {frame}")

    return state


def run_pipeline_and_post(
    pipeline: dict,
    img_source: str or list[str],
    output_path: Path = None,
    fov: str = None,
    overwrite: bool = True,
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
    if output_path is None:
        output_path = Path(pipeline["io"]["output_path"])

    steps_dir = output_path / "steps" / fov
    profiles_file = output_path / "profiles" / f"{fov}.parquet"

    profiles = None
    post_results = None
    ntps = pipeline["io"].get("ntps", 2)

    # Main processing loop
    if overwrite or not profiles_file.exists():
        # print(f"Processing {fov}")
        state = run_pipeline_return_state(pipeline, img_source, steps_dir=steps_dir)

        # Aggregate profiles from the state output
        profiles = get_profiles_from_state(state, pipeline)

        # Save files
        profiles_file.parent.mkdir(parents=True, exist_ok=True)
        pyarrow.parquet.write_table(profiles, profiles_file)

        # Run global processing steps (post-processing)
        post_results = {}

        ntps = pipeline["io"]["ntps"]
        if ntps == 1:  # Temporarily do not perform global operations on non-timeseries
            return profiles, post_results

        for step_name, parameters in pipeline["global_steps"].items():
            associated_data = [
                x for x in pipeline["global_passed_data"] if x.startswith(step_name)
            ]
            assert len(associated_data), (
                f"Incorrect pipeline: Missing information of which data to ingest for step {step_name}"
            )
            # TODO if nahual data is present, validate that the maximum target_label is below
            # the maximum label in the profiles
            for output_name in associated_data:
                state["fn"] = init_step(step_name, parameters)

                input_data = get_step_output(
                    state["data"], pipeline["global_passed_data"][output_name]
                )
                post_results[output_name] = state["fn"](input_data=input_data)

        # Save global steps into files (steps are saved as they go, not at the end)
        if step_name in pipeline["save"]:
            write_fn = dispatch_write_fn(step_name)
            for output_pathname in post_results:
                if output_pathname.startswith(step_name):
                    write_fn(
                        post_results[output_pathname],
                        output_path,
                        subpath=output_pathname,
                        filename=fov,
                    )
    else:
        print(f"Skipping {fov}, as it exists")

    return profiles, post_results


def get_profiles_from_state(state: dict, pipeline: dict) -> pyarrow.Table:
    # Isolate features for every time point
    # We assume that extraction steps start with "ext"
    profiles = pyarrow.Table.from_pylist(
        [],
        schema=pa.schema([
            pa.field("tile", pa.int64()),
            pa.field("label", pa.int64()),
            pa.field("branch", pa.string()),
            pa.field("metric", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("object", pa.string()),
            pa.field("tp", pa.int64()),
        ]),
    )
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
