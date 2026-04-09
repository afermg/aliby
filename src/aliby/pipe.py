#!/usr/bin/env jupyter
"""
New and simpler pipeline that uses dictionaries as parameters and to define variable and between-step method execution.
"""

from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Callable

import numcodecs
import numpy
import pyarrow
import pyarrow as pa
from imagecodecs.numcodecs import Jpegxl
from loguru import logger


from aliby.global_steps import dispatch_global_step
from aliby.io.image import dispatch_image
from aliby.io.write import dispatch_write_fn
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import dispatch_tiler
from aliby.track.dispatch import dispatch_tracker
from extraction.extract import (
    extract_tree,
    extract_tree_multi,
    format_extraction,
    process_tree_masks,
    process_tree_masks_overlap,
)

numcodecs.register_codec(Jpegxl)

# from aliby.tile.tiler import Tiler, TilerParameters,


def configure_logging(file):
    # Remove default standard library logging handler
    logger.remove()

    # Configure file logging with rotation and compression
    logger.add(
        file,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    configure_logging()


def init_step(
    step_name: str,
    parameters: dict[str, str or callable or int or dict],
    other_steps: dict = None,
) -> callable:
    """
    Set up the parameters for any step. This mostly includes dispatching a specific step subtype.

    If we need to perform some validation before commiting to a pipeline (e.g., checking the
    servers for a nahual process), this is the place to do it.
    """
    if other_steps is None:
        other_steps = {}

    match step_name:
        case s if s.startswith("tile"):
            image_kwargs = parameters.pop("image_kwargs", None)
            if image_kwargs is None:
                raise ValueError(
                    f"Step '{step_name}' is missing required 'image_kwargs'."
                )
            if "source" not in image_kwargs:
                raise ValueError(
                    f"Step '{step_name}' 'image_kwargs' is missing required 'source'."
                )
            tiler_constructor = dispatch_tiler(parameters.pop("kind", None), parameters)
            image_type = dispatch_image(source=image_kwargs["source"])
            image = image_type(**image_kwargs)
            step = tiler_constructor(image)
        case s if s.startswith("segment"):
            seg_kwargs = parameters.get("segmenter_kwargs", {})
            if seg_kwargs.get("kind", "").endswith("baby"):  # Baby needs a tiler inside
                tiler_step = next(
                    (v for k, v in other_steps.items() if k.startswith("tile")), None
                )
                if tiler_step is None:
                    raise ValueError(
                        f"Step '{step_name}' using 'baby' requires a preceding 'tile' step."
                    )
                seg_kwargs["tiler"] = tiler_step
            if "channel_to_segment" not in parameters:
                raise ValueError(
                    f"Step '{step_name}' is missing required 'channel_to_segment'."
                )
            step = dispatch_segmenter(
                channel_to_segment=parameters["channel_to_segment"],
                **seg_kwargs,
            )
        case s if s.startswith("track"):
            if parameters.get("kind", "").endswith(
                "baby"
            ):  # tracker needs to pull info from baby crawler
                segment_step = next(
                    (v for k, v in other_steps.items() if k.startswith("segment")), None
                )
                if segment_step is None:
                    raise ValueError(
                        f"Step '{step_name}' using 'baby' tracking requires a preceding 'segment' step."
                    )
                parameters["crawler"] = segment_step.crawler
            step = dispatch_tracker(**parameters)
        case s if s.startswith("extract_"):
            if "tree" not in parameters:
                raise ValueError(f"Step '{step_name}' is missing required 'tree'.")
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
            if "tree" not in parameters:
                raise ValueError(f"Step '{step_name}' is missing required 'tree'.")
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
            address = parameters.get("address")
            if address is None:
                raise ValueError(
                    f"If using Nahual you must have an address, currently it is None in step '{step_name}'"
                )

            if step_name.startswith("nahual_embed"):
                if "setup_params" not in parameters:
                    raise ValueError(
                        f"Nahual embed step '{step_name}' is missing 'setup_params'."
                    )
                if "model_group" not in parameters:
                    raise ValueError(
                        f"Nahual embed step '{step_name}' is missing 'model_group'."
                    )

                from nahual.process import dispatch_setup_process

                setup_params = parameters["setup_params"]
                model_group = parameters["model_group"]

                setup, process = dispatch_setup_process(model_group)

                # If channel_ids exist subindex the input array
                selected_channels = parameters.get("selected_channels")
                if selected_channels:
                    process = partial(
                        slice_channels_process,
                        process=process,
                        selected_channels=selected_channels,
                    )

                info = setup(
                    setup_params,
                    address=address,
                )
                print(f"Embedder via nahual set up. Remote returned {info}")

                return partial(process, address=address)
            if step_name.startswith("nahual_track"):
                if "parameters" not in parameters:
                    raise ValueError(
                        f"Nahual track step '{step_name}' is missing 'parameters'."
                    )
                setup, process = dispatch_global_step(step_name)
                # print(f"NAHUAL: Setting up remote process on {parameters['address']}.")
                setup_output = setup(parameters["parameters"], address=address)
                print(f"NAHUAL: Remote process set up, returned {setup_output}.")

                # For the final step we provide the address used for setting the remote up
                step = partial(process, address=address)
        case _:
            raise ValueError(f"Invalid step name {step_name=}")

    return step


def slice_channels_process(
    data: numpy.ndarray,
    process: Callable,
    selected_channels: list[int] | numpy.ndarray,
    **kwargs,
) -> numpy.ndarray:
    """
    Apply a processing function to a subset of channels in a NumPy array.

    Parameters
    ----------
    data : numpy.ndarray
        The input array, expected to be at least 2D where the second
        dimension represents channels.
    process : callable
        A function to be applied to the sliced data. It should accept the
        sliced array as its first argument.
    selected_channels : list of int or numpy.ndarray
        Indices of the channels to select from the input data.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `process` function.

    Returns
    -------
    any
        The result of the `process` function applied to the selected data.
    """
    return process(data[:, selected_channels], **kwargs)


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
    passed_methods = pipeline.get("passed_methods", {})

    tp = list(state.get("tps", {None: 0}).values())[0]
    if not tp:  # Initialise steps
        state = {"tps": dict(zip(steps, cycle([0]))), "data": {}, "fn": {}}

    for step_name, parameters in steps.items():
        # Get or initialise step
        if step_name not in state["data"]:
            state["data"][step_name] = []
        if step_name not in state["fn"]:
            state["fn"][step_name] = init_step(step_name, parameters, state["fn"])
        step = state["fn"][step_name]

        # Pass input data if available
        this_step_receives = pipeline["passed_data"].get(step_name, {})

        # Specifies method calls between steps to get data.
        # Format: {consumer_step: (producer_step, method_name, parameter_key)}
        # parameter_key is pulled from the "parameters" subdict
        passed_data = {}
        for kwd, from_step, *varname in this_step_receives:
            passed_value = state["data"].get(from_step, [])

            step_argname = kwd
            if len(varname):  # Use a different variable name to match the step
                step_argname = varname[0]

            if len(passed_value):
                if (
                    step_name == "track" and kwd == "masks"
                ):  # Only tracking  masks require multiple time points
                    # Convert tp,tile,y,x -> tile,tp,y,x for stitch tracking
                    passed_data[step_argname] = [
                        [tp_tiles[tile] for tp_tiles in passed_value[-2:]]
                        for tile in range(len(passed_value[-1]))
                    ]
                else:  # We only care about the last time point
                    last_value = passed_value[-1]
                    if isinstance(last_value, dict):  # Select a subfield of the data
                        last_value = last_value[kwd]
                    passed_data[step_argname] = last_value

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
        if (
            step_name.startswith("segment")
            and parameters["segmenter_kwargs"]["kind"] != "baby"
        ):  # Pass correct images from tiler
            source_step, method = passed_methods.get(step_name)
            args = (
                getattr(state["fn"][source_step], method)(tp),
            )  # This assumes that the source step is an object that contains the data

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


def validate_pipeline(pipeline: dict) -> None:
    """
    Sanity checks before computationally expensive operations.
    Validates pipeline structure, dependencies, and parameters.
    """
    if not isinstance(pipeline, dict):
        raise TypeError("Pipeline configuration must be a dictionary.")

    if "steps" not in pipeline or not isinstance(pipeline["steps"], dict):
        raise ValueError(
            "Pipeline must contain a 'steps' dictionary mapping step names to parameters."
        )

    steps = pipeline["steps"]

    # Validate passed_data existence since pipeline_step accesses it directly
    if "passed_data" not in pipeline or not isinstance(pipeline["passed_data"], dict):
        raise ValueError("Pipeline must contain a 'passed_data' dictionary.")

    passed_data = pipeline["passed_data"]
    for target_step, dependencies in passed_data.items():
        if not isinstance(dependencies, (list, tuple)):
            raise TypeError(
                f"'passed_data' dependencies for step '{target_step}' must be a sequence."
            )
        for dep in dependencies:
            if not isinstance(dep, (list, tuple)) or len(dep) < 2:
                raise ValueError(
                    f"Invalid dependency format in 'passed_data' for '{target_step}': {dep}"
                )
            from_step = dep[1]
            if from_step not in steps:
                raise ValueError(
                    f"Step '{target_step}' expects data from '{from_step}', but '{from_step}' is not defined in 'steps'."
                )

    # Validate passed_methods
    passed_methods = pipeline.get("passed_methods", {})
    if not isinstance(passed_methods, dict):
        raise TypeError("'passed_methods' must be a dictionary.")
    for target_step, method_dep in passed_methods.items():
        if not isinstance(method_dep, (list, tuple)) or len(method_dep) < 2:
            raise ValueError(
                f"Invalid method dependency format for '{target_step}': {method_dep}"
            )
        from_step = method_dep[0]
        if from_step not in steps:
            raise ValueError(
                f"Step '{target_step}' expects a method from '{from_step}', but '{from_step}' is not defined in 'steps'."
            )

    steps_to_write = pipeline.get("save")
    if steps_to_write is not None:
        if not isinstance(steps_to_write, (list, tuple, set)):
            raise TypeError("'save' must be a sequence of step names.")
        for step in steps_to_write:
            if step not in steps and step not in pipeline.get("global_steps", {}):
                raise ValueError(
                    f"Step '{step}' listed in 'save' is not defined in the pipeline 'steps' or 'global_steps'."
                )

    for k, params in steps.items():
        if not isinstance(params, dict):
            raise TypeError(f"Parameters for step '{k}' must be a dictionary.")
        if k.startswith("nahual"):
            if "address" not in params:
                raise ValueError(
                    f"Nahual-deployed step '{k}' must provide an 'address' parameter."
                )

    # Validate global_steps structure if present
    global_steps = pipeline.get("global_steps", {})
    if global_steps:
        if "global_passed_data" not in pipeline:
            raise ValueError(
                "Pipeline defines 'global_steps' but is missing 'global_passed_data'."
            )
        if not isinstance(pipeline["global_passed_data"], dict):
            raise TypeError("'global_passed_data' must be a dictionary.")


def run_pipeline_return_state(pipeline: dict, steps_dir: str = None) -> dict:
    validate_pipeline(pipeline)

    state = {}

    # TODO Add dimensionality to autodefine max_ntps
    # TODO add assertion of ntps <= max_ntps
    ntps = pipeline.get("ntps", 1)  # {"ntps": 1})["ntps"]
    for _ in range(ntps):
        # print(f"Processing frame {frame}")
        state = pipeline_step(pipeline, state, steps_dir=steps_dir)
        # print(f"Finished frame {frame}")

    return state


def run_pipeline_and_post(
    pipeline: dict,
    pipeline_name: str,
    output_path: str | Path,
    overwrite: bool = True,
    # img_source: str | Path | list[str],
    # logger=None,
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
    output_path = Path(output_path)
    steps_dir = output_path / "steps" / pipeline_name
    profiles_file = output_path / "profiles" / f"{pipeline_name}.parquet"

    profiles = None
    post_results = None
    # pipeline["io"].get("ntps", 2) #

    # Main processing loop
    if overwrite or not profiles_file.exists():
        state = run_pipeline_return_state(pipeline, steps_dir=steps_dir)

        # Aggregate profiles from the state output
        profiles = get_profiles_from_state(state, pipeline)

        # Save files
        if len(profiles):
            profiles_file.parent.mkdir(parents=True, exist_ok=True)
            pyarrow.parquet.write_table(profiles, profiles_file, compression="zstd")

        # Run global processing steps (post-processing)
        post_results = {}

        # ntps = pipeline["io"]["ntps"]
        # if ntps == 1:  # Temporarily do not perform global operations on non-timeseries
        #     return profiles, post_results

        for step_name, parameters in pipeline.get("global_steps", {}).items():
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
                post_result = state["fn"](input_data=input_data)
                post_results[output_name] = post_result

            # Save global steps into files (per-tp steps are saved as they go, not at the end)
            if step_name in pipeline["save"]:
                write_fn = dispatch_write_fn(step_name)
                for output_subdir in post_results:
                    if output_subdir.startswith(step_name):
                        write_fn(
                            post_result,
                            output_path,
                            subpath=output_subdir,
                            filename=pipeline_name,
                        )
    else:
        logger.info(f"Skipping {pipeline_name}")
        profiles, post_results = None, None

    return profiles, post_results


def get_profiles_from_state(state: dict, pipeline: dict) -> pyarrow.Table:
    # Isolate features for every time point
    # We assume that extraction steps start with "ext"
    profiles = pyarrow.Table.from_pylist(
        [],
        schema=pa.schema(
            [
                pa.field("metadata_tile", pa.int64()),
                pa.field("metadata_label", pa.int64()),
                pa.field("metadata_object", pa.string()),
                pa.field("metadata_tp", pa.int64()),
            ]
        ),
    )
    feature_steps = [
        step_name
        for step_name in pipeline["steps"]
        if step_name.startswith("extract") or step_name.startswith("nahual_embed")
    ]
    # We will concatenate tables with the same number of colums horizontally
    # and then join them on time point, tile and object columns
    data = {k.split("_")[0]: [] for k in feature_steps}
    for ext_step in feature_steps:
        step_prefix = ext_step.split("_")[0]

        # Data is stored in the following order: step -> time points -> tiles
        for tp, ext_output in enumerate(state["data"][ext_step]):
            if isinstance(ext_output, numpy.ndarray):  # Format arbitrary embedders
                # We give it empty instructions that will be ignored
                ext_output = (cycle((("__", "__"),)), (ext_output,))

            table: pyarrow.PyTable = format_extraction(ext_output)
            # Renamae map
            rename_map = {
                "tile": "metadata_tile",
                "label": "metadata_label",
            }
            new_names = [rename_map.get(c, c) for c in table.column_names]
            table = table.rename_columns(new_names)

            if len(table):  # Cover case whence measurements are empty
                # object name (cp_measure) or model name (nahual embedders)
                table = table.append_column(
                    "metadata_object",
                    pyarrow.array(
                        [ext_step.split("_")[-1]] * len(table), pyarrow.string()
                    ),
                )
                table = table.append_column(
                    "metadata_tp",
                    pyarrow.array([tp] * len(table), pyarrow.uint8()),
                )
                data[step_prefix].append(table)

    all_wide_tables = []
    for k, wide_tables in data.items():
        if len(wide_tables):
            all_wide_tables.append(pyarrow.concat_tables(wide_tables))

    # Create one wide table to rule them all, the final profiles
    if all_wide_tables:
        profiles = all_wide_tables[0]
        for table in all_wide_tables[1:]:
            profiles = profiles.join(
                table,
                keys=[f"metadata_{k}" for k in ("tp", "tile", "object", "label")],
            )

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
