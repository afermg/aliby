#!/usr/bin/env jupyter
"""
Pipeline core: shared, segmenter-agnostic engine.

Pipelines (pipe, pipe_baby) build on top of this by providing their
own ``init_step`` dispatcher and binding ``run_pipeline_and_post`` with a
pipeline-specific ``post_state_hook``.
"""

from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Callable, Sequence

import numcodecs
import numpy
import pyarrow
import pyarrow as pa
from imagecodecs.numcodecs import Jpegxl
from loguru import logger

from aliby.global_steps import dispatch_global_step
from aliby.io.image import dispatch_image
from aliby.io.write import dispatch_write_fn
from aliby.tile.tiler import dispatch_tiler
from extraction.extract import (
    extract_tree,
    extract_tree_multi,
    format_extraction,
    process_tree_masks,
    process_tree_masks_overlap,
)

numcodecs.register_codec(Jpegxl)


def configure_logging(file):
    logger.remove()
    logger.add(
        file,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )


# ---------------------------------------------------------------------------
# Shared init helpers (used by both pipelines' init_step dispatchers)
# ---------------------------------------------------------------------------


def _init_tile(step_name: str, parameters: dict) -> Callable:
    image_kwargs = parameters.pop("image_kwargs", None)
    if image_kwargs is None:
        raise ValueError(f"Step '{step_name}' is missing required 'image_kwargs'.")
    if "source" not in image_kwargs:
        raise ValueError(
            f"Step '{step_name}' 'image_kwargs' is missing required 'source'."
        )
    tiler_constructor = dispatch_tiler(parameters.pop("kind", None), parameters)
    image_type = dispatch_image(source=image_kwargs["source"])
    image = image_type(**image_kwargs)
    return tiler_constructor(image)


def _init_extract(step_name: str, parameters: dict, *, overlap: bool) -> Callable:
    if "tree" not in parameters:
        raise ValueError(f"Step '{step_name}' is missing required 'tree'.")
    process = process_tree_masks
    measure_fn = extract_tree
    if overlap:
        process = process_tree_masks_overlap
        measure_fn = partial(extract_tree, overlap=True)
    return partial(
        process,
        measure_fn=measure_fn,
        tree=parameters["tree"],
        **parameters.get("kwargs", {}),
    )


def _init_extract_multi(step_name: str, parameters: dict) -> Callable:
    if "tree" not in parameters:
        raise ValueError(f"Step '{step_name}' is missing required 'tree'.")
    return partial(
        process_tree_masks,
        measure_fn=extract_tree_multi,
        tree=parameters["tree"],
        **parameters.get("kwargs", {}),
    )


def _init_nahual_embed(step_name: str, parameters: dict) -> Callable:
    address = parameters.get("address")
    if address is None:
        raise ValueError(
            f"If using Nahual you must have an address, currently it is None in step '{step_name}'"
        )
    if "setup_params" not in parameters:
        raise ValueError(f"Nahual embed step '{step_name}' is missing 'setup_params'.")
    if "model_group" not in parameters:
        raise ValueError(f"Nahual embed step '{step_name}' is missing 'model_group'.")

    from nahual.process import dispatch_setup_process

    setup, process = dispatch_setup_process(parameters["model_group"])

    selected_channels = parameters.get("selected_channels")
    if selected_channels:
        process = partial(
            slice_channels_process,
            process=process,
            selected_channels=selected_channels,
        )

    info = setup(parameters["setup_params"], address=address)
    print(f"Embedder via nahual set up. Remote returned {info}")
    return partial(process, address=address)


def _init_nahual_track(step_name: str, parameters: dict) -> Callable:
    address = parameters.get("address")
    if address is None:
        raise ValueError(
            f"If using Nahual you must have an address, currently it is None in step '{step_name}'"
        )
    if "parameters" not in parameters:
        raise ValueError(f"Nahual track step '{step_name}' is missing 'parameters'.")
    setup, process = dispatch_global_step(step_name)
    setup_output = setup(parameters["parameters"], address=address)
    print(f"NAHUAL: Remote process set up, returned {setup_output}.")
    return partial(process, address=address)


def slice_channels_process(
    data: numpy.ndarray,
    process: Callable,
    selected_channels: list[int] | numpy.ndarray,
    **kwargs,
) -> numpy.ndarray:
    """Apply a processing function to a subset of channels in a NumPy array."""
    return process(data[:, selected_channels], **kwargs)


def run_step(step, *args, **kwargs):
    if hasattr(step, "run_tp"):  # older OO-style
        result = step.run_tp(*args, **kwargs)
    else:
        if "tp" in kwargs:
            del kwargs["tp"]
        result = step(*args, **kwargs)
    return result


# ---------------------------------------------------------------------------
# Per-tp loop and post-processing
# ---------------------------------------------------------------------------


def pipeline_step(
    pipeline: dict,
    state: dict | None,
    steps_dir: str | None,
    init_step_fn: Callable,
) -> dict:
    """Run one timepoint of the pipeline using the provided init_step_fn."""
    if state is None:
        state = {}

    steps = pipeline["steps"]
    passed_methods = pipeline.get("passed_methods", {})

    if not state:
        state = {"tps": dict(zip(steps, cycle([0]))), "data": {}, "fn": {}}
    tp = next(iter(state["tps"].values()))

    for step_name, parameters in steps.items():
        if step_name not in state["data"]:
            state["data"][step_name] = []
        if step_name not in state["fn"]:
            state["fn"][step_name] = init_step_fn(step_name, parameters, state["fn"])
        step = state["fn"][step_name]

        # Pull data from previous steps via passed_data spec.
        # Format: {consumer_step: [(arg_name, producer_step, *opt_var), ...]}
        this_step_receives = pipeline["passed_data"].get(step_name, {})
        passed_data = {}
        for kwd, from_step, *varname in this_step_receives:
            passed_value = state["data"].get(from_step, [])
            step_argname = varname[0] if varname else kwd

            if len(passed_value):
                if step_name == "track" and kwd == "masks":
                    # tracker reads last 2 timepoints; reshape tp,tile,y,x -> tile,tp,y,x
                    passed_data[step_argname] = [
                        [tp_tiles[tile] for tp_tiles in passed_value[-2:]]
                        for tile in range(len(passed_value[-1]))
                    ]
                else:
                    last_value = passed_value[-1]
                    if isinstance(last_value, dict):
                        last_value = last_value[kwd]
                    passed_data[step_argname] = last_value

        # Pull pixels from a method on a previous-step object when configured.
        # Cellpose builder emits passed_methods[segment_*] = ("tile", "get_fczyx").
        # BABY builder does NOT emit one for segment steps because BABY pulls
        # pixels through its embedded tiler (injected at init time).
        args = ()
        method_spec = passed_methods.get(step_name)
        if method_spec is not None and step_name.startswith("segment"):
            source_step, method = method_spec
            args = (getattr(state["fn"][source_step], method)(tp),)

        step_result = run_step(step, *args, tp=tp, **passed_data)

        # Save outputs that are listed in pipeline["save"]
        steps_to_write = pipeline.get("save") or []
        save_interval = pipeline.get("save_interval", 1)
        should_save = (
            bool(steps_to_write) and save_interval > 0 and (tp % save_interval) == 0
        )
        if should_save and step_name in steps_to_write:
            print(f"Saving {step_name} to {steps_dir}")
            write_fn = dispatch_write_fn(step_name)
            write_fn(step_result, steps_dir=steps_dir, subpath=step_name, tp=tp)

        state["data"][step_name].append(step_result)
        if step_name not in state["fn"]:
            state["fn"][step_name] = step
        state["tps"][step_name] = tp + 1

    # End-of-tp memory hygiene.
    # Drop the raw pixel block from the last tile entry: tile pixels are only
    # consumed within the same tp via passed_data and never re-read after.
    for step_name in state["data"]:
        if step_name.startswith("tile"):
            entry = state["data"][step_name][-1] if state["data"][step_name] else None
            if isinstance(entry, dict) and "pixels" in entry:
                del entry["pixels"]

    # Trim per-step history per the pipeline's "retain" config.
    retain_cfg = pipeline.get("retain", {})
    for step_name, history in state["data"].items():
        keep = retain_cfg.get(step_name, "all")
        if isinstance(keep, int) and keep >= 0 and len(history) > keep:
            del history[: len(history) - keep]

    return state


def validate_pipeline(pipeline: dict) -> None:
    if not isinstance(pipeline, dict):
        raise TypeError("Pipeline configuration must be a dictionary.")

    if "steps" not in pipeline or not isinstance(pipeline["steps"], dict):
        raise ValueError(
            "Pipeline must contain a 'steps' dictionary mapping step names to parameters."
        )

    steps = pipeline["steps"]

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

    if "save_interval" in pipeline:
        save_interval = pipeline["save_interval"]
        if (
            not isinstance(save_interval, int)
            or isinstance(save_interval, bool)
            or save_interval < 1
        ):
            raise ValueError(
                f"'save_interval' must be a positive int, got {save_interval!r}."
            )

    retain = pipeline.get("retain", {})
    if not isinstance(retain, dict):
        raise TypeError(
            "'retain' must be a dictionary mapping step name to int or 'all'."
        )
    for step_name, keep in retain.items():
        if step_name not in steps:
            raise ValueError(
                f"'retain' references step '{step_name}' not defined in 'steps'."
            )
        if keep != "all" and not (
            isinstance(keep, int) and not isinstance(keep, bool) and keep >= 0
        ):
            raise ValueError(
                f"'retain[{step_name}]' must be a non-negative int or 'all', got {keep!r}."
            )
        track_reads_segment = any(
            from_step == step_name
            for target, deps in passed_data.items()
            if target.startswith("track")
            for dep in deps
            if (from_step := dep[1])
        )
        if track_reads_segment and isinstance(keep, int) and keep < 2:
            raise ValueError(
                f"'retain[{step_name}]' = {keep} is too small; per-tp 'track' step "
                f"reads the last 2 timepoints of '{step_name}'."
            )

    for k, params in steps.items():
        if not isinstance(params, dict):
            raise TypeError(f"Parameters for step '{k}' must be a dictionary.")
        if k.startswith("nahual"):
            if "address" not in params:
                raise ValueError(
                    f"Nahual-deployed step '{k}' must provide an 'address' parameter."
                )

    global_steps = pipeline.get("global_steps", {})
    if global_steps:
        if "global_passed_data" not in pipeline:
            raise ValueError(
                "Pipeline defines 'global_steps' but is missing 'global_passed_data'."
            )
        if not isinstance(pipeline["global_passed_data"], dict):
            raise TypeError("'global_passed_data' must be a dictionary.")


def run_pipeline_return_state(
    pipeline: dict,
    steps_dir: str | None,
    init_step_fn: Callable,
) -> dict:
    validate_pipeline(pipeline)
    state = {}
    ntps = pipeline.get("ntps", 1)
    for _ in range(ntps):
        state = pipeline_step(pipeline, state, steps_dir, init_step_fn)
    return state


def _run_pipeline_and_post_impl(
    pipeline: dict,
    pipeline_name: str,
    output_path: str | Path,
    overwrite: bool = True,
    *,
    init_step_fn: Callable,
    post_state_hook: Callable | None = None,
) -> tuple[pyarrow.Table, dict | None]:
    """Run a step-based pipeline and any post-processing global steps.

    Parameters
    ----------
    init_step_fn
        Pipeline-specific step initialiser (cellpose or baby flavour).
    post_state_hook
        Optional callable ``(state, pipeline, output_path, pipeline_name) -> None``
        invoked after profiles are written and before global steps. Used by the
        BABY pipeline to extract tracking/lineage from segmenter metadata.
    """
    output_path = Path(output_path)
    steps_dir = output_path / "steps" / pipeline_name
    profiles_file = output_path / "profiles" / f"{pipeline_name}.parquet"

    profiles = None
    post_results = None

    if overwrite or not profiles_file.exists():
        state = run_pipeline_return_state(pipeline, steps_dir, init_step_fn)
        profiles = get_profiles_from_state(state, pipeline)

        profiles_file.parent.mkdir(parents=True, exist_ok=True)
        pyarrow.parquet.write_table(profiles, profiles_file, compression="zstd")

        if post_state_hook is not None:
            post_state_hook(state, pipeline, output_path, pipeline_name)

        post_results = {}
        for step_name, parameters in pipeline.get("global_steps", {}).items():
            associated_data = [
                x for x in pipeline["global_passed_data"] if x.startswith(step_name)
            ]
            assert len(associated_data), (
                f"Incorrect pipeline: Missing information of which data to ingest for step {step_name}"
            )
            for output_name in associated_data:
                step_fn = init_step_fn(step_name, parameters)
                input_data = get_step_output(
                    state["data"],
                    pipeline["global_passed_data"][output_name],
                    steps_dir=steps_dir,
                )
                post_result = step_fn(input_data=input_data)
                post_results[output_name] = post_result

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
    data = {k.split("_")[0]: [] for k in feature_steps}
    for ext_step in feature_steps:
        step_prefix = ext_step.split("_")[0]
        for tp, ext_output in enumerate(state["data"][ext_step]):
            if isinstance(ext_output, numpy.ndarray):  # arbitrary embedders
                ext_output = (cycle((("__", "__"),)), (ext_output,))
            table: pyarrow.Table = format_extraction(ext_output)
            rename_map = {"tile": "metadata_tile", "label": "metadata_label"}
            new_names = [rename_map.get(c, c) for c in table.column_names]
            table = table.rename_columns(new_names)

            if len(table):
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

    if all_wide_tables:
        profiles = all_wide_tables[0]
        for table in all_wide_tables[1:]:
            profiles = profiles.join(
                table,
                keys=[f"metadata_{k}" for k in ("tp", "tile", "object", "label")],
            )

    return profiles


def get_step_output(
    state_data: dict,
    fetchers: tuple[Callable | str],
    steps_dir: Path | None = None,
) -> numpy.ndarray:
    """Aggregate outputs across timepoints from in-memory state or per-tp .npz files."""
    combined_outputs = []
    for fetcher in fetchers:
        if isinstance(fetcher, str):
            if fetcher.startswith("from_disk:"):
                if steps_dir is None:
                    raise ValueError(
                        "from_disk fetcher requires steps_dir; pass it through "
                        "get_step_output(..., steps_dir=...)"
                    )
                step_name = fetcher.removeprefix("from_disk:")
                aggregated_output = _load_per_tp_masks(Path(steps_dir) / step_name)
            else:
                # Monotile assumption (mirrored by _load_per_tp_masks for disk path)
                aggregated_output = [x[0] for x in state_data[fetcher]]
        elif isinstance(fetcher, Callable):
            aggregated_output = fetcher(state_data)
        else:
            raise Exception(
                f"Invalid type, expected Callable or string, got {type(fetcher)}"
            )
        combined_outputs.append(aggregated_output)

    return numpy.asarray(combined_outputs)


def _load_per_tp_masks(step_dir: Path) -> list[numpy.ndarray]:
    """Read per-tp .npz mask files written by io.write.write_ndarray.

    Two on-disk formats are supported:
      - baby segmenters: keys ``tile_0``, ``tile_1``, ... (one per tile)
      - other segmenters: a single key ``arr_0`` holding a stacked
        (tiles, Y, X) array
    """
    files = sorted(step_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No per-tp .npz files found under {step_dir}; ensure this step "
            f"is listed in pipeline['save']."
        )
    masks = []
    for f in files:
        with numpy.load(f) as npz:
            keys = list(npz.keys())
            if "tile_0" in keys:
                masks.append(npz["tile_0"])
            elif keys == ["arr_0"]:
                stacked = npz["arr_0"]
                masks.append(stacked[0])
            else:
                raise ValueError(f"Unrecognised .npz layout in {f}: keys={keys}")
    return masks


# ---------------------------------------------------------------------------
# Builder helper shared by both pipeline builders
# ---------------------------------------------------------------------------


def _attach_trackastra(
    base_pipeline: dict,
    channels_to_segment: Sequence[str],
    trackastra_address: str,
    trackastra_parameters: dict | None,
) -> None:
    """Wire a nahual_trackastra global step into ``base_pipeline`` in place.

    Disk-backed: per-tp segment masks are saved by the main loop, then
    nahual_trackastra reads them back via the ``from_disk:`` fetcher. This
    keeps state["data"]["segment_*"] bounded by retain=2 (per-tp tracker
    needs the last 2 timepoints in RAM).
    """
    seg_step_names = [f"segment_{obj}" for obj in channels_to_segment]
    for seg in seg_step_names:
        if seg not in base_pipeline["save"]:
            base_pipeline["save"].append(seg)
    base_pipeline["save"].append("nahual_trackastra")

    base_pipeline["global_steps"] = {
        "nahual_trackastra": dict(
            address=trackastra_address,
            parameters=trackastra_parameters or {},
        ),
    }
    base_pipeline["global_passed_data"] = {
        f"nahual_trackastra_{obj}": (f"from_disk:segment_{obj}",)
        for obj in channels_to_segment
    }

    retain = base_pipeline.setdefault("retain", {})
    for seg in seg_step_names:
        retain.setdefault(seg, 2)
    retain.setdefault("tile", 1)
