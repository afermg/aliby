#!/usr/bin/env jupyter
"""
BABY (nahual_baby) pipeline.

Runs BABY remotely via Nahual for segmentation and tracking. BABY produces
overlapping per-tile masks plus tracking/lineage metadata that is extracted
post-run via the ``_save_baby_tracking_lineage`` post-state hook.

Still supports Nahual embedders alongside BABY segmentation.
"""

from functools import partial
from pathlib import Path
from typing import Callable

import pyarrow
from loguru import logger

from aliby.io.roi_provenance import (
    ProvenanceRunLock,
    _write_roi_provenance_locked,
    acquire_provenance_run_lock,
    baby_segment_steps,
    provenance_enabled,
    validate_provenance_preflight,
)
from aliby.pipe_core import (
    _init_extract,
    _init_nahual_embed,
    _init_nahual_track,
    _init_tile,
    _run_pipeline_and_post_impl,
)
from aliby.segment.dispatch import dispatch_segmenter
from aliby.track.dispatch import dispatch_tracker


def _init_segment_baby(step_name: str, parameters: dict, other_steps: dict) -> Callable:
    """BABY segmenter — requires a preceding ``tile`` step; the tiler instance
    is injected into the segmenter so BABY can pull pixels itself."""
    seg_kwargs = parameters.get("segmenter_kwargs", {})
    tiler_step = next((v for k, v in other_steps.items() if k.startswith("tile")), None)
    if tiler_step is None:
        raise ValueError(
            f"Step '{step_name}' using 'baby' requires a preceding 'tile' step."
        )
    seg_kwargs["tiler"] = tiler_step
    if "channel_to_segment" not in parameters:
        raise ValueError(
            f"Step '{step_name}' is missing required 'channel_to_segment'."
        )
    return dispatch_segmenter(
        channel_to_segment=parameters["channel_to_segment"],
        **seg_kwargs,
    )


def _init_track_baby(step_name: str, parameters: dict, other_steps: dict) -> Callable:
    """BABY tracker — pulls the crawler from the segment step."""
    segment_step = next(
        (v for k, v in other_steps.items() if k.startswith("segment")), None
    )
    if segment_step is None:
        raise ValueError(
            f"Step '{step_name}' using 'baby' tracking requires a preceding 'segment' step."
        )
    parameters["crawler"] = segment_step.crawler
    return dispatch_tracker(**parameters)


def init_step(
    step_name: str,
    parameters: dict,
    other_steps: dict | None = None,
) -> Callable:
    """Set up parameters for any step in the BABY pipeline."""
    if other_steps is None:
        other_steps = {}

    match step_name:
        case s if s.startswith("tile"):
            return _init_tile(s, parameters)
        case s if s.startswith("segment"):
            return _init_segment_baby(s, parameters, other_steps)
        case s if s.startswith("track"):
            return _init_track_baby(s, parameters, other_steps)
        case s if s.startswith("extract_"):
            return _init_extract(s, parameters, overlap=True)
        case s if s.startswith("extractmulti_"):
            raise ValueError(
                "Multi-channel colocalization extraction is not supported with "
                "BABY's overlapping masks."
            )
        case s if s.startswith("nahual_embed"):
            return _init_nahual_embed(s, parameters)
        case s if s.startswith("nahual_track"):
            return _init_nahual_track(s, parameters)
        case _:
            raise ValueError(f"Invalid step name {step_name=}")


def _save_baby_tracking_lineage(
    state: dict,
    pipeline: dict,
    output_path: Path,
    pipeline_name: str,
    *,
    provenance_lock: ProvenanceRunLock | None = None,
) -> None:
    """Extract and save BABY tracking/lineage from segment metadata across timepoints."""
    publish = provenance_enabled(pipeline)
    if publish:
        if provenance_lock is None:
            raise RuntimeError("Enabled provenance requires an authenticated run lock.")
        provenance_lock.verify()
    segment_steps = baby_segment_steps(pipeline, validate_names=publish)
    tiler = None
    if publish:
        tilers = [
            step
            for step_name, step in state.get("fn", {}).items()
            if step_name.startswith("tile")
        ]
        if len(tilers) != 1:
            raise ValueError(
                "BABY ROI provenance requires exactly one live tile step; "
                f"found {len(tilers)}."
            )
        tiler = tilers[0]
        timepoints = pipeline.get("ntps", 1)
        for step_name in segment_steps:
            step_data = state.get("data", {}).get(step_name, [])
            if len(step_data) != timepoints or any(
                not isinstance(result, dict) or "metadata" not in result
                for result in step_data
            ):
                raise ValueError(
                    f"BABY segment state {step_name!r} must contain exactly "
                    f"{timepoints} metadata-bearing results."
                )

    for step_name in segment_steps:
        step_data = state["data"].get(step_name, [])
        baby_meta_history = [
            tp_result["metadata"]
            for tp_result in step_data
            if isinstance(tp_result, dict) and "metadata" in tp_result
        ]
        if not baby_meta_history:
            continue

        from aliby.segment.baby_parser import (
            accumulate_lineage,
            accumulate_tracking,
            baby_tracking_to_table,
        )

        tracking = accumulate_tracking(baby_meta_history)
        lineage = accumulate_lineage(baby_meta_history)
        table = baby_tracking_to_table(tracking, lineage)

        if not len(table):
            if publish:
                raise ValueError(
                    f"BABY tracking for {step_name!r} is empty; provenance requires "
                    "a released tracking artifact."
                )
            continue
        tracking_dir = output_path / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        out_file = tracking_dir / f"{pipeline_name}_{step_name}.parquet"
        pyarrow.parquet.write_table(table, out_file, compression="zstd")
        logged_file = out_file
        if provenance_lock is not None:
            logged_file = (
                provenance_lock.lexical_output_path
                / "tracking"
                / f"{pipeline_name}_{step_name}.parquet"
            )
        logger.info(f"Saved baby tracking/lineage to {logged_file}")

    if publish:
        provenance_lock.verify()
        npz_path, json_path = _write_roi_provenance_locked(
            tiler, pipeline, pipeline_name, provenance_lock
        )
        logger.info(f"Saved live ROI provenance to {npz_path} and {json_path}")


def run_pipeline_and_post(
    pipeline: dict,
    pipeline_name: str,
    output_path: str | Path,
    overwrite: bool = True,
    *,
    backend: str = "sequential",
    max_workers: int | None = None,
    resource_limits: dict[str, int] | None = None,
) -> tuple[pyarrow.Table, dict | None]:
    """Run BABY with opt-in ROI provenance preflighted before any output write."""
    publish = provenance_enabled(pipeline)
    validate_provenance_preflight(pipeline, output_path, pipeline_name)
    lock_context = None
    if publish:
        lock_context = acquire_provenance_run_lock(output_path)
        try:
            # A concurrent winner may have committed while this process waited.
            validate_provenance_preflight(
                pipeline, lock_context.lexical_output_path, pipeline_name
            )
            lock_context.verify()
        except Exception:
            lock_context.release()
            raise
    try:
        post_state_hook = _save_baby_tracking_lineage
        core_output_path = output_path
        if lock_context is not None:
            post_state_hook = partial(
                _save_baby_tracking_lineage, provenance_lock=lock_context
            )
            core_output_path = lock_context.descriptor_output_path
        result = _run_pipeline_and_post_impl(
            pipeline,
            pipeline_name,
            core_output_path,
            overwrite,
            init_step_fn=init_step,
            post_state_hook=post_state_hook,
            backend=backend,
            max_workers=max_workers,
            resource_limits=resource_limits,
        )
        if lock_context is not None:
            lock_context.verify()
        return result
    finally:
        if lock_context is not None:
            lock_context.release()
