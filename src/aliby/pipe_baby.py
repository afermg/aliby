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
    state: dict, pipeline: dict, output_path: Path, pipeline_name: str
) -> None:
    """Extract and save BABY tracking/lineage from segment metadata across timepoints."""
    for step_name in pipeline["steps"]:
        if not step_name.startswith("segment"):
            continue
        seg_kwargs = pipeline["steps"][step_name].get("segmenter_kwargs", {})
        if not seg_kwargs.get("kind", "").endswith("baby"):
            continue

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

        if len(table):
            tracking_dir = output_path / "tracking"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            out_file = tracking_dir / f"{pipeline_name}_{step_name}.parquet"
            pyarrow.parquet.write_table(table, out_file, compression="zstd")
            logger.info(f"Saved baby tracking/lineage to {out_file}")


run_pipeline_and_post = partial(
    _run_pipeline_and_post_impl,
    init_step_fn=init_step,
    post_state_hook=_save_baby_tracking_lineage,
)
