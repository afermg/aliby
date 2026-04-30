#!/usr/bin/env jupyter
"""
Cellpose + cp_measure pipeline (the standard).

Segments with Cellpose (local or remote via Nahual) and extracts features with
cp_measure. Also supports Nahual embedders and optional Nahual trackastra.

For BABY (overlapping masks, BABY-specific tracking/lineage), see
``aliby.pipe_baby``.
"""

from functools import partial
from typing import Callable

from aliby.pipe_core import (
    _init_extract,
    _init_extract_multi,
    _init_nahual_embed,
    _init_nahual_track,
    _init_tile,
    _run_pipeline_and_post_impl,
    configure_logging,
    get_profiles_from_state,
    get_step_output,
    pipeline_step,
    run_pipeline_return_state,
    run_step,
    slice_channels_process,
    validate_pipeline,
)
from aliby.segment.dispatch import dispatch_segmenter
from aliby.track.dispatch import dispatch_tracker


def _init_segment_cellpose(
    step_name: str, parameters: dict, other_steps: dict
) -> Callable:
    seg_kwargs = parameters.get("segmenter_kwargs", {})
    if "channel_to_segment" not in parameters:
        raise ValueError(
            f"Step '{step_name}' is missing required 'channel_to_segment'."
        )
    return dispatch_segmenter(
        channel_to_segment=parameters["channel_to_segment"],
        **seg_kwargs,
    )


def _init_track_cellpose(
    step_name: str, parameters: dict, other_steps: dict
) -> Callable:
    return dispatch_tracker(**parameters)


def init_step(
    step_name: str,
    parameters: dict,
    other_steps: dict | None = None,
) -> Callable:
    """Set up parameters for any step in the cellpose pipeline."""
    if other_steps is None:
        other_steps = {}

    match step_name:
        case s if s.startswith("tile"):
            return _init_tile(s, parameters)
        case s if s.startswith("segment"):
            return _init_segment_cellpose(s, parameters, other_steps)
        case s if s.startswith("track"):
            return _init_track_cellpose(s, parameters, other_steps)
        case s if s.startswith("extract_"):
            return _init_extract(s, parameters, overlap=False)
        case s if s.startswith("extractmulti_"):
            return _init_extract_multi(s, parameters)
        case s if s.startswith("nahual_embed"):
            return _init_nahual_embed(s, parameters)
        case s if s.startswith("nahual_track"):
            return _init_nahual_track(s, parameters)
        case _:
            raise ValueError(f"Invalid step name {step_name=}")


run_pipeline_and_post = partial(
    _run_pipeline_and_post_impl, init_step_fn=init_step, post_state_hook=None
)


__all__ = [
    "configure_logging",
    "get_profiles_from_state",
    "get_step_output",
    "init_step",
    "pipeline_step",
    "run_pipeline_and_post",
    "run_pipeline_return_state",
    "run_step",
    "slice_channels_process",
    "validate_pipeline",
]
