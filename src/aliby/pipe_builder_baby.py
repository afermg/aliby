#!/usr/bin/env python
"""
Builder for the BABY (nahual_baby) pipeline.

Hard-wired to ``kind="nahual_baby"``. Always uses overlapping-mask extraction.
Multi-channel colocalization extraction is not emitted (BABY's overlapping masks
break the coloc routine). Segment steps do NOT receive ``passed_methods`` —
BABY pulls pixels via the tiler injected at init time.
"""

from typing import Sequence

from aliby.pipe_core import _attach_trackastra


def build_pipeline_steps(
    baby_address: str,
    baby_modelset: str,
    channels_to_segment: dict[str, int] | None = None,
    channels_to_extract: Sequence[int] | None = None,
    features_to_extract: Sequence[str] = (
        "radial_zernikes",
        "intensity",
        "feret",
        "texture",
        "radial_distribution",
        "zernike",
    ),
    extract_ncores: int | None = None,
    steps_to_write: Sequence[str] | None = None,
    trackastra_address: str | None = None,
    trackastra_parameters: dict | None = None,
) -> dict:
    """Build a BABY pipeline definition (no IO).

    Parameters
    ----------
    baby_address : str
        Address of the Nahual server hosting BABY (required).
    baby_modelset : str
        BABY modelset name to load (required).
    channels_to_segment : dict of {str: int} or None
        Mapping object name → channel index. Defaults to ``{"nuclei": 1, "cell": 0}``.
    channels_to_extract, features_to_extract, extract_ncores, steps_to_write,
    trackastra_address, trackastra_parameters
        See ``pipe_builder.build_pipeline_steps``.
    """
    if channels_to_segment is None:
        channels_to_segment = {"nuclei": 1, "cell": 0}

    if channels_to_extract is None:
        channels_to_extract = list(channels_to_segment.values())

    seg_params = {}
    for obj, ch_id in channels_to_segment.items():
        step_name = f"segment_{obj}"
        seg_params[step_name] = dict(
            segmenter_kwargs=dict(
                kind="nahual_baby",
                address=baby_address,
                modelset=baby_modelset,
            ),
            channel_to_segment=ch_id,
        )

    # overlap=True is enforced by pipe_baby.init_step, not via the params dict.
    extract_base = dict(
        tree={"None": {"None": ("sizeshape",)}},
        kwargs=dict(ncores=extract_ncores),
    )
    for i in channels_to_extract:
        extract_base["tree"][i] = {"max": features_to_extract}

    # BABY emits one extract step per object; no extractmulti_* variant.
    ext_params = {f"extract_{obj}": extract_base for obj in channels_to_segment}

    base_pipeline = {
        "steps": dict(
            tile=dict(tile_size=None),
            **seg_params,
            **ext_params,
        ),
        "passed_data": {
            f"extract_{obj}": [
                ("masks", f"segment_{obj}"),
                ("pixels", "tile"),
            ]
            for obj in channels_to_segment
        },
        # No passed_methods for segment steps: BABY's tiler is injected at init
        # time and BABY pulls pixels itself.
        "passed_methods": {},
        "save": [f"segment_{obj}" for obj in channels_to_segment.keys()],
        "save_interval": 1,
    }

    if steps_to_write is not None:
        base_pipeline["save"] = list(steps_to_write)

    if trackastra_address is not None:
        _attach_trackastra(
            base_pipeline,
            channels_to_segment,
            trackastra_address,
            trackastra_parameters,
        )

    return base_pipeline
