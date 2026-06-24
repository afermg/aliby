#!/usr/bin/env python
"""
Builder for the Cellpose + cp_measure pipeline (the standard).

Defaults to local Cellpose. If ``nahual_addresses`` is provided, switches to
``nahual_cellpose``. Always includes single-channel feature extraction and
multi-channel colocalization. Optional Nahual trackastra is wired in via
``pipe_core._attach_trackastra``.

For BABY, see ``aliby.pipe_builder_baby``.
"""

from itertools import combinations, product
from typing import Sequence

from aliby.pipe_core import _attach_trackastra


def _create_extract_multich_tree(
    channels: Sequence[int],
    extract_ncores: int | None,
    cp_measure_feature_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Build the extract_multich tree dict for colocalization.

    ``cp_measure_feature_kwargs`` (optional) lands under
    ``kwargs["cp_measure_kwargs"]`` so the extractor sees per-feature
    options (e.g. ``{"intensity": {"edge_measurements": False}}``).
    """
    kwargs: dict = {"ncores": extract_ncores}
    if cp_measure_feature_kwargs:
        kwargs["cp_measure_kwargs"] = dict(cp_measure_feature_kwargs)
    return {
        "tree": {
            pair: {
                "None": {
                    "max": ["pearson", "costes", "manders_fold", "rwc"],
                },
            }
            for pair in combinations(channels, r=2)
        },
        "kwargs": kwargs,
    }


def build_pipeline_steps(
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
    nahual_addresses: str | Sequence[str] | None = None,
    steps_to_write: Sequence[str] | None = None,
    trackastra_address: str | None = None,
    trackastra_parameters: dict | None = None,
    cp_measure_feature_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Build a Cellpose + cp_measure pipeline definition (no IO).

    Parameters
    ----------
    channels_to_segment : dict of {str: int} or None
        Mapping object name → channel index. Defaults to ``{"nuclei": 1, "cell": 0}``.
    channels_to_extract : Sequence of int or None
        Channels to extract features from. Defaults to all values in ``channels_to_segment``.
    features_to_extract : Sequence of str
        cp_measure feature names to compute per extracted channel.
    extract_ncores : int or None
        Cores for joblib-parallelised extraction.
    nahual_addresses : str or Sequence of str or None
        If provided, segmentation runs remotely via Nahual (kind=``nahual_cellpose``).
    steps_to_write : Sequence of str or None
        Step names whose outputs are written to disk. Defaults to segment steps.
    trackastra_address, trackastra_parameters
        If both provided, attach a ``nahual_trackastra`` global step.
    cp_measure_feature_kwargs : dict of {feature_name: kwargs_dict} or None
        Optional per-feature kwargs forwarded to the underlying
        ``cp_measure`` functions, e.g.
        ``{"intensity": {"edge_measurements": False}}`` to skip the
        expensive boundary-pixel pass. Lands under
        ``steps["extract*_<obj>"]["kwargs"]["cp_measure_kwargs"]`` for
        both single-channel and multi-channel extract steps. Defaults
        to ``None`` → no change in behaviour for existing callers.
    """
    if channels_to_segment is None:
        channels_to_segment = {"nuclei": 1, "cell": 0}

    use_nahual = nahual_addresses is not None
    segmenter_kind = "nahual_cellpose" if use_nahual else "cellpose"

    if channels_to_extract is None:
        channels_to_extract = list(channels_to_segment.values())

    seg_params = {}
    for obj, ch_id in channels_to_segment.items():
        step_name = f"segment_{obj}"
        seg_params[step_name] = dict(
            segmenter_kwargs=dict(kind=segmenter_kind),
            channel_to_segment=ch_id,
        )

    # cp_measure per-feature kwargs land in extract step ``kwargs``,
    # picked up by ``extraction.extract.process_tree_masks`` →
    # ``extract_tree``/``extract_tree_multi`` and used to build a fresh
    # per-step ``CELL_FUNS`` overlay.
    extract_kwargs: dict = dict(ncores=extract_ncores)
    if cp_measure_feature_kwargs:
        extract_kwargs["cp_measure_kwargs"] = dict(cp_measure_feature_kwargs)
    extract_base = dict(
        tree={"None": {"None": ("sizeshape",)}},
        kwargs=extract_kwargs,
    )
    for i in channels_to_extract:
        extract_base["tree"][i] = {"max": features_to_extract}

    extract_multich_base = _create_extract_multich_tree(
        channels_to_extract,
        extract_ncores,
        cp_measure_feature_kwargs=cp_measure_feature_kwargs,
    )

    extract_variants = [("", extract_base), ("multi", extract_multich_base)]
    ext_params = {
        f"extract{name}_{obj}": var
        for (name, var), obj in product(extract_variants, channels_to_segment)
        if len(var)
    }

    base_pipeline = {
        "steps": dict(
            tile=dict(tile_size=None),
            **seg_params,
            **ext_params,
        ),
        "passed_data": {
            f"extract{multi}_{obj}": [
                ("masks", f"segment_{obj}"),
                ("pixels", "tile"),
            ]
            for obj in channels_to_segment
            for multi in (n for n, _ in extract_variants)
        },
        "passed_methods": {
            f"segment_{obj}": ("tile", "get_fczyx") for obj in channels_to_segment
        },
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
