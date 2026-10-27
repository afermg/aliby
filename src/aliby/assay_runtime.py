"""Explicit bindings from accepted assay decisions to existing pipeline adapters.

No model discovery/download or batch admission. BABY identities are declarations
that a real-run validator must attest against its owned service. Local Cellpose
weights are hash-checked before constructing the model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from aliby.assay import ChannelInput, select_inputs


def _canonical(value):
    if value is None or type(value) in (str, int, bool):
        return [type(value).__name__, value]
    if type(value) is float and math.isfinite(value):
        return ["float", value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_canonical(v) for v in value]]
    if isinstance(value, dict):
        # Pipeline insertion order affects stable execution/write order.
        return ["dict", [[_canonical(k), _canonical(v)] for k, v in value.items()]]
    raise ValueError(
        f"Bound configuration must contain portable finite data, not {type(value).__name__}"
    )


def binding_fingerprint(pipeline):
    copy = deepcopy(pipeline)
    copy["assay_binding"].pop("pipeline_sha256", None)
    return hashlib.sha256(
        json.dumps(_canonical(copy), separators=(",", ":")).encode()
    ).hexdigest()


def validate_assay_binding(pipeline):
    binding = pipeline.get("assay_binding")
    marked = any(
        isinstance(p, dict)
        and isinstance(p.get("segmenter_kwargs"), dict)
        and p["segmenter_kwargs"].get("assay_bound")
        for p in pipeline.get("steps", {}).values()
    )
    if binding is None:
        if marked:
            raise ValueError("assay_bound requires a complete assay binding")
        return
    if binding.get("schema") != "aliby.assay-binding.v1" or binding.get(
        "pipeline_sha256"
    ) != binding_fingerprint(pipeline):
        raise ValueError(
            "Assay binding/configuration changed; bind and review it again before execution"
        )


def bind_assay_pipeline(pipeline, decision, *, model_identities):
    """Return a deep-copied pipeline with explicit inputs and extractor trees.

    Supports BABY 2D overlapping masks and local Cellpose 2D labels. Exactly one
    tiler and conventional segment_<object>/extract_<object> names are required.
    Each object's measurements may use one selection per original channel.
    Unsupported bindings fail before any model initialization or IO.
    """
    decision.require_ready()
    if "assay_binding" in pipeline:
        raise ValueError("Bind a fresh unbound pipeline, not a previous attempt")
    p = deepcopy(pipeline)
    if type(p.get("ntps", 1)) is not int or p.get("ntps", 1) != decision.request.ntps:
        raise ValueError("Pipeline and assay ntps differ")
    tiles = [name for name in p["steps"] if name.startswith("tile")]
    if len(tiles) != 1:
        raise ValueError("Assay binding requires one unambiguous tiler")
    tile = tiles[0]
    capabilities = {m.model_id: m for m in decision.models}
    selected = dict(decision.selected_models)
    declared = {f"segment_{s.object_name}" for s in decision.request.segmentations}
    kinds = {
        p["steps"].get(n, {}).get("segmenter_kwargs", {}).get("kind") for n in declared
    }
    if len(kinds) > 1:
        raise ValueError(
            "Mixed segmenter families within one site are not yet supported"
        )
    if {n for n in p["steps"] if n.startswith("segment")} != declared:
        raise ValueError("Pipeline segment steps must match the declared assay objects")
    inputs = {}
    measurements = {}
    model_records = {}
    for seg in decision.request.segmentations:
        obj = seg.object_name
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", obj):
            raise ValueError(
                "Legacy profile object names must be alphanumeric without underscores"
            )
        name = f"segment_{obj}"
        params = p["steps"][name]
        if set(params) - {"segmenter_kwargs", "channel_to_segment"}:
            raise ValueError(f"Unsupported segment step parameters for {name}")
        kw = params.get("segmenter_kwargs", {})
        model = capabilities[selected[obj]]
        if model_identities.get(model.model_id) != model.identity:
            raise ValueError(f"Missing/mismatched model identity for {model.model_id}")
        kind = kw.get("kind")
        if kind == "nahual_baby":
            allowed = {"kind", "address", "modelset", "extra_args"}
            if (
                len(seg.inputs) != 1
                or model.masks != "overlapping"
                or not model.stateful
            ):
                raise ValueError(
                    "BABY binding requires one channel, overlapping masks, and stateful capability"
                )
            if kw.get("modelset") != model.model_id or not kw.get("address"):
                raise ValueError("BABY modelset/address must match the selected model")
            if any(i.reduction == "mean" for i in seg.inputs):
                raise ValueError(
                    "BABY transport does not accept floating-point mean projections"
                )
        elif kind == "cellpose":
            allowed = {"kind", "setup_params", "eval_params"}
            if model.masks != "labels" or model.z_planes != 1 or model.stateful:
                raise ValueError(
                    "Local Cellpose adapter supports stateless 2D labels only"
                )
            setup = kw.get("setup_params", {})
            if not setup.get("pretrained_model") or set(setup) - {
                "pretrained_model",
                "gpu",
                "use_bfloat16",
            }:
                raise ValueError(
                    "Cellpose requires explicit local weights and supported setup options"
                )
            evaluation = kw.get("eval_params", {})
            if set(evaluation) - {
                "batch_size",
                "normalize",
                "invert",
                "diameter",
                "flow_threshold",
                "cellprob_threshold",
                "min_size",
                "max_size_fraction",
                "niter",
                "augment",
                "tile_overlap",
                "bsize",
            }:
                raise ValueError(
                    "Unsupported Cellpose evaluation option (no implicit channel/Z remapping)"
                )
        else:
            raise ValueError(f"No validated assay adapter for {kind!r}")
        if set(kw) - allowed:
            raise ValueError(f"Unsupported {kind} options: {sorted(set(kw) - allowed)}")
        kw["assay_bound"] = True
        kw["assay_model_identity"] = model.identity
        params["channel_to_segment"] = 0  # Selected original channel becomes C0.
        p.setdefault("passed_methods", {})[name] = (tile, "get_fczyx")
        if p.setdefault("passed_data", {}).get(name):
            raise ValueError(
                "Bound segment pixels must have a single explicit method source"
            )
        inputs[name] = [asdict(i) for i in seg.inputs]
        model_records[name] = asdict(model)

        mono = [
            m
            for m in decision.request.measurements
            if m.object_name == obj and len(m.inputs) == 1
        ]
        multi = [
            m
            for m in decision.request.measurements
            if m.object_name == obj and len(m.inputs) > 1
        ]
        for prefix, requests in (("extract", mono), ("extractmulti", multi)):
            ext = f"{prefix}_{obj}"
            if not requests:
                p["steps"].pop(ext, None)
                p["passed_data"].pop(ext, None)
                p.get("passed_methods", {}).pop(ext, None)
                if ext in (p.get("save") or ()):
                    p["save"] = [n for n in p["save"] if n != ext]
                p.get("retain", {}).pop(ext, None)
                p.get("step_resources", {}).pop(ext, None)
                continue
            chosen = {}
            for measurement in requests:
                for inp in measurement.inputs:
                    if inp.channel in chosen and chosen[inp.channel] != inp:
                        raise ValueError(
                            "One selection per original channel/object/extractor is supported"
                        )
                    chosen[inp.channel] = inp
            ordered = tuple(chosen.values())
            remap = {i.channel: n for n, i in enumerate(ordered)}
            tree = {}
            from extraction.extract import CELL_FUNS

            for measurement in requests:
                colocalization = {"pearson", "costes", "manders_fold", "rwc"}
                if any(
                    (f in colocalization) != (prefix == "extractmulti")
                    for f in measurement.features
                ):
                    raise ValueError(
                        "Feature input arity does not match mono/colocalization measurement"
                    )
                if any(f not in CELL_FUNS for f in measurement.features):
                    raise ValueError(
                        f"Unsupported extractor feature in {measurement.name}"
                    )
                channel_key = (
                    remap[measurement.inputs[0].channel]
                    if prefix == "extract"
                    else tuple(remap[i.channel] for i in measurement.inputs)
                )
                if prefix == "extractmulti" and len(channel_key) != 2:
                    raise ValueError(
                        "Current colocalization adapter requires exactly two channels"
                    )
                branch = tree.setdefault(channel_key, {})
                if prefix == "extractmulti":
                    branch = branch.setdefault("None", {})
                previous = branch.get("max", ())
                branch["max"] = tuple(dict.fromkeys((*previous, *measurement.features)))
            old = p["steps"].get(ext, {})
            if set(old.get("kwargs", {})) - {
                "ncores",
                "progress_bar",
                "cp_measure_kwargs",
            }:
                raise ValueError(f"Unsupported extractor options for {ext}")
            p["steps"][ext] = {
                "tree": tree,
                "kwargs": deepcopy(old.get("kwargs", {"ncores": None})),
                "assay_overlap": model.masks == "overlapping",
            }
            p["passed_data"][ext] = [("masks", name), ("pixels", tile)]
            inputs[ext] = [asdict(i) for i in ordered]
            measurements[ext] = {
                "names": [m.name for m in requests],
                "overlapping": model.masks == "overlapping",
            }
    allowed_extract = set(measurements)
    if any(n.startswith("extract") and n not in allowed_extract for n in p["steps"]):
        raise ValueError("Undeclared extraction step in bound pipeline")
    p["assay_binding"] = {
        "schema": "aliby.assay-binding.v1",
        "decision_sha256": decision.fingerprint,
        "decision": decision.to_dict(),
        "inputs": inputs,
        "models": model_records,
        "measurements": measurements,
    }
    p["assay_binding"]["pipeline_sha256"] = binding_fingerprint(p)
    from aliby.pipe_core import validate_pipeline

    validate_pipeline(p)
    return p


def prepare_bound_pixels(pipeline, step_name, pixels):
    binding = pipeline["assay_binding"]
    records = binding["inputs"][step_name]
    array = np.asarray(pixels)
    channels = {c["index"]: c for c in binding["decision"]["request"]["channels"]}
    if (
        array.ndim != 5
        or not array.shape[0]
        or any(array.shape[2] != channels[i["channel"]]["stored_z"] for i in records)
    ):
        raise ValueError(
            "Actual nonempty FCZYX input differs from the bound raw layout"
        )
    selected = select_inputs(array, tuple(ChannelInput(**i) for i in records))
    if step_name in binding["models"]:
        if not np.any(selected) or any(
            not np.any(selected[:, c, z])
            for c in range(selected.shape[1])
            for z in range(selected.shape[2])
        ):
            raise ValueError(
                "Zero segmentation plane in actual submitted input; review required"
            )
        kind = pipeline["steps"][step_name]["segmenter_kwargs"]["kind"]
        if kind == "nahual_baby" and selected.dtype not in (np.uint8, np.uint16):
            raise ValueError(
                "BABY input must be uint8 or uint16; no implicit dtype conversion"
            )
    return selected


def restore_measurement_labels(pipeline, step_name, result):
    """Keep original channel/Z identity in metrics after compact C remapping."""
    records = pipeline["assay_binding"]["inputs"][step_name]
    labels = [
        f"c{i['channel']}[z={','.join(map(str, i['z']))};{i['reduction']}]"
        for i in records
    ]
    instructions, *rest = result
    restored = []
    for ids, branch in instructions:
        channel, *tail = branch
        label = (
            labels[channel]
            if isinstance(channel, int)
            else "|".join(labels[c] for c in channel)
        )
        restored.append((ids, (label, *tail)))
    return (tuple(restored), *rest)


def finish_bound_result(pipeline, step_name, result):
    if step_name not in pipeline.get("assay_binding", {}).get("measurements", {}):
        return result
    if inspect.isawaitable(result):

        async def finish():
            return restore_measurement_labels(pipeline, step_name, await result)

        return finish()
    return restore_measurement_labels(pipeline, step_name, result)


def require_guarded_baby_client():
    from nahual.client import baby

    if any(
        not hasattr(baby, name)
        for name in (
            "_UNSAFE_SESSIONS",
            "_SESSION_LOCK",
            "_checked_get",
            "HTTP_TIMEOUT",
            "POLL_TIMEOUT",
        )
    ):
        raise RuntimeError(
            "BABY assay execution requires the guarded, no-retry Nahual client"
        )
    deadlines = (*baby.HTTP_TIMEOUT, baby.POLL_TIMEOUT)
    if any(
        isinstance(t, bool)
        or not isinstance(t, (int, float))
        or not math.isfinite(t)
        or t <= 0
        for t in deadlines
    ):
        raise ValueError("BABY transport deadlines must be finite and positive")


def init_bound_segmenter(kind, channel_to_segment, address=None, **kwargs):
    identity = kwargs.pop("assay_model_identity")
    if channel_to_segment != 0:
        raise ValueError("Bound segmentation must use remapped C0")
    if kind == "nahual_baby":
        from aliby.segment.dispatch import dispatch_segmenter

        require_guarded_baby_client()
        delegate = dispatch_segmenter(kind, channel_to_segment, address, **kwargs)

        def checked_baby(pixels):
            result = delegate(pixels)
            masks, metadata = result["masks"], result["metadata"]
            if len(masks) != len(pixels) or len(metadata) != len(pixels):
                raise ValueError("BABY output tile count differs from submitted pixels")
            for mask, meta in zip(masks, metadata, strict=True):
                mask = np.asarray(mask)
                labels = meta.get("cell_label", [])
                if not isinstance(labels, (list, tuple)) or any(
                    type(label) is not int or label <= 0 for label in labels
                ):
                    raise ValueError("BABY tracking labels must be positive integers")
                if (
                    mask.ndim != 3
                    or mask.shape[-2:] != pixels.shape[-2:]
                    or mask.dtype.kind not in "iu"
                ):
                    raise ValueError("Invalid BABY layered-mask shape/dtype")
                observed = [
                    int(label)
                    for layer in mask
                    for label in np.unique(layer)
                    if label != 0
                ]
                if len(set(observed)) != len(observed) or sorted(observed) != sorted(
                    labels
                ):
                    raise ValueError("BABY mask labels do not match tracking metadata")
            return result

        return checked_baby
    if kind != "cellpose":
        raise ValueError(f"Unsupported bound segmenter {kind}")
    setup = dict(kwargs["setup_params"])
    weights = Path(setup["pretrained_model"])
    if not weights.is_file():
        raise ValueError(
            "Local Cellpose weights are required; automatic downloads are disabled"
        )
    with weights.open("rb") as handle:
        before = os.fstat(handle.fileno())
        observed = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
        if identity != observed:
            raise ValueError(
                "Cellpose weight identity does not match the assay decision"
            )
        from cellpose.models import CellposeModel

        # Pin the inode during loading; a replaced filename must not cause an
        # implicit fallback/download or load different weights after hashing.
        setup["pretrained_model"] = f"/proc/self/fd/{handle.fileno()}"
        model = CellposeModel(**setup)
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Cellpose weights changed while the model was loading")
    evaluation = dict(kwargs.get("eval_params", {}))

    def segment(pixels):
        if pixels.ndim != 5 or pixels.shape[2] != 1:
            raise ValueError("Cellpose binding requires explicit FC1YX pixels")
        # A list prevents treating F as Z or collapsing labels across tiles.
        images = list(np.moveaxis(pixels[:, :, 0], 1, -1))
        masks, *_ = model.eval(images, channel_axis=-1, do_3D=False, **evaluation)
        if len(masks) != len(images):
            raise ValueError("Cellpose returned a different tile count")
        checked = []
        for mask in masks:
            mask = np.asarray(mask)
            if (
                mask.shape != pixels.shape[-2:]
                or mask.dtype.kind not in "iu"
                or np.any(mask < 0)
                or np.any(mask >= np.iinfo(np.uint16).max)
            ):
                raise ValueError("Invalid Cellpose label shape/dtype/range")
            checked.append(mask.astype(np.uint16, copy=False))
        return checked

    return segment
