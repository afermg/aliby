"""Pure, opt-in assay decisions; no discovery, model loading, or execution.

Evidence adapters must distinguish unreadable/missing planes from stored zeros.
An accepted decision means sampled input compatibility, not biological approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
import hashlib
import json

SCHEMA = "aliby.assay-decision.v1"
PLANE_STATES = {"nonzero", "zero", "missing", "unreadable", "unknown"}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _sequence(instance, name):
    value = getattr(instance, name)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list")
    object.__setattr__(instance, name, tuple(value))


def _indices(values, name, *, empty=False):
    if not values and not empty:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _integer(value, name)
    if tuple(sorted(set(values))) != values:
        raise ValueError(
            f"{name} must be unique and increasing; no synthetic duplication"
        )


@dataclass(frozen=True)
class PlaneSample:
    timepoint: int
    states: tuple[str, ...]
    evidence_id: str

    def __post_init__(self):
        _integer(self.timepoint, "timepoint")
        _sequence(self, "states")
        if not self.states or any(
            not isinstance(s, str) or s not in PLANE_STATES for s in self.states
        ):
            raise ValueError("states must contain explicit plane observation states")
        _text(self.evidence_id, "sample evidence_id")


@dataclass(frozen=True)
class ChannelEvidence:
    index: int
    name: str
    modality: str
    stored_z: int
    samples: tuple[PlaneSample, ...]
    # None = optional sidecar unavailable/unusable. () = explicitly not acquired.
    acquired_z: tuple[int, ...] | None = None
    metadata_evidence: tuple[str, ...] = ()
    metadata_conflicts: tuple[str, ...] = ()
    metadata_warnings: tuple[str, ...] = ()

    def __post_init__(self):
        _integer(self.index, "channel index")
        _integer(self.stored_z, "stored_z", 1)
        _text(self.name, "channel name")
        _text(self.modality, "modality")
        for field in (
            "samples",
            "metadata_evidence",
            "metadata_conflicts",
            "metadata_warnings",
        ):
            _sequence(self, field)
        for field in ("metadata_evidence", "metadata_conflicts", "metadata_warnings"):
            for item in getattr(self, field):
                _text(item, field)
        if self.acquired_z is not None:
            _sequence(self, "acquired_z")
            _indices(self.acquired_z, "acquired_z", empty=True)
            if any(z >= self.stored_z for z in self.acquired_z):
                raise ValueError("acquired_z exceeds storage layout")
            if not self.metadata_evidence:
                raise ValueError("acquired_z requires a metadata evidence identity")
        if any(
            not isinstance(s, PlaneSample) or len(s.states) != self.stored_z
            for s in self.samples
        ):
            raise ValueError("samples must describe every stored Z slot")
        times = [s.timepoint for s in self.samples]
        if len(times) != len(set(times)):
            raise ValueError("duplicate sampled timepoint")
        object.__setattr__(
            self, "samples", tuple(sorted(self.samples, key=lambda s: s.timepoint))
        )


@dataclass(frozen=True)
class ChannelInput:
    channel: int
    z: tuple[int, ...]
    reduction: str = "stack"

    def __post_init__(self):
        _integer(self.channel, "input channel")
        _sequence(self, "z")
        _indices(self.z, "input z")
        if self.reduction not in ("stack", "max", "mean"):
            raise ValueError("reduction must be stack, max, or mean")

    @property
    def depth(self):
        return len(self.z) if self.reduction == "stack" else 1


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    identity: str
    modalities: tuple[str, ...]
    z_planes: int
    masks: str = "labels"
    stateful: bool = False
    assays: tuple[str, ...] = dataclass_field(kw_only=True)
    targets: tuple[str, ...] = dataclass_field(kw_only=True)
    reductions: tuple[str, ...] = ("stack",)

    def __post_init__(self):
        _text(self.model_id, "model_id")
        _text(self.identity, "model identity")
        for name in ("modalities", "assays", "targets", "reductions"):
            _sequence(self, name)
            if not getattr(self, name):
                raise ValueError(f"model {name} must not be empty")
            for item in getattr(self, name):
                _text(item, f"model {name}")
        if any(r not in ("stack", "max", "mean") for r in self.reductions):
            raise ValueError("unknown model input reduction")
        _integer(self.z_planes, "model z_planes", 1)
        if self.masks not in ("labels", "overlapping"):
            raise ValueError(
                "only 2D label and overlapping mask capabilities are supported"
            )
        if type(self.stateful) is not bool:
            raise ValueError("stateful must be boolean")


def _inputs(instance):
    _sequence(instance, "inputs")
    if not instance.inputs or any(
        not isinstance(x, ChannelInput) for x in instance.inputs
    ):
        raise ValueError("inputs must contain ChannelInput records")
    if len({x.channel for x in instance.inputs}) != len(instance.inputs):
        raise ValueError("a channel cannot be duplicated in one input")


@dataclass(frozen=True)
class SegmentationRequest:
    object_name: str
    inputs: tuple[ChannelInput, ...]
    model_id: str | None = None
    target: str = "cell"

    def __post_init__(self):
        _text(self.object_name, "object_name")
        _text(self.target, "segmentation target")
        _inputs(self)
        if self.model_id is not None:
            _text(self.model_id, "requested model_id")


@dataclass(frozen=True)
class MeasurementRequest:
    name: str
    object_name: str
    inputs: tuple[ChannelInput, ...]
    features: tuple[str, ...] = ("intensity",)

    def __post_init__(self):
        _text(self.name, "measurement name")
        _text(self.object_name, "measurement object_name")
        _inputs(self)
        _sequence(self, "features")
        if not self.features:
            raise ValueError("features must be explicit and nonempty")
        for feature in self.features:
            _text(feature, "feature")


@dataclass(frozen=True)
class AssayRequest:
    site_id: str
    source_identity: str
    ntps: int
    channels: tuple[ChannelEvidence, ...]
    segmentations: tuple[SegmentationRequest, ...]
    measurements: tuple[MeasurementRequest, ...] = ()
    assay_type: str = dataclass_field(kw_only=True)

    def __post_init__(self):
        _text(self.assay_type, "assay_type")
        _text(self.site_id, "site_id")
        _text(self.source_identity, "raw source_identity")
        _integer(self.ntps, "ntps", 1)
        for field, cls in (
            ("channels", ChannelEvidence),
            ("segmentations", SegmentationRequest),
            ("measurements", MeasurementRequest),
        ):
            _sequence(self, field)
            if any(not isinstance(x, cls) for x in getattr(self, field)):
                raise ValueError(f"invalid {field} record")
        if not self.channels or not self.segmentations:
            raise ValueError("channels and segmentations must be explicit and nonempty")
        indices = [c.index for c in self.channels]
        objects = [s.object_name for s in self.segmentations]
        names = [m.name for m in self.measurements]
        if any(len(x) != len(set(x)) for x in (indices, objects, names)):
            raise ValueError(
                "channel indices, object names, and measurement names must be unique"
            )
        object.__setattr__(
            self, "channels", tuple(sorted(self.channels, key=lambda c: c.index))
        )
        for channel in self.channels:
            if any(s.timepoint >= self.ntps for s in channel.samples):
                raise ValueError(
                    "sample timepoint outside requested interval [0, ntps)"
                )
        for operation in (*self.segmentations, *self.measurements):
            if any(i.channel not in indices for i in operation.inputs):
                raise ValueError("input references an unknown channel index")
        if any(m.object_name not in objects for m in self.measurements):
            raise ValueError("measurement references an unknown segmented object")


@dataclass(frozen=True)
class DecisionIssue:
    severity: str
    code: str
    target: str
    detail: str


@dataclass(frozen=True)
class AssayDecision:
    request: AssayRequest
    models: tuple[ModelCapability, ...]
    selected_models: tuple[tuple[str, str], ...]
    issues: tuple[DecisionIssue, ...]

    @property
    def ready(self):
        return not any(i.severity == "review" for i in self.issues)

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "status": "compatible_sampled_input" if self.ready else "review_required",
            **asdict(self),
        }

    @property
    def fingerprint(self):
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def require_ready(self):
        # Recompute rather than trusting an edited/persisted decision's status.
        checked = decide_assay(self.request, self.models)
        if checked != self or not checked.ready:
            reasons = sorted({i.code for i in checked.issues if i.severity == "review"})
            raise ValueError(f"assay decision is not ready or has changed: {reasons}")
        return self


def decide_assay(
    request: AssayRequest, models: tuple[ModelCapability, ...]
) -> AssayDecision:
    """Resolve declared capabilities, never infer a model from its name/camera.

    The supplied catalogue is the caller's candidate set, not a model search.
    Multiple compatible candidates require an explicit choice. Missing optional
    metadata does not prevent pixel-evidence-only decisions.
    """
    models = tuple(models)
    if any(not isinstance(m, ModelCapability) for m in models):
        raise ValueError("models must contain ModelCapability records")
    if len({m.model_id for m in models}) != len(models):
        raise ValueError("model IDs must be unique")
    models = tuple(sorted(models, key=lambda m: m.model_id))
    channels = {c.index: c for c in request.channels}
    issues = []
    selected = []

    def issue(severity, code, target, detail):
        issues.append(DecisionIssue(severity, code, target, detail))

    def check_inputs(inputs, target, segmentation):
        if len(inputs) > 1:
            times = [{s.timepoint for s in channels[i.channel].samples} for i in inputs]
            if not set.intersection(*times):
                issue(
                    "review",
                    "no_joint_channel_sample",
                    target,
                    "Multichannel inputs need at least one jointly observed timepoint.",
                )
        for inp in inputs:
            channel = channels[inp.channel]
            scope = f"{target}/channel:{channel.index}"
            if any(z >= channel.stored_z for z in inp.z):
                issue("review", "selected_z_out_of_bounds", scope, str(inp.z))
                continue
            for conflict in channel.metadata_conflicts:
                issue("review", "acquisition_metadata_conflict", scope, conflict)
            for warning in channel.metadata_warnings:
                issue("warning", "optional_metadata_warning", scope, warning)
            if channel.acquired_z is None:
                issue(
                    "warning",
                    "optional_metadata_pixel_fallback",
                    scope,
                    "Acquisition sidecar is not required.",
                )
            elif not set(inp.z).issubset(channel.acquired_z):
                issue(
                    "review",
                    "selected_plane_not_acquired",
                    scope,
                    str(channel.acquired_z),
                )
            if not channel.samples:
                issue(
                    "review",
                    "raw_evidence_missing",
                    scope,
                    "No sampled pixel evidence.",
                )
            for sample in channel.samples:
                if channel.acquired_z is not None:
                    unexpected = [
                        z
                        for z, state in enumerate(sample.states)
                        if state == "nonzero" and z not in channel.acquired_z
                    ]
                    if unexpected:
                        issue(
                            "review",
                            "acquisition_pixel_conflict",
                            scope,
                            f"tp{sample.timepoint}: nonzero planes outside acquisition declaration: {unexpected}",
                        )
                states = [sample.states[z] for z in inp.z]
                if any(s in ("missing", "unreadable", "unknown") for s in states):
                    issue(
                        "review",
                        "selected_plane_unobservable",
                        scope,
                        f"tp{sample.timepoint}: {states}",
                    )
                if "zero" in states:
                    if segmentation:
                        issue(
                            "review",
                            "zero_segmentation_plane",
                            scope,
                            f"tp{sample.timepoint}: review actual input, do not synthesize Z.",
                        )
                    else:
                        issue(
                            "warning",
                            "zero_measurement_not_biological_absence",
                            scope,
                            f"tp{sample.timepoint}: stored zeros, not proof of biological absence.",
                        )

    for seg in request.segmentations:
        check_inputs(seg.inputs, f"segment:{seg.object_name}", True)
        modalities = tuple(channels[i.channel].modality for i in seg.inputs)
        compatible = [
            m
            for m in models
            if request.assay_type in m.assays
            and seg.target in m.targets
            and m.modalities == modalities
            and all(
                i.depth == m.z_planes and i.reduction in m.reductions
                for i in seg.inputs
            )
        ]
        if seg.model_id is not None:
            matches = [m for m in models if m.model_id == seg.model_id]
            if not matches:
                issue(
                    "review",
                    "requested_model_unavailable",
                    seg.object_name,
                    seg.model_id,
                )
                continue
            if matches[0] not in compatible:
                issue(
                    "review", "model_input_incompatible", seg.object_name, seg.model_id
                )
                continue
        else:
            matches = compatible
            if len(matches) != 1:
                issue(
                    "review",
                    "model_choice_ambiguous" if matches else "no_compatible_model",
                    seg.object_name,
                    str([m.model_id for m in matches]),
                )
                continue
        selected.append((seg.object_name, matches[0].model_id))
        if matches[0].masks == "overlapping" and any(
            m.object_name == seg.object_name and len(m.inputs) > 1
            for m in request.measurements
        ):
            issue(
                "review",
                "overlapping_colocalization_unsupported",
                seg.object_name,
                "Current overlapping-mask extractor does not support multichannel colocalization.",
            )

    for measurement in request.measurements:
        check_inputs(measurement.inputs, f"measure:{measurement.name}", False)
        if any(i.depth != 1 for i in measurement.inputs):
            issue(
                "review",
                "measurement_requires_2d_input",
                measurement.name,
                "Declare a plane or explicit projection; volumetric extraction needs a separate capability.",
            )
    issue(
        "warning",
        "sampled_compatibility_only",
        request.site_id,
        "No guarantee of full-cadence usability, segmentation quality, identities, calibration, or biology. A fresh smoke and output validation remain required.",
    )
    return AssayDecision(request, models, tuple(selected), tuple(dict.fromkeys(issues)))


def select_inputs(pixels, inputs: tuple[ChannelInput, ...]):
    """Materialize explicit FCZYX inputs, preserving source and singleton axes.

    Separate calls for segmentation and measurement prevent a one-plane BF
    selection from truncating fluorescence. This helper does not run a model
    and is not automatically installed into the legacy pipeline dispatcher.
    """
    import numpy as np

    array = np.asarray(pixels)
    if array.ndim != 5:
        raise ValueError("pixels must have explicit FCZYX axes")
    inputs = tuple(inputs)
    if not inputs or any(not isinstance(i, ChannelInput) for i in inputs):
        raise ValueError("inputs must contain ChannelInput records")
    if len({i.channel for i in inputs}) != len(inputs):
        raise ValueError("input channels must not be duplicated")
    selected = []
    for inp in inputs:
        if inp.channel >= array.shape[1] or max(inp.z) >= array.shape[2]:
            raise ValueError("selection exceeds actual FCZYX pixels")
        part = np.take(array[:, inp.channel : inp.channel + 1], inp.z, axis=2)
        if not np.isfinite(part).all():
            raise ValueError(
                "selected pixels contain nonfinite values; no filling is allowed"
            )
        if inp.reduction == "max":
            part = part.max(axis=2, keepdims=True)
        elif inp.reduction == "mean":
            part = part.mean(axis=2, keepdims=True)
        selected.append(part)
    if len({p.shape[2] for p in selected}) != 1:
        raise ValueError("selected channels must have equal effective Z depth")
    result = np.concatenate(selected, axis=1)
    result.flags.writeable = False
    return result
