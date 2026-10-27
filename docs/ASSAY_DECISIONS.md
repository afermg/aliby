# Assay decisions before async orchestration

## Status and architecture decision

**Planning foundation with an opt-in runtime implementation in this worktree.**
The planning records remain separate from execution. Sequential stays the
reference/default, and production outputs are unchanged. The experimental
runtime bindings and asyncio backend reuse the existing graph; see
[Assay bindings and async execution](ASSAY_ASYNC_RUNTIME.md) for testing, explicit
validators, limitations, and real-data admission requirements.

Separate four responsibilities:

1. **Evidence adapters:** observe raw pixels/layout and optional acquisition
   metadata; retain source identities, exact-site matches, and conflicts.
2. **Assay decisions:** pure, immutable, serializable records describing what to
   segment and measure, why a candidate is compatible, and what requires review.
3. **Backend bindings:** implement and verify those decisions against concrete
   image readers, models, feature extractors, and trackers.
4. **Scheduling/publication:** arrange bound operations under dependency,
   resource, session, failure, and release constraints.

`aliby.assay` implements the second layer and a small explicit array-selection
helper. `aliby.execution_contract` derives a **non-executable target contract**
for the fourth layer from the existing pipeline dictionary. Neither module does
network I/O, starts a service, discovers files, installs models, or launches a
reconstruction. `aliby.assay_runtime` now supplies explicit BABY/local-Cellpose
and measurement bindings, and `aliby.assay_release` provides fresh staged
publication with mandatory supplied preflight/output validators. Raw-format
adapters and real model/domain validation still need site-specific integration.

This deliberately prevents replacing a nested loop with asyncio while leaving
assay choices and hidden mutable state embedded inside the scheduler.

## Decisions we are fixing in place

| Concern | Rule | Reason |
|---|---|---|
| Acquisition metadata | Helpful, not mandatory | Missing/unsupported/unreadable/nonmatching sidecars permit a pixel-evidence fallback. |
| Channel identity | Explicit index, name, modality, assay, and source | A channel name, filename, or model name is not enough to infer biology or optics. |
| Z | Per-channel acquired indices, stored extent, and observed planes are separate | Global Z=5 does not establish five BF acquisitions. |
| Input choice | Explicit increasing unique Z indices and explicit reduction | No duplicated planes, synthetic pixels, automatic focus choice, or implicit projection. |
| Models | Declared assay/target/modalities/Z/reduction capabilities and identity | Shape compatibility alone is insufficient; multiple compatible candidates require a choice. |
| Fluorescence | Independent measurement selections | Selecting BF Z0 must not truncate sectioned mCherry or project zero-padded GFP. |
| Missing versus zero | Missing/unreadable/unknown selected pixels require review | Stored-zero fluorescence is observable data, not proof of no cells or no expression. |
| Biological validation | Separate from computational compatibility | A compatible sampled input is not a validated segmentation, identity, phenotype, or usable interval. |
| Stateful processing | Fresh attempt/session, ordered frames, no automatic retry | An indeterminate POST may already have advanced the tracker. |
| Publication | Computation → commit → final validation → release | A returned mask or partially written directory is not a successful dataset. |

`Z sect.=0` in the current acquisition format maps to `(0,)`, **not** `()`.
An empty acquired-plane tuple means explicitly not acquired. `None` means the
optional acquisition declaration is unavailable or unusable. Only an evidence
adapter with a verified exact-site match should populate `acquired_z`.

Model capabilities describe a caller-supplied, scoped candidate catalogue, not
an automatic search over every installed model. Assay and target declarations
must come from the model's intended use or an explicitly scoped diagnostic.
They do not independently establish camera, magnification, calibration, or
accuracy. Default model reductions accept only a stack/slice; a single-plane
shape does **not** automatically approve a five-plane max/mean projection.

## Public interfaces

### Pure decision layer

- `PlaneSample(timepoint, states, evidence_id)` describes **every stored Z slot**
  at that sampled timepoint. States: `nonzero`, `zero`, `missing`, `unreadable`,
  `unknown`.
- `ChannelEvidence` separates stored Z, optional acquired indices, sampled
  evidence, metadata conflicts, metadata warnings, and evidence identities.
- `ChannelInput(channel, z, reduction="stack")` is a selection, not a raw-data
  rewrite. `stack`, `max`, and `mean` are explicit transformations.
- `ModelCapability` declares model identity, eligible assays/targets, ordered
  modalities, required effective Z depth, allowed reductions, overlapping versus
  label masks, and statefulness. No capability is inferred from a model name.
- `AssayRequest` contains site/raw identity, requested prefix `[0, ntps)`, assay
  type, segmentations, and measurements.
- `decide_assay(request, models)` returns an immutable `AssayDecision` with
  stable issue codes, selected model IDs, JSON-compatible `to_dict()`, and a
  SHA256 `fingerprint` covering the full decision/evidence/candidate catalogue.
- `decision.require_ready()` recomputes the decision; stripping its review
  issues or editing a selected model cannot turn it into an accepted decision.

`ready` means **compatible sampled input**, not permission to execute or release.
Sidecar warnings alone do not block. Incompatible/missing/ambiguous models,
conflicting acquisition evidence, unavailable selected pixels, zero BF input,
and unsupported overlapping-mask colocalization require review. Malformed
configuration raises `ValueError` before work begins.

The first implementation covers 2D segmentation/measurements, including models
that consume a Z stack to produce 2D masks. Volumetric masks/measurements and
embedding-only assays need separate capabilities; they are not silently
reinterpreted as a supported assay. Feature names are declarations, not proof a
particular extractor implements them. Runtime adapters must validate them.

### Minimal synthetic example

This is a **synthetic policy example**, not evidence approving a real model or
position. Replace identity strings and observations with verified artifacts.

```python
from aliby.assay import (
    AssayRequest, ChannelEvidence, ChannelInput, MeasurementRequest,
    ModelCapability, PlaneSample, SegmentationRequest, decide_assay,
)

bf = ChannelEvidence(
    index=0, name="Brightfield", modality="brightfield", stored_z=5,
    acquired_z=(0,), metadata_evidence=("example:exact-site-sidecar",),
    samples=(PlaneSample(0, ("nonzero", "zero", "zero", "zero", "zero"),
                         "example:BF-pixel-digest"),),
)
red = ChannelEvidence(
    index=1, name="mCherry", modality="fluorescence", stored_z=5,
    # No sidecar declaration: pixel-only fallback remains available.
    samples=(PlaneSample(0, ("nonzero",) * 5, "example:red-pixel-digest"),),
)
request = AssayRequest(
    site_id="example-site", source_identity="example:raw-archive-digest",
    assay_type="yeast_microfluidics", ntps=12, channels=(bf, red),
    segmentations=(SegmentationRequest("cell", (ChannelInput(0, (0,)),)),),
    measurements=(MeasurementRequest(
        "red_intensity", "cell", (ChannelInput(1, (0, 1, 2, 3, 4), "max"),)
    ),),
)
model = ModelCapability(
    model_id="example-one-plane-model", identity="example:weights-and-config",
    assays=("yeast_microfluidics",), targets=("cell",),
    modalities=("brightfield",), z_planes=1,
    masks="overlapping", stateful=True,
)
decision = decide_assay(request, (model,))
assert decision.ready
assert decision.selected_models == (("cell", "example-one-plane-model"),)
report = decision.to_dict()  # Explicit caller-controlled persistence.
```

`select_inputs(pixels, inputs)` materializes read-only **FCZYX** arrays without
changing the input array or squeezing singleton dimensions. Call it separately
for segmentation and each measurement. It validates actual bounds, finite
pixels, unique channels, and equal resulting channel depths. It does not cast,
normalize, interpolate, or fill missing/nonfinite data. Mean reduction produces
floating-point pixels; a backend must explicitly support the resulting dtype.

After selection, channel positions are the order of `inputs`: selecting original
BF channel 2 alone produces output channel **0**. A future backend adapter must
bind that new channel index consistently; passing the old channel 2 to the
client would be wrong. Extraction must still receive its own selection of the
original raw pixels.

Do **not** add an unrecognized selection keyword to an existing segmenter and
assume it is honored. Use the fingerprinted `bind_assay_pipeline` path described
in [the runtime guide](ASSAY_ASYNC_RUNTIME.md). Unbound legacy behavior remains
unchanged.

## Target execution contract

`compile_execution_contract(pipeline, decision, attempt_id=...,
pipeline_identity=..., max_workers=..., resource_limits=...)`:

- Requires a recomputed compatible decision and matching `ntps`.
- Reuses `compile_pipeline_graph` and existing resource validation. Missing
  nodes, cycles, unsupported resource capacities, and ambiguous BABY tiler
  bindings fail before initialization.
- Expands existing per-frame graph edges. Every root of frame `t+1` depends on
  the complete commit/cleanup barrier for frame `t`.
- Gives tiling/drift, BABY, selected stateful models, and trackers logical
  session scopes containing **site, fresh attempt, and step**.
- Preserves independent within-frame branches, while preventing a subsequent
  frame from changing the mutable tiler/crawler behind a prior frame's method
  call. Global work remains serial after the full time series.
- Records worker/resource limits, no-retry policy, restart-from-zero failure
  policy, final validation, and publication dependencies.
- Produces `aliby.execution-contract.v1` with `executable: false` and a stable
  fingerprint. This is an inspectable migration target, **not a runnable DAG**.

The caller supplies the identity of the frozen configuration/code binding;
Python callables are not implicitly serialized or hashed. An attempt ID is a
logical isolation key, not a created remote session or an acquired run lock.
No execution backend currently enforces this new record directly.

The future commit operation must preserve stable write order and perform
retention/tiler cleanup only after all frame consumers finish. This is a target
contract, not a claim that today's sequential runner is transactional: it may
write intermediate results before a later step fails. Even an async adapter
cannot roll back a remote POST or arbitrary side effects inside user functions.

## Tests as the migration gate

`tests/test_assay_decisions.py` covers:

- Single-plane BF in five-slot storage with five-plane mCherry and Z0-only GFP.
- One/three/five-plane inputs with no acquisition sidecar; unusable sidecars;
  exact conflicts and not-acquired declarations.
- Explicit/ambiguous/incompatible models, assay/target mismatches, and projection
  compatibility rather than model-name guessing.
- Cell Painting nuclei/cell models, multichannel inputs, and colocalization.
- Unobservable versus zero data; raw/metadata bounds; immutable, identity-bound
  decisions; no plane duplication or nonfinite filling.
- Independent nonmutating selections, including negative GFP values that would
  incorrectly become zero under an all-Z max of padded storage.

`tests/test_assay_execution_contract.py` uses a **test-only asyncio interpreter**
with fake nodes, not a production scheduler. It checks:

- Successful-result parity with both existing sequential and threaded runners.
- Independent branch overlap and descendants starting before another branch
  finishes, without requiring a particular arbitrary task start order.
- Stateful temporal ordering, attempt-specific session scopes, and resources.
- Failure, timeout, cancellation, failed commit, and failed final validation:
  no retry, no later frame, no dependent extraction, and no release publication.
- Global barriers, missing/cyclic references, and rejected edited decisions.

These tests validate policy and dependency semantics. They do not establish
real RPC cancellation, numerical model quality, cp_measure integration, or
transactional filesystem behavior.

## Staged replacement plan

1. **Evidence adapters:** convert current raw/archive/channel-Z checks into these
   records, retaining source hashes, requested interval, channel-specific
   acquisition flags and missing-chunk observations. Never turn a Zarr fill value
   for a missing chunk into an observed zero. Keep ROI selection/mappings and
   retrospective censoring separate from biological eligibility.
2. **Legacy bindings:** lower accepted selections into current BABY/Cellpose and
   measurement adapters. Validate actual loaded model shape, input dtype and
   normalization, channel remapping, transmitted Z dimensions, features,
   output mask layout, ROI geometry, and model/code identities. Fail closed on
   unsupported options. Run small fresh real-data diagnostics before any batch.
3. **Frame-scoped artifacts:** replace implicit `state['data'][-1]` dependencies
   and live tiler-method reads with explicit `(site, attempt, frame, step)` values
   or immutable artifacts. Preserve lineage/mask/profile semantic keys and
   distinguish absent output, saved empty segmentation, and failed execution.
4. **Async adapter:** bind concrete operations to the derived graph. Await
   resource admission before service initialization/submission; enforce deadlines,
   bounded queues, session serialization, and failure ownership. Reuse the
   conformance suite, then add actual transport, thread/process cancellation,
   memory bounds, lost-response, restart, and crash-after-write tests.
5. **Release gate:** implement real ordered commits and release validation with
   owned staging roots, locks, manifests, hashes, HTTP receipts, ROI evidence,
   tracking/profile consistency, and repeated-mask checks. Never publish a
   release marker after failure. Empty outputs require explicit interpretation,
   not automatic biological-negative labels.
6. **Opt-in parity rollout:** run both backends on separately named small runs,
   compare semantic outputs and provenance, then widen only after reviewed
   parity. Keep sequential execution as a reference/fallback chosen before a
   run, never a transparent retry of a failed stateful session.
7. **Only later relax frame barriers:** permit cross-frame overlap once readers
   and all downstream inputs are immutable and temporal dependencies are
   explicit. A semaphore alone does not enforce chronological tracking.

Production reconstruction, acquisition/model calibration review, and the old
forecasting quarantine are not changed or approved by this planning work.
