# Experimental assay bindings and async execution

This worktree implements opt-in runtime bindings and an asyncio backend on top
of the existing graph. **Sequential remains the default.** Nothing here starts
a production batch, approves a held dataset, installs a model, or removes the
forecasting quarantine. Test results use synthetic images and stub neural models
with real ALIBY extraction, Parquet/NPZ writing, and provenance code.

## Real-data follow-up

After explicit staged-processing authorization, paired real BABY trials passed
on three experiment19307 representatives. See [ASSAY_REAL_TRIAL.md](ASSAY_REAL_TRIAL.md)
for exact scope, parity results, limitations, and the live expansion status path.
The historical synthetic checkpoint below is retained as such.

## Validation checkpoint

On 2026-10-28, **290 targeted tests passed in each of two existing runtimes**
(the project environment and the retained guarded pipeline environment).
This adds **23 integrity regression cases** to the 267-test checkpoint from
2026-10-27. Ruff and whitespace checks passed. This was not a full repository
test run. Four existing fork deprecation warnings remain; the guarded environment
reports 1,208 warnings in total, mostly upstream scikit-image/NumPy reshape
deprecations. No dependencies were installed/upgraded.

The additional regressions cover four repaired defects:

- Bound feature histories cannot be truncated and silently renumbered from
  timepoint zero. Explicit retention must cover all requested frames; bounded
  pixel/mask history remains supported where dependencies permit it.
- Optional feature groups use a full outer join on the semantic keys for bound
  pipelines. Objects without colocalization measurements remain present with
  nulls, regardless of step insertion order. Unbound legacy behavior is unchanged.
- Existing computed artifacts and audit records are snapshotted before output
  validation and rechecked both afterwards and immediately before publication.
  Validators may add separate reports, not modify, replace, or delete outputs.
- Failure-evidence IO errors cannot mask the original failure or cancellation.
  Existing evidence is not overwritten; a recording error is attached as an
  exception note instead.

Reproductions, both runtime logs, source identities, and preservation checks are
retained in `/work/users/amunoz/aliby_baby/analyses/assay-async-regression-20261028`.

The worktree is isolated from the original checkout. Real neural inference,
live HTTP/server cancellation, and biological segmentation quality remain
untested; the next stage is a jointly reviewed small fresh diagnostic, not an
automatic corpus continuation.

## Start with the local tests

From this worktree (not the original checkout):

```sh
cd /work/users/amunoz/worktrees/aliby-assay-async-20261027
PYTHONPATH="$PWD/src" /work/users/amunoz/projects/aliby/.venv/bin/python -m pytest -q \
  tests/test_async_backend.py tests/test_assay_runtime.py tests/test_assay_release.py \
  tests/test_assay_integrity_edges.py
```

For the full targeted regression set, add:

```sh
  tests/test_assay_decisions.py tests/test_assay_execution_contract.py \
  tests/test_dag_executor.py tests/test_roi_selection.py tests/test_roi_provenance.py \
  tests/test_tiler_publication_controls.py tests/test_tiler_center_fallback.py
```

The tests do not need a model server, weights download, or raw corpus access.
The weights used by model stubs are explicitly fake test files. Do not interpret
a passing synthetic model test as segmentation validation.

### Important environment distinction

The ordinary project's `.venv` currently has an **unguarded Nahual BABY client**
(source SHA256 `43916990c9cbf31174a2a6067dcef05453ed48fc6ba03a36d74ee811db63960e`).
It can run the stubbed tests, but the new live BABY adapter deliberately rejects
that client before loading a session. No shared environment was upgraded.

The retained repaired runtime has the guarded client (SHA256
`28ab657afa5bac0712105f8236487c8a114e4018cf44a0665a6eb1e11428dc26`):

```sh
PYTHONPATH="$PWD/src" \
  /work/users/amunoz/aliby_baby/analyses/gcd2-repair-20261021/runtime/pipeline-venv/bin/python \
  -m pytest -q tests/test_async_backend.py tests/test_assay_runtime.py tests/test_assay_release.py \
  tests/test_assay_integrity_edges.py
```

Using `PYTHONPATH` here deliberately tests this worktree without modifying either
installed environment. For a real diagnostic, freeze a separately identified
runtime/source binding; do not describe the old installed distribution revision
as the new worktree's execution revision. Runtime receipts record actual source
file paths/hashes as well as distribution versions.

## Binding a decision

```python
from aliby.assay_runtime import bind_assay_pipeline

# pipeline and decision are explicitly prepared for the same site and ntps.
# Identity values come from verified artifacts/owned-service attestation, not
# guessed model names or a user-interface approval checkbox.
bound = bind_assay_pipeline(
    pipeline,
    decision,
    model_identities={model.model_id: model.identity for model in decision.models},
)
```

`bind_assay_pipeline` calls `decision.require_ready()`, deep-copies the existing
pipeline, and produces `aliby.assay-binding.v1`. It does not initialize models.
The binding fingerprints the decision, model declarations, source configuration,
explicit selections, feature trees, and graph wiring. Changing the bound
configuration requires binding/review again; the runner rejects stale bindings.
Configuration must contain portable finite data; use string paths rather than
`Path` objects or executable callables in bound configuration. Preserve the
Python pipeline configuration/recipe separately; `assay-binding.json` is an
audit record, not a decoder for arbitrary pipeline dictionaries.

Implemented adapters:

- **BABY:** one explicitly selected BF channel; model-specific 1/3/5-Z stack as
  declared in the capability; overlapping masks and persistent labels. Original
  channel 2, for example, becomes C0 in the compact submitted array. The modelset
  must match the selected model ID. Actual finite input, dtype, Z extent, nonzero
  planes, tile count, layered mask shape, and mask/tracking labels are checked.
  The guarded client's failure-session/deadline interface is required; its
  transport logic is not replaced and no retry is introduced. The supplied
  remote model identity is a declaration until preflight/validation attests it
  against the owned service and its immutable assets.
- **Local Cellpose:** explicit existing weights whose SHA256 matches the model
  identity; no default-model lookup/download. The weights inode is held while
  loading via `/proc/self/fd`, and concurrent modification is rejected. Explicit
  2D, one- or multichannel input is submitted as a list of per-tile YXC images,
  preventing the legacy F/Z ambiguity and tile-label collapse. Allowed setup:
  `pretrained_model`, `gpu`, `use_bfloat16`. Only the listed numerical eval
  options in `assay_runtime.py` are accepted; no implicit channel/Z transform.
- **Measurements:** independent original-channel/Z selections for each object's
  mono-channel and two-channel colocalization extractors. Actual `cp_measure`
  functions run on the selected/projected pixels, not a truncated segmentation
  input. Features are checked against the extractor registry and input arity.
  Output metric prefixes retain original channel/Z identity, e.g.
  `c1[z=0;stack]/max/intensity/...`. The inner singleton-Z `max` is an identity
  reduction after the explicitly recorded selection, not an all-storage max.
  Different object feature schemas are unioned with nulls, not invented zeros.

Current restrictions fail closed: one tiler; conventional alphanumeric object
names; one selection per original channel/object/extractor; no mixed segmenter
families in one site; no unsupported remote Cellpose/Spotiflow binding; no
volumetric mask adapter; no overlapping-mask colocalization. Input selection
helpers do not cast, interpolate, duplicate Z, fill nonfinite data, or guess
physical calibration. Mean projections are not accepted by the integer-only
BABY transport. Legacy unbound behavior is unchanged.

## Running the async backend

For synchronous CLI code, the existing entry point accepts `backend="async"`:

```python
from aliby.pipe import run_pipeline_and_post

profiles, post = run_pipeline_and_post(
    bound, pipeline_name="site", output_path="fresh-diagnostic-output",
    backend="async", max_workers=4, resource_limits={"cpu": 4},
    step_timeout=600,
)
```

For BABY tracking/ROI publication, use the corresponding entry point from
`aliby.pipe_baby`. These direct artifact-writer APIs retain partial outputs on
failure, like the legacy API; **an existing profiles file is not a release gate**.

In an existing event loop, await `aliby.pipe.run_pipeline_and_post_async` or
`aliby.pipe_baby.run_pipeline_and_post_async` instead; omit `backend` in the
native async call. The lower-level
`aliby.pipe_core.run_pipeline_return_state_async` supports an explicit
initializer and native async step callables. Calling the synchronous async
wrapper from an active event loop fails before initialization.

Semantics:

- Existing graph wiring and resource validation are reused, with a conservative
  async default of four workers (the legacy thread-backend default is unchanged).
  Independent branches
  and ready descendants can overlap **within one frame**.
- Initializers are serialized, with resource slots reserved **before** model or
  session initialization. Context variables propagate to blocking adapters.
- Timepoints and global steps remain serial. A later frame cannot mutate a live
  tiler/crawler while an earlier consumer still reads it. Each run initializes
  fresh step instances; no failed-session resume or state splicing is offered.
- Native async callbacks are awaited. Blocking adapters run in a private bounded
  thread pool. Successful results are published to dependent nodes as ready;
  executor-owned frame writes occur in stable graph order only after all nodes
  succeed.
- On an observed failure, stop admitting work, cancel cooperative async work,
  drain admitted blocking work, and abort the site. No retry, later frame,
  incomplete-frame commit, or automatic fallback backend.
- `step_timeout` bounds callback waiting during initialization/computation/writes,
  postprocessing, and supplied release checks. **It is not a hard thread kill.**
  A running blocking call must finish before ownership/run locks are released;
  wall time can therefore exceed the timeout while draining. Transport adapters
  need finite deadlines, and native async callbacks must cooperate with
  cancellation. A hung plugin still requires process-level containment.
- Slots coordinate local admission, not device affinity, memory budgets, native
  library thread counts, or service-side workers. Declare explicit GPU/endpoint
  resources and retain the existing site/batch budgets in an external driver.

The earlier `execution_contract.v1` remains a non-executable inspection/testing
record. The actual backend consumes the same existing compiled graph and
implements its conservative frame/resource/failure rules; it does not execute
arbitrary task names from a saved contract.

## Fresh, validation-gated publication

For an experimental accepted release, prefer:

```python
from aliby.assay_release import run_assay_release_async

release = await run_assay_release_async(
    bound, "site", "/approved/parent/new-release-name",
    preflight=verify_inputs_before_execution,
    validator=validate_actual_outputs,
    max_workers=4, resource_limits={"cpu": 4}, step_timeout=600,
)
```

Both callbacks receive `(stage, binding)` and may be sync or async. There are
**no default approvals**. Each must return JSON-compatible data such as
`{"accepted": true, "checks": {...}, "evidence": {...}}`; every declared check
must be literal `True` and all required check names must be present.

- Preflight runs before any model initialization. Required checks:
  `raw_identity`, `model_identity`; additionally `roi_selection` for BABY.
  Use real archive/source identities and the existing full-series ROI selection
  preflight, not merely the strings in the decision.
- Post-run validation requires `model_identity`, `input_layout`, `semantic_keys`;
  BABY also requires `transport`, `roi_geometry`, `repeated_masks`. Reuse the
  existing guarded receipts, executed-ROI validation, full-frame repeat audit,
  and profile/tracking key checks. A callback returning named booleans is an
  attestation interface, not an implementation of those scientific/domain checks.
- BABY releases additionally require existing `publish_roi_provenance=True` and
  its valid `run_binding`, saved masks, and full-run lock. Existing restrictions,
  including no BABY provenance/global-step combination, remain in effect.
- The library checks nonempty profiles, rejects nonregular artifacts, hashes and
  fsyncs files, records runtime/source identities, and rechecks artifact and
  directory identities before publishing. Validation receipt and manifest are
  included in the staged evidence, along with effective execution/resource options.
  All pre-existing files and directories are immutable during output validation.
  Write any additional reports into a new root-level directory such as
  `stage / "validation-artifacts"`, not inside existing output directories.
  Service logs must likewise not keep modifying pre-existing staged files.
- Output parent must exist. A private fresh sibling attempt directory is used.
  An existing destination, including a symlink, is never replaced. Atomic
  publication uses Linux `renameat2(RENAME_NOREPLACE)` and fails closed if that
  operation is unavailable. Private staging is owned by this run; external
  writers must not write into it.
- Failure/cancellation retains the hidden attempt and writes
  `.name.attempt-<uuid>/failure.json` when possible; it does not create a final
  release. If evidence recording fails, the original exception carries a note
  identifying that failure and the attempt path; inspect the retained directory.
  Do not resume the attempt. If
  an OS error occurs after the rename, inspect the destination/manifest before
  taking any action; do not retry into or overwrite it.
- `completion.json` says `computationally_validated`,
  `experimental_backend: "async"`, and `biological_approval: false`.
  It is **not** a human approval, calibrated assay, or forecasting admission.

## Before testing on real data together

1. Review this worktree diff and the synthetic regression results.
2. Select a small, explicitly authorized fresh diagnostic and pinned compatible
   model; confirm its channel-specific Z and preserve all fluorescence planes.
   Missing acquisition sidecars remain acceptable with explicit configuration
   and pixel evidence. No single-plane model was downloaded or tested here.
3. Prepare an isolated guarded runtime/service and real preflight/release
   validators. Preserve original ROI IDs/exclusions and transport evidence.
4. Compare sequential and async **fresh sessions** in separate roots, including
   mask/tracking/profile semantic equality and overlays. A failed/indeterminate
   session must not be retried in the other backend.
5. Only widen scope after that review. Keep the 19 held positions, failed
   attempts, completed outputs, and old forecast quarantine unchanged meanwhile.
