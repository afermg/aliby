# Bounded real-data rollout (2026-10-28/29)

The user authorized a small real-data trial, progressive expansion on success,
and repairs followed by fresh runs on failure. This does not authorize bypassing
integrity gates, resuming failed stateful sessions, replacing completed outputs,
or admitting the quarantined forecasting benchmark.

## Small real trials completed

Using the unchanged assay runtime from `c045492`, the retained guarded pipeline
runtime and pinned BABY service `885ce5e`, the official model
`yeast-alcatras-brightfield-sCMOS-60x-1z` was downloaded into a private trial cache.
Its immutable effective model identity is
`8b9588c562d73dc75ece01fe0699a8eaa2fd0207fc08c8fc6a7bcd1892e93abb`.
The actual loaded CNN input is `[None,128,128,1]`; BABY's declared default image
crop remains 117×117. An initial inspection incorrectly equated the CNN's spatial
shape with crop size; that failed inspection is retained. No model was retuned.
Camera/pixel calibration remain unverified.

Segmentation submits original BF channel2, Z0. Measurements independently use
mCherry channel0 Z0–4 (explicit max projection) and GFP channel1 Z0. Raw storage
is unchanged. All runs use the same whole-series geometric ROI selection within
each site, with original ROI mappings/exclusion reasons retained.

Fresh sequential and async processes/services/sessions passed exact parity:

| Site | Frames | Profile/tracking rows per run |
|---|---:|---:|
| `phluorin_dot6_001` | 0–2 | 190 |
| `phluorin_dot6_001` | 0–11 | 788 |
| `phluorin_nrg1_014` | 0–11 | 1,853 |
| `phluorin_tod6_019` | 0–11 | 2,031 |

Submitted pixel hashes, mask arrays, and sorted semantic profile/tracking tables
were exactly equal. Every run passed raw identity, input layout, ROI provenance,
executed geometry, transport, and repeated-mask gates. Each owned service was
stopped after its attempt. These timings do not demonstrate an async speedup:
this pipeline remains dominated by serial/stateful work and validation overhead.

Assistant inspection of fixed geometric targets at tp0/6/11 found cell-like
outlines rather than systematically empty/stale masks. Small/boundary detections,
ROI-clipped cells and Tod6 lower-field shading remain caveats. This is not an
independent or human biological validation. Initial review images counted mask
layers rather than unique labels; v2 figures correct counts, while the semantic
validators were already label-based. Original figures are retained.

## Expansion

The root is
`/work/users/amunoz/aliby_baby/analyses/assay-real-trial-20261028`.
`PLAN.md` and `evidence/expansion.json` record scope, evidence hashes, and review.
`expansion/status.json` is authoritative for current progress; do not infer full
completion from the small-trial table above.

The bounded driver first runs all 180 frames of Dot6_001 in a fresh async
session. Only on success does it proceed through the other 18 previously held
experiment19307 sites. Each non-representative site must pass its own fresh
three-frame smoke before its full run. All final directories are new. The 85
completed batch positions, separate Gcd2 repair, and 282 out-of-scope positions
are not reprocessed.

Limits: one site at a time, four BABY CPU workers/site, four hours/full site,
48 hours/batch, at least 300 GiB free. The driver stops on the **first** failure
(stricter than the prior three-consecutive-failure ceiling). It never retries or
fixes code autonomously: inspect retained failure evidence, repair in a worktree,
rerun regression/parity gates if identities changed, then explicitly prepare a
new continuation. No rerun over a completed or failed attempt.

Create `STOP` in the trial root for an orderly stop of owned processes. The
launcher handles termination by stopping its own worker and service groups;
it does not adopt or stop unrelated services. Live logs and PID records remain
in each attempt. A completion marker denotes computational validation only.

## Code and regression evidence

The host-specific examples are not a portable production CLI:

- `examples/assay_real_trial_models.py`: bounded private official-model download.
- `examples/assay_real_trial.py`: raw/ROI preparation, fresh owned services,
  sequential/async execution, real validators, exact parity comparison.
- `examples/assay_real_trial_review.py`: fixed-target overlays, unique-label counts.
- `examples/assay_real_expansion.py`: one-shot gated, bounded expansion driver.
- `tests/test_real_trial_driver.py`: 17 synthetic scope/stop/disk/evidence/parity
  gate cases. **307 targeted tests passed in each existing runtime**, with four
  project warnings and 1,208 guarded-runtime warnings. Not a full repository run.

Frozen launch scripts and logs live under the trial root. The initial catalogue
request failed because stdlib urllib lacked CA trust; the subsequent request used
requests' configured CA trust with TLS verification enabled. No shared cache or
environment was upgraded. The original checkout remains untouched.
