# Astra review — 2026-11-01

## Ownership and deployment boundary

Reviewed directly by the current `gpt-6-astra` model. No second implementation owner or subagent was launched. Changes are isolated on `fix/astra-review-20261101`, based on the existing `c613bae` expansion commit, in `/work/users/amunoz/worktrees/aliby-astra-review-20261101`. They are **not deployed into the sealed running campaign**. Original edits, index, contributor history, stacked bases and outputs are preserved. No merge, push, service deployment, schedule/session change, shared-model change, or external message occurred.

Evidence root: `/work/users/amunoz/aliby_baby/analyses/astra-review-20261101/`.

## Actual evidence reviewed

Read the campaign driver, trial wrapper, tiler/TileLocations, BABY mask/lineage parser, guarded transport, frozen geometric preflight/validator, plan/checkpoint/seal artifacts, current process command lines and logs, existing diagnostic outputs, and the raw-tail investigation verdict. This was not a summary-only review.

The active expansion was still on its first full representative, `14190137:ura7ha_001`. The actual 3- and 12-frame sequential/async comparisons were recomputed read-only: exact submitted pixels, masks and semantic tables still match (709 and 3,133 profile/tracking rows). Completed diagnostic inventories and owned shutdown receipts were checked. No new model inference or service request was made for this review.

A separate read-only audit independently reconstructed raw BF crops and the existing uint8/NYXZ/F-order wire bytes at timepoints 0, 1, 2, 11, 150 and 300. All six submitted hashes match. Sampled phase increments match the recorded trajectory, and mask positive-label sets match tracking metadata for every retained ROI at those frames. The selection retains 79 of 89 ROIs and preserves all ten exclusions. An independent synthetic known-motion/label-plane test confirms signed cumulative registration and coordinate alignment; the existing drift-corrected path passes.

These are computational checks, **not biological approval or validated tracking identity**. At tp300, 45/79 masks are empty, compared with 0/79 at tp150; the remaining tiles contain 805 positive labels in total. This occupancy change is recorded for deferred review, not relabeled as biological absence or used to change ROIs/models. Physical calibration, fluorescence spectral identity and biological suitability remain unverified.

## Findings and isolated fixes

1. **P1 — incomplete evidence admission.** A typo/unknown backend skipped completion verification; an attempt could point to another directory's release; parent-directory symlinks were followed; extra files were ignored. These defects were reproduced. Verification now recognizes only the two explicit backends, binds the output to its owning attempt, requires decision identity agreement, reads inventory files through anchored no-follow descriptors, checks ordinary-file/stability properties, and verifies exact output inventory membership. Fresh sequential diagnostics now have a computational-only inventory manifest as well.

2. **P1 — execution seal and preparation checkpoint gaps.** Deleting the seal disabled that check. `prepared.json` endorsed its own mutable inventory/path hashes without a controller-held identity. Execution now requires a complete seal, and the controller detects seal replacement/removal. A parent-observed preparation digest is recorded and passed into child processes; the worker checks the same immutable bytes before admission and rechecks preparation/runtime evidence at preflight and validation. Source trees now cover the actually imported core, analysis helpers, transport and measurement code, plus interpreter identity and relevant dependency versions—not a hardcoded tree unrelated to loaded code. This is stronger provenance, not a claim of a fully hermetic native-library environment.

3. **P2 — broken uncorrected-image API.** `get_tp_channel(..., drift=False)` raised `UnboundLocalError`, while `get_fczyx(..., drift=False)` silently ignored the flag. Both failures were reproduced. The flag is now forwarded and uses the initial crop geometry without mutating drift history. The default corrected path and mask labels are unchanged.

4. **P2 — misleading/stale operational checkpoints and delayed cancellation.** The active status timestamp/elapsed field had not advanced since full-stage admission, and `held` included the running site. Future checkpoints add a heartbeat, mutually exclusive counts, and an explicit non-resumable designation. Failed-stage admission closes before potentially slow cleanup; SIGINT follows the same cancellation path as SIGTERM. Deadlines are checked before launching and after a child exits. Isolated fake-process tests exercise cleanup ordering, expired admission and signal registration without creating real jobs.

5. **P2 — failure evidence could replace the primary exception or skip cleanup.** The trial wrapper wrote failure receipts without protecting the original exception, and a worker cleanup exception could skip service cleanup. Failure-recording errors now attach notes to the primary exception. Both owned processes and the socket receive cleanup attempts; shutdown evidence cannot claim success after an unconfirmed stop. A standalone cleanup/evidence failure still propagates. Tests inject full-disk and worker-stop failures without touching live processes.

The initial reproduction suite had **7 failures / 29 passes**. After fixes and additional coverage, **360 targeted tests pass in each existing runtime**, with Ruff and whitespace checks passing. Warnings remain 1,208 in the guarded runtime and four in the project runtime, consistent with existing upstream warnings. This is not the full repository suite or an independent review of this patch.

## Active-run risk and remaining work

The active first full run has slowed to about 50 seconds/frame in recent completed-file receipts. Its controller's four-hour total-site ceiling is approximately **2026-11-01T04:55:15Z**, earlier than the launcher's individual full-attempt ceiling. At the observed rate it may time out before frame 450. This is a projection, **not an observed failure**. The sealed limits, session, worker processes and outputs were not altered. If it stops, retain the partial attempt and diagnose performance before any fresh restart; never resume its tracker state or silently increase the ceiling.

No actual checkpoint corruption, input substitution or source/seal modification was found during this review. Nevertheless the isolated hardening is not retroactive protection for the active controller. The legacy active trial wrapper still has its original failure-evidence/cleanup behavior; the isolated fixes do not alter it. No evidence of such an IO failure occurred here. Controller-level journal storage failures can still prevent durable progress reporting; the checkpoint is deliberately not a stateful-session recovery mechanism.

`gcd7_020` remains excluded: the inspected investigation still establishes complete acquired data only through tp182, partial tp183 and missing tp184–239, with no verified alternate recovery source. The protected original 104 positions and separate Gcd2 remain excluded. Human/biological review remains deferred; forecasting remains quarantined.

**Adoption is deferred.** Do not copy this patch into an active sealed root or replay the all-282 plan in another root. After active work is safely terminal, adoption needs a separately pinned remaining-only continuation plan excluding all newly completed positions, a new seal, an explicit matching `ALIBY_SOURCE_WORKTREE`, and fresh short parity/integrity gates for the changed source. Planning-only scope checks may explicitly use `require_seal=False`; production admission may not. Existing code-review/approval gates remain intact.
