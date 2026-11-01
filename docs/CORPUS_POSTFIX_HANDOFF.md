# Authorized post-fix continuation

The user requested pushing the fixes to a branch and ensuring computational analysis proceeds under them. The existing sealed full-series attempt must not be hot-patched, duplicated or statefully resumed.

`corpus_assay_handoff.py` is a one-shot **waiting supervisor, not an inference controller**. It waits for the watched full attempt's owned pipeline/service shutdown receipt. Only after that boundary may it create the legacy root's operator STOP marker to prevent further legacy admission. A pre-existing/operator-changed STOP is never overridden. It also accepts natural predecessor termination (including its original timeout), without changing limits or terminating the live attempt itself.

Before launching anything, the handoff requires a terminal predecessor, an available predecessor ownership lock, and no live predecessor command lines or recorded process groups. It validates full published releases, including releases that landed before the legacy completion ledger was committed, to avoid duplicate processing across that race. Ambiguous published output blocks handoff. Genuine predecessor failures are deferred for diagnosis; this change does not pretend that provenance hardening fixes a slow 450-frame analysis.

A new `aliby.corpus-continuation.v1` plan is the original authorized 282-position complement minus validated predecessor completions and explicitly deferred genuine failures. Old outputs and partial attempts are retained unchanged. The fixed controller validates that partition, the predecessor terminal/input hashes, and completion ownership. A separate shared controller lock prevents overlapping new controllers across output roots. The old controller predates that lock, hence the additional predecessor process checks.

Before waiting, the handoff's scripts, metadata, model-evidence records, source identity and configuration are pinned by a pending seal whose digest is held in the supervisor's environment. They are rechecked while waiting and before transition; generating the execution seal cannot silently authorize intervening changes.

Every eligible position uses fresh preparation, services, sessions and outputs. The first representative is a short single-plane family, followed by the existing 3/12-frame sequential/async parity and full-series gates. Subsequent work remains bounded to four sites, four BABY workers/site, four total hours/site, 48 hours/batch and a 300-GiB reserve. No automatic retry of genuine failures or bypass of a gate is added. The controller records predecessor completions and deferred cases separately from its current counts.

The new root is `/work/users/amunoz/aliby_baby/analyses/corpus-postfix-20261101`. During waiting, `handoff-status.json` is authoritative and **does not mean fixed inference has started**. After confirmed startup, `campaign/status.json` is authoritative. The watcher confirms the new controller PID and active stage before reporting `fixed_runtime_running`; failed startup does not trigger a duplicate/retry. Its maximum waiting period is six hours. An operator can cancel the pending handoff using the new root's STOP file.

`gcd7_020` and the original 104 completions plus separate Gcd2 remain excluded. Human/biological review stays deferred. Forecasting remains quarantined. No monitor/session settings are changed by the handoff.
