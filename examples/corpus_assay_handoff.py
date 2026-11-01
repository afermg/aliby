"""One-shot boundary handoff: no hot patch, no overlapping analysis controllers.

Wait for the already-running full attempt to stop its owned workers. Only then
request legacy admission stop, reconcile published outputs, and start fresh
fixed-runtime work on the eligible remainder. Genuine predecessor failures are
held for diagnosis, not blindly retried. This watcher performs no inference.
"""

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import corpus_assay_trial as trial


def live_owned_processes(parent):
    groups = set()
    for pattern in (
        "jobs/*/attempts/*/service-process.json",
        "jobs/*/attempts/*/pipeline-process.json",
    ):
        for path in parent.glob(pattern):
            groups.add(trial.load(path)["pid"])
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if int(entry.name) in groups:
                raise RuntimeError("Cannot establish predecessor process shutdown")
            continue
        if fields[0] == "Z":
            continue
        named = str(parent).encode() in argv and any(
            Path(os.fsdecode(arg)).name
            in ("corpus_assay_expansion.py", "corpus_assay_trial.py")
            for arg in argv
            if arg
        )
        if named or int(fields[2]) in groups:
            found.append(int(entry.name))
    return found


def assert_quiescent(parent):
    terminal = trial.load(parent / "campaign/terminal.json")
    if terminal["status"] not in ("completed", "stopped") or terminal.get("current"):
        raise ValueError("Predecessor is not terminal")
    if live_owned_processes(parent):
        raise RuntimeError("Predecessor still owns live processes")
    with (parent / "campaign/owner.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def reconcile(parent):
    """Adopt validated published full outputs even across a ledger-write race."""
    assert_quiescent(parent)
    plan = trial.load(parent / "plan.json")
    terminal = trial.load(parent / "campaign/terminal.json")
    expected_source = trial.load(parent / "source-identity.json")
    completed = []
    for job in plan["jobs"]:
        matches = []
        for attempt in (parent / "jobs" / job["id"] / "attempts").glob("*"):
            identity_path = attempt / "attempt.json"
            if not identity_path.exists():
                continue
            identity = trial.load(identity_path)
            if identity["ntps"] != job["ntps"] or identity["backend"] != "async":
                continue
            release = attempt / "release"
            if not release.exists():
                continue
            # An ambiguous published output is never silently reprocessed.
            result, receipt = trial.verify_attempt(attempt)
            if (
                result["key"] != job["key"]
                or result["ntps"] != job["ntps"]
                or result["source_identity"] != expected_source
            ):
                raise ValueError("Predecessor published identity mismatch")
            matches.append(
                {
                    "key": job["key"],
                    "attempt": str(attempt),
                    "completion": str(release / "completion.json"),
                    "completion_sha256": trial.sha(release / "completion.json"),
                    "result_sha256": trial.sha(attempt / "result.json"),
                    "summary": receipt["summary"],
                }
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple published full attempts; manual reconciliation required"
            )
        completed.extend(matches)
    done = {row["key"] for row in completed}
    if not {row["key"] for row in terminal["completed"]} <= done:
        raise ValueError("Ledger completion has no validated published output")
    deferred = []
    for row in terminal["failed"]:
        if row["key"] in done:
            continue
        # A boundary STOP can race with bookkeeping/preparation, not validation.
        if row["error"] == "Operator STOP present":
            continue
        deferred.append(
            {
                "key": row["key"],
                "reason": "predecessor_failure_requires_diagnosis",
                "failure": row,
            }
        )
    return completed, deferred


def remaining_plan(parent_plan, completed, deferred):
    plan = deepcopy(parent_plan)
    excluded = {r["key"] for r in completed + deferred}
    plan["schema"] = "aliby.corpus-continuation.v1"
    plan["jobs"] = [j for j in plan["jobs"] if j["key"] not in excluded]
    jobs = {j["key"]: j for j in plan["jobs"]}
    families = []
    for family in plan["families"]:
        family["jobs"] = [key for key in family["jobs"] if key in jobs]
        if not family["jobs"]:
            continue
        family["representative"] = min(
            family["jobs"], key=lambda key: (jobs[key]["ntps"], key)
        )
        families.append(family)
    # Start with a short 1z family; do not immediately replay the failed 450-frame site.
    families.sort(
        key=lambda family: (
            jobs[family["representative"]]["declared_planes"]["Brightfield"] != 1,
            jobs[family["representative"]]["ntps"],
            family["id"],
        )
    )
    plan["families"] = families
    plan["predecessor_completed"] = completed
    plan["deferred"] = deferred
    return plan


def prepare_continuation(root, parent):
    trial.verify_seal(parent)
    completed, deferred = reconcile(parent)
    parent_plan = trial.load(parent / "plan.json")
    plan = remaining_plan(parent_plan, completed, deferred)
    receipt = {
        "schema": "aliby.corpus-handoff.v1",
        "parent_root": str(parent),
        "parent_files": {
            name: trial.sha(parent / name)
            for name in (
                "plan.json",
                "sealed-inputs.json",
                "source-identity.json",
                "campaign/terminal.json",
            )
        },
        "completed": completed,
        "deferred": deferred,
        "policy": "Fresh sessions only; prior published full outputs excluded; genuine failures held.",
    }
    trial.save(root / "continuation.json", receipt)
    trial.save(root / "plan.json", plan)
    trial.check_scope(root, require_seal=False)
    files = [
        root / name
        for name in (
            "plan.json",
            "metadata.json",
            "source-identity.json",
            "continuation.json",
            "handoff-config.json",
        )
    ]
    files.extend(sorted((root / "scripts").glob("*.py")))
    files.extend(sorted((root / "evidence").glob("yeast*.json")))
    trial.save(
        root / "sealed-inputs.json",
        {str(p.relative_to(root)): trial.sha(p) for p in files},
    )
    trial.check_scope(root)
    return plan


def main(root):
    config = trial.load(root / "handoff-config.json")
    parent = Path(config["parent_root"])
    watched = Path(config["watched_attempt"])
    if root.resolve() == parent.resolve() or parent not in watched.parents:
        raise ValueError("Invalid handoff paths")
    if not 0 < config["maximum_wait_seconds"] <= 6 * 3600:
        raise ValueError("Unbounded handoff wait")
    if trial.source_identity() != trial.load(root / "source-identity.json"):
        raise ValueError("Fixed runtime changed before handoff")
    with (root / "handoff-owner.lock").open("x") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trial.save(
            root / "handoff-start.json",
            {
                "pid": os.getpid(),
                "parent_root": str(parent),
                "inference_running": False,
            },
        )
        deadline = time.monotonic() + config["maximum_wait_seconds"]
        phase = "waiting_for_predecessor_full_attempt_shutdown"
        process = None

        def status(**extra):
            tmp = root / "handoff-status.tmp"
            tmp.write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "pid": os.getpid(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **extra,
                    },
                    indent=2,
                )
                + "\n"
            )
            tmp.replace(root / "handoff-status.json")

        try:
            while True:
                if (root / "STOP").exists():
                    raise RuntimeError("Operator cancelled pending handoff")
                if (parent / "STOP").exists():
                    owned_stop = root / "evidence/boundary-stop.json"
                    if not owned_stop.exists() or trial.load(owned_stop).get(
                        "stop_sha256"
                    ) != trial.sha(parent / "STOP"):
                        raise RuntimeError(
                            "Predecessor STOP is not owned by this handoff; preserve operator stop"
                        )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Handoff wait expired; no processing launched")
                if (
                    trial.sha(parent / "sealed-inputs.json")
                    != config["parent_seal_sha256"]
                ):
                    raise ValueError("Predecessor seal changed")
                terminal = parent / "campaign/terminal.json"
                if terminal.exists():
                    try:
                        assert_quiescent(parent)
                        break
                    except (RuntimeError, BlockingIOError):
                        pass
                else:
                    shutdown = watched / "shutdown.json"
                    if shutdown.exists():
                        try:
                            receipt = trial.load(shutdown)
                        except json.JSONDecodeError:
                            receipt = {}
                        if (
                            receipt.get("owned_pipeline_stopped") is True
                            and receipt.get("owned_service_stopped") is True
                        ):
                            # No active inference is interrupted. Published outputs
                            # are reconciled even if legacy ledger commit is racing.
                            try:
                                with (parent / "STOP").open("x") as handle:
                                    handle.write(
                                        "Operator-authorized fixed-runtime handoff after full-attempt shutdown. Preserve outputs; do not resume.\n"
                                    )
                                trial.save(
                                    root / "evidence/boundary-stop.json",
                                    {
                                        "watched_attempt": str(watched),
                                        "shutdown_sha256": trial.sha(shutdown),
                                        "stop_sha256": trial.sha(parent / "STOP"),
                                    },
                                )
                            except FileExistsError:
                                pass
                            phase = "waiting_for_predecessor_controller_and_owned_groups_to_exit"
                status()
                time.sleep(2)
            phase = "reconciling_published_outputs"
            status()
            plan = prepare_continuation(root, parent)
            if not plan["jobs"]:
                phase = "nothing_eligible_to_launch"
                status(
                    completed=len(plan["predecessor_completed"]),
                    deferred=plan["deferred"],
                )
                return
            assert_quiescent(parent)
            if trial.source_identity() != trial.load(root / "source-identity.json"):
                raise ValueError("Fixed runtime changed while waiting")
            env = trial.environment(f"{trial.WORKTREE / 'src'}:{trial.ANALYSIS}")
            env["ALIBY_SOURCE_WORKTREE"] = str(trial.WORKTREE)
            with (root / "evidence/driver.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        str(trial.PIPE_PY),
                        str(root / "scripts/corpus_assay_expansion.py"),
                        str(root),
                    ],
                    env=env,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            trial.save(
                root / "evidence/launch.json",
                {
                    "pid": process.pid,
                    "fixed_runtime": True,
                    "eligible_jobs": len(plan["jobs"]),
                    "predecessor_completed": len(plan["predecessor_completed"]),
                    "deferred": plan["deferred"],
                },
            )
            phase = "checking_fixed_controller_startup"
            status(controller_pid=process.pid)
            startup_deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Fixed controller exited {process.returncode}; inspect driver.log, do not duplicate"
                    )
                checkpoint = root / "campaign/status.json"
                if checkpoint.exists():
                    state = trial.load(checkpoint)
                    if (
                        state["pid"] == process.pid
                        and state["status"] == "running"
                        and state["current"]
                    ):
                        break
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError(
                        "Fixed controller startup not confirmed; preserve process and inspect before any retry"
                    )
                time.sleep(1)
            phase = "fixed_runtime_running"
            status(
                controller_pid=process.pid,
                eligible_jobs=len(plan["jobs"]),
                deferred=plan["deferred"],
            )
        except BaseException as error:
            phase = "handoff_blocked"
            status(
                error=str(error),
                error_type=type(error).__name__,
                controller_launch_attempted=process is not None,
                controller_pid=process.pid if process is not None else None,
            )
            raise


if __name__ == "__main__":
    main(Path(sys.argv[1]))
