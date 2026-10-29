"""One-shot bounded expansion after recorded paired real-data diagnostic gates.

Stops on the first failure (stricter than the three-consecutive-failure ceiling).
Never retries, resumes a prior batch, or removes partial/completed output.
"""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def verify_gate(root, trial):
    gate = trial.load(root / "evidence/expansion.json")
    if (
        gate["model_sha256"]
        != trial.load(root / "evidence/loaded-model.json")["sha256"]
    ):
        raise ValueError("Model identity changed since representative diagnostics")
    if gate["source_identity"] != trial.source_identity():
        raise ValueError("Runtime source changed; rerun small parity diagnostics")
    for site, entry in gate["representatives"].items():
        if (
            trial.sha(entry["parity_path"]) != entry["parity_sha256"]
            or trial.sha(entry["panel_path"]) != entry["panel_sha256"]
        ):
            raise ValueError("Diagnostic evidence changed")
        comparison = trial.load(entry["parity_path"])
        if comparison["status"] != "exact_semantic_parity" or comparison["ntps"] != 12:
            raise ValueError("Missing representative twelve-frame parity")
        for key in ("left", "right"):
            attempt = Path(comparison[key])
            if trial.load(attempt / "attempt.json")["site"] != site:
                raise ValueError("Diagnostic site mismatch")
            if (
                trial.load(attempt / "result.json")["source_identity"]
                != gate["source_identity"]
            ):
                raise ValueError("Diagnostic source mismatch")
    if set(gate["representatives"]) != {
        "phluorin_dot6_001",
        "phluorin_nrg1_014",
        "phluorin_tod6_019",
    }:
        raise ValueError("Missing reporter-group diagnostics")
    return gate


def main(root):
    spec = importlib.util.spec_from_file_location(
        "trial", root / "scripts/assay_real_trial_v2.py"
    )
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)
    gate = verify_gate(root, trial)
    trial.reserve(root)
    directory = root / "expansion"
    directory.mkdir(exist_ok=False)
    lock = (directory / "owner.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    deadline = started + 48 * 3600
    state = {
        "status": "running",
        "pid": os.getpid(),
        "completed": [],
        "failed": [],
        "held": list(trial.HELD),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": gate["limits"],
        "stop_on_first_failure": True,
        "biological_approval": False,
    }

    def status():
        state["elapsed_seconds"] = time.monotonic() - started
        temp = directory / "status.tmp"
        temp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temp.replace(directory / "status.json")

    def event(data):
        with (directory / "events.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}
                )
                + "\n"
            )
        status()

    def stage(site, name, args):
        trial.reserve(root)
        if time.monotonic() >= deadline:
            raise TimeoutError("48-hour batch deadline exhausted")
        proc = None
        with (directory / f"{site}-{name}.log").open("xb") as log:
            try:
                proc = subprocess.Popen(
                    [
                        str(trial.PIPE_PY),
                        str(root / "scripts/assay_real_trial_v2.py"),
                        str(root),
                        *args,
                    ],
                    env=trial.environment(f"{trial.WORKTREE / 'src'}:{trial.ANALYSIS}"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                event(
                    {"site": site, "stage": name, "state": "started", "pid": proc.pid}
                )
                # Preparation also bounded; per-run site limits enforced inside launcher.
                limit = min(
                    deadline,
                    time.monotonic() + (4 * 3600 + 240 if name == "full" else 3600),
                )
                while proc.poll() is None:
                    trial.reserve(root)
                    if time.monotonic() >= limit:
                        raise TimeoutError("Stage/batch wall limit exhausted")
                    time.sleep(1)
                if proc.returncode:
                    raise RuntimeError(
                        f"{site} {name} exited {proc.returncode}; inspect expansion log"
                    )
            finally:
                if proc is not None:
                    # launch's SIGTERM handler drains/stops its owned service and worker.
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=40)
                    except subprocess.TimeoutExpired:
                        trial.stop(proc)
        event({"site": site, "stage": name, "state": "passed"})

    def attempt_for(site, ntps):
        candidates = list((root / "attempts").glob(f"{site}-{ntps:03d}-async-*"))
        if len(candidates) != 1:
            raise ValueError("Ambiguous or absent fresh attempt; no implicit resume")
        attempt = candidates[0]
        result = trial.load(attempt / "result.json")
        receipt = trial.load(attempt / "validated.json")
        if result["status"] != "computationally_validated" or not all(
            receipt["checks"].values()
        ):
            raise ValueError("Attempt did not pass computational gates")
        release = Path(result["output"])
        completion = trial.load(release / "completion.json")
        for name, expected in completion["artifacts_sha256"].items():
            if trial.sha(release / name) != expected:
                raise ValueError("Released artifact changed")
        if result["source_identity"] != gate["source_identity"]:
            raise ValueError("Source changed during expansion")
        return attempt, result, receipt

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    current = None
    status()
    try:
        for current in trial.HELD:
            trial.reserve(root)
            state["current"] = current
            status()
            if not (root / "prepared" / current / "prepared.json").exists():
                stage(current, "prepare", ["prepare", "--site", current])
            # Representative twelve-frame trials already passed. Other sites
            # get their own fresh smoke before a new full session is admitted.
            if current not in gate["representatives"]:
                stage(
                    current,
                    "smoke",
                    ["launch", "--site", current, "--ntps", "3", "--backend", "async"],
                )
                attempt_for(current, 3)
            stage(
                current,
                "full",
                ["launch", "--site", current, "--ntps", "180", "--backend", "async"],
            )
            attempt, result, receipt = attempt_for(current, 180)
            subprocess.run(
                [
                    str(trial.PIPE_PY),
                    str(root / "scripts/assay_real_trial_review_v2.py"),
                    str(attempt),
                ],
                env=trial.environment(f"{trial.WORKTREE / 'src'}:{trial.ANALYSIS}"),
                check=True,
                timeout=120,
            )
            state["completed"].append(
                {
                    "site": current,
                    "attempt": str(attempt),
                    "output": result["output"],
                    "summary": receipt["summary"],
                }
            )
            state["held"].remove(current)
            event(
                {
                    "site": current,
                    "state": "full_series_completed",
                    "summary": receipt["summary"],
                }
            )
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "stopped_on_failure"
        state["failed"].append(
            {"site": current, "type": type(error).__name__, "error": str(error)}
        )
        trial.save(
            directory / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "no_retry": True,
            },
        )
        raise
    finally:
        status()
        trial.save(directory / "terminal.json", state)
        lock.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
