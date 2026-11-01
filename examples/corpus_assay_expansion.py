"""One-shot campaign, fixed disjoint batches, fresh stateful sessions, first-failure stop.

Four-hour total site budgets and 48-hour per-batch ceilings; <=4 sites in flight.
Human/biological review is explicitly deferred, never fabricated as a release gate.
"""

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import corpus_assay_trial as trial

OWNER_LOCK = Path("/work/users/amunoz/aliby_baby/analyses/.corpus-assay-owner.lock")


class AdmissionStopped(RuntimeError):
    pass


def batches(plan):
    jobs = {j["key"]: j for j in plan["jobs"]}
    representatives = [f["representative"] for f in plan["families"]]
    members = [key for f in plan["families"] for key in f["jobs"]]
    if (
        not jobs
        or len(jobs) != len(plan["jobs"])
        or len(members) != len(jobs)
        or set(members) != set(jobs)
        or len(set(representatives)) != len(representatives)
    ):
        raise ValueError("Family membership does not partition authorized scope")
    for family in plan["families"]:
        if family["representative"] not in family["jobs"] or any(
            jobs[k]["family"] != family["id"] for k in family["jobs"]
        ):
            raise ValueError("Family/representative mismatch")
    # First site is serial. Only after its entire staged workflow passes may
    # representative cohorts execute concurrently, then nonrepresentatives.
    groups = [representatives[:1]]
    for part in (
        representatives[1:],
        [j["key"] for j in plan["jobs"] if j["key"] not in representatives],
    ):
        groups.extend(
            [
                part[i : i + plan["limits"]["workers"]]
                for i in range(0, len(part), plan["limits"]["workers"])
            ]
        )
    return groups


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Four-hour site/48-hour batch wall limit exhausted")


def checkpoint_counts(state):
    completed = {v["key"] for v in state["completed"]}
    failed = {v["key"] for v in state["failed"]}
    interrupted = {v["key"] for v in state["interrupted"]}
    pending = set(state["held"])
    current = set(state["current"])
    groups = (completed, failed, interrupted, pending)
    if (
        any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
        or not current <= pending
    ):
        raise ValueError("Overlapping or inconsistent checkpoint states")
    return {
        "completed": len(completed),
        "running": len(current),
        "failed": len(failed),
        "interrupted": len(interrupted),
        "unstarted": len(pending - current),
    }


def main(root):
    plan = trial.check_scope(root)
    trial.reserve(root)
    # New controllers share a campaign-independent ownership lock. The legacy
    # controller predates it, so continuations also require OS-level quiescence.
    owner = os.fdopen(
        os.open(OWNER_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+"
    )
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if plan.get("schema") == "aliby.corpus-continuation.v1":
        from corpus_assay_handoff import assert_quiescent

        assert_quiescent(Path(trial.load(root / "continuation.json")["parent_root"]))
    source = trial.source_identity()
    if source != trial.load(root / "source-identity.json"):
        raise ValueError("Source changed before launch")
    if (
        Path(source["core-tree"]["path"]).resolve()
        != (trial.WORKTREE / "src").resolve()
    ):
        raise ValueError("Loaded runtime differs from configured worker source")
    sealed_checksum = trial.sha(root / "sealed-inputs.json")
    sealed = trial.verify_seal(root)
    for name, expected in sealed.items():
        if trial.sha(root / name) != expected:
            raise ValueError("Sealed input changed: " + name)
    directory = root / "campaign"
    directory.mkdir(exist_ok=False)
    lock = (directory / "owner.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    cancel = threading.Event()
    mutex = threading.RLock()
    jobs = {j["key"]: j for j in plan["jobs"]}
    representatives = {f["representative"] for f in plan["families"]}
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": [],
        "failed": [],
        "interrupted": [],
        "held": list(jobs),
        "current": {},
        "family_gates": {},
        "prepared_checkpoints": {},
        "predecessor_completed": plan.get("predecessor_completed", []),
        "deferred": plan.get("deferred", []),
        "resumable": False,
        "checkpoint_semantics": "Observational progress only; never a resumable tracker/session checkpoint.",
        "limits": plan["limits"],
        "biological_approval": False,
        "review_policy": "Human/biological review deferred until computational completion; not an execution gate.",
        "source_identity_sha256": trial.sha(root / "source-identity.json"),
        "plan_sha256": trial.sha(root / "plan.json"),
        "batches": batches(plan),
    }

    last_checkpoint = started

    def status():
        nonlocal last_checkpoint
        with mutex:
            last_checkpoint = time.monotonic()
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["counts"] = checkpoint_counts(state)
            state["elapsed_seconds"] = last_checkpoint - started
            temporary = directory / "status.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(directory / "status.json")

    def event(data):
        with mutex:
            with (directory / "events.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            **data,
                        }
                    )
                    + "\n"
                )
            status()

    def check():
        if cancel.is_set():
            raise AdmissionStopped(
                "Campaign admission stopped after another failure or cancellation"
            )
        trial.reserve(root)
        if trial.sha(root / "sealed-inputs.json") != sealed_checksum:
            raise ValueError("Execution seal changed during campaign")
        with mutex:
            if time.monotonic() - last_checkpoint >= 15:
                status()
        for name, expected in sealed.items():
            if trial.sha(root / name) != expected:
                raise ValueError("Sealed campaign input changed: " + name)

    def stage(key, name, args, deadline):
        check()
        check_deadline(deadline)
        env = trial.environment(f"{trial.WORKTREE / 'src'}:{trial.ANALYSIS}")
        env["ALIBY_SOURCE_WORKTREE"] = str(trial.WORKTREE)
        # Bind child attempts to the parent's observed preparation, not a
        # mutable receipt that can endorse its own replacement.
        with mutex:
            prepared_digest = state["prepared_checkpoints"].get(key)
        if prepared_digest is not None:
            env["ALIBY_PREPARED_SHA256"] = prepared_digest
        else:
            env.pop("ALIBY_PREPARED_SHA256", None)
        proc = None
        with mutex:
            state["current"][key] = name
        with (directory / f"{jobs[key]['id']}-{name}.log").open("xb") as log:
            try:
                proc = subprocess.Popen(
                    [
                        str(trial.PIPE_PY),
                        str(root / "scripts/corpus_assay_trial.py"),
                        str(root),
                        *args,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                event({"key": key, "stage": name, "state": "started", "pid": proc.pid})
                while proc.poll() is None:
                    check()
                    check_deadline(deadline)
                    time.sleep(1)
                if proc.returncode:
                    raise RuntimeError(
                        f"{key} {name} exited {proc.returncode}; inspect preserved stage log"
                    )
                check_deadline(deadline)
            except BaseException:
                # Close admission before cleanup; peers must not start another
                # stage during the potentially 45-second shutdown wait.
                cancel.set()
                raise
            finally:
                if proc is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        trial.common.stop(proc)
        event({"key": key, "stage": name, "state": "passed"})

    def attempt_for(key, n, backend):
        matches = list(
            (root / "jobs" / jobs[key]["id"] / "attempts").glob(f"{n:04d}-{backend}-*")
        )
        if len(matches) != 1:
            raise ValueError(
                "Absent/ambiguous fresh attempt; no retries or implicit resume"
            )
        path = matches[0]
        result, receipt = trial.verify_attempt(path)
        if (
            result["key"] != key
            or result["ntps"] != n
            or result["backend"] != backend
            or result["source_identity"] != trial.load(root / "source-identity.json")
        ):
            raise ValueError("Attempt identity mismatch")
        return path, result, receipt

    def process(key, batch_deadline):
        deadline = min(
            batch_deadline, time.monotonic() + plan["limits"]["site_hours"] * 3600
        )
        try:
            stage(key, "prepare", ["prepare", "--key", key], deadline)
            prepared = root / "jobs" / jobs[key]["id"] / "prepared"
            checksum = trial.evidence_sha(prepared, "prepared.json")
            if trial.load(prepared / "prepared.json").get("key") != key:
                raise ValueError("Prepared checkpoint site mismatch")
            trial.save(
                directory / f"{jobs[key]['id']}-prepared-checkpoint.json",
                {"key": key, "sha256": checksum, "resumable": False},
            )
            with mutex:
                state["prepared_checkpoints"][key] = checksum
            if key in representatives:
                for n in (3, 12):
                    pair = []
                    for backend in ("sequential", "async"):
                        stage(
                            key,
                            f"{n}-{backend}",
                            [
                                "launch",
                                "--key",
                                key,
                                "--ntps",
                                str(n),
                                "--backend",
                                backend,
                            ],
                            deadline,
                        )
                        pair.append(attempt_for(key, n, backend)[0])
                    report = directory / f"{jobs[key]['id']}-parity-{n}.json"
                    stage(
                        key,
                        f"compare-{n}",
                        [
                            "compare",
                            "--left",
                            str(pair[0]),
                            "--right",
                            str(pair[1]),
                            "--report",
                            str(report),
                        ],
                        deadline,
                    )
                    comparison = trial.load(report)
                    if (
                        comparison["status"] != "exact_semantic_parity"
                        or comparison["ntps"] != n
                    ):
                        raise ValueError("Representative parity gate missing")
            else:
                family = next(f for f in plan["families"] if key in f["jobs"])
                rep = family["representative"]
                # A representative must have a completed full run as well as
                # freshly reverified parity evidence before family expansion.
                with mutex:
                    completed = {r["key"] for r in state["completed"]}
                if rep not in completed:
                    raise ValueError("Family representative has not completed")
                gate = state["family_gates"][family["id"]]
                if trial.sha(gate["path"]) != gate["sha256"]:
                    raise ValueError("Family gate receipt changed")
                trial.verify_files(root, trial.load(gate["path"]))
                for n in (3, 12):
                    for backend in ("sequential", "async"):
                        attempt_for(rep, n, backend)
                    report = trial.load(
                        directory / f"{jobs[rep]['id']}-parity-{n}.json"
                    )
                    if report["status"] != "exact_semantic_parity":
                        raise ValueError("Family parity invalid")
                attempt_for(rep, jobs[rep]["ntps"], "async")
                stage(
                    key,
                    "3-async",
                    ["launch", "--key", key, "--ntps", "3", "--backend", "async"],
                    deadline,
                )
                attempt_for(key, 3, "async")
            check()
            stage(
                key,
                "full",
                [
                    "launch",
                    "--key",
                    key,
                    "--ntps",
                    str(jobs[key]["ntps"]),
                    "--backend",
                    "async",
                ],
                deadline,
            )
            attempt, result, receipt = attempt_for(key, jobs[key]["ntps"], "async")
            completion = Path(result["output"]) / "completion.json"
            if key in representatives:
                inventory = {}
                for path in (root / "jobs" / jobs[key]["id"] / "attempts").iterdir():
                    for name, checksum in trial.snapshot_files(path).items():
                        inventory[str((path / name).relative_to(root))] = checksum
                for n in (3, 12):
                    report = directory / f"{jobs[key]['id']}-parity-{n}.json"
                    inventory[str(report.relative_to(root))] = trial.sha(report)
                gate_path = directory / f"family-{jobs[key]['family']}-gate.json"
                trial.save(gate_path, inventory)
                with mutex:
                    state["family_gates"][jobs[key]["family"]] = {
                        "path": str(gate_path),
                        "sha256": trial.sha(gate_path),
                    }
            with mutex:
                state["completed"].append(
                    {
                        "key": key,
                        "attempt": str(attempt),
                        "ntps": jobs[key]["ntps"],
                        "completion": str(completion),
                        "completion_sha256": trial.sha(completion),
                        "summary": receipt["summary"],
                        "biological_approval": False,
                    }
                )
                state["held"].remove(key)
                state["current"].pop(key, None)
            event({"key": key, "stage": "site", "state": "completed"})
        except BaseException as error:
            cancel.set()
            with mutex:
                category = (
                    "interrupted" if isinstance(error, AdmissionStopped) else "failed"
                )
                state[category].append(
                    {
                        "key": key,
                        "stage": state["current"].get(key),
                        "error": str(error),
                        "type": type(error).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
                if key in state["held"]:
                    state["held"].remove(key)
                state["current"].pop(key, None)
            event({"key": key, "state": "stopped", "error": str(error)})
            raise

    def terminate(signum, frame):
        cancel.set()
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    status()
    try:
        for index, group in enumerate(state["batches"]):
            check()
            batch_deadline = time.monotonic() + plan["limits"]["batch_hours"] * 3600
            state["batch_index"] = index
            event({"batch": index, "keys": group, "state": "started"})
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
                futures = [pool.submit(process, key, batch_deadline) for key in group]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            event({"batch": index, "state": "completed"})
        state["status"] = "completed"
    except BaseException as error:
        cancel.set()
        state["status"] = "stopped"
        state["stop_reason"] = str(error)
    finally:
        status()
        trial.save(directory / "terminal.json", state)
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
