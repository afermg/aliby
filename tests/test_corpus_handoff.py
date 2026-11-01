"""Remaining-only handoff and no-overlap checks, without real processing."""

import fcntl
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import corpus_assay_trial as trial
import corpus_assay_handoff as handoff
import corpus_assay_expansion as driver

sys.path.pop(0)


@pytest.fixture
def continuation(tmp_path):
    parent, root = tmp_path / "parent", tmp_path / "new"
    (parent / "campaign").mkdir(parents=True)
    root.mkdir()
    jobs = [
        {
            "key": key,
            "id": key,
            "ntps": 450 if key == "b" else 180,
            "family": key,
            "declared_planes": {"Brightfield": 1 if key == "c" else 5},
        }
        for key in ("a", "b", "c")
    ]
    plan = {
        "schema": "aliby.corpus-expansion.v1",
        "jobs": jobs,
        "families": [
            {"id": j["key"], "representative": j["key"], "jobs": [j["key"]]}
            for j in jobs
        ],
        "limits": {"workers": 4},
    }
    trial.save(parent / "plan.json", plan)
    trial.save(parent / "sealed-inputs.json", {})
    trial.save(parent / "source-identity.json", {})
    terminal = {
        "status": "stopped",
        "current": {},
        "completed": [{"key": "a"}],
        "failed": [{"key": "b", "error": "deadline"}],
    }
    trial.save(parent / "campaign/terminal.json", terminal)
    (parent / "campaign/owner.lock").touch()
    attempt = parent / "jobs/a/attempts/full"
    (attempt / "release").mkdir(parents=True)
    trial.save(attempt / "release/completion.json", {})
    trial.save(attempt / "result.json", {"key": "a", "ntps": 180, "backend": "async"})
    completed = [
        {
            "key": "a",
            "attempt": str(attempt),
            "completion": str(attempt / "release/completion.json"),
            "completion_sha256": trial.sha(attempt / "release/completion.json"),
            "result_sha256": trial.sha(attempt / "result.json"),
        }
    ]
    deferred = [{"key": "b", "reason": "predecessor_failure_requires_diagnosis"}]
    receipt = {
        "parent_root": str(parent),
        "completed": completed,
        "deferred": deferred,
        "parent_files": {
            name: trial.sha(parent / name)
            for name in (
                "plan.json",
                "sealed-inputs.json",
                "source-identity.json",
                "campaign/terminal.json",
            )
        },
    }
    trial.save(root / "continuation.json", receipt)
    return root, parent, plan, receipt


def replace(path, data):
    import json

    path.write_text(json.dumps(data))


def test_remaining_excludes_completed_and_genuinely_failed(continuation):
    root, _, plan, receipt = continuation
    assert trial.continuation_remaining(root, {"a", "b", "c"}) == {"c"}
    pending = handoff.remaining_plan(plan, receipt["completed"], receipt["deferred"])
    assert [j["key"] for j in pending["jobs"]] == ["c"]
    assert driver.batches(pending) == [["c"]]
    assert len(plan["jobs"]) == 3  # predecessor plan is never changed


def test_predecessor_evidence_change_blocks_handoff(continuation):
    root, parent, _, _ = continuation
    (parent / "campaign/terminal.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        trial.continuation_remaining(root, {"a", "b", "c"})


def test_no_silent_omission_of_unattempted_work(continuation):
    root, _, _, receipt = continuation
    receipt["deferred"].append({"key": "c"})
    replace(root / "continuation.json", receipt)
    with pytest.raises(ValueError, match="failure evidence"):
        trial.continuation_remaining(root, {"a", "b", "c"})


def test_ledger_completion_cannot_be_reprocessed(continuation):
    root, _, _, receipt = continuation
    receipt["completed"] = []
    replace(root / "continuation.json", receipt)
    with pytest.raises(ValueError, match="omitted"):
        trial.continuation_remaining(root, {"a", "b", "c"})


def test_scope_cannot_expand(continuation):
    root, _, _, _ = continuation
    with pytest.raises(ValueError, match="scope"):
        trial.continuation_remaining(root, {"a", "b", "c", "extra"})


def test_live_predecessor_prevents_new_owner(continuation, monkeypatch):
    _, parent, _, _ = continuation
    monkeypatch.setattr(handoff, "live_owned_processes", lambda root: [123])
    with pytest.raises(RuntimeError, match="live"):
        handoff.assert_quiescent(parent)


def test_predecessor_owner_lock_prevents_overlap(continuation, monkeypatch):
    _, parent, _, _ = continuation
    monkeypatch.setattr(handoff, "live_owned_processes", lambda root: [])
    with (parent / "campaign/owner.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            handoff.assert_quiescent(parent)
    handoff.assert_quiescent(parent)


def test_published_output_is_adopted_across_ledger_write_race(
    continuation, monkeypatch
):
    _, parent, _, _ = continuation
    attempt = parent / "jobs/a/attempts/full"
    trial.save(attempt / "attempt.json", {"ntps": 180, "backend": "async"})
    terminal = trial.load(parent / "campaign/terminal.json")
    terminal["completed"] = []
    terminal["failed"].append({"key": "a", "error": "Operator STOP present"})
    replace(parent / "campaign/terminal.json", terminal)
    monkeypatch.setattr(handoff, "assert_quiescent", lambda parent: None)
    monkeypatch.setattr(
        trial,
        "verify_attempt",
        lambda attempt: (
            {"key": "a", "ntps": 180, "source_identity": {}},
            {"summary": {}},
        ),
    )
    completed, deferred = handoff.reconcile(parent)
    assert [row["key"] for row in completed] == ["a"]
    assert [row["key"] for row in deferred] == ["b"]


def test_short_single_plane_representative_is_first(continuation):
    _, _, plan, _ = continuation
    result = handoff.remaining_plan(plan, [], [])
    assert result["families"][0]["representative"] == "c"


@pytest.mark.parametrize("shut_down", [False, True])
@pytest.mark.parametrize("operator_stop", [False, True])
def test_watcher_never_stops_running_attempt_or_launches_competing_controller(
    tmp_path, monkeypatch, shut_down, operator_stop
):
    parent, root = tmp_path / "parent", tmp_path / "new"
    watched = parent / "jobs/one/attempts/full"
    watched.mkdir(parents=True)
    root.mkdir()
    (root / "evidence").mkdir()
    trial.save(parent / "sealed-inputs.json", {})
    trial.save(root / "source-identity.json", {})
    trial.save(
        root / "handoff-config.json",
        {
            "parent_root": str(parent),
            "watched_attempt": str(watched),
            "maximum_wait_seconds": 60,
            "parent_seal_sha256": trial.sha(parent / "sealed-inputs.json"),
        },
    )
    if shut_down:
        trial.save(
            watched / "shutdown.json",
            {"owned_pipeline_stopped": True, "owned_service_stopped": True},
        )
    if operator_stop:
        (parent / "STOP").write_text("operator pause; do not override")
    monkeypatch.setattr(trial, "source_identity", lambda: {})

    class StopObservation(Exception):
        pass

    def stop_after_first_check(seconds):
        raise StopObservation("end test observation")

    monkeypatch.setattr(handoff.time, "sleep", stop_after_first_check)

    def forbidden(*args, **kwargs):
        pytest.fail("Competing controller was launched")

    monkeypatch.setattr(handoff.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError if operator_stop else StopObservation):
        handoff.main(root)
    assert (parent / "STOP").exists() is (shut_down or operator_stop)
    if operator_stop:
        assert (parent / "STOP").read_text() == "operator pause; do not override"
    assert (
        trial.load(root / "handoff-status.json")["controller_launch_attempted"] is False
    )
