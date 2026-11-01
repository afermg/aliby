"""Isolated runtime/observational-checkpoint regression tests; never launch jobs."""

import json
from pathlib import Path
import signal
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import corpus_assay_trial as trial
import corpus_assay_expansion as driver

sys.path.pop(0)


def test_source_tree_tracks_previously_unpinned_helper(tmp_path):
    (tmp_path / "roi_selection.py").write_text("version = 1")
    original = trial.python_tree_identity(tmp_path)
    (tmp_path / "roi_selection.py").write_text("version = 2")
    assert original != trial.python_tree_identity(tmp_path)


def test_source_tree_tracks_added_module(tmp_path):
    (tmp_path / "first.py").write_text("x = 1")
    original = trial.python_tree_identity(tmp_path)
    (tmp_path / "new_helper.py").write_text("x = 2")
    assert original != trial.python_tree_identity(tmp_path)


def test_source_tree_rejects_empty_and_linked_inputs(tmp_path):
    with pytest.raises(ValueError, match="Empty"):
        trial.python_tree_identity(tmp_path)
    (tmp_path / "source.txt").write_text("x = 1")
    (tmp_path / "linked.py").symlink_to(tmp_path / "source.txt")
    with pytest.raises(ValueError, match="Unsafe"):
        trial.python_tree_identity(tmp_path)


def test_prepared_checkpoint_pinned_by_parent_not_self(tmp_path, monkeypatch):
    path = tmp_path / "prepared.json"
    trial.save(path, {"key": "one", "inventory_sha256": "initial"})
    monkeypatch.setenv("ALIBY_PREPARED_SHA256", trial.sha(path))
    assert trial.checked_prepared(tmp_path, "one")["inventory_sha256"] == "initial"
    path.write_text(
        json.dumps(
            {"key": "one", "inventory_sha256": "rewritten-to-match-changed-files"}
        )
    )
    with pytest.raises(ValueError, match="checkpoint"):
        trial.checked_prepared(tmp_path, "one")


def test_prepared_checkpoint_requires_parent_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIBY_PREPARED_SHA256", raising=False)
    with pytest.raises(ValueError, match="checkpoint"):
        trial.checked_prepared(tmp_path, "one")


def test_prepared_checkpoint_cannot_change_site(tmp_path, monkeypatch):
    trial.save(tmp_path / "prepared.json", {"key": "other"})
    monkeypatch.setenv("ALIBY_PREPARED_SHA256", trial.sha(tmp_path / "prepared.json"))
    with pytest.raises(ValueError, match="site mismatch"):
        trial.checked_prepared(tmp_path, "one")


def test_seal_requires_coverage(tmp_path):
    trial.save(tmp_path / "sealed-inputs.json", {})
    with pytest.raises(ValueError, match="Incomplete"):
        trial.verify_seal(tmp_path)


def test_complete_seal_rejects_later_change(tmp_path):
    names = [
        "plan.json",
        "metadata.json",
        "source-identity.json",
        "scripts/corpus_assay_trial.py",
        "scripts/corpus_assay_expansion.py",
        "scripts/assay_real_trial.py",
    ]
    names += [f"evidence/{model}.json" for model in trial.MODELS.values()]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial")
    (tmp_path / "plan.json").write_text(
        json.dumps({"schema": "aliby.corpus-expansion.v1"})
    )
    trial.save(
        tmp_path / "sealed-inputs.json",
        {name: trial.sha(tmp_path / name) for name in names},
    )
    trial.verify_seal(tmp_path)
    (tmp_path / "metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        trial.verify_seal(tmp_path)


def test_checkpoint_counts_do_not_double_count_running():
    state = {
        "completed": [{"key": "done"}],
        "failed": [{"key": "failed"}],
        "interrupted": [{"key": "cancelled"}],
        "held": ["active", "pending"],
        "current": {"active": "full"},
    }
    assert driver.checkpoint_counts(state) == {
        "completed": 1,
        "failed": 1,
        "interrupted": 1,
        "running": 1,
        "unstarted": 1,
    }
    state["current"]["done"] = "stale"
    with pytest.raises(ValueError, match="inconsistent"):
        driver.checkpoint_counts(state)


def test_checkpoint_rejects_terminal_overlap():
    with pytest.raises(ValueError, match="Overlapping"):
        driver.checkpoint_counts(
            {
                "completed": [{"key": "same"}],
                "failed": [],
                "interrupted": [],
                "held": ["same"],
                "current": {},
            }
        )


@pytest.mark.parametrize("expired", [False, True])
def test_campaign_closes_admission_before_cleanup_and_handles_sigint(
    tmp_path, monkeypatch, expired
):
    plan = {
        "jobs": [{"key": "one", "id": "one", "ntps": 20}],
        "families": [{"representative": "one"}],
        "limits": {"site_hours": 0 if expired else 4, "batch_hours": 48},
    }
    source = {"core-tree": {"path": str(trial.WORKTREE / "src")}}
    trial.save(tmp_path / "source-identity.json", source)
    trial.save(tmp_path / "plan.json", plan)
    trial.save(tmp_path / "sealed-inputs.json", {})
    monkeypatch.setattr(trial, "check_scope", lambda root: plan)
    monkeypatch.setattr(trial, "verify_seal", lambda root: {})
    monkeypatch.setattr(trial, "source_identity", lambda: source)
    monkeypatch.setattr(trial, "reserve", lambda root: None)
    monkeypatch.setattr(driver, "batches", lambda plan: [["one"]])
    monkeypatch.setattr(driver, "OWNER_LOCK", tmp_path / "global-owner.lock")
    registrations = {}
    monkeypatch.setattr(
        driver.signal, "signal", lambda number, fn: registrations.update({number: fn})
    )
    events = []
    original_event = threading.Event

    def event_factory():
        event = original_event()
        events.append(event)
        return event

    monkeypatch.setattr(driver.threading, "Event", event_factory)
    processes = []

    class FailedProcess:
        pid = 424242
        returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def popen(*args, **kwargs):
        process = FailedProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    cleanup = []

    def killpg(pid, signum):
        assert events[0].is_set(), "Admission remained open during failed-stage cleanup"
        cleanup.append(pid)

    monkeypatch.setattr(driver.os, "killpg", killpg)
    with pytest.raises(SystemExit):
        driver.main(tmp_path)
    terminal = trial.load(tmp_path / "campaign/terminal.json")
    assert terminal["resumable"] is False
    assert terminal["counts"]["failed"] == 1
    assert terminal["counts"]["running"] == 0
    assert terminal["counts"]["unstarted"] == 0
    assert signal.SIGINT in registrations and signal.SIGTERM in registrations
    assert len(processes) == (0 if expired else 1)
    assert cleanup == ([] if expired else [424242])
