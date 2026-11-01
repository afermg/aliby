"""Failure recording must not lose primary errors or skip another owned cleanup."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import corpus_assay_trial as trial

sys.path.pop(0)


def test_failed_evidence_write_preserves_primary(monkeypatch, tmp_path):
    primary = RuntimeError("original pipeline failure")

    def broken(*args):
        raise OSError("full disk")

    monkeypatch.setattr(trial, "save", broken)
    trial.save_failure(primary, tmp_path / "failure.json", {})
    assert str(primary) == "original pipeline failure"
    assert "full disk" in primary.__notes__[0]


def test_worker_cleanup_error_does_not_skip_service_or_mask_primary(
    monkeypatch, tmp_path
):
    class Process:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    class Socket:
        closed = False

        def close(self):
            self.closed = True

    worker, service, sock = Process(None), Process(0), Socket()
    calls = []

    def stop(process):
        calls.append(process)
        if process is worker:
            raise TimeoutError("worker did not stop")

    monkeypatch.setattr(trial.common, "stop", stop)
    primary = RuntimeError("original pipeline failure")
    trial.cleanup_owned(worker, service, sock, tmp_path, primary=primary)
    assert calls == [worker, service]
    assert sock.closed
    assert "worker did not stop" in primary.__notes__[0]
    assert trial.load(tmp_path / "shutdown.json") == {
        "owned_pipeline_stopped": False,
        "owned_service_stopped": True,
    }


def test_missing_shutdown_evidence_is_not_success(monkeypatch, tmp_path):
    class Socket:
        def close(self):
            pass

    def broken(*args):
        raise OSError("cannot save shutdown")

    monkeypatch.setattr(trial, "save", broken)
    with pytest.raises(OSError, match="cannot save shutdown"):
        trial.cleanup_owned(None, None, Socket(), tmp_path)
