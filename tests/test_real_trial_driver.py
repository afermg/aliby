"""Synthetic tests of the bounded, host-specific real-trial example's gates."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def module(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


trial = module("assay_real_trial")
expansion = module("assay_real_expansion")


def test_scope_is_exactly_the_previously_held_nineteen():
    assert len(trial.HELD) == len(set(trial.HELD)) == 19
    assert trial.HELD[0] == "phluorin_dot6_001"
    assert "gcd7_020" not in trial.HELD
    assert "gcd2_002" not in trial.HELD


def test_stop_and_disk_reserve_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trial.shutil, "disk_usage", lambda _: SimpleNamespace(free=299 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="reserve"):
        trial.reserve(tmp_path)
    monkeypatch.setattr(
        trial.shutil, "disk_usage", lambda _: SimpleNamespace(free=301 * 1024**3)
    )
    trial.reserve(tmp_path)
    (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError, match="STOP"):
        trial.reserve(tmp_path)


def test_evidence_write_is_no_replace(tmp_path):
    target = tmp_path / "evidence.json"
    trial.save(target, {"original": True})
    with pytest.raises(FileExistsError):
        trial.save(target, {"original": False})
    assert trial.load(target) == {"original": True}


@pytest.fixture
def gate(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sources = {"module": {"sha256": "frozen"}}
    trial.save(evidence / "loaded-model.json", {"sha256": "model"})
    representatives = {}
    for site in ("phluorin_dot6_001", "phluorin_nrg1_014", "phluorin_tod6_019"):
        path = tmp_path / site
        path.mkdir()
        trial.save(path / "attempt.json", {"site": site})
        trial.save(path / "result.json", {"source_identity": sources})
        report = evidence / f"{site}.json"
        trial.save(
            report,
            {
                "status": "exact_semantic_parity",
                "ntps": 12,
                "left": str(path),
                "right": str(path),
            },
        )
        panel = path / "panel.png"
        panel.write_bytes(b"synthetic image placeholder")
        representatives[site] = {
            "parity_path": str(report),
            "parity_sha256": trial.sha(report),
            "panel_path": str(panel),
            "panel_sha256": trial.sha(panel),
        }
    value = {
        "model_sha256": "model",
        "source_identity": sources,
        "representatives": representatives,
    }
    trial.save(evidence / "expansion.json", value)
    return value, SimpleNamespace(
        load=trial.load, sha=trial.sha, source_identity=lambda: sources
    )


def test_expansion_gate_accepts_unchanged_representative_evidence(tmp_path, gate):
    value, adapter = gate
    assert expansion.verify_gate(tmp_path, adapter) == value


@pytest.mark.parametrize(
    "change",
    ["model", "source", "panel", "parity", "reporter", "site", "diagnostic_source"],
)
def test_expansion_gate_rejects_changed_or_missing_evidence(tmp_path, gate, change):
    value, adapter = gate
    representative = value["representatives"]["phluorin_dot6_001"]
    if change == "model":
        (tmp_path / "evidence/loaded-model.json").write_text('{"sha256":"other"}')
    elif change == "source":
        adapter.source_identity = lambda: {"different": "source"}
    elif change == "panel":
        Path(representative["panel_path"]).write_bytes(b"different")
    elif change == "parity":
        Path(representative["parity_path"]).write_text("{}")
    elif change == "reporter":
        value["representatives"].pop("phluorin_tod6_019")
        (tmp_path / "evidence/expansion.json").write_text(json.dumps(value))
    elif change == "site":
        (tmp_path / "phluorin_dot6_001/attempt.json").write_text('{"site":"other"}')
    else:
        (tmp_path / "phluorin_dot6_001/result.json").write_text(
            '{"source_identity":{}}'
        )
    with pytest.raises(ValueError):
        expansion.verify_gate(tmp_path, adapter)


@pytest.mark.parametrize(
    "change", [None, "pixels", "masks", "profiles", "tracking", "identity"]
)
def test_real_comparison_requires_exact_semantic_equality(tmp_path, change):
    paths = [tmp_path / name for name in ("sequential", "async")]
    for index, path in enumerate(paths):
        path.mkdir()
        output = path / "output"
        (output / "steps/site/segment_cell").mkdir(parents=True)
        (output / "profiles").mkdir()
        (output / "tracking").mkdir()
        trial.save(
            path / "result.json",
            {
                "backend": path.name,
                "ntps": 3,
                "decision_sha256": "decision"
                if change != "identity" or index == 0
                else "different",
                "source_identity": {},
                "output": str(output),
            },
        )
        trial.save(path / "attempt.json", {"site": "site"})
        with (path / "baby-http.jsonl").open("w") as handle:
            for tp in range(3):
                handle.write(
                    json.dumps(
                        {
                            "method": "post",
                            "event": "request_started",
                            "submitted_pixel_sha256": str(tp)
                            if change != "pixels" or index == 0
                            else "different",
                        }
                    )
                    + "\n"
                )
                np.savez(
                    output / "steps/site/segment_cell" / f"{tp:04d}.npz",
                    tile_0=np.array([index if change == "masks" else 1]),
                )
        pd.DataFrame(
            {
                "metadata_tp": [0, 1, 2],
                "metadata_tile": [0, 0, 0],
                "metadata_object": ["cell"] * 3,
                "metadata_label": [1, 1, 1],
                "value": [index if change == "profiles" else 1] * 3,
            }
        ).to_parquet(output / "profiles/site.parquet")
        pd.DataFrame(
            {
                "tp": [0, 1, 2],
                "tile": [0, 0, 0],
                "cell_label": [1, 1, 1],
                "mother_label": [index if change == "tracking" else 0] * 3,
            }
        ).to_parquet(output / "tracking/site_segment_cell.parquet")
    if change is None:
        trial.compare(*paths, tmp_path / "parity.json")
        assert trial.load(tmp_path / "parity.json")["status"] == "exact_semantic_parity"
    else:
        with pytest.raises((ValueError, AssertionError)):
            trial.compare(*paths, tmp_path / "parity.json")
        assert not (tmp_path / "parity.json").exists()
