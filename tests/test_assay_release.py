import asyncio
import hashlib
import json
import threading

import pyarrow.parquet as pq
import pytest

from aliby.assay_release import run_assay_release as _run_assay_release
from aliby.assay_release import run_assay_release_async as _run_assay_release_async
from assay_runtime_support import fixture


def synthetic_preflight(stage, binding):
    assert binding["decision"]["request"]["source_identity"] == "fixture:raw"
    assert not (stage / "profiles").exists()
    return {
        "accepted": True,
        "checks": {"raw_identity": True, "model_identity": True, "roi_selection": True},
        "scope": "synthetic test attestation only; not a real archive/model verification",
    }


def run_assay_release(*args, **kwargs):
    kwargs.setdefault("preflight", synthetic_preflight)
    return _run_assay_release(*args, **kwargs)


async def run_assay_release_async(*args, **kwargs):
    kwargs.setdefault("preflight", synthetic_preflight)
    return await _run_assay_release_async(*args, **kwargs)


def test_missing_or_rejected_preflight_never_initializes_models(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, save=True)
    with pytest.raises(ValueError, match="preflight"):
        _run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=validator
        )
    assert not attempts(tmp_path)

    def reject(stage, binding):
        assert not f.log["constructed"]
        return {"accepted": False, "checks": {"raw_identity": False}}

    with pytest.raises(ValueError, match="Preflight"):
        run_assay_release(
            f.bound,
            "fixture",
            tmp_path / "released",
            preflight=reject,
            validator=validator,
        )
    assert not f.log["constructed"] and not (tmp_path / "released").exists()
    (stage,) = attempts(tmp_path)
    assert json.loads((stage / "preflight.json").read_text())["accepted"] is False


def validator(stage, binding):
    table = pq.read_table(stage / "profiles/fixture.parquet")
    assert len(table) == 4
    assert len(list((stage / "steps/fixture/segment_cell").glob("*.npz"))) == 2
    keys = list(
        zip(
            *(
                table[c].to_pylist()
                for c in (
                    "metadata_tp",
                    "metadata_tile",
                    "metadata_label",
                    "metadata_object",
                )
            ),
            strict=True,
        )
    )
    return {
        "accepted": True,
        "checks": {
            "model_identity": binding["models"]["segment_cell"]["identity"].startswith(
                "sha256:"
            ),
            "input_layout": binding["inputs"]["segment_cell"][0]["z"] == (0,),
            "semantic_keys": len(set(keys)) == len(table),
        },
        "scope": "synthetic fixtures only; model stub, real cp_measure",
    }


def attempts(tmp_path):
    return list(tmp_path.glob(".released.attempt-*"))


def test_success_is_validated_then_atomically_published_with_hashes(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)
    destination = tmp_path / "released"

    def validate(stage, binding):
        assert not destination.exists()
        assert not (stage / "completion.json").exists()
        return validator(stage, binding)

    actual = run_assay_release(
        f.bound, "fixture", destination, validator=validate, max_workers=3
    )
    assert actual == destination
    assert not attempts(tmp_path)
    manifest = json.loads((destination / "completion.json").read_text())
    assert manifest["biological_approval"] is False
    assert manifest["pipeline_sha256"] == f.bound["assay_binding"]["pipeline_sha256"]
    for name, expected in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    "receipt",
    [
        False,
        {},
        {"accepted": True},
        {"accepted": True, "checks": {"toy": True}},
        {"accepted": True, "checks": {"model_identity": False}},
        {"accepted": True, "checks": {"model_identity": 1}},
    ],
)
def test_failed_or_incomplete_validation_never_publishes(
    tmp_path, monkeypatch, receipt
):
    f = fixture(tmp_path, monkeypatch, save=True)
    with pytest.raises(ValueError, match="validator"):
        run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=lambda *a: receipt
        )
    assert not (tmp_path / "released").exists()
    (stage,) = attempts(tmp_path)
    assert (stage / "failure.json").exists()
    assert not (stage / "completion.json").exists()
    assert (stage / "profiles/fixture.parquet").exists()


def test_model_failure_preserves_failed_attempt_without_automatic_retry(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)
    f.weights.write_bytes(b"wrong weights")
    from aliby.executor import PipelineStepError

    with pytest.raises(PipelineStepError, match="identity"):
        run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=validator
        )
    (stage,) = attempts(tmp_path)
    assert (
        json.loads((stage / "failure.json").read_text())["restart"]
        == "new_attempt_from_frame_zero"
    )
    assert not (tmp_path / "released").exists()
    assert not f.log["images"]


@pytest.mark.parametrize("existing", ["directory", "symlink"])
def test_existing_destination_is_never_overwritten_or_initialized(
    tmp_path, monkeypatch, existing
):
    f = fixture(tmp_path, monkeypatch, save=True)
    target = tmp_path / "released"
    if existing == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        run_assay_release(f.bound, "fixture", target, validator=validator)
    assert not f.log["constructed"] and not attempts(tmp_path)
    assert target.is_symlink() if existing == "symlink" else target.is_dir()


def test_destination_race_cannot_replace_another_directory(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, save=True)
    target = tmp_path / "released"

    def race(stage, binding):
        result = validator(stage, binding)
        target.mkdir()
        (target / "owner.txt").write_text("other owner")
        return result

    with pytest.raises(FileExistsError):
        run_assay_release(f.bound, "fixture", target, validator=race)
    assert (target / "owner.txt").read_text() == "other owner"
    assert not (target / "completion.json").exists()
    (stage,) = attempts(tmp_path)
    assert (stage / "failure.json").exists() and not (
        stage / "completion.json"
    ).exists()


def test_nonregular_artifacts_are_rejected(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, save=True)

    def validate(stage, binding):
        result = validator(stage, binding)
        (stage / "outside").symlink_to(f.weights)
        return result

    with pytest.raises(ValueError, match="Nonregular"):
        run_assay_release(f.bound, "fixture", tmp_path / "released", validator=validate)
    assert not (tmp_path / "released").exists()


def test_cancelled_blocking_validator_is_drained_before_failure_is_recorded(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)
    started, release, ended = threading.Event(), threading.Event(), threading.Event()

    def blocking(stage, binding):
        started.set()
        assert release.wait(5)
        (stage / "late-validator-output.txt").write_text(
            "retained before terminal failure"
        )
        ended.set()
        return validator(stage, binding)

    async def check():
        task = asyncio.create_task(
            run_assay_release_async(
                f.bound, "fixture", tmp_path / "released", validator=blocking
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert ended.is_set()

    asyncio.run(check())
    assert not (tmp_path / "released").exists()
    (stage,) = attempts(tmp_path)
    assert (stage / "late-validator-output.txt").exists()
    assert (
        json.loads((stage / "failure.json").read_text())["error_type"]
        == "CancelledError"
    )


def test_pending_cancellation_after_preflight_cannot_initialize_models(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)

    async def cancel(stage, binding):
        asyncio.current_task().cancel()
        return synthetic_preflight(stage, binding)

    async def check():
        task = asyncio.create_task(
            run_assay_release_async(
                f.bound,
                "fixture",
                tmp_path / "released",
                preflight=cancel,
                validator=validator,
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    assert not f.log["constructed"] and not (tmp_path / "released").exists()


def test_async_validator_is_awaited(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, save=True)

    async def validate(stage, binding):
        return validator(stage, binding)

    assert asyncio.run(
        run_assay_release_async(
            f.bound, "fixture", tmp_path / "released", validator=validate
        )
    ).exists()


def test_baby_release_cannot_bypass_existing_provenance_safeguards(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, kind="nahual_baby", save=True)
    with pytest.raises(ValueError, match="provenance"):
        run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=validator
        )
    assert not f.log["constructed"] and not attempts(tmp_path)


@pytest.mark.parametrize("mutation", ["modify", "extra", "replace"])
def test_artifact_changes_after_hashing_block_publication(
    tmp_path, monkeypatch, mutation
):
    f = fixture(tmp_path, monkeypatch, save=True)
    import aliby.assay_release as release

    original = release._inventory

    def inventory(stage):
        result = original(stage)
        target = stage / "profiles/fixture.parquet"
        if mutation == "extra":
            (stage / "unexpected").write_text("extra")
        elif mutation == "replace":
            data = target.read_bytes()
            target.unlink()
            target.write_bytes(data)
        else:
            with target.open("ab") as handle:
                handle.write(b"changed")
        return result

    monkeypatch.setattr(release, "_inventory", inventory)
    with pytest.raises(RuntimeError, match="changed after"):
        run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=validator
        )
    assert not (tmp_path / "released").exists()
    (stage,) = attempts(tmp_path)
    assert (stage / "failure.json").exists() and not (
        stage / "completion.json"
    ).exists()


def test_full_baby_release_retains_actual_roi_tracking_guard(tmp_path, monkeypatch):
    import numpy as np
    from aliby.assay_runtime import bind_assay_pipeline
    from aliby.tile.tiles import TileLocations
    import aliby.pipe_baby as baby

    f = fixture(tmp_path, monkeypatch, kind="nahual_baby", save=True)
    p = f.pipeline
    p["publish_roi_provenance"] = True
    p["run_binding"] = {"source": "synthetic-fixture-only"}
    bound = bind_assay_pipeline(
        p, f.decision, model_identities={f.model.model_id: f.model.identity}
    )
    original = baby.init_step

    def initialize(name, parameters, others=None):
        result = original(name, parameters, others)
        if name == "tile":
            result.tile_locs = TileLocations(
                [[10, 10], [20, 20]],
                tile_size=(9, 9),
                max_size=(32, 32),
                drifts=[[0.0, 0.0], [0.0, 0.0]],
                roi_source="detected",
            )
            result.pixels = np.broadcast_to(
                f.pixels[:, 0, :, :, :1, :1], (2, 3, 5, 32, 32)
            )
            result.ref_channel, result.ref_z = 2, 0
            result.track_drift, result.fallback_to_center = True, False
            run = result.run_tp

            def run_tp(tp):
                output = run(tp)
                result.no_processed = tp + 1
                return output

            result.run_tp = run_tp
        return result

    monkeypatch.setattr(baby, "init_step", initialize)

    def validate(stage, binding):
        receipt = validator(stage, binding)
        tracking = pq.read_table(stage / "tracking/fixture_segment_cell.parquet")
        with np.load(stage / "tiling/fixture/state.npz") as state:
            geometry = state["crop_origins_yx"]
        receipt["checks"].update(
            {
                "transport": len(f.log["images"]) == 2,
                "roi_geometry": geometry.shape == (2, 2, 2),
                "repeated_masks": True,
            }
        )
        assert tracking["cell_label"].to_pylist() == [17, 17, 18, 18]
        return receipt

    destination = run_assay_release(
        bound, "fixture", tmp_path / "released", validator=validate
    )
    assert (destination / "completion.json").exists()
    assert (destination / "tiling/fixture/manifest.json").exists()


def test_rename_not_available_fails_closed_with_evidence(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, save=True)
    import aliby.io.roi_provenance as roi

    monkeypatch.setattr(roi, "_get_renameat2", lambda: None)
    with pytest.raises(RuntimeError, match="unsafe publication"):
        run_assay_release(
            f.bound, "fixture", tmp_path / "released", validator=validator
        )
    assert not (tmp_path / "released").exists()
    (stage,) = attempts(tmp_path)
    assert (stage / "failure.json").exists() and not (
        stage / "completion.json"
    ).exists()
