"""Additional integrity regressions using model stubs, not real inference."""

import asyncio
from copy import deepcopy
import json
from dataclasses import replace

import numpy as np
import pytest

from aliby.assay import ChannelInput, MeasurementRequest, decide_assay
from aliby.assay_runtime import bind_assay_pipeline, binding_fingerprint
from aliby.assay_release import run_assay_release
from aliby.pipe_core import get_profiles_from_state, run_pipeline_return_state
from assay_runtime_support import fixture
from test_assay_release import synthetic_preflight, validator


@pytest.mark.parametrize("keep", [0, 1])
def test_profile_history_cannot_be_truncated_and_renumbered(
    tmp_path, monkeypatch, keep
):
    f = fixture(tmp_path, monkeypatch, ntps=2)
    f.pipeline["retain"] = {"extract_cell": keep}
    with pytest.raises(ValueError, match="retain|history"):
        bind_assay_pipeline(
            f.pipeline,
            f.decision,
            model_identities={f.model.model_id: f.model.identity},
        )
    assert not f.log["constructed"]


@pytest.mark.parametrize("keep", ["all", 2, 3])
def test_complete_feature_history_and_bounded_pixel_history_are_allowed(
    tmp_path, monkeypatch, keep
):
    f = fixture(tmp_path, monkeypatch, ntps=2)
    f.pipeline["retain"] = {"extract_cell": keep, "segment_cell": 1, "tile": 1}
    bound = bind_assay_pipeline(
        f.pipeline, f.decision, model_identities={f.model.model_id: f.model.identity}
    )
    state = run_pipeline_return_state(bound, None, f.init, backend="async")
    assert len(state["data"]["segment_cell"]) == 1
    profiles = get_profiles_from_state(state, bound)
    assert sorted(profiles["metadata_tp"].to_pylist()) == [0, 0, 1, 1]


@pytest.mark.parametrize("backend", ["sequential", "concurrent", "async"])
def test_older_fingerprinted_truncating_binding_cannot_execute(
    tmp_path, monkeypatch, backend
):
    f = fixture(tmp_path, monkeypatch)
    # Reproduce an admissible v1 binding created before the retention guard.
    f.bound["retain"] = {"extract_cell": 1}
    f.bound["assay_binding"]["pipeline_sha256"] = binding_fingerprint(f.bound)
    with pytest.raises(ValueError, match="retain|history"):
        run_pipeline_return_state(f.bound, None, f.init, backend=backend)
    assert not f.log["constructed"]


@pytest.mark.parametrize("backend", ["sequential", "concurrent", "async"])
@pytest.mark.parametrize("coloc_first", [False, True])
def test_optional_colocalization_preserves_other_objects(
    tmp_path, monkeypatch, backend, coloc_first
):
    f = fixture(tmp_path, monkeypatch)
    pattern = np.arange(81, dtype=np.uint16).reshape(9, 9)
    f.pixels[:, :, 0] += pattern
    f.pixels[:, :, 1, 0] += 2 * pattern
    red = ChannelInput(0, tuple(range(5)), "max")
    green = ChannelInput(1, (0,))
    request = replace(
        f.decision.request,
        segmentations=(
            *f.decision.request.segmentations,
            replace(f.decision.request.segmentations[0], object_name="other"),
        ),
        measurements=(
            *f.decision.request.measurements,
            MeasurementRequest("other-green", "other", (green,)),
            MeasurementRequest("cell-coloc", "cell", (red, green), ("pearson",)),
        ),
    )
    decision = decide_assay(request, (f.model,))
    f.pipeline["steps"]["segment_other"] = deepcopy(f.pipeline["steps"]["segment_cell"])
    if coloc_first:
        f.pipeline["steps"] = {"extractmulti_cell": {}, **f.pipeline["steps"]}
    bound = bind_assay_pipeline(
        f.pipeline, decision, model_identities={f.model.model_id: f.model.identity}
    )
    state = run_pipeline_return_state(
        bound, None, f.init, backend=backend, max_workers=4
    )
    profiles = get_profiles_from_state(state, bound)
    assert len(profiles) == 8, (
        "Optional features must not silently drop another object's observations"
    )
    assert set(profiles["metadata_object"].to_pylist()) == {"cell", "other"}
    col = next(n for n in profiles.column_names if "|" in n and "pearson" in n)
    assert profiles[col].null_count == 4
    keys = list(
        zip(
            *(
                profiles[c].to_pylist()
                for c in (
                    "metadata_tp",
                    "metadata_tile",
                    "metadata_object",
                    "metadata_label",
                )
            ),
            strict=True,
        )
    )
    assert len(set(keys)) == 8


@pytest.mark.parametrize("mutation", ["modify", "remove", "replace"])
def test_validator_cannot_rewrite_already_produced_artifacts(
    tmp_path, monkeypatch, mutation
):
    f = fixture(tmp_path, monkeypatch, save=True)
    target = tmp_path / "release"

    def mutate(stage, binding):
        receipt = validator(stage, binding)
        if mutation == "remove":
            (stage / "steps/fixture/segment_cell/0001.npz").unlink()
        else:
            profile = stage / "profiles/fixture.parquet"
            data = profile.read_bytes()
            if mutation == "replace":
                profile.unlink()
            profile.write_bytes(
                data + (b"invalid extra data" if mutation == "modify" else b"")
            )
        return receipt

    with pytest.raises(RuntimeError, match="changed|modified|validation"):
        run_assay_release(
            f.bound, "fixture", target, preflight=synthetic_preflight, validator=mutate
        )
    assert not target.exists()
    (stage,) = tmp_path.glob(".release.attempt-*")
    assert (stage / "failure.json").exists()
    assert not (stage / "completion.json").exists()


def test_validator_can_add_separate_evidence_without_rewriting_outputs(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)

    def validate(stage, binding):
        receipt = validator(stage, binding)
        reports = stage / "validation-artifacts"
        reports.mkdir()
        (reports / "review.txt").write_text("synthetic computational evidence only")
        return receipt

    target = run_assay_release(
        f.bound,
        "fixture",
        tmp_path / "release",
        preflight=synthetic_preflight,
        validator=validate,
    )
    manifest = json.loads((target / "completion.json").read_text())
    assert "validation-artifacts/review.txt" in manifest["artifacts_sha256"]
    assert manifest["biological_approval"] is False


def test_changes_after_validation_but_before_hashing_also_block_publication(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, save=True)
    import aliby.assay_release as release

    original = release._inventory

    def inventory(stage):
        # This change would be incorporated into the final inventory itself.
        # The pre-validation snapshot must still reject it.
        with (stage / "profiles/fixture.parquet").open("ab") as handle:
            handle.write(b"late change")
        return original(stage)

    monkeypatch.setattr(release, "_inventory", inventory)
    with pytest.raises(RuntimeError, match="changed during validation"):
        run_assay_release(
            f.bound,
            "fixture",
            tmp_path / "release",
            preflight=synthetic_preflight,
            validator=validator,
        )
    assert not (tmp_path / "release").exists()
    (stage,) = tmp_path.glob(".release.attempt-*")
    assert (stage / "failure.json").exists()
    assert not (stage / "completion.json").exists()


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("recording_error", ["collision", "io"])
def test_diagnostic_recording_error_does_not_hide_original_failure(
    tmp_path, monkeypatch, error_type, recording_error
):
    f = fixture(tmp_path, monkeypatch, save=True)
    import aliby.assay_release as release

    original = release._write_json

    def write(path, value):
        if path.name == "failure.json" and recording_error == "io":
            raise OSError("simulated evidence IO failure")
        return original(path, value)

    monkeypatch.setattr(release, "_write_json", write)

    def fail(stage, binding):
        if recording_error == "collision":
            (stage / "failure.json").write_text("reserved-name collision")
        raise error_type("original validation failure")

    with pytest.raises(error_type, match="original validation failure") as caught:
        run_assay_release(
            f.bound,
            "fixture",
            tmp_path / "release",
            preflight=synthetic_preflight,
            validator=fail,
        )
    assert any(
        "failure evidence" in note for note in getattr(caught.value, "__notes__", ())
    )
    assert not (tmp_path / "release").exists()
    (stage,) = tmp_path.glob(".release.attempt-*")
    assert not (stage / "completion.json").exists()
    if recording_error == "collision":
        assert (stage / "failure.json").read_text() == "reserved-name collision"
