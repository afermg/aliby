import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from aliby.io.roi_provenance import (
    NPZ_KEYS,
    collect_live_roi_state,
    validate_provenance_preflight,
    write_roi_provenance,
)
from aliby.pipe_baby import _save_baby_tracking_lineage, run_pipeline_and_post
from aliby.tile.tiles import TileLocations


def make_live_tiler(*, drifts=None, centres=None, roi_source="detected"):
    if centres is None:
        centres = [[20, 30], [40, 50]]
    if drifts is None:
        drifts = [[0.0, 0.0], [1.2, -2.7], [-0.8, 1.1]]
    locations = TileLocations(
        centres,
        tile_size=(6, 10),
        max_size=(80, 120),
        drifts=drifts,
        roi_source=roi_source,
    )
    return SimpleNamespace(
        tile_locs=locations,
        no_processed=3,
        pixels=np.zeros((3, 3, 1, 80, 120), dtype=np.uint8),
        ref_channel=2,
        ref_z=0,
        track_drift=True,
        fallback_to_center=False,
    )


def publication_pipeline(**updates):
    pipeline = {
        "ntps": 3,
        "publish_roi_provenance": True,
        "run_binding": {"acquisition": "run-7", "labels": [1, True, None]},
        "steps": {"segment_cell": {"segmenter_kwargs": {"kind": "nahual_baby"}}},
        "save": ["segment_cell"],
    }
    pipeline.update(updates)
    return pipeline


def make_released_artifacts(root, pipeline=None, site="site-A"):
    pipeline = pipeline or publication_pipeline()
    paths = []
    for step_name in pipeline["steps"]:
        if not step_name.startswith("segment"):
            continue
        segment_dir = root / "steps" / site / step_name
        segment_dir.mkdir(parents=True)
        for tp in range(pipeline["ntps"]):
            npz = segment_dir / f"{tp:04d}.npz"
            metadata = segment_dir / f"{tp:04d}_meta.json"
            npz.write_bytes(f"mask-{step_name}-{tp}".encode())
            metadata.write_text(json.dumps({"tp": tp}))
            paths.extend((npz, metadata))
    profile = root / "profiles" / f"{site}.parquet"
    profile.parent.mkdir()
    profile.write_bytes(b"profiles")
    paths.append(profile)
    tracking = root / "tracking" / f"{site}_segment_cell.parquet"
    tracking.parent.mkdir()
    tracking.write_bytes(b"tracking")
    paths.append(tracking)
    return paths


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_exact_keys_dtypes_yx_geometry_and_atomic_inventory(tmp_path):
    tiler = make_live_tiler()
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)

    npz_path, json_path = write_roi_provenance(tiler, pipeline, tmp_path, "site-A")

    assert npz_path == tmp_path / "tiling" / "site-A" / "state.npz"
    assert json_path == tmp_path / "tiling" / "site-A" / "manifest.json"
    assert {path.name for path in npz_path.parent.iterdir()} == {
        "state.npz",
        "manifest.json",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        assert tuple(archive.files) == NPZ_KEYS
        assert archive["roi_id"].dtype == np.dtype("int64")
        assert archive["t0_centres_yx"].dtype == np.dtype("int64")
        assert archive["incremental_registration_shift_yx"].dtype == np.dtype("float64")
        assert archive["crop_origins_yx"].dtype == np.dtype("int64")
        for key in ("tile_shape_yx", "raw_shape_yx", "max_size_yx"):
            assert archive[key].dtype == np.dtype("int64")
        np.testing.assert_array_equal(archive["roi_id"], [0, 1])
        np.testing.assert_array_equal(archive["t0_centres_yx"], [[20, 30], [40, 50]])
        np.testing.assert_array_equal(archive["tile_shape_yx"], [6, 10])
        np.testing.assert_array_equal(archive["raw_shape_yx"], [80, 120])
        np.testing.assert_array_equal(archive["max_size_yx"], [80, 120])
        np.testing.assert_array_equal(
            archive["crop_origins_yx"][:, 0],
            [[17, 25], [15, 27], [16, 26]],
        )

    manifest = json.loads(json_path.read_text())
    assert manifest["schema_version"] == "aliby.roi-provenance.v1"
    assert manifest["site"] == "site-A"
    assert manifest["npz"] == {
        "basename": "state.npz",
        "sha256": sha256(npz_path),
    }
    assert manifest["coordinate_convention"] == {
        "axes": "YX",
        "index_base": 0,
        "space": "raw array",
        "crops": "NumPy half-open [start, stop)",
    }
    assert manifest["registration"]["formula"] == (
        "effective centre = trunc(t0 - cumsum(shifts through t))"
    )
    assert manifest["timepoints"] == 3
    assert manifest["rois"] == 2
    assert manifest["roi_source"] == "detected"
    assert manifest["reference"] == {"channel": 2, "z": 0}
    assert manifest["controls"] == {
        "track_drift": True,
        "fallback_to_center": False,
    }
    assert manifest["run_binding"] == pipeline["run_binding"]
    assert "global outputs" in manifest["artifact_scope"]["excluded"]
    assert npz_path.stat().st_mode & 0o222 == 0
    assert json_path.stat().st_mode & 0o222 == 0
    assert npz_path.parent.stat().st_mode & 0o222 == 0


def test_manifest_hashes_exact_sorted_core_inventory(tmp_path):
    pipeline = publication_pipeline()
    released = make_released_artifacts(tmp_path, pipeline)
    ignored = tmp_path / "steps" / "site-A" / "extract_cell" / "0000.npz"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not in core inventory")

    _, json_path = write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")

    artifacts = json.loads(json_path.read_text())["artifacts_sha256"]
    assert list(artifacts) == sorted(artifacts)
    assert artifacts == {
        path.relative_to(tmp_path).as_posix(): sha256(path) for path in released
    }
    assert all(not name.startswith("tiling/") for name in artifacts)


@pytest.mark.parametrize(
    ("drifts", "message"),
    [
        ([[0.0, 0.0], [1.0, 2.0]], "one numeric YX row"),
        ([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "exactly zero"),
        ([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]], "non-finite"),
        ([[0.0, 0.0], [1.0], [0.0, 0.0]], "one numeric YX row"),
    ],
)
def test_rejects_missing_malformed_nonfinite_and_nonzero_t0_drift(drifts, message):
    tiler = make_live_tiler(drifts=drifts)
    with pytest.raises(ValueError, match=message):
        collect_live_roi_state(tiler, 3)


def test_rejects_crop_origin_inconsistent_with_live_cumulative_semantics(monkeypatch):
    tiler = make_live_tiler()
    tile = tiler.tile_locs.tiles[0]
    actual = tile.as_range

    def wrong_range(tp):
        y, x = actual(tp)
        return slice(y.start + (tp == 2), y.stop + (tp == 2)), x

    monkeypatch.setattr(tile, "as_range", wrong_range)
    with pytest.raises(ValueError, match="do not reconstruct"):
        collect_live_roi_state(tiler, 3)


def test_writer_refuses_existing_final_directory_without_modifying_it(tmp_path):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)
    final = tmp_path / "tiling" / "site-A"
    final.mkdir(parents=True)
    existing = final / "manifest.json"
    existing.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")

    assert existing.read_text() == "keep"
    assert list(final.iterdir()) == [existing]


def test_stale_hidden_temp_is_ignored_and_safe_retry_commits(tmp_path):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)
    stale = tmp_path / "tiling" / ".site-A.roi-provenance-stale"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("interrupted")

    state_path, manifest_path = write_roi_provenance(
        make_live_tiler(), pipeline, tmp_path, "site-A"
    )

    assert state_path.is_file() and manifest_path.is_file()
    assert (stale / "partial").read_text() == "interrupted"


def test_python_error_removes_uncommitted_temp_directory(tmp_path, monkeypatch):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest write failed")

    monkeypatch.setattr("aliby.io.roi_provenance._write_fsynced", fail_manifest)

    with pytest.raises(OSError, match="manifest write failed"):
        write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")

    tiling = tmp_path / "tiling"
    assert not (tiling / "site-A").exists()
    assert list(tiling.iterdir()) == []


def test_concurrent_final_appearing_at_rename_is_not_replaced(tmp_path, monkeypatch):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)
    final = tmp_path / "tiling" / "site-A"
    real_rename = os.rename

    def race_rename(source, destination):
        final.mkdir()
        (final / "winner").write_text("other publisher")
        return real_rename(source, destination)

    monkeypatch.setattr("aliby.io.roi_provenance.os.rename", race_rename)

    with pytest.raises(OSError):
        write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")

    assert (final / "winner").read_text() == "other publisher"
    assert not any("roi-provenance-" in path.name for path in final.parent.iterdir())


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"tuple": (1, 2)},
        {"array": np.array([1])},
        {"bad_key": {1: "value"}},
        {"infinite": float("inf")},
    ],
)
def test_rejects_empty_or_non_json_safe_run_binding_without_output(tmp_path, binding):
    pipeline = publication_pipeline(run_binding=binding)
    with pytest.raises((TypeError, ValueError), match="run_binding"):
        validate_provenance_preflight(pipeline, tmp_path, "site-A")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("site", ["", ".", "..", "../site", "site/x", "/site"])
def test_preflight_rejects_unsafe_site_without_output(tmp_path, site):
    with pytest.raises(ValueError, match="pipeline_name"):
        validate_provenance_preflight(publication_pipeline(), tmp_path, site)
    assert list(tmp_path.iterdir()) == []


def test_preflight_rejects_unsafe_dynamic_segment_name(tmp_path):
    pipeline = publication_pipeline(
        steps={"segment/escape": {"segmenter_kwargs": {"kind": "nahual_baby"}}},
        save=["segment/escape"],
    )
    with pytest.raises(ValueError, match="step name"):
        validate_provenance_preflight(pipeline, tmp_path, "site-A")


@pytest.mark.parametrize("location", ["root", "tiling", "final"])
def test_preflight_rejects_symlinked_output_directories(tmp_path, location):
    real = tmp_path / "real"
    real.mkdir()
    output = tmp_path / "output"
    if location == "root":
        output.symlink_to(real, target_is_directory=True)
    else:
        output.mkdir()
        if location == "tiling":
            (output / "tiling").symlink_to(real, target_is_directory=True)
        else:
            tiling = output / "tiling"
            tiling.mkdir()
            (tiling / "site-A").symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_provenance_preflight(publication_pipeline(), output, "site-A")


@pytest.mark.parametrize("target", ["steps", "site", "segment", "profile", "tracking"])
def test_writer_rejects_symlinked_source_paths(tmp_path, target):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = {
        "steps": tmp_path / "steps",
        "site": tmp_path / "steps" / "site-A",
        "segment": tmp_path / "steps" / "site-A" / "segment_cell",
        "profile": tmp_path / "profiles" / "site-A.parquet",
        "tracking": tmp_path / "tracking" / "site-A_segment_cell.parquet",
    }
    path = paths[target]
    if path.is_dir():
        moved = tmp_path / f"moved-{target}"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    else:
        path.unlink()
        path.symlink_to(outside)

    with pytest.raises((ValueError, FileNotFoundError), match="symlink|inventory"):
        write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")
    assert not (tmp_path / "tiling" / "site-A").exists()


def test_rejects_missing_extra_and_stale_core_artifacts(tmp_path):
    pipeline = publication_pipeline()
    make_released_artifacts(tmp_path, pipeline)
    extra = tmp_path / "steps" / "site-A" / "segment_cell" / "stale.json"
    extra.write_text("stale")

    with pytest.raises(ValueError, match="inventory mismatch"):
        write_roi_provenance(make_live_tiler(), pipeline, tmp_path, "site-A")


def test_baby_hook_is_opt_in_and_generic_tracking_behavior_remains(
    tmp_path, monkeypatch
):
    state = {
        "fn": {},
        "data": {
            "segment_cell": [{"metadata": [{"cell_label": [1], "mother_assign": [0]}]}]
        },
    }
    pipeline = {
        "steps": {"segment_cell": {"segmenter_kwargs": {"kind": "nahual_baby"}}}
    }
    monkeypatch.setattr(
        "aliby.pipe_baby.write_roi_provenance",
        lambda *_args: pytest.fail("disabled publication must not be called"),
    )

    _save_baby_tracking_lineage(state, pipeline, tmp_path, "site-A")

    assert (tmp_path / "tracking" / "site-A_segment_cell.parquet").is_file()
    assert not (tmp_path / "tiling").exists()


def test_opt_in_hook_rejects_empty_tracking_without_publication(tmp_path):
    pipeline = publication_pipeline(ntps=1)
    state = {
        "fn": {"tile": object()},
        "data": {"segment_cell": [{"metadata": [{"cell_label": []}]}]},
    }

    with pytest.raises(ValueError, match="tracking.*empty"):
        _save_baby_tracking_lineage(state, pipeline, tmp_path, "site-A")

    assert not (tmp_path / "tracking").exists()
    assert not (tmp_path / "tiling").exists()


def test_baby_hook_publishes_only_after_tracking_and_requires_complete_state(
    tmp_path, monkeypatch
):
    tile_step = object()
    state = {
        "fn": {"tile": tile_step},
        "data": {
            "segment_cell": [{"metadata": [{"cell_label": [1], "mother_assign": [0]}]}]
        },
    }
    pipeline = publication_pipeline(ntps=1)
    calls = []

    def publish(tiler, passed_pipeline, output_path, pipeline_name):
        tracking = output_path / "tracking" / "site-A_segment_cell.parquet"
        assert tracking.is_file()
        calls.append((tiler, passed_pipeline, pipeline_name))
        return (
            output_path / "tiling/site-A/state.npz",
            output_path / "tiling/site-A/manifest.json",
        )

    monkeypatch.setattr("aliby.pipe_baby.write_roi_provenance", publish)
    _save_baby_tracking_lineage(state, pipeline, tmp_path, "site-A")
    assert calls == [(tile_step, pipeline, "site-A")]

    state["data"]["segment_cell"] = []
    with pytest.raises(ValueError, match="metadata-bearing"):
        _save_baby_tracking_lineage(state, pipeline, tmp_path, "site-B")


def test_disabled_wrapper_preserves_return_and_backend_kwargs(tmp_path, monkeypatch):
    pipeline = {"steps": {}, "publish_roi_provenance": False}
    expected = (object(), {"result": object()})
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr("aliby.pipe_baby._run_pipeline_and_post_impl", run)

    result = run_pipeline_and_post(
        pipeline,
        "legacy/site-name",
        tmp_path,
        False,
        backend="concurrent",
        max_workers=3,
        resource_limits={"cpu": 2},
    )

    assert result is expected
    assert calls[0][0] == (pipeline, "legacy/site-name", tmp_path, False)
    assert calls[0][1]["backend"] == "concurrent"
    assert calls[0][1]["max_workers"] == 3
    assert calls[0][1]["resource_limits"] == {"cpu": 2}


def test_preflight_existing_release_blocks_pipeline_before_any_artifact_mutation(
    tmp_path, monkeypatch
):
    pipeline = publication_pipeline()
    artifacts = make_released_artifacts(tmp_path, pipeline)
    before = {path: path.read_bytes() for path in artifacts}
    final = tmp_path / "tiling" / "site-A"
    final.mkdir(parents=True)
    (final / "state.npz").write_bytes(b"released state")
    (final / "manifest.json").write_bytes(b"released manifest")
    called = False

    def must_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        for path in artifacts:
            path.write_bytes(b"mutated")

    monkeypatch.setattr("aliby.pipe_baby._run_pipeline_and_post_impl", must_not_run)

    with pytest.raises(FileExistsError, match="already exists"):
        run_pipeline_and_post(pipeline, "site-A", tmp_path)

    assert called is False
    assert {path: path.read_bytes() for path in artifacts} == before
