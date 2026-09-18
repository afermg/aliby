import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from aliby.io.roi_provenance import (
    NPZ_KEYS,
    collect_live_roi_state,
    write_roi_provenance,
)
from aliby.pipe_baby import _save_baby_tracking_lineage
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_exact_keys_dtypes_yx_geometry_and_enumeration(tmp_path):
    tiler = make_live_tiler()
    pipeline = {
        "ntps": 3,
        "run_binding": {"acquisition": "run-7", "labels": [1, True, None]},
    }

    npz_path, json_path = write_roi_provenance(tiler, pipeline, tmp_path, "site-A")

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
        "basename": "site-A.npz",
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
    assert npz_path.stat().st_mode & 0o222 == 0
    assert json_path.stat().st_mode & 0o222 == 0


def test_manifest_hashes_only_released_artifacts_in_sorted_relative_inventory(tmp_path):
    segment_dir = tmp_path / "steps" / "site-A" / "segment_cell"
    segment_dir.mkdir(parents=True)
    segment_npz = segment_dir / "0000.npz"
    segment_json = segment_dir / "0000_meta.json"
    segment_npz.write_bytes(b"mask")
    segment_json.write_text('{"metadata": true}')
    profiles = tmp_path / "profiles" / "site-A.parquet"
    profiles.parent.mkdir()
    profiles.write_bytes(b"profiles")
    tracking = tmp_path / "tracking" / "site-A_segment_cell.parquet"
    tracking.parent.mkdir()
    tracking.write_bytes(b"tracking")
    ignored = tmp_path / "steps" / "site-A" / "extract_cell" / "0000.npz"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not released by this schema")

    _, json_path = write_roi_provenance(
        make_live_tiler(), {"ntps": 3}, tmp_path, "site-A"
    )

    artifacts = json.loads(json_path.read_text())["artifacts_sha256"]
    assert list(artifacts) == sorted(artifacts)
    assert artifacts == {
        path.relative_to(tmp_path).as_posix(): sha256(path)
        for path in (profiles, segment_npz, segment_json, tracking)
    }
    assert all(not name.startswith("tiling/") for name in artifacts)
    assert not any(".roi-provenance-" in name for name in artifacts)


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


def test_writer_refuses_existing_destination_without_modifying_it(tmp_path):
    tiling = tmp_path / "tiling"
    tiling.mkdir()
    existing = tiling / "site-A.json"
    existing.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        write_roi_provenance(make_live_tiler(), {"ntps": 3}, tmp_path, "site-A")

    assert existing.read_text() == "keep"
    assert not (tiling / "site-A.npz").exists()


@pytest.mark.parametrize(
    "binding",
    [
        {"tuple": (1, 2)},
        {"array": np.array([1])},
        {"bad_key": {1: "value"}},
        {"infinite": float("inf")},
    ],
)
def test_rejects_non_json_safe_run_binding_without_publication(tmp_path, binding):
    with pytest.raises((TypeError, ValueError), match="run_binding"):
        write_roi_provenance(
            make_live_tiler(),
            {"ntps": 3, "run_binding": binding},
            tmp_path,
            "site-A",
        )
    assert list((tmp_path / "tiling").iterdir()) == []


def test_baby_hook_publishes_provenance_after_tracking(tmp_path, monkeypatch):
    tile_step = object()
    state = {
        "fn": {"tile": tile_step},
        "data": {
            "segment_cell": [
                {
                    "metadata": [
                        {"cell_label": [1], "mother_assign": [0]},
                    ]
                }
            ]
        },
    }
    pipeline = {
        "steps": {"segment_cell": {"segmenter_kwargs": {"kind": "nahual_baby"}}}
    }
    calls = []

    def publish(tiler, passed_pipeline, output_path, pipeline_name):
        tracking = output_path / "tracking" / "site-A_segment_cell.parquet"
        assert tracking.is_file()
        calls.append((tiler, passed_pipeline, pipeline_name))
        return output_path / "tiling/site-A.npz", output_path / "tiling/site-A.json"

    monkeypatch.setattr("aliby.pipe_baby.write_roi_provenance", publish)

    _save_baby_tracking_lineage(state, pipeline, tmp_path, "site-A")

    assert calls == [(tile_step, pipeline, "site-A")]
