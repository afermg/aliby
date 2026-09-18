"""Publish the exact live ROI and registration state used by a pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "aliby.roi-provenance.v1"
NPZ_KEYS = (
    "roi_id",
    "t0_centres_yx",
    "incremental_registration_shift_yx",
    "crop_origins_yx",
    "tile_shape_yx",
    "raw_shape_yx",
    "max_size_yx",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int64_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array with shape {shape}.")
    if array.dtype.kind in "iu":
        if array.dtype.kind == "u" and (array > np.iinfo(np.int64).max).any():
            raise ValueError(f"{name} must contain finite signed int64 values.")
        return array.astype(np.int64)

    numeric = np.asarray(array, dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.trunc(numeric)).all()
        or (numeric < -(2**63)).any()
        or (numeric >= 2**63).any()
    ):
        raise ValueError(f"{name} must contain finite signed int64 values.")
    return numeric.astype(np.int64)


def _json_safe(value: Any, path: str = "run_binding") -> Any:
    """Copy JSON data while rejecting coercions and non-finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, list):
        return [
            _json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key.")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}.")


def _manifest_scalar(value: Any, name: str) -> str | int | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise TypeError(f"{name} must be a string, integer, or null.")


def _effective_bool(
    tiler: Any, current: str, legacy: str | None, default: bool
) -> bool:
    if legacy is not None and hasattr(tiler, legacy):
        value = getattr(tiler, legacy)
    else:
        value = getattr(tiler, current, default)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Effective {current} control must be boolean.")
    return bool(value)


def collect_live_roi_state(tiler: Any, timepoints: int) -> dict[str, np.ndarray]:
    """Validate and encode the final live ``Tiler`` coordinate state."""
    if (
        not isinstance(timepoints, int)
        or isinstance(timepoints, bool)
        or timepoints < 1
    ):
        raise ValueError("timepoints must be a positive integer.")
    tile_locs = getattr(tiler, "tile_locs", None)
    if tile_locs is None:
        raise ValueError("Tiler has no live tile locations to publish.")
    tiles = list(getattr(tile_locs, "tiles", ()))
    if not tiles:
        raise ValueError("Tiler has no live tiles to publish.")
    if getattr(tiler, "no_processed", None) != timepoints:
        raise ValueError(
            "Tiler processed-timepoint count is inconsistent with publication."
        )

    n_rois = len(tiles)
    centres = _int64_array(
        [tile.centre for tile in tiles], "t0_centres_yx", (n_rois, 2)
    )
    initial = _int64_array(tile_locs.initial_location, "initial_location", (n_rois, 2))
    if not np.array_equal(initial, centres):
        raise ValueError("Tile enumeration and initial locations are inconsistent.")

    try:
        drift_values = np.asarray(getattr(tile_locs, "drifts", None))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "incremental_registration_shift_yx must have one numeric YX row per timepoint."
        ) from error
    if drift_values.shape != (timepoints, 2) or drift_values.dtype.kind not in "iuf":
        raise ValueError(
            "incremental_registration_shift_yx must have one numeric YX row per timepoint."
        )
    shifts = np.asarray(drift_values, dtype=np.float64)
    if not np.isfinite(shifts).all():
        raise ValueError(
            "incremental_registration_shift_yx contains non-finite values."
        )
    if not np.equal(shifts[0], 0.0).all():
        raise ValueError("Timepoint-zero registration shift must be exactly zero.")

    tile_sizes = [_int64_array(tile.size, "tile size", (2,)) for tile in tiles]
    if any(not np.array_equal(size, tile_sizes[0]) for size in tile_sizes[1:]):
        raise ValueError("All published tiles must have one consistent YX shape.")
    tile_shape = tile_sizes[0]
    if (tile_shape <= 0).any():
        raise ValueError("Tile shape must be positive.")

    raw_shape = _int64_array(tiler.pixels.shape[-2:], "raw shape", (2,))
    max_size = _int64_array(tile_locs.max_size, "max size", (2,))
    if (raw_shape <= 0).any() or (max_size <= 0).any():
        raise ValueError("Raw and maximum YX shapes must be positive.")
    if int(tiler.pixels.shape[0]) < timepoints:
        raise ValueError(
            "Raw pixels contain fewer timepoints than the published state."
        )

    origins = np.empty((timepoints, n_rois, 2), dtype=np.int64)
    for roi_id, tile in enumerate(tiles):
        if tile.parent_class is not tile_locs:
            raise ValueError(
                "Tile parent state is inconsistent with live tile locations."
            )
        if not np.array_equal(
            _int64_array(tile.max_size, "tile max size", (2,)), max_size
        ):
            raise ValueError(
                "Tile maximum size is inconsistent with live tile locations."
            )
        for tp in range(timepoints):
            ranges = tile.as_range(tp)
            if len(ranges) != 2 or any(item.start is None for item in ranges):
                raise ValueError("Tile crop range is missing a YX start coordinate.")
            origins[tp, roi_id] = [ranges[0].start, ranges[1].start]
            extents = [item.stop - item.start for item in ranges]
            if extents != tile_shape.tolist():
                raise ValueError("Tile crop range is inconsistent with tile shape.")

    cumulative = np.cumsum(shifts, axis=0)
    effective_centres = np.trunc(
        centres[np.newaxis, :, :].astype(np.float64) - cumulative[:, np.newaxis, :]
    ).astype(np.int64)
    expected_origins = effective_centres - tile_shape // 2
    if not np.array_equal(origins, expected_origins):
        raise ValueError(
            "Crop origins do not reconstruct under cumulative-shift truncation semantics."
        )

    return {
        "roi_id": np.arange(n_rois, dtype=np.int64),
        "t0_centres_yx": centres,
        "incremental_registration_shift_yx": shifts,
        "crop_origins_yx": origins,
        "tile_shape_yx": tile_shape,
        "raw_shape_yx": raw_shape,
        "max_size_yx": max_size,
    }


def _released_artifact_hashes(output_path: Path, pipeline_name: str) -> dict[str, str]:
    candidates: set[Path] = set()
    steps_dir = output_path / "steps" / pipeline_name
    if steps_dir.is_dir():
        for segment_dir in steps_dir.iterdir():
            if segment_dir.is_dir() and segment_dir.name.startswith("segment"):
                candidates.update(
                    path
                    for path in segment_dir.rglob("*")
                    if path.is_file() and path.suffix in {".npz", ".json"}
                )
    profile = output_path / "profiles" / f"{pipeline_name}.parquet"
    if profile.is_file():
        candidates.add(profile)
    tracking_dir = output_path / "tracking"
    if tracking_dir.is_dir():
        candidates.update(
            path
            for path in tracking_dir.glob(f"{pipeline_name}_*.parquet")
            if path.is_file()
        )
    return {
        path.relative_to(output_path).as_posix(): _sha256(path)
        for path in sorted(
            candidates, key=lambda item: item.relative_to(output_path).as_posix()
        )
    }


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".roi-provenance-", suffix=suffix, dir=directory
    )
    os.close(descriptor)
    return Path(name)


def write_roi_provenance(
    tiler: Any,
    pipeline: dict,
    output_path: str | Path,
    pipeline_name: str,
) -> tuple[Path, Path]:
    """Atomically publish non-overwriting NPZ and JSON ROI provenance files."""
    output_path = Path(output_path)
    output_dir = output_path / "tiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{pipeline_name}.npz"
    json_path = output_dir / f"{pipeline_name}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(
            f"ROI provenance destination already exists for {pipeline_name!r}."
        )

    timepoints = pipeline.get("ntps", 1)
    arrays = collect_live_roi_state(tiler, timepoints)
    run_binding = _json_safe(pipeline.get("run_binding", {}))
    roi_source = getattr(tiler.tile_locs, "roi_source", None)
    if not isinstance(roi_source, str) or not roi_source:
        raise ValueError("Live tile locations have no valid roi_source.")

    npz_temp = _temporary_path(output_dir, ".npz")
    json_temp = _temporary_path(output_dir, ".json")
    linked_npz = False
    linked_json = False
    try:
        with npz_temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        npz_sha256 = _sha256(npz_temp)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "site": pipeline_name,
            "npz": {"basename": npz_path.name, "sha256": npz_sha256},
            "coordinate_convention": {
                "axes": "YX",
                "index_base": 0,
                "space": "raw array",
                "crops": "NumPy half-open [start, stop)",
            },
            "registration": {
                "shift_kind": "incremental",
                "shift_axes": "YX",
                "shift_sign": "shift registering timepoint t to timepoint t-1",
                "formula": "effective centre = trunc(t0 - cumsum(shifts through t))",
            },
            "timepoints": int(arrays["incremental_registration_shift_yx"].shape[0]),
            "rois": int(arrays["roi_id"].shape[0]),
            "roi_source": roi_source,
            "reference": {
                "channel": _manifest_scalar(
                    getattr(tiler, "ref_channel", None), "reference channel"
                ),
                "z": _manifest_scalar(getattr(tiler, "ref_z", 0), "reference z"),
            },
            "controls": {
                "track_drift": _effective_bool(
                    tiler, "track_drift", "calculate_drift", False
                ),
                "fallback_to_center": _effective_bool(
                    tiler, "fallback_to_center", None, True
                ),
            },
            "run_binding": run_binding,
            "artifacts_sha256": _released_artifact_hashes(output_path, pipeline_name),
        }
        encoded = (
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        with json_temp.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

        # Hard-link publication is atomic and fails rather than replacing an
        # independently-created destination. Temps are removed below.
        os.link(npz_temp, npz_path)
        linked_npz = True
        os.link(json_temp, json_path)
        linked_json = True
        os.chmod(npz_path, 0o444)
        os.chmod(json_path, 0o444)
    except Exception:
        if linked_json:
            json_path.unlink(missing_ok=True)
        if linked_npz:
            npz_path.unlink(missing_ok=True)
        raise
    finally:
        npz_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)

    return npz_path, json_path
