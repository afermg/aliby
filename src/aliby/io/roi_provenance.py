"""Publish the exact live ROI and registration state used by a pipeline."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
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


SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def validate_safe_component(value: Any, name: str) -> str:
    """Require one non-dot path component with a conservative spelling."""
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} must be one nonempty safe path component.")
    if value in {".", ".."}:
        raise ValueError(f"{name} must not be a dot path component.")
    return value


def provenance_enabled(pipeline: dict) -> bool:
    value = pipeline.get("publish_roi_provenance", False)
    if not isinstance(value, bool):
        raise TypeError("publish_roi_provenance must be boolean.")
    return value


def baby_segment_steps(pipeline: dict, *, validate_names: bool = True) -> list[str]:
    result = []
    for step_name, parameters in pipeline.get("steps", {}).items():
        if validate_names:
            validate_safe_component(step_name, "pipeline step name")
        if step_name.startswith("segment"):
            kind = parameters.get("segmenter_kwargs", {}).get("kind", "")
            if isinstance(kind, str) and kind.endswith("baby"):
                result.append(step_name)
    return result


def _require_directory(path: Path, name: str, *, required: bool = True) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"Required {name} does not exist: {path}")
        return
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"{name} must not be a symlink: {path}")
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"{name} must be a directory: {path}")


def _require_regular_file(path: Path, name: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required {name} does not exist: {path}") from error
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"{name} must not be a symlink: {path}")
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")


def _validate_existing_regular_file(path: Path, name: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"{name} must not be a symlink: {path}")
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")


def _enforce_containment(output_path: Path, candidate: Path) -> None:
    root = output_path.absolute()
    path = candidate.absolute()
    if os.path.commonpath((root, path)) != str(root):
        raise ValueError(f"Provenance path escapes output root: {candidate}")


def _sha256_nofollow(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Artifact must be a non-symlink regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"Artifact changed while opening it: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (
            before.st_size,
            before.st_mtime_ns,
        ):
            raise RuntimeError(f"Artifact changed while hashing it: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def validate_provenance_preflight(
    pipeline: dict, output_path: str | Path, pipeline_name: str
) -> None:
    """Fail before pipeline output mutation when opt-in publication is unsafe."""
    if not provenance_enabled(pipeline):
        return
    pipeline_name = validate_safe_component(pipeline_name, "pipeline_name")
    if not isinstance(pipeline.get("run_binding"), dict) or not pipeline["run_binding"]:
        raise ValueError("run_binding must be a nonempty dictionary for publication.")
    _json_safe(pipeline["run_binding"])

    segments = baby_segment_steps(pipeline)
    if not segments:
        raise ValueError("ROI provenance requires at least one BABY segment step.")
    saved = pipeline.get("save") or ()
    missing = [step for step in segments if step not in saved]
    if missing:
        raise ValueError(f"BABY segment steps must be saved: {missing}")

    timepoints = pipeline.get("ntps", 1)
    if (
        not isinstance(timepoints, int)
        or isinstance(timepoints, bool)
        or timepoints < 1
    ):
        raise ValueError("ntps must be a positive integer for publication.")

    output_path = Path(output_path)
    _require_directory(output_path, "output root", required=False)
    tiling_dir = output_path / "tiling"
    steps_dir = output_path / "steps"
    site_steps_dir = steps_dir / pipeline_name
    profiles_dir = output_path / "profiles"
    tracking_dir = output_path / "tracking"
    for path, name in (
        (tiling_dir, "tiling directory"),
        (steps_dir, "steps directory"),
        (site_steps_dir, "site steps directory"),
        (profiles_dir, "profiles directory"),
        (tracking_dir, "tracking directory"),
    ):
        _enforce_containment(output_path, path)
        _require_directory(path, name, required=False)
    for step_name in segments:
        segment_dir = site_steps_dir / step_name
        _enforce_containment(output_path, segment_dir)
        _require_directory(
            segment_dir, f"segment directory {step_name}", required=False
        )
        for tp in range(timepoints):
            for filename in (f"{tp:04d}.npz", f"{tp:04d}_meta.json"):
                artifact = segment_dir / filename
                _enforce_containment(output_path, artifact)
                _validate_existing_regular_file(artifact, "released segment artifact")
    for artifact in (
        profiles_dir / f"{pipeline_name}.parquet",
        *(tracking_dir / f"{pipeline_name}_{step}.parquet" for step in segments),
    ):
        _enforce_containment(output_path, artifact)
        _validate_existing_regular_file(artifact, "released table artifact")

    final_dir = tiling_dir / pipeline_name
    _enforce_containment(output_path, final_dir)
    if _lexists(final_dir):
        details = final_dir.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(
                f"final provenance destination must not be a symlink: {final_dir}"
            )
        raise FileExistsError(
            f"ROI provenance destination already exists for {pipeline_name!r}."
        )


def _expected_artifacts(
    pipeline: dict, output_path: Path, pipeline_name: str
) -> list[Path]:
    timepoints = pipeline.get("ntps", 1)
    if (
        not isinstance(timepoints, int)
        or isinstance(timepoints, bool)
        or timepoints < 1
    ):
        raise ValueError("ntps must be a positive integer for publication.")
    segments = baby_segment_steps(pipeline)

    _require_directory(output_path, "output root")
    steps_dir = output_path / "steps"
    site_steps_dir = steps_dir / pipeline_name
    profiles_dir = output_path / "profiles"
    tracking_dir = output_path / "tracking"
    for path, name in (
        (steps_dir, "steps directory"),
        (site_steps_dir, "site steps directory"),
        (profiles_dir, "profiles directory"),
        (tracking_dir, "tracking directory"),
    ):
        _enforce_containment(output_path, path)
        _require_directory(path, name)

    expected = [profiles_dir / f"{pipeline_name}.parquet"]
    expected_tracking_names = {
        f"{pipeline_name}_{step_name}.parquet" for step_name in segments
    }
    actual_tracking_names = {
        entry.name
        for entry in os.scandir(tracking_dir)
        if entry.name.startswith(f"{pipeline_name}_")
    }
    if actual_tracking_names != expected_tracking_names:
        raise ValueError(
            "Tracking artifact inventory mismatch: "
            f"expected {sorted(expected_tracking_names)}, got {sorted(actual_tracking_names)}."
        )
    expected.extend(tracking_dir / name for name in sorted(expected_tracking_names))

    for step_name in segments:
        segment_dir = site_steps_dir / step_name
        _enforce_containment(output_path, segment_dir)
        _require_directory(segment_dir, f"segment directory {step_name}")
        expected_names = {
            name
            for tp in range(timepoints)
            for name in (f"{tp:04d}.npz", f"{tp:04d}_meta.json")
        }
        actual_names = {entry.name for entry in os.scandir(segment_dir)}
        if actual_names != expected_names:
            raise ValueError(
                f"Segment artifact inventory mismatch for {step_name}: "
                f"expected {sorted(expected_names)}, got {sorted(actual_names)}."
            )
        expected.extend(segment_dir / name for name in sorted(expected_names))

    for path in expected:
        _enforce_containment(output_path, path)
        _require_regular_file(path, "released artifact")
    return sorted(expected, key=lambda path: path.relative_to(output_path).as_posix())


def _released_artifact_hashes(
    pipeline: dict, output_path: Path, pipeline_name: str
) -> dict[str, str]:
    return {
        path.relative_to(output_path).as_posix(): _sha256_nofollow(path)
        for path in _expected_artifacts(pipeline, output_path, pipeline_name)
    }


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_roi_provenance(
    tiler: Any,
    pipeline: dict,
    output_path: str | Path,
    pipeline_name: str,
) -> tuple[Path, Path]:
    """Publish a validated immutable state directory with one atomic rename."""
    if not provenance_enabled(pipeline):
        raise ValueError("ROI provenance publication is not enabled.")
    validate_provenance_preflight(pipeline, output_path, pipeline_name)
    pipeline_name = validate_safe_component(pipeline_name, "pipeline_name")
    output_path = Path(output_path)
    timepoints = pipeline.get("ntps", 1)
    arrays = collect_live_roi_state(tiler, timepoints)
    run_binding = _json_safe(pipeline["run_binding"])
    roi_source = getattr(tiler.tile_locs, "roi_source", None)
    if not isinstance(roi_source, str) or not roi_source:
        raise ValueError("Live tile locations have no valid roi_source.")
    artifact_hashes = _released_artifact_hashes(pipeline, output_path, pipeline_name)

    tiling_dir = output_path / "tiling"
    try:
        tiling_dir.mkdir(mode=0o755)
    except FileExistsError:
        pass
    _require_directory(tiling_dir, "tiling directory")
    final_dir = tiling_dir / pipeline_name
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{pipeline_name}.roi-provenance-", dir=tiling_dir)
    )
    state_path = temp_dir / "state.npz"
    manifest_path = temp_dir / "manifest.json"
    try:
        with state_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        npz_sha256 = _sha256(state_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "site": pipeline_name,
            "npz": {"basename": "state.npz", "sha256": npz_sha256},
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
            "artifact_scope": {
                "included": "BABY segment arrays/metadata, profiles, and BABY tracking",
                "excluded": "global outputs; this hook runs before global steps",
            },
            "artifacts_sha256": artifact_hashes,
        }
        _write_fsynced(
            manifest_path,
            (
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode(),
        )
        os.chmod(state_path, 0o444)
        os.chmod(manifest_path, 0o444)
        directory_fd = os.open(temp_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.chmod(temp_dir, 0o555)

        # Serialise cooperating publishers on the parent directory. The final
        # existence check plus rename gives a single all-or-none visibility point.
        parent_fd = os.open(tiling_dir, os.O_RDONLY)
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            if _lexists(final_dir):
                raise FileExistsError(
                    f"ROI provenance destination already exists for {pipeline_name!r}."
                )
            os.rename(temp_dir, final_dir)
            os.fsync(parent_fd)
        finally:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
            os.close(parent_fd)
    except Exception:
        if _lexists(temp_dir):
            os.chmod(temp_dir, 0o755)
            shutil.rmtree(temp_dir)
        raise

    return final_dir / "state.npz", final_dir / "manifest.json"
