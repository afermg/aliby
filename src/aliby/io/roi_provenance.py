"""Publish the exact live ROI and registration state used by a pipeline."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
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


_LOCK_AUTHORITY = object()


@dataclass
class ProvenanceRunLock:
    """Authenticated ownership of one lexical output root and its run lock."""

    lexical_output_path: Path
    root_fd: int
    tiling_fd: int
    root_identity: tuple[int, int, int]
    tiling_identity: tuple[int, int, int]
    owner_pid: int
    authority: object
    active: bool = True

    @property
    def descriptor_output_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.root_fd}")

    def verify(self) -> None:
        if (
            not self.active
            or self.owner_pid != os.getpid()
            or self.authority is not _LOCK_AUTHORITY
        ):
            raise RuntimeError("Provenance run lock is inactive or not owned here.")
        root_fd_identity = _directory_identity(os.fstat(self.root_fd))
        tiling_fd_identity = _directory_identity(os.fstat(self.tiling_fd))
        root_path_identity = _directory_identity(
            os.stat(self.lexical_output_path, follow_symlinks=False)
        )
        tiling_path_identity = _directory_identity(
            os.stat("tiling", dir_fd=self.root_fd, follow_symlinks=False)
        )
        if (
            root_fd_identity != self.root_identity
            or root_path_identity != self.root_identity
            or tiling_fd_identity != self.tiling_identity
            or tiling_path_identity != self.tiling_identity
        ):
            raise RuntimeError("Lexical output root or tiling lock binding changed.")

    def release(self) -> None:
        if not self.active:
            return
        if self.owner_pid != os.getpid():
            raise RuntimeError(
                "Provenance run lock cannot be released by another process."
            )
        try:
            fcntl.flock(self.tiling_fd, fcntl.LOCK_UN)
        finally:
            os.close(self.tiling_fd)
            os.close(self.root_fd)
            self.active = False


def acquire_provenance_run_lock(output_path: str | Path) -> ProvenanceRunLock:
    """Create/open the real tiling directory and exclusively lock it."""
    lexical_output_path = Path(output_path)
    lexical_output_path.mkdir(parents=True, exist_ok=True)
    root_fd = os.open(lexical_output_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    tiling_fd = None
    try:
        try:
            os.mkdir("tiling", mode=0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        tiling_fd = os.open(
            "tiling",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        fcntl.flock(tiling_fd, fcntl.LOCK_EX)
        context = ProvenanceRunLock(
            lexical_output_path=lexical_output_path,
            root_fd=root_fd,
            tiling_fd=tiling_fd,
            root_identity=_directory_identity(os.fstat(root_fd)),
            tiling_identity=_directory_identity(os.fstat(tiling_fd)),
            owner_pid=os.getpid(),
            authority=_LOCK_AUTHORITY,
        )
        context.verify()
        return context
    except Exception:
        if tiling_fd is not None:
            try:
                fcntl.flock(tiling_fd, fcntl.LOCK_UN)
            finally:
                os.close(tiling_fd)
        os.close(root_fd)
        raise


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
    tile_steps = [name for name in pipeline["steps"] if name.startswith("tile")]
    if len(tile_steps) != 1:
        raise ValueError(
            "ROI provenance requires exactly one configured tile step; "
            f"found {len(tile_steps)}."
        )
    save_interval = pipeline.get("save_interval", 1)
    if type(save_interval) is not int or save_interval != 1:
        raise ValueError("ROI provenance requires save_interval exactly 1.")
    if pipeline.get("global_steps") or pipeline.get("global_passed_data"):
        raise ValueError(
            "ROI provenance does not support global steps because publication "
            "precedes global outputs."
        )
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


@dataclass
class _HeldDirectory:
    name: str | None
    fd: int
    parent: "_HeldDirectory | None"
    path: Path
    opened_stat: os.stat_result


@dataclass
class _HeldArtifact:
    relative_path: str
    name: str
    fd: int
    parent: _HeldDirectory
    opened_stat: os.stat_result


def _metadata_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _directory_identity(details: os.stat_result) -> tuple[int, int, int]:
    return details.st_dev, details.st_ino, details.st_mode


def _open_child_directory(
    parent: _HeldDirectory, name: str, path: Path
) -> _HeldDirectory:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent.fd)
    except OSError as error:
        raise ValueError(f"Required directory is missing or unsafe: {path}") from error
    return _HeldDirectory(name, descriptor, parent, path, os.fstat(descriptor))


def _open_artifact(
    parent: _HeldDirectory, name: str, relative_path: str
) -> _HeldArtifact:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent.fd)
    except OSError as error:
        raise ValueError(
            f"Required artifact is missing or unsafe: {relative_path}"
        ) from error
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode):
        os.close(descriptor)
        raise ValueError(f"Artifact must be a regular file: {relative_path}")
    return _HeldArtifact(relative_path, name, descriptor, parent, details)


class _HeldInventory:
    """No-follow artifact and directory descriptors held through publication."""

    def __init__(
        self, lock_context: ProvenanceRunLock, pipeline: dict, pipeline_name: str
    ):
        lock_context.verify()
        self.output_path = lock_context.lexical_output_path
        self.directories: list[_HeldDirectory] = []
        self.artifacts: list[_HeldArtifact] = []
        self.segment_inventories: list[tuple[_HeldDirectory, set[str]]] = []
        self.tracking_prefix = ""
        self.expected_tracking_names: set[str] = set()
        try:
            self.root = self._duplicate_root(lock_context)
            self.steps = self._child(self.root, "steps")
            self.site = self._child(self.steps, pipeline_name)
            self.profiles = self._child(self.root, "profiles")
            self.tracking = self._child(self.root, "tracking")
            self.tiling = self._duplicate_tiling(lock_context)
            self._open_expected_artifacts(pipeline, pipeline_name)
        except Exception:
            self.close()
            raise

    def _duplicate_root(self, lock_context: ProvenanceRunLock) -> _HeldDirectory:
        descriptor = os.dup(lock_context.root_fd)
        details = os.fstat(descriptor)
        if _directory_identity(details) != lock_context.root_identity:
            os.close(descriptor)
            raise RuntimeError("Held inventory root does not match the run lock.")
        result = _HeldDirectory(
            None, descriptor, None, lock_context.lexical_output_path, details
        )
        self.directories.append(result)
        return result

    def _duplicate_tiling(self, lock_context: ProvenanceRunLock) -> _HeldDirectory:
        descriptor = os.dup(lock_context.tiling_fd)
        details = os.fstat(descriptor)
        if _directory_identity(details) != lock_context.tiling_identity:
            os.close(descriptor)
            raise RuntimeError(
                "Held inventory tiling directory does not match the lock."
            )
        result = _HeldDirectory(
            "tiling",
            descriptor,
            self.root,
            lock_context.lexical_output_path / "tiling",
            details,
        )
        self.directories.append(result)
        return result

    def _child(self, parent: _HeldDirectory, name: str) -> _HeldDirectory:
        result = _open_child_directory(parent, name, parent.path / name)
        self.directories.append(result)
        return result

    def _artifact(self, parent: _HeldDirectory, name: str, relative_path: str) -> None:
        self.artifacts.append(_open_artifact(parent, name, relative_path))

    def _open_expected_artifacts(self, pipeline: dict, pipeline_name: str) -> None:
        timepoints = pipeline.get("ntps", 1)
        segments = baby_segment_steps(pipeline)
        profile_name = f"{pipeline_name}.parquet"
        self._artifact(self.profiles, profile_name, f"profiles/{profile_name}")

        self.tracking_prefix = f"{pipeline_name}_"
        self.expected_tracking_names = {
            f"{pipeline_name}_{step_name}.parquet" for step_name in segments
        }
        actual_tracking_names = {
            name
            for name in os.listdir(self.tracking.fd)
            if name.startswith(self.tracking_prefix)
        }
        if actual_tracking_names != self.expected_tracking_names:
            raise ValueError(
                "Tracking artifact inventory mismatch: "
                f"expected {sorted(self.expected_tracking_names)}, "
                f"got {sorted(actual_tracking_names)}."
            )
        for name in sorted(self.expected_tracking_names):
            self._artifact(self.tracking, name, f"tracking/{name}")

        for step_name in segments:
            segment = self._child(self.site, step_name)
            expected_names = {
                name
                for tp in range(timepoints)
                for name in (f"{tp:04d}.npz", f"{tp:04d}_meta.json")
            }
            actual_names = set(os.listdir(segment.fd))
            if actual_names != expected_names:
                raise ValueError(
                    f"Segment artifact inventory mismatch for {step_name}: "
                    f"expected {sorted(expected_names)}, got {sorted(actual_names)}."
                )
            self.segment_inventories.append((segment, expected_names))
            for name in sorted(expected_names):
                relative = f"steps/{pipeline_name}/{step_name}/{name}"
                self._artifact(segment, name, relative)

        self.artifacts.sort(key=lambda artifact: artifact.relative_path)

    def hashes(self) -> dict[str, str]:
        return {
            artifact.relative_path: _hash_held_artifact(artifact)
            for artifact in self.artifacts
        }

    def revalidate(self) -> None:
        root_path_stat = os.stat(self.output_path, follow_symlinks=False)
        root_fd_stat = os.fstat(self.root.fd)
        if _directory_identity(root_path_stat) != _directory_identity(
            root_fd_stat
        ) or _directory_identity(root_fd_stat) != _directory_identity(
            self.root.opened_stat
        ):
            raise RuntimeError("Output root identity changed before publication.")

        for directory in self.directories[1:]:
            current_path = os.stat(
                directory.name,
                dir_fd=directory.parent.fd,
                follow_symlinks=False,
            )
            current_fd = os.fstat(directory.fd)
            if _directory_identity(current_path) != _directory_identity(
                current_fd
            ) or _directory_identity(current_fd) != _directory_identity(
                directory.opened_stat
            ):
                raise RuntimeError(
                    f"Directory identity changed before publication: {directory.path}"
                )

        current_tracking_names = {
            name
            for name in os.listdir(self.tracking.fd)
            if name.startswith(self.tracking_prefix)
        }
        if current_tracking_names != self.expected_tracking_names:
            raise RuntimeError(
                "Tracking artifact inventory changed before publication."
            )
        for segment, expected_names in self.segment_inventories:
            if set(os.listdir(segment.fd)) != expected_names:
                raise RuntimeError(
                    f"Segment artifact inventory changed before publication: {segment.path}"
                )

        for artifact in self.artifacts:
            current_path = os.stat(
                artifact.name,
                dir_fd=artifact.parent.fd,
                follow_symlinks=False,
            )
            current_fd = os.fstat(artifact.fd)
            expected = _metadata_identity(artifact.opened_stat)
            if (
                _metadata_identity(current_path) != expected
                or _metadata_identity(current_fd) != expected
            ):
                raise RuntimeError(
                    "Artifact identity or metadata changed before publication: "
                    f"{artifact.relative_path}"
                )

    def close(self) -> None:
        for artifact in reversed(self.artifacts):
            try:
                os.close(artifact.fd)
            except OSError:
                pass
        self.artifacts.clear()
        for directory in reversed(self.directories):
            try:
                os.close(directory.fd)
            except OSError:
                pass
        self.directories.clear()


def _hash_held_artifact(artifact: _HeldArtifact) -> str:
    expected = _metadata_identity(artifact.opened_stat)
    if _metadata_identity(os.fstat(artifact.fd)) != expected:
        raise RuntimeError(f"Artifact changed before hashing: {artifact.relative_path}")
    os.lseek(artifact.fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(artifact.fd, 1024 * 1024):
        digest.update(chunk)
    if _metadata_identity(os.fstat(artifact.fd)) != expected:
        raise RuntimeError(f"Artifact changed while hashing: {artifact.relative_path}")
    return digest.hexdigest()


def _write_fsynced_at(
    directory_fd: int, name: str, content: bytes, *, mode: int = 0o600
) -> int:
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        mode,
        dir_fd=directory_fd,
    )
    try:
        remaining = memoryview(content)
        while remaining:
            remaining = remaining[os.write(descriptor, remaining) :]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _get_renameat2():
    return getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)


def _rename_noreplace(
    source_dir_fd: int, source_name: str, destination_dir_fd: int, destination_name: str
) -> None:
    """Linux atomic rename with mandatory no-replace semantics."""
    renameat2 = _get_renameat2()
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing unsafe publication.")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source_name),
        destination_dir_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError(
            "renameat2 RENAME_NOREPLACE is unsupported; refusing unsafe publication."
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _before_provenance_commit() -> None:
    """Test seam immediately before final descriptor revalidation and commit."""


def _cleanup_temp(tiling_fd: int, temp_fd: int, temp_name: str) -> None:
    try:
        os.fchmod(temp_fd, 0o700)
        for name in os.listdir(temp_fd):
            os.unlink(name, dir_fd=temp_fd)
    finally:
        os.close(temp_fd)
    os.rmdir(temp_name, dir_fd=tiling_fd)


def _write_roi_provenance_locked(
    tiler: Any,
    pipeline: dict,
    pipeline_name: str,
    lock_context: ProvenanceRunLock,
) -> tuple[Path, Path]:
    """Publish using an authenticated lock already held across the pipeline."""
    if not provenance_enabled(pipeline):
        raise ValueError("ROI provenance publication is not enabled.")
    lock_context.verify()
    pipeline_name = validate_safe_component(pipeline_name, "pipeline_name")
    output_path = lock_context.lexical_output_path
    timepoints = pipeline.get("ntps", 1)
    arrays = collect_live_roi_state(tiler, timepoints)
    run_binding = _json_safe(pipeline["run_binding"])
    roi_source = getattr(tiler.tile_locs, "roi_source", None)
    if not isinstance(roi_source, str) or not roi_source:
        raise ValueError("Live tile locations have no valid roi_source.")

    inventory: _HeldInventory | None = None
    temp_fd: int | None = None
    state_fd: int | None = None
    manifest_fd: int | None = None
    temp_name: str | None = None
    committed = False
    try:
        inventory = _HeldInventory(lock_context, pipeline, pipeline_name)
        artifact_hashes = inventory.hashes()
        temp_name = f".{pipeline_name}.roi-provenance-{uuid.uuid4().hex}"
        os.mkdir(temp_name, mode=0o700, dir_fd=inventory.tiling.fd)
        temp_fd = os.open(
            temp_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=inventory.tiling.fd,
        )

        state_fd = os.open(
            "state.npz",
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=temp_fd,
        )
        with os.fdopen(state_fd, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
        os.fsync(state_fd)
        os.lseek(state_fd, 0, os.SEEK_SET)
        npz_digest = hashlib.sha256()
        while chunk := os.read(state_fd, 1024 * 1024):
            npz_digest.update(chunk)
        npz_sha256 = npz_digest.hexdigest()
        os.fchmod(state_fd, 0o444)
        os.fsync(state_fd)

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
                "included": "complete BABY core outputs: segment, profiles, and tracking",
                "global_steps": "forbidden because publication precedes global outputs",
            },
            "artifacts_sha256": artifact_hashes,
        }
        encoded = (
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        manifest_fd = _write_fsynced_at(temp_fd, "manifest.json", encoded)
        os.fchmod(manifest_fd, 0o444)
        os.fsync(manifest_fd)
        os.fchmod(temp_fd, 0o555)
        os.fsync(temp_fd)

        _before_provenance_commit()
        lock_context.verify()
        inventory.revalidate()
        # Same-user malicious mutation after this check is outside the contract;
        # renameat2 still guarantees that a destination can never be replaced.
        _rename_noreplace(
            inventory.tiling.fd,
            temp_name,
            inventory.tiling.fd,
            pipeline_name,
        )
        committed = True
        os.fsync(inventory.tiling.fd)
    finally:
        if state_fd is not None:
            os.close(state_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        if inventory is not None:
            if temp_fd is not None:
                if committed:
                    os.close(temp_fd)
                else:
                    try:
                        _cleanup_temp(inventory.tiling.fd, temp_fd, temp_name)
                    except FileNotFoundError:
                        pass
            elif temp_name is not None and not committed:
                try:
                    os.rmdir(temp_name, dir_fd=inventory.tiling.fd)
                except FileNotFoundError:
                    pass
            inventory.close()

    final_dir = output_path / "tiling" / pipeline_name
    return final_dir / "state.npz", final_dir / "manifest.json"


def write_roi_provenance(
    tiler: Any,
    pipeline: dict,
    output_path: str | Path,
    pipeline_name: str,
) -> tuple[Path, Path]:
    """Acquire the full run lock and publish through the descriptor-bound writer."""
    if not provenance_enabled(pipeline):
        raise ValueError("ROI provenance publication is not enabled.")
    validate_provenance_preflight(pipeline, output_path, pipeline_name)
    lock_context = acquire_provenance_run_lock(output_path)
    try:
        validate_provenance_preflight(
            pipeline, lock_context.lexical_output_path, pipeline_name
        )
        lock_context.verify()
        return _write_roi_provenance_locked(
            tiler, pipeline, pipeline_name, lock_context
        )
    finally:
        lock_context.release()
