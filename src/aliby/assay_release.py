"""Fresh, validation-gated experimental releases; never overwrite or resume."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import uuid

from aliby.assay_runtime import validate_assay_binding
from aliby.async_executor import (
    AsyncCalls,
    run_sync,
    validate_timeout,
    resolve_async_resources,
)
from aliby.executor import compile_pipeline_graph
from aliby.io.roi_provenance import _rename_noreplace, validate_safe_component
from aliby.pipe_core import validate_pipeline


def _write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stamp(details):
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _snapshot_processed(root):
    """Freeze existing artifacts before a validator may add separate evidence."""
    stamps = {}
    for path in root.rglob("*"):
        details = path.lstat()
        name = str(path.relative_to(root))
        if not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)):
            raise ValueError(f"Nonregular artifact in release: {name}")
        stamps[name] = _stamp(details)
    return stamps


def _verify_processed(root, stamps):
    # New root-level reports/directories are allowed, but existing files and
    # directories are immutable. Never silently hash a validator's corrections
    # as though they were the originally computed scientific outputs.
    for name, expected in stamps.items():
        try:
            observed = _stamp((root / name).lstat())
        except FileNotFoundError as error:
            raise RuntimeError(f"Artifact removed during validation: {name}") from error
        if observed != expected:
            raise RuntimeError(f"Artifact changed during validation: {name}")


def _inventory(root):
    hashes, stamps = {}, {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if name == "completion.json":
            raise ValueError("Reserved completion marker exists before publication")
        details = path.lstat()
        stamps[name] = _stamp(details)
        if stat.S_ISDIR(details.st_mode):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            continue
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"Nonregular artifact in release: {name}")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            if _stamp(os.fstat(handle.fileno())) != stamps[name]:
                raise RuntimeError(f"Artifact replaced before hashing: {name}")
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
            os.fsync(handle.fileno())
            if _stamp(os.fstat(handle.fileno())) != stamps[name]:
                raise RuntimeError(f"Artifact changed during hashing: {name}")
    return hashes, stamps


def _verify_inventory(root, stamps):
    observed = {
        str(p.relative_to(root)): _stamp(p.lstat())
        for p in root.rglob("*")
        if p.relative_to(root).as_posix() != "completion.json"
    }
    if observed != stamps:
        raise RuntimeError("Release artifacts changed after validation/hashing")


def _runtime_identity():
    names = (
        "aliby.assay",
        "aliby.assay_runtime",
        "aliby.async_executor",
        "aliby.assay_release",
        "aliby.pipe_core",
        "aliby.pipe",
        "aliby.pipe_baby",
        "aliby.segment.dispatch",
        "extraction.extract",
        "nahual.client.baby",
        "cellpose.models",
    )
    modules = {}
    for name in names:
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        digest = None
        if filename:
            with Path(filename).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        modules[name] = {
            "path": filename,
            "sha256": digest,
            "loaded": module is not None,
        }
    versions = {}
    for name in ("aliby", "numpy", "pyarrow", "cp-measure", "nahual", "cellpose"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "modules": modules, "distributions": versions}


def _accept(receipt, required, label):
    if (
        not isinstance(receipt, dict)
        or receipt.get("accepted") is not True
        or not isinstance(receipt.get("checks"), dict)
        or not receipt["checks"]
        or any(v is not True for v in receipt["checks"].values())
    ):
        raise ValueError(
            f"{label} validator did not explicitly accept every declared check"
        )
    if not required.issubset(receipt["checks"]):
        raise ValueError(
            f"{label} validator omitted required checks: {sorted(required - set(receipt['checks']))}"
        )


async def run_assay_release_async(
    pipeline,
    pipeline_name,
    output_path,
    *,
    validator,
    preflight=None,
    max_workers=None,
    resource_limits=None,
    step_timeout=None,
):
    """Run a bound pipeline in fresh staging and atomically publish on approval.

    ``preflight(stage, binding)`` checks raw/model identities (and BABY ROI
    selection) before any model initializes. ``validator(stage, binding)``
    checks actual model/transport/ROI/profile/tracking outputs. Both must return
    JSON-compatible mappings with ``accepted is True`` and all required
    ``checks`` explicitly True; no defaults or human approval are inferred.
    This function checks nonempty profiles and artifact integrity, not biology.
    Failure/cancellation retains the hidden attempt with failure.json. No retry,
    directory replacement, model download, or production batch is performed.
    """
    if not callable(preflight) or not callable(validator):
        raise ValueError(
            "Explicit preflight and computational release validators are required"
        )
    if "assay_binding" not in pipeline:
        raise ValueError("A reviewed assay binding is required")
    validate_timeout(step_timeout)
    validate_pipeline(pipeline)
    validate_assay_binding(pipeline)
    workers, limits, requirements = resolve_async_resources(
        pipeline, compile_pipeline_graph(pipeline), max_workers, resource_limits
    )
    pipeline_name = validate_safe_component(pipeline_name, "pipeline_name")
    pipeline = deepcopy(pipeline)
    kinds = {
        p.get("segmenter_kwargs", {}).get("kind")
        for n, p in pipeline["steps"].items()
        if n.startswith("segment")
    }
    required_checks = {"model_identity", "input_layout", "semantic_keys"}
    preflight_checks = {"raw_identity", "model_identity"}
    if "nahual_baby" in kinds:
        if pipeline.get("publish_roi_provenance") is not True:
            raise ValueError(
                "BABY assay releases require existing ROI/tracking provenance safeguards"
            )
        required_checks.update({"transport", "roi_geometry", "repeated_masks"})
        preflight_checks.add("roi_selection")
        from aliby.io.roi_provenance import validate_provenance_preflight

        validate_provenance_preflight(pipeline, output_path, pipeline_name)
    output_path = Path(output_path).absolute()
    validate_safe_component(output_path.name, "release name")
    parent = output_path.parent.resolve(strict=True)
    destination = parent / output_path.name
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    stage_name = f".{output_path.name}.attempt-{uuid.uuid4().hex}"
    stage_fd = None
    calls = AsyncCalls(1)
    published = False
    stage = None
    try:
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        stage_fd = os.open(
            stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        stage = Path(f"/proc/self/fd/{stage_fd}")
        lexical_stage = parent / stage_name
        binding = pipeline["assay_binding"]
        _write_json(stage / "assay-binding.json", binding)
        _write_json(
            stage / "execution-options.json",
            {
                "backend": "async",
                "max_workers": workers,
                "resource_limits": limits,
                "requirements": requirements,
                "step_timeout": step_timeout,
                "timepoints_serial": True,
                "automatic_retries": False,
            },
        )
        preflight_receipt = await calls.timed_call(
            step_timeout, preflight, stage, deepcopy(binding)
        )
        _write_json(stage / "preflight.json", preflight_receipt)
        _accept(preflight_receipt, preflight_checks, "Preflight")
        if "nahual_baby" in kinds:
            # Native BABY wrapper owns its existing full-run provenance lock.
            from aliby.pipe_baby import run_pipeline_and_post_async

            run_path = lexical_stage
        else:
            from aliby.pipe import run_pipeline_and_post_async

            run_path = stage
        profiles, _ = await run_pipeline_and_post_async(
            pipeline,
            pipeline_name,
            run_path,
            overwrite=True,
            max_workers=max_workers,
            resource_limits=resource_limits,
            step_timeout=step_timeout,
        )
        if profiles is None or len(profiles) == 0:
            raise ValueError(
                "Empty profiles require review; they are not a biological negative"
            )
        processed = await calls.call(_snapshot_processed, stage)
        receipt = await calls.timed_call(
            step_timeout, validator, stage, deepcopy(binding)
        )
        await calls.call(_verify_processed, stage, processed)
        _write_json(stage / "validation.json", receipt)
        _accept(receipt, required_checks, "Release")
        _write_json(stage / "runtime-identity.json", _runtime_identity())
        hashes, stamps = await calls.call(_inventory, stage)
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        # Revalidate held directory identities immediately before publication.
        held = os.fstat(stage_fd)
        named = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        parent_named = parent.stat()
        parent_held = os.fstat(parent_fd)
        if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino) or (
            parent_named.st_dev,
            parent_named.st_ino,
        ) != (parent_held.st_dev, parent_held.st_ino):
            raise RuntimeError("Release staging/parent directory identity changed")
        _write_json(
            stage / "completion.json",
            {
                "schema": "aliby.assay-release.v1",
                "status": "computationally_validated",
                "biological_approval": False,
                "experimental_backend": "async",
                "pipeline_sha256": binding["pipeline_sha256"],
                "decision_sha256": binding["decision_sha256"],
                "artifacts_sha256": hashes,
            },
        )
        os.fsync(stage_fd)
        _verify_inventory(stage, stamps)
        _verify_processed(stage, processed)
        _rename_noreplace(parent_fd, stage_name, parent_fd, destination.name)
        published = True
        os.fsync(parent_fd)
        return destination
    except BaseException as error:
        # Drain before writing failure evidence: no abandoned worker may keep
        # producing output after the attempt is reported terminal.
        try:
            await calls.close()
        finally:
            if stage is not None and not published:
                try:
                    marker = stage / "completion.json"
                    if marker.exists():
                        marker.unlink()
                    _write_json(
                        stage / "failure.json",
                        {
                            "schema": "aliby.assay-attempt-failure.v1",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "restart": "new_attempt_from_frame_zero",
                            "published": False,
                        },
                    )
                except Exception as evidence_error:
                    # Preserve both the primary error and any pre-existing
                    # evidence when disk/permissions/reserved names prevent IO.
                    error.add_note(
                        f"Could not record failure evidence in {lexical_stage}: "
                        f"{type(evidence_error).__name__}: {evidence_error}"
                    )
        raise
    finally:
        try:
            await calls.close()
        finally:
            if stage_fd is not None:
                os.close(stage_fd)
            os.close(parent_fd)


def run_assay_release(*args, **kwargs):
    """Synchronous CLI entry point; notebooks should await the async version."""
    return run_sync(run_assay_release_async, *args, **kwargs)
