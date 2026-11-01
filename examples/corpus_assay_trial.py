"""Explicit 282-position complement: fresh, pinned, no-retry real assay trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import uuid

import assay_real_trial as common

WORKTREE = Path(
    os.environ.get(
        "ALIBY_SOURCE_WORKTREE",
        "/work/users/amunoz/worktrees/aliby-corpus-expansion-20261101",
    )
)
ANALYSIS = common.ANALYSIS
PIPE_PY = common.PIPE_PY
BABY_PY = common.BABY_PY
BABY_SOURCE = common.BABY_SOURCE
save, load, sha, digest = common.save, common.load, common.sha, common.digest
MODELS = {n: f"yeast-alcatras-brightfield-sCMOS-60x-{n}z" for n in (1, 5)}


def verify_seal(root):
    seal = root / "sealed-inputs.json"
    if not seal.is_file() or seal.is_symlink():
        raise ValueError("Missing or unsafe execution seal")
    inventory = load(seal)
    required = {
        "plan.json",
        "metadata.json",
        "source-identity.json",
        "scripts/corpus_assay_trial.py",
        "scripts/corpus_assay_expansion.py",
        "scripts/assay_real_trial.py",
    }
    required.update(f"evidence/{model}.json" for model in MODELS.values())
    if not isinstance(inventory, dict) or not required.issubset(inventory):
        raise ValueError("Incomplete execution seal")
    verify_files(root, inventory)
    if (
        load(root / "plan.json").get("schema") == "aliby.corpus-continuation.v1"
        and "continuation.json" not in inventory
    ):
        raise ValueError("Incomplete continuation seal")
    return inventory


def check_scope(root, *, require_seal=True):
    if require_seal:
        verify_seal(root)
    plan = load(root / "plan.json")
    if (
        plan["schema"]
        not in ("aliby.corpus-expansion.v1", "aliby.corpus-continuation.v1")
        or plan["authorized_sites"] != 282
    ):
        raise ValueError("Invalid authorized scope")
    if sha(root / "metadata.json") != plan["metadata_sha256"]:
        raise ValueError("Metadata changed")
    prior = Path(
        "/work/users/amunoz/aliby_baby/analyses/corpus-rebuild-20261021/batch-plan-v1.json"
    )
    if Path(plan["previous_plan"]) != prior:
        raise ValueError("Protected plan path changed")
    if sha(prior) != plan["previous_plan_sha256"]:
        raise ValueError("Protected batch identity changed")
    manifest = Path("/work/users/amunoz/aliby_baby/cache/zenodo_manifest.json")
    if sha(manifest) != plan["source_manifest_sha256"]:
        raise ValueError("Primary corpus identity changed")
    protected = {j["record_key"] for j in load(prior)["jobs"]} | {
        "14191444:gcd2_002",
        "14190058:gcd7_020",
    }
    universe = {
        f"{r['deposition']}:{Path(r['key']).stem}"
        for r in load(manifest)
        if r["key"].endswith(".zip")
    }
    expected = universe - protected
    if plan["schema"] == "aliby.corpus-continuation.v1":
        expected = continuation_remaining(root, expected)
    keys = [j["key"] for j in plan["jobs"]]
    if (
        len(keys) != len(expected)
        or len(set(keys)) != len(expected)
        or set(keys) != expected
    ):
        raise ValueError(
            "Not exactly the authorized complement; protected site or duplicate"
        )
    limits = plan["limits"]
    if not (
        1 <= limits["workers"] <= 4
        and 1 <= limits["baby_workers"] <= 4
        and 0 < limits["site_hours"] <= 4
        and 0 < limits["batch_hours"] <= 48
        and limits["minimum_free_gib"] >= 300
    ):
        raise ValueError("Unsafe resource limits")
    return plan


def continuation_remaining(root, authorized):
    receipt = load(root / "continuation.json")
    parent = Path(receipt["parent_root"])
    if parent.resolve() == root.resolve():
        raise ValueError("Continuation cannot reference itself")
    for name, expected in receipt["parent_files"].items():
        if evidence_sha(parent, name) != expected:
            raise ValueError("Predecessor evidence changed: " + name)
    required = {
        "plan.json",
        "sealed-inputs.json",
        "source-identity.json",
        "campaign/terminal.json",
    }
    if not required.issubset(receipt["parent_files"]):
        raise ValueError("Incomplete predecessor identity")
    terminal = load(parent / "campaign/terminal.json")
    if terminal["status"] not in ("completed", "stopped") or terminal.get("current"):
        raise ValueError("Predecessor is not terminal")
    parent_plan = load(parent / "plan.json")
    if (
        parent_plan["schema"] != "aliby.corpus-expansion.v1"
        or {j["key"] for j in parent_plan["jobs"]} != authorized
    ):
        raise ValueError("Predecessor scope differs from authorized complement")
    done = [row["key"] for row in receipt["completed"]]
    deferred = [row["key"] for row in receipt["deferred"]]
    if (
        len(set(done)) != len(done)
        or len(set(deferred)) != len(deferred)
        or set(done) & set(deferred)
        or not (set(done) | set(deferred)) <= authorized
    ):
        raise ValueError("Invalid continuation partition")
    if not {row["key"] for row in terminal["completed"]} <= set(done):
        raise ValueError("Continuation omitted a predecessor completion")
    failed = {row["key"] for row in terminal["failed"]}
    if not set(deferred) <= failed - set(done):
        raise ValueError("Deferred positions lack predecessor failure evidence")
    jobs = {job["key"]: job for job in parent_plan["jobs"]}
    for row in receipt["completed"]:
        attempt = Path(row["attempt"])
        owned = parent / "jobs" / jobs[row["key"]]["id"] / "attempts"
        if (
            attempt.parent != owned
            or Path(row["completion"]) != attempt / "release/completion.json"
        ):
            raise ValueError("Predecessor completion is not owned by its site")
        if (
            sha(row["completion"]) != row["completion_sha256"]
            or sha(attempt / "result.json") != row["result_sha256"]
        ):
            raise ValueError("Predecessor completion changed")
        result = load(attempt / "result.json")
        if (
            result["key"] != row["key"]
            or result["ntps"] != jobs[row["key"]]["ntps"]
            or result["backend"] != "async"
        ):
            raise ValueError("Predecessor result identity mismatch")
    return authorized - set(done) - set(deferred)


def config(root, key):
    from aliby_analyses.image_only_pipeline import load_metadata

    plan = check_scope(root)
    jobs = [j for j in plan["jobs"] if j["key"] == key]
    if len(jobs) != 1:
        raise ValueError("Site not in authorized complement")
    job = jobs[0]
    records, _ = load_metadata(root / "metadata.json")
    record = records[key]
    if (
        job["id"] != key.replace(":", "--")
        or record.modelset not in MODELS.values()
        or record.modelset != job["model"]
    ):
        raise ValueError("Job/model identity mismatch")
    if (
        record.expected_timepoints != job["ntps"]
        or record.channels.index("Brightfield") != job["bf_index"]
    ):
        raise ValueError("Job layout mismatch")
    return plan, job, record


def source_identity():
    result = common.source_identity()
    for path in (Path(__file__), Path(common.__file__)):
        result[path.name] = {"sha256": sha(path)}
    # Identify the code actually imported, never an unrelated hardcoded tree.
    core = Path(result["aliby.pipe_core"]["path"]).resolve().parents[1]
    analysis = Path(result["aliby_analyses.output_validation"]["path"]).resolve().parent
    nahual = Path(result["nahual.client.baby"]["path"]).resolve().parents[1]
    measure = importlib.util.find_spec("cp_measure")
    if measure is None or not measure.submodule_search_locations:
        raise ValueError("Cannot identify measurement source")
    roots = {
        "core-tree": core,
        "analysis-tree": analysis,
        "nahual-tree": nahual,
        "measurement-tree": Path(next(iter(measure.submodule_search_locations))),
    }
    for name, path in roots.items():
        result[name] = python_tree_identity(path)
    result["runtime"] = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "executable_sha256": sha(Path(sys.executable)),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-image",
                "pandas",
                "pyarrow",
                "dask",
                "zarr",
            )
        },
    }
    return result


def python_tree_identity(directory):
    files = sorted(directory.rglob("*.py"))
    if not files:
        raise ValueError("Empty runtime source tree")
    return {
        "path": str(directory.resolve()),
        "files": {
            str(path.relative_to(directory)): evidence_sha(
                directory, str(path.relative_to(directory))
            )
            for path in files
        },
    }


def checked_prepared(directory, key):
    expected = os.environ.get("ALIBY_PREPARED_SHA256")
    if not expected:
        raise ValueError("Prepared checkpoint lacks controller admission receipt")
    data = _evidence_value(directory, "prepared.json", lambda handle: handle.read())
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(
            "Prepared checkpoint differs from controller admission receipt"
        )
    raw = json.loads(data)
    if raw.get("key") != key:
        raise ValueError("Prepared checkpoint site mismatch")
    return raw


def environment(source):
    return common.environment(source)


def reserve(root):
    common.reserve(root)


def prepare(root, key):
    import numpy as np
    from aliby_analyses.channel_z import acquisition_evidence
    from aliby_analyses.extract import extracted_zarr
    from aliby_analyses.image_only_pipeline import ReadOnlyArchive
    from aliby_analyses.pipeline import wrap_zarr_as_group
    from aliby_analyses.roi_selection import make_selection
    from aliby_analyses.time_series_qc import PlaneReader

    reserve(root)
    _, job, record = config(root, key)
    directory = root / "jobs" / job["id"] / "prepared"
    directory.mkdir(parents=True, exist_ok=False)
    archive = Path(job["archive"])
    ReadOnlyArchive(archive, record, job["ntps"]).fetch(
        f"{record.record_id}/{record.archive.key}"
    )
    selection = make_selection(record, archive, job["ntps"])
    save(directory / "roi-selection.json", selection)
    samples = sorted(
        {t for t in (0, 1, 2, 11, job["ntps"] // 2, job["ntps"] - 1) if t < job["ntps"]}
    )
    channels = []
    reader = PlaneReader(archive)
    try:
        for index, name in enumerate(record.channels):
            metadata = acquisition_evidence(archive, record.site, name)
            if (
                metadata["declared_planes"] != job["declared_planes"][name]
                or metadata["conflicts"]
            ):
                raise ValueError("Channel acquisition evidence changed or conflicts")
            if job["acquisition_sha256"] not in [
                e.get("sha256") for e in metadata["evidence"]
            ]:
                raise ValueError("Acquisition sidecar changed")
            sampled = []
            for tp in samples:
                states, hashes = [], []
                for z in range(record.shape[2]):
                    pixels = reader.plane(tp, index, z)
                    states.append("nonzero" if np.any(pixels) else "zero")
                    hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                sampled.append(
                    {
                        "timepoint": tp,
                        "states": states,
                        "evidence_id": digest(hashes),
                        "plane_sha256": hashes,
                    }
                )
            channels.append(
                {"index": index, "name": name, "metadata": metadata, "samples": sampled}
            )
    finally:
        reader.close()
    save(directory / "channel-evidence.json", channels)
    with extracted_zarr(archive, record, root / "extracted", keep=True) as zarr:
        wrap_zarr_as_group(zarr)
        inventory = {
            str(p.relative_to(zarr)): sha(p)
            for p in sorted(zarr.rglob("*"))
            if p.is_file()
        }
        save(directory / "extracted-inventory.json", inventory)
        save(
            directory / "prepared.json",
            {
                "key": key,
                "archive": str(archive),
                "archive_sha256": sha(archive),
                "zarr": str(zarr),
                "plan_sha256": sha(root / "plan.json"),
                "metadata_sha256": sha(root / "metadata.json"),
                "roi_sha256": sha(directory / "roi-selection.json"),
                "channels_sha256": sha(directory / "channel-evidence.json"),
                "inventory_sha256": sha(directory / "extracted-inventory.json"),
            },
        )
    print(
        json.dumps(
            {
                "key": key,
                "retained": len(selection["selected_original_roi_ids"]),
                "detected": len(selection["original_centres_yx"]),
            }
        ),
        flush=True,
    )


def attest(address, model):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            address + "/provenance", timeout=(3, 5), allow_redirects=False
        )
        if response.status_code != 200:
            raise ValueError("Unexpected provenance HTTP status")
        result = response.json()
    if (
        result["api_version"] != "baby-http-v1"
        or result["service_revision"] != model["service_revision"]
        or result["models"].get(model["modelset"], {}).get("sha256") != model["sha256"]
    ):
        raise ValueError("Owned model/service attestation mismatch")
    return result


def service(model, fd, port):
    from baby import server

    sock = socket.socket(fileno=fd)
    original = server.web.run_app

    def owned(app, **kwargs):
        kwargs.pop("port", None)
        kwargs.pop("host", None)
        return original(app, sock=sock, **kwargs)

    server.web.run_app = owned
    server.main(["--port", str(port), "--njobs", "4", "--modelset", model])


def run(root, key, ntps, backend, address, attempt):
    from aliby.assay import (
        AssayRequest,
        ChannelEvidence,
        ChannelInput,
        MeasurementRequest,
        ModelCapability,
        PlaneSample,
        SegmentationRequest,
        decide_assay,
    )
    from aliby.assay_runtime import bind_assay_pipeline
    from aliby.assay_release import run_assay_release
    from aliby.pipe_baby import run_pipeline_and_post
    from aliby_analyses.dataset import validate_zarr
    from aliby_analyses.image_only_pipeline import (
        ReadOnlyArchive,
        checked_baby_http,
        cp_measure_identity,
        segmentation_review,
    )
    from aliby_analyses.output_validation import (
        RoiExpectation,
        write_run_metadata,
        validate_site_output,
        summary_dict,
    )
    from aliby_analyses.pipeline import build_pipeline
    from aliby_analyses.roi_selection import (
        validate_selection,
        verify_executed_selection,
    )

    reserve(root)
    plan, job, record = config(root, key)
    if ntps not in (3, 12, job["ntps"]) or ntps > job["ntps"]:
        raise ValueError("Unapproved prefix")
    prepared = root / "jobs" / job["id"] / "prepared"
    raw = checked_prepared(prepared, key)
    for filename, hashkey in [
        ("roi-selection.json", "roi_sha256"),
        ("channel-evidence.json", "channels_sha256"),
        ("extracted-inventory.json", "inventory_sha256"),
    ]:
        if sha(prepared / filename) != raw[hashkey]:
            raise ValueError("Prepared input evidence changed")
    if (
        sha(root / "plan.json") != raw["plan_sha256"]
        or sha(root / "metadata.json") != raw["metadata_sha256"]
    ):
        raise ValueError("Prepared configuration changed")
    model = load(root / "evidence" / f"{record.modelset}.json")
    depth = job["declared_planes"]["Brightfield"]
    if model["actual_input_shape"][-1] != depth or depth not in (1, 5):
        raise ValueError("Actual loaded model/input mismatch")
    selection = load(prepared / "roi-selection.json")
    validate_selection(selection, record, ntps)
    channel_records = load(prepared / "channel-evidence.json")
    channels = []
    for item in channel_records:
        meta = item["metadata"]
        warnings = list(meta["warnings"])
        if key in plan["channel_order_overrides"]:
            warnings.append(
                "Explicit BF/GFP stored-axis override; acquisition-table ordering conflict retained in metadata/proposals."
            )
        channels.append(
            ChannelEvidence(
                index=item["index"],
                name=item["name"],
                modality="brightfield"
                if item["name"] == "Brightfield"
                else "fluorescence",
                stored_z=record.shape[2],
                samples=tuple(
                    PlaneSample(s["timepoint"], tuple(s["states"]), s["evidence_id"])
                    for s in item["samples"]
                    if s["timepoint"] < ntps
                ),
                acquired_z=tuple(range(meta["declared_planes"]))
                if meta["declared_planes"]
                else None,
                metadata_evidence=tuple(
                    e["sha256"]
                    for e in meta["evidence"]
                    if e.get("status") == "matched"
                ),
                metadata_conflicts=tuple(meta["conflicts"]),
                metadata_warnings=tuple(warnings),
            )
        )
    capability = ModelCapability(
        record.modelset,
        "sha256:" + model["sha256"],
        ("brightfield",),
        depth,
        masks="overlapping",
        stateful=True,
        assays=("yeast_alcatras",),
        targets=("cell",),
    )
    measurements = []
    for i, name in enumerate(record.fluorescence_channels):
        z = tuple(range(job["declared_planes"][name]))
        selection_input = ChannelInput(
            record.channels.index(name), z, "max" if len(z) > 1 else "stack"
        )
        measurements.append(
            MeasurementRequest(
                name,
                "cell",
                (selection_input,),
                ("intensity", "sizeshape") if i == 0 else ("intensity",),
            )
        )
    request = AssayRequest(
        key,
        raw["archive_sha256"],
        ntps,
        tuple(channels),
        (
            SegmentationRequest(
                "cell",
                (ChannelInput(job["bf_index"], tuple(range(depth))),),
                record.modelset,
            ),
        ),
        tuple(measurements),
        assay_type="yeast_alcatras",
    )
    decision = decide_assay(request, (capability,))
    save(attempt / "decision.json", decision.to_dict())
    decision.require_ready()
    sources = source_identity()
    if sources != load(root / "source-identity.json"):
        raise ValueError("Pinned runtime source changed")
    software = {
        "sources": sources,
        "cp_measure": cp_measure_identity(),
        "service": attest(address, model),
        "model_actual_input_shape": model["actual_input_shape"],
        "plan_sha256": sha(root / "plan.json"),
    }
    save(attempt / "software.json", software)
    fingerprint = digest(
        {
            "decision": decision.fingerprint,
            "sources": sources,
            "roi": raw["roi_sha256"],
            "backend": backend,
        }
    )
    expectation = RoiExpectation.from_record(
        record,
        work_fingerprint=fingerprint,
        timepoints=ntps,
        segmentation_index=job["bf_index"],
        segmentation_name="Brightfield",
        fluorescence_names=record.fluorescence_channels,
    )
    schema = validate_zarr(Path(raw["zarr"]), record, requested_ntps=ntps)
    pipeline = build_pipeline(
        Path(raw["zarr"]),
        record=record,
        schema=schema,
        address=address,
        roi_expectation=expectation,
    )
    pipeline["steps"]["tile"]["roi_indices"] = selection["selected_original_roi_ids"]
    pipeline["steps"]["extract_cell"]["kwargs"] = {
        "ncores": None,
        "progress_bar": False,
    }
    pipeline["retain"] = {
        "tile": 1
    }  # Frame barrier permits bounded raw history; feature/lineage histories remain complete.
    bound = bind_assay_pipeline(
        pipeline, decision, model_identities={record.modelset: capability.identity}
    )
    save(attempt / "binding.json", bound["assay_binding"])
    audit = attempt / "baby-http.jsonl"

    def verify_runtime():
        if checked_prepared(prepared, key) != raw or source_identity() != sources:
            raise ValueError("Prepared checkpoint/runtime changed during attempt")
        verify_seal(root)

    def preflight(stage, binding):
        verify_runtime()
        ReadOnlyArchive(Path(raw["archive"]), record, ntps).fetch(
            f"{record.record_id}/{record.archive.key}"
        )
        if sha(prepared / "extracted-inventory.json") != raw["inventory_sha256"]:
            raise ValueError("Prepared inventory changed")
        verify_files(Path(raw["zarr"]), load(prepared / "extracted-inventory.json"))
        validate_selection(selection, record, ntps)
        save(stage / "roi-selection.json", selection)
        save(stage / "channel-evidence.json", channel_records)
        return {
            "accepted": True,
            "checks": {
                "raw_identity": True,
                "model_identity": True,
                "roi_selection": True,
            },
            "service": attest(address, model),
            "raw_sha256": raw["archive_sha256"],
        }

    def validator(stage, binding):
        verify_runtime()
        stage = stage.resolve(strict=True)
        verify_executed_selection(stage, record.site, selection, ntps)
        write_run_metadata(stage, expectation=expectation, software_stack=software)
        summary = validate_site_output(
            stage, expectation=expectation, software_stack=software
        )
        if not summary.profile_rows:
            raise ValueError(
                "Empty output requires computational review, not biological-negative labeling"
            )
        review = segmentation_review(stage, Path(raw["archive"]), record, ntps)
        save(stage / "segmentation-review.json", review)
        if review["state"] != "no_repeat_gate_trigger":
            raise ValueError(
                "Repeat/empty integrity gate: " + str(review["review_reasons"])
            )
        events = [json.loads(line) for line in audit.read_text().splitlines()]
        posts = [
            e
            for e in events
            if e["method"] == "post" and e["event"] == "request_started"
        ]
        responses = [
            e
            for e in events
            if e["method"] == "post" and e["event"] == "response_received"
        ]
        dims = [len(selection["selected_original_roi_ids"]), 117, 117, depth]
        if (
            len(posts) != ntps
            or len(responses) != ntps
            or any(e["status"] != 200 for e in responses)
        ):
            raise ValueError("POST count/status mismatch")
        if any(e["submitted_dimensions"] != dims for e in posts):
            raise ValueError("Submitted layout mismatch")
        if len({e["submitted_pixel_sha256"] for e in posts}) != ntps:
            raise ValueError("Repeated submitted inputs require review")
        if any(e["event"] == "request_failed" for e in events):
            raise ValueError("Failed HTTP event")
        shutil.copyfile(audit, stage / "baby-http.jsonl")
        receipt = {
            "accepted": True,
            "checks": {
                n: True
                for n in (
                    "raw_identity",
                    "model_identity",
                    "input_layout",
                    "semantic_keys",
                    "transport",
                    "roi_geometry",
                    "repeated_masks",
                )
            },
            "summary": summary_dict(summary),
            "service": attest(address, model),
            "submitted_dimensions": dims,
            "biological_approval": False,
            "human_reviewed": False,
        }
        save(attempt / "validated.json", receipt)
        return receipt

    with checked_baby_http(audit):
        if backend == "async":
            output = run_assay_release(
                bound,
                record.site,
                attempt / "release",
                preflight=preflight,
                validator=validator,
                max_workers=4,
                resource_limits={"cpu": 4},
                step_timeout=600,
            )
        else:
            output = attempt / "sequential-diagnostic"
            output.mkdir()
            save(output / "preflight.json", preflight(output, bound["assay_binding"]))
            run_pipeline_and_post(bound, record.site, output, backend="sequential")
            save(output / "validation.json", validator(output, bound["assay_binding"]))
            save(
                output / "diagnostic-completion.json",
                {
                    "schema": "aliby.assay-diagnostic.v1",
                    "status": "computationally_validated",
                    "decision_sha256": decision.fingerprint,
                    "biological_approval": False,
                    "artifacts_sha256": snapshot_files(output),
                },
            )
    save(
        attempt / "result.json",
        {
            "key": key,
            "site": record.site,
            "backend": backend,
            "ntps": ntps,
            "status": "computationally_validated",
            "output": str(output),
            "decision_sha256": decision.fingerprint,
            "source_identity": sources,
            "biological_approval": False,
        },
    )


def launch(root, key, ntps, backend):
    import requests

    reserve(root)
    _, job, record = config(root, key)
    checked_prepared(root / "jobs" / job["id"] / "prepared", key)
    if ntps not in (3, 12, job["ntps"]):
        raise ValueError("Unapproved prefix")
    attempt = (
        root
        / "jobs"
        / job["id"]
        / "attempts"
        / f"{ntps:04d}-{backend}-{uuid.uuid4().hex[:12]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    model = load(root / "evidence" / f"{record.modelset}.json")
    started = time.monotonic()
    budget = 4 * 3600 if ntps == job["ntps"] else 3600
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    save(
        attempt / "attempt.json",
        {
            "key": key,
            "site": record.site,
            "ntps": ntps,
            "backend": backend,
            "address": address,
            "budget_seconds": budget,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    proc = worker = None
    try:
        with (
            (attempt / "service.log").open("xb") as sl,
            (attempt / "pipeline.log").open("xb") as pl,
        ):
            env = environment(BABY_SOURCE)
            env["BABY_MODELSETS_PATH"] = str(root / "models")
            proc = subprocess.Popen(
                [
                    str(BABY_PY),
                    __file__,
                    str(root),
                    "service",
                    "--model",
                    record.modelset,
                    "--fd",
                    str(sock.fileno()),
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=sl,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(sock.fileno(),),
            )
            sock.close()
            save(attempt / "service-process.json", {"pid": proc.pid})
            until = time.monotonic() + 180
            while True:
                reserve(root)
                if proc.poll() is not None or time.monotonic() > until:
                    raise RuntimeError("Owned service startup failed")
                try:
                    receipt = attest(address, model)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    time.sleep(0.5)
            save(attempt / "service-attestation.json", receipt)
            worker = subprocess.Popen(
                [
                    str(PIPE_PY),
                    __file__,
                    str(root),
                    "run",
                    "--key",
                    key,
                    "--ntps",
                    str(ntps),
                    "--backend",
                    backend,
                    "--address",
                    address,
                    "--attempt",
                    str(attempt),
                ],
                env=environment(f"{WORKTREE / 'src'}:{ANALYSIS}"),
                stdout=pl,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            save(attempt / "pipeline-process.json", {"pid": worker.pid})
            while worker.poll() is None:
                reserve(root)
                if time.monotonic() - started > budget:
                    raise TimeoutError("Site attempt deadline exceeded")
                time.sleep(1)
            if worker.returncode:
                raise RuntimeError(
                    f"Pipeline exited {worker.returncode}; evidence={attempt}"
                )
        save(
            attempt / "terminal.json",
            {"status": "passed", "elapsed_seconds": time.monotonic() - started},
        )
        print(json.dumps({"attempt": str(attempt), "status": "passed"}), flush=True)
    except BaseException as error:
        save_failure(
            error,
            attempt / "driver-failure.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        cleanup_owned(worker, proc, sock, attempt, primary=sys.exception())


def save_failure(primary, path, payload):
    try:
        save(path, payload)
    except Exception as error:
        primary.add_note(f"Could not record failure evidence at {path}: {error!r}")


def cleanup_owned(worker, service_process, sock, attempt, *, primary=None):
    errors = []
    receipt = {}
    for name, process in (("pipeline", worker), ("service", service_process)):
        try:
            if process is not None:
                common.stop(process)
        except Exception as error:
            errors.append(error)
        try:
            receipt[f"owned_{name}_stopped"] = (
                process is None or process.poll() is not None
            )
        except Exception as error:
            receipt[f"owned_{name}_stopped"] = False
            errors.append(error)
    try:
        sock.close()
    except Exception as error:
        errors.append(error)
    try:
        save(attempt / "shutdown.json", receipt)
    except Exception as error:
        errors.append(error)
    if errors:
        cause = primary if primary is not None else errors[0]
        for error in errors:
            cause.add_note(f"Owned cleanup/evidence failure: {error!r}")
        if primary is None:
            raise cause


def snapshot_files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in evidence tree")
        if path.is_file():
            result[str(path.relative_to(directory))] = sha(path)
    return result


def evidence_sha(directory, name):
    return _evidence_value(
        directory,
        name,
        lambda handle: hashlib.file_digest(handle, "sha256").hexdigest(),
    )


def _evidence_value(directory, name, read_value):
    if not isinstance(name, str):
        raise ValueError("Unsafe evidence path")
    path = Path(name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError("Unsafe evidence path")
    descriptors = []
    try:
        descriptors.append(
            os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        )
        for part in path.parts[:-1]:
            descriptors.append(
                os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptors[-1],
                )
            )
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptors[-1],
        )
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Unsafe nonregular evidence file")
            value = read_value(handle)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("Evidence changed while hashing")
            return value
    except OSError as error:
        raise ValueError("Unsafe or missing evidence path: " + name) from error
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def verify_files(directory, inventory, *, allowed_extra=None):
    if not isinstance(inventory, dict):
        raise ValueError("Invalid evidence inventory")
    for name, expected in inventory.items():
        if evidence_sha(directory, name) != expected:
            raise ValueError("Evidence changed: " + name)
    if allowed_extra is not None:
        paths = list(directory.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise ValueError("Unsafe evidence symlink")
        actual = {str(p.relative_to(directory)) for p in paths if p.is_file()}
        if actual != set(inventory) | set(allowed_extra):
            raise ValueError("Output inventory does not match directory")


def verify_attempt(attempt):
    result = load(attempt / "result.json")
    receipt = load(attempt / "validated.json")
    required = {
        "raw_identity",
        "model_identity",
        "input_layout",
        "semantic_keys",
        "transport",
        "roi_geometry",
        "repeated_masks",
    }
    if (
        result["status"] != "computationally_validated"
        or result.get("biological_approval") is not False
        or receipt["accepted"] is not True
        or not required.issubset(receipt["checks"])
        or any(v is not True for v in receipt["checks"].values())
    ):
        raise ValueError("Unvalidated attempt")
    backend = result["backend"]
    if backend not in ("async", "sequential"):
        raise ValueError("Unknown attempt backend")
    release = Path(result["output"])
    owned = attempt / ("release" if backend == "async" else "sequential-diagnostic")
    if release.absolute() != owned.absolute() or owned.is_symlink():
        raise ValueError("Attempt output is not owned by this attempt")
    manifest_name = (
        "completion.json" if backend == "async" else "diagnostic-completion.json"
    )
    evidence_sha(release, manifest_name)
    manifest = load(release / manifest_name)
    if (
        manifest["status"] != "computationally_validated"
        or manifest["biological_approval"] is not False
        or manifest.get("decision_sha256") != result.get("decision_sha256")
        or not result.get("decision_sha256")
    ):
        raise ValueError("Invalid completion identity")
    if not manifest["artifacts_sha256"]:
        raise ValueError("Empty completion inventory")
    verify_files(release, manifest["artifacts_sha256"], allowed_extra={manifest_name})
    shutdown = load(attempt / "shutdown.json")
    if (
        shutdown.get("owned_service_stopped") is not True
        or shutdown.get("owned_pipeline_stopped") is not True
    ):
        raise ValueError("Owned processes not stopped")
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "command", choices=("prepare", "service", "launch", "run", "compare")
    )
    parser.add_argument("--key")
    parser.add_argument("--ntps", type=int)
    parser.add_argument("--backend", choices=("sequential", "async"))
    parser.add_argument("--model", choices=tuple(MODELS.values()))
    parser.add_argument("--fd", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--address")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "service":
        service(args.model, args.fd, args.port)
    elif args.command == "prepare":
        prepare(args.root, args.key)
    elif args.command == "compare":
        common.compare(args.left, args.right, args.report)
    elif args.command == "launch":

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Termination {signum}")

        signal.signal(signal.SIGTERM, terminate)
        launch(args.root, args.key, args.ntps, args.backend)
    else:
        try:
            run(
                args.root, args.key, args.ntps, args.backend, args.address, args.attempt
            )
        except BaseException as error:
            save_failure(
                error,
                args.attempt / "worker-failure.json",
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
