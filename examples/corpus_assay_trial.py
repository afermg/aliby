"""Explicit 282-position complement: fresh, pinned, no-retry real assay trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import uuid

import assay_real_trial as common

WORKTREE = Path("/work/users/amunoz/worktrees/aliby-corpus-expansion-20261101")
ANALYSIS = common.ANALYSIS
PIPE_PY = common.PIPE_PY
BABY_PY = common.BABY_PY
BABY_SOURCE = common.BABY_SOURCE
save, load, sha, digest = common.save, common.load, common.sha, common.digest
MODELS = {n: f"yeast-alcatras-brightfield-sCMOS-60x-{n}z" for n in (1, 5)}


def check_scope(root):
    plan = load(root / "plan.json")
    seal = root / "sealed-inputs.json"
    if seal.exists():
        for name, expected in load(seal).items():
            if sha(root / name) != expected:
                raise ValueError("Sealed campaign input changed: " + name)
    if plan["schema"] != "aliby.corpus-expansion.v1" or plan["authorized_sites"] != 282:
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
    keys = [j["key"] for j in plan["jobs"]]
    if len(keys) != 282 or len(set(keys)) != 282 or set(keys) != universe - protected:
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
    # Pin all core modules, not merely imported front-end dispatch functions.
    result["core-tree"] = {
        str(p.relative_to(WORKTREE / "src")): sha(p)
        for p in sorted((WORKTREE / "src").rglob("*.py"))
    }
    return result


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
    raw = load(prepared / "prepared.json")
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

    def preflight(stage, binding):
        ReadOnlyArchive(Path(raw["archive"]), record, ntps).fetch(
            f"{record.record_id}/{record.archive.key}"
        )
        for name, expected in load(prepared / "extracted-inventory.json").items():
            if sha(Path(raw["zarr"]) / name) != expected:
                raise ValueError("Extracted raw changed: " + name)
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
        save(
            attempt / "driver-failure.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if worker is not None:
            common.stop(worker)
        if proc is not None:
            common.stop(proc)
        sock.close()
        save(
            attempt / "shutdown.json",
            {
                "owned_service_stopped": proc is None or proc.poll() is not None,
                "owned_pipeline_stopped": worker is None or worker.poll() is not None,
            },
        )


def snapshot_files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in evidence tree")
        if path.is_file():
            result[str(path.relative_to(directory))] = sha(path)
    return result


def verify_files(directory, inventory):
    for name, expected in inventory.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (directory / path).is_symlink():
            raise ValueError("Unsafe evidence path")
        if sha(directory / path) != expected:
            raise ValueError("Evidence changed: " + name)


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
    if result["backend"] == "async":
        release = Path(result["output"])
        manifest = load(release / "completion.json")
        if (
            manifest["status"] != "computationally_validated"
            or manifest["biological_approval"] is not False
        ):
            raise ValueError("Invalid completion")
        if not manifest["artifacts_sha256"]:
            raise ValueError("Empty completion inventory")
        verify_files(release, manifest["artifacts_sha256"])
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
            save(
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
