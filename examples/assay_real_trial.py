"""Bounded real-data trials; explicit CLI stages, fresh sessions, no retries.

Uses pinned existing analysis validators without changing their production runner.
Only the 19 previously held experiment19307 sites are eligible. Every output is
beneath a fresh trial root. This is computational validation, not biological approval.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import uuid

MODEL = "yeast-alcatras-brightfield-sCMOS-60x-1z"
REPAIR = Path("/work/users/amunoz/aliby_baby/analyses/gcd2-repair-20261021")
PIPE_PY = REPAIR / "runtime/pipeline-venv/bin/python"
BABY_PY = REPAIR / "runtime/baby-venv/bin/python"
WORKTREE = Path("/work/users/amunoz/worktrees/aliby-assay-async-20261027")
ANALYSIS = Path("/work/users/amunoz/worktrees/aliby-channel-z-20261027/src")
BABY_SOURCE = Path("/work/users/amunoz/projects/baby/python")
METADATA = Path(
    "/work/users/amunoz/aliby_baby/analyses/corpus-rebuild-20261021/metadata-v1.json"
)
HELD = tuple(f"phluorin_dot6_{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)) + (
    "phluorin_nrg1_014",
    "phluorin_nrg1_017",
    "phluorin_nrg1_018",
    "phluorin_nrg1_s14",
    "phluorin_tod6_019",
    "phluorin_tod6_020",
    "phluorin_tod6_022",
    "phluorin_tod6_023",
)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def record_for(site):
    from aliby_analyses.image_only_pipeline import load_metadata

    if site not in HELD:
        raise ValueError("Site outside approved held-site scope")
    records, _ = load_metadata(METADATA)
    return replace(records[f"14189201:{site}"], modelset=MODEL)


def reserve(root):
    if (root / "STOP").exists():
        raise RuntimeError("Operator STOP present")
    if shutil.disk_usage(root).free < 300 * 1024**3:
        raise RuntimeError("300 GiB reserve reached")


def environment(source):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(source),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        TF_NUM_INTRAOP_THREADS="4",
        TF_NUM_INTEROP_THREADS="2",
    )
    return env


def source_identity():
    import aliby.assay_release
    import aliby.assay_runtime
    import aliby.async_executor
    import aliby.pipe_core
    import aliby.pipe_baby
    import aliby.segment.dispatch
    import aliby_analyses.output_validation
    import aliby_analyses.image_only_pipeline
    import nahual.client.baby

    modules = (
        aliby.assay_release,
        aliby.assay_runtime,
        aliby.async_executor,
        aliby.pipe_core,
        aliby.pipe_baby,
        aliby.segment.dispatch,
        aliby_analyses.output_validation,
        aliby_analyses.image_only_pipeline,
        nahual.client.baby,
    )
    return {
        m.__name__: {"path": m.__file__, "sha256": sha(m.__file__)} for m in modules
    }


def prepare(root, site):
    import numpy as np
    from aliby_analyses.channel_z import acquisition_evidence
    from aliby_analyses.extract import extracted_zarr
    from aliby_analyses.image_only_pipeline import ReadOnlyArchive
    from aliby_analyses.roi_selection import make_selection
    from aliby_analyses.time_series_qc import PlaneReader
    from aliby_analyses.pipeline import wrap_zarr_as_group

    reserve(root)
    record = record_for(site)
    directory = root / "prepared" / site
    directory.mkdir(parents=True, exist_ok=False)
    archive = Path("/work/users/amunoz/aliby_baby/cache/14189201") / record.archive.key
    ReadOnlyArchive(archive, record, record.expected_timepoints).fetch(
        f"14189201/{record.archive.key}"
    )
    selection = make_selection(record, archive, record.expected_timepoints)
    save(directory / "roi-selection.json", selection)
    channels = []
    reader = PlaneReader(archive)
    try:
        for index, name in enumerate(record.channels):
            meta = acquisition_evidence(archive, site, name)
            samples = []
            for tp in (0, 1, 2, 11, 30, 90, 179):
                states, hashes = [], []
                for z in range(record.shape[2]):
                    pixels = reader.plane(tp, index, z)
                    states.append("nonzero" if np.any(pixels) else "zero")
                    hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                samples.append(
                    {
                        "timepoint": tp,
                        "states": states,
                        "evidence_id": digest(hashes),
                        "plane_sha256": hashes,
                    }
                )
            channels.append(
                {"index": index, "name": name, "metadata": meta, "samples": samples}
            )
    finally:
        reader.close()
    save(directory / "channel-evidence.json", channels)
    with extracted_zarr(archive, record, root / "extracted", keep=True) as zarr:
        wrap_zarr_as_group(zarr)
        inventory = {
            str(p.relative_to(zarr)): {"sha256": sha(p), "size": p.stat().st_size}
            for p in sorted(zarr.rglob("*"))
            if p.is_file()
        }
        save(directory / "extracted-inventory.json", inventory)
        save(
            directory / "prepared.json",
            {
                "site": site,
                "archive": str(archive),
                "archive_md5": record.archive.md5,
                "archive_sha256": sha(archive),
                "zarr": str(zarr),
                "source_metadata": str(METADATA),
                "source_metadata_sha256": sha(METADATA),
                "model_override": MODEL,
                "correction": "Historical five-plane BF interpretation is superseded by channel-specific acquisition and pixel evidence. BF/GFP acquired Z0; mCherry acquired Z0-4. Physical calibration unverified.",
                "roi_selection_sha256": sha(directory / "roi-selection.json"),
                "channel_evidence_sha256": sha(directory / "channel-evidence.json"),
                "extracted_inventory_sha256": sha(
                    directory / "extracted-inventory.json"
                ),
            },
        )
    print(
        json.dumps(
            {
                "prepared": site,
                "retained_rois": len(selection["selected_original_roi_ids"]),
                "detected_rois": len(selection["original_centres_yx"]),
                "excluded": selection["excluded_rois"],
            }
        ),
        flush=True,
    )


def service(root, fd, port, attempt):
    from baby import server

    sock = socket.socket(fileno=fd)
    original = server.web.run_app

    def owned(app, **kwargs):
        kwargs.pop("port", None)
        kwargs.pop("host", None)
        return original(app, sock=sock, **kwargs)

    server.web.run_app = owned
    server.main(["--port", str(port), "--njobs", "4", "--modelset", MODEL])


def attest(address, expected):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            address + "/provenance", timeout=(3, 5), allow_redirects=False
        )
        response.raise_for_status()
        result = response.json()
    if (
        result["service_revision"] != expected["service_revision"]
        or result["api_version"] != "baby-http-v1"
        or result["models"][MODEL]["sha256"] != expected["sha256"]
    ):
        raise ValueError("Owned service/model attestation mismatch")
    return result


def run(root, site, ntps, backend, address, attempt):
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

    if ntps not in (3, 12, 180):
        raise ValueError("Only predeclared 3/12/180-frame stages permitted")
    reserve(root)
    record = record_for(site)
    prepared = root / "prepared" / site
    raw = load(prepared / "prepared.json")
    model = load(root / "evidence/loaded-model.json")
    if sha(METADATA) != raw["source_metadata_sha256"]:
        raise ValueError("Source metadata changed")
    selection = load(prepared / "roi-selection.json")
    if sha(prepared / "roi-selection.json") != raw["roi_selection_sha256"]:
        raise ValueError("ROI selection changed")
    validate_selection(selection, record, ntps)
    channel_records = load(prepared / "channel-evidence.json")
    if sha(prepared / "channel-evidence.json") != raw["channel_evidence_sha256"]:
        raise ValueError("Channel evidence changed")
    channels = []
    for item in channel_records:
        meta = item["metadata"]
        channels.append(
            ChannelEvidence(
                index=item["index"],
                name=item["name"],
                modality="brightfield" if item["index"] == 2 else "fluorescence",
                stored_z=5,
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
                metadata_warnings=tuple(meta["warnings"]),
            )
        )
    capability = ModelCapability(
        MODEL,
        "sha256:" + model["sha256"],
        ("brightfield",),
        1,
        masks="overlapping",
        stateful=True,
        assays=("yeast_alcatras",),
        targets=("cell",),
    )
    request = AssayRequest(
        record.key,
        raw["archive_sha256"],
        ntps,
        tuple(channels),
        (SegmentationRequest("cell", (ChannelInput(2, (0,)),), MODEL),),
        (
            MeasurementRequest(
                "mCherry",
                "cell",
                (ChannelInput(0, (0, 1, 2, 3, 4), "max"),),
                ("intensity", "sizeshape"),
            ),
            MeasurementRequest("GFP", "cell", (ChannelInput(1, (0,)),), ("intensity",)),
        ),
        assay_type="yeast_alcatras",
    )
    decision = decide_assay(request, (capability,))
    save(attempt / "decision.json", decision.to_dict())
    decision.require_ready()
    source = source_identity()
    software = {
        "sources": source,
        "cp_measure": cp_measure_identity(),
        "service": attest(address, model),
    }
    save(attempt / "software.json", software)
    fingerprint = digest(
        {
            "decision": decision.fingerprint,
            "sources": source,
            "roi": raw["roi_selection_sha256"],
            "backend": backend,
        }
    )
    schema = validate_zarr(Path(raw["zarr"]), record, requested_ntps=ntps)
    expectation = RoiExpectation.from_record(
        record,
        work_fingerprint=fingerprint,
        timepoints=ntps,
        segmentation_index=2,
        segmentation_name="Brightfield",
        fluorescence_names=("mCherry", "GFP"),
    )
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
    bound = bind_assay_pipeline(
        pipeline, decision, model_identities={MODEL: capability.identity}
    )
    save(attempt / "binding.json", bound["assay_binding"])
    audit = attempt / "baby-http.jsonl"

    def preflight(stage, binding):
        ReadOnlyArchive(Path(raw["archive"]), record, ntps).fetch(
            f"14189201/{record.archive.key}"
        )
        if (
            sha(prepared / "extracted-inventory.json")
            != raw["extracted_inventory_sha256"]
        ):
            raise ValueError("Extracted inventory changed")
        for name, info in load(prepared / "extracted-inventory.json").items():
            if sha(Path(raw["zarr"]) / name) != info["sha256"]:
                raise ValueError("Extracted raw chunk changed: " + name)
        receipt = attest(address, model)
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
            "service": receipt,
            "archive_sha256": raw["archive_sha256"],
        }

    def validator(stage, binding):
        # Existing analysis validation uses no-follow directory opens. Resolve
        # our own held fd path to the owned lexical directory for that validator.
        stage = stage.resolve(strict=True)
        verify_executed_selection(stage, site, selection, ntps)
        write_run_metadata(stage, expectation=expectation, software_stack=software)
        summary = validate_site_output(
            stage, expectation=expectation, software_stack=software
        )
        if not summary.profile_rows:
            raise ValueError("Empty profiles are not a biological negative")
        review = segmentation_review(stage, Path(raw["archive"]), record, ntps)
        save(stage / "segmentation-review.json", review)
        if review["state"] != "no_repeat_gate_trigger":
            raise ValueError(
                "Segmentation repeat/empty gate: " + str(review["review_reasons"])
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
        if (
            len(posts) != ntps
            or len(responses) != ntps
            or any(e["status"] != 200 for e in responses)
        ):
            raise ValueError("POST count/status mismatch")
        dims = [len(selection["selected_original_roi_ids"]), 117, 117, 1]
        if any(e["submitted_dimensions"] != dims for e in posts):
            raise ValueError("Submitted channel/Z geometry mismatch")
        if len({e["submitted_pixel_sha256"] for e in posts}) != ntps:
            raise ValueError("Repeated submitted input requires review")
        if any(e["event"] == "request_failed" for e in events):
            raise ValueError("Failed transport recorded")
        shutil.copyfile(audit, stage / "baby-http.jsonl")
        receipt = {
            "accepted": True,
            "checks": {
                key: True
                for key in (
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
                site,
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
            run_pipeline_and_post(bound, site, output, backend="sequential")
            save(output / "validation.json", validator(output, bound["assay_binding"]))
    save(
        attempt / "result.json",
        {
            "status": "computationally_validated",
            "backend": backend,
            "output": str(output),
            "ntps": ntps,
            "decision_sha256": decision.fingerprint,
            "source_identity": source,
            "biological_approval": False,
        },
    )


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def launch(root, site, ntps, backend):
    reserve(root)
    record_for(site)
    attempt = root / "attempts" / f"{site}-{ntps:03d}-{backend}-{uuid.uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    expected = load(root / "evidence/loaded-model.json")
    proc = worker = None
    started = time.monotonic()
    budget = 3600 if ntps < 180 else 4 * 3600
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    save(
        attempt / "attempt.json",
        {
            "site": site,
            "ntps": ntps,
            "backend": backend,
            "address": address,
            "budget_seconds": budget,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    try:
        with (
            (attempt / "service.log").open("xb") as service_log,
            (attempt / "pipeline.log").open("xb") as pipeline_log,
        ):
            env = environment(BABY_SOURCE)
            env["BABY_MODELSETS_PATH"] = str(root / "models")
            proc = subprocess.Popen(
                [
                    str(BABY_PY),
                    __file__,
                    str(root),
                    "service",
                    "--fd",
                    str(sock.fileno()),
                    "--port",
                    str(port),
                    "--attempt",
                    str(attempt),
                ],
                stdout=service_log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                pass_fds=(sock.fileno(),),
            )
            sock.close()
            save(attempt / "service-process.json", {"pid": proc.pid})
            until = time.monotonic() + 180
            while True:
                reserve(root)
                if proc.poll() is not None or time.monotonic() > until:
                    raise RuntimeError("Owned service failed startup")
                try:
                    attestation = attest(address, expected)
                    break
                except (OSError, ConnectionError):
                    time.sleep(0.5)
                except Exception as error:
                    import requests

                    if isinstance(error, requests.ConnectionError):
                        time.sleep(0.5)
                    else:
                        raise
            save(attempt / "service-attestation.json", attestation)
            env = environment(f"{WORKTREE / 'src'}:{ANALYSIS}")
            worker = subprocess.Popen(
                [
                    str(PIPE_PY),
                    __file__,
                    str(root),
                    "run",
                    "--site",
                    site,
                    "--ntps",
                    str(ntps),
                    "--backend",
                    backend,
                    "--address",
                    address,
                    "--attempt",
                    str(attempt),
                ],
                stdout=pipeline_log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            save(attempt / "pipeline-process.json", {"pid": worker.pid})
            while worker.poll() is None:
                reserve(root)
                if time.monotonic() - started > budget:
                    raise TimeoutError("Attempt wall budget exhausted")
                time.sleep(1)
            if worker.returncode:
                raise RuntimeError(
                    f"Pipeline exited {worker.returncode}; inspect {attempt / 'pipeline.log'}"
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
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        if worker is not None:
            stop(worker)
        if proc is not None:
            stop(proc)
        sock.close()
        save(
            attempt / "shutdown.json",
            {
                "owned_service_stopped": proc is None or proc.poll() is not None,
                "owned_pipeline_stopped": worker is None or worker.poll() is not None,
            },
        )


def compare(left, right, destination):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    a, b = load(left / "result.json"), load(right / "result.json")
    if (
        a["backend"] != "sequential"
        or b["backend"] != "async"
        or a["ntps"] != b["ntps"]
    ):
        raise ValueError("Invalid comparison pair")
    if (
        a["decision_sha256"] != b["decision_sha256"]
        or a["source_identity"] != b["source_identity"]
    ):
        raise ValueError("Comparison identities differ")
    site = load(left / "attempt.json")["site"]
    if load(right / "attempt.json")["site"] != site:
        raise ValueError("Sites differ")
    outputs = [Path(a["output"]), Path(b["output"])]

    def submissions(path):
        return [
            e["submitted_pixel_sha256"]
            for e in map(
                json.loads, (path / "baby-http.jsonl").read_text().splitlines()
            )
            if e["method"] == "post" and e["event"] == "request_started"
        ]

    assert submissions(left) == submissions(right), "Submitted inputs differ"
    for tp in range(a["ntps"]):
        paths = [p / "steps" / site / "segment_cell" / f"{tp:04d}.npz" for p in outputs]
        with np.load(paths[0]) as aa, np.load(paths[1]) as bb:
            assert aa.files == bb.files
            for key in aa.files:
                np.testing.assert_array_equal(aa[key], bb[key])
    rows = {}
    for name, keys in (
        (
            f"profiles/{site}.parquet",
            ["metadata_tp", "metadata_tile", "metadata_object", "metadata_label"],
        ),
        (f"tracking/{site}_segment_cell.parquet", ["tp", "tile", "cell_label"]),
    ):
        tables = [
            pq.read_table(p / name)
            .to_pandas()
            .sort_values(keys)
            .reset_index(drop=True)
            .sort_index(axis=1)
            for p in outputs
        ]
        pd.testing.assert_frame_equal(*tables, check_exact=True)
        rows[name] = len(tables[0])
    save(
        destination,
        {
            "status": "exact_semantic_parity",
            "left": str(left),
            "right": str(right),
            "ntps": a["ntps"],
            "submitted_inputs_equal": True,
            "all_mask_arrays_equal": True,
            "tables_equal": rows,
            "biological_approval": False,
        },
    )
    print(json.dumps(load(destination)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "command", choices=("prepare", "service", "run", "launch", "compare")
    )
    parser.add_argument("--site", choices=HELD)
    parser.add_argument("--ntps", type=int, choices=(3, 12, 180))
    parser.add_argument("--backend", choices=("sequential", "async"))
    parser.add_argument("--address")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.site)
    elif args.command == "service":
        service(args.root, args.fd, args.port, args.attempt)
    elif args.command == "launch":

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Received termination signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        launch(args.root, args.site, args.ntps, args.backend)
    elif args.command == "compare":
        compare(args.left, args.right, args.report)
    else:
        try:
            run(
                args.root,
                args.site,
                args.ntps,
                args.backend,
                args.address,
                args.attempt,
            )
        except BaseException as error:
            save(
                args.attempt / "worker-failure.json",
                {
                    "error": str(error),
                    "type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
