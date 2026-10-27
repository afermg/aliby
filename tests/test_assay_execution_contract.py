"""Conformance probes, not a production asyncio executor or remote service."""

import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from graphlib import CycleError
import json

import numpy as np
import pytest

from aliby.assay import (
    AssayRequest,
    ChannelEvidence,
    ChannelInput,
    ModelCapability,
    PlaneSample,
    SegmentationRequest,
    decide_assay,
    select_inputs,
)
from aliby.execution_contract import compile_execution_contract
from aliby.pipe_core import run_pipeline_return_state


def decision(ntps=3):
    request = AssayRequest(
        "site-a",
        "sha256:raw",
        ntps,
        (
            ChannelEvidence(
                0,
                "BF",
                "brightfield",
                1,
                (PlaneSample(0, ("nonzero",), "sha256:pixels"),),
            ),
        ),
        (SegmentationRequest("cell", (ChannelInput(0, (0,)),)),),
        assay_type="yeast_microfluidics",
    )
    return decide_assay(
        request,
        (
            ModelCapability(
                "baby-one",
                "sha256:weights",
                ("brightfield",),
                1,
                "overlapping",
                True,
                assays=("yeast_microfluidics",),
                targets=("cell",),
            ),
        ),
    )


class Node:
    def __init__(self, kind, fail_at=None):
        self.kind = kind
        self.seen = []
        self.fail_at = fail_at

    def run_tp(self, tp, **kwargs):
        self.seen.append(tp)
        if self.kind == "tile":
            assert self.seen == list(range(tp + 1))
            return {"pixels": np.full((2, 1, 1, 2, 2), tp + 1, dtype=np.uint16)}
        if self.kind == "segment_cell":
            assert self.seen == list(range(tp + 1))  # Stateful crawler cannot reorder.
            if tp == self.fail_at:
                raise RuntimeError(
                    "indeterminate POST: session may already have advanced"
                )
            pixels = select_inputs(kwargs["pixels"], (ChannelInput(0, (0,)),))
            return {"masks": int(pixels.sum()), "labels": tp + 1}
        if self.kind == "embed":
            return int(kwargs["pixels"].sum()) * 3
        return kwargs["masks"] + int(kwargs["pixels"].sum())


def pipeline(ntps=3, fail_at=None):
    # Purposely non-topological declaration, with an independent branch.
    return {
        "ntps": ntps,
        "steps": {
            name: {"node": Node(name, fail_at)}
            for name in ("extract_cell", "segment_cell", "embed", "tile")
        },
        "passed_data": {
            "segment_cell": [("pixels", "tile")],
            "embed": [("pixels", "tile")],
            "extract_cell": [("masks", "segment_cell"), ("pixels", "tile")],
        },
        "passed_methods": {},
        "save": [],
    }


def compile_plan(p, **kwargs):
    return compile_execution_contract(
        p,
        decision(p["ntps"]),
        attempt_id="fresh-attempt",
        pipeline_identity="sha256:frozen-config-and-code",
        max_workers=4,
        **kwargs,
    )


async def reference_async(contract, p, hook=None):
    """Tiny test-only interpreter: deps + slots + no retry, with fake nodes.

    No files, model services, release writer, or durable recovery semantics.
    A future executor must pass these probes with real adapters as well.
    """
    futures = {}
    values = {}
    started = []
    finished = []
    workers = asyncio.Semaphore(contract.max_workers)
    resources = {
        name: asyncio.Semaphore(count) for name, count in contract.resource_limits
    }

    async def run(task):
        await asyncio.gather(*(futures[d] for d in task.dependencies))
        acquired = []
        async with workers:
            try:
                for resource, amount in task.resources:
                    for _ in range(amount):
                        await resources[resource].acquire()
                        acquired.append(resource)
                started.append(task.key)
                if hook:
                    await hook(task)
                if task.kind == "step":
                    kwargs = {}
                    for key, source, *alias in p["passed_data"].get(task.step, ()):
                        value = values[f"frame/{task.timepoint}/step/{source}"]
                        kwargs[alias[0] if alias else key] = (
                            value[key] if isinstance(value, dict) else value
                        )
                    value = p["steps"][task.step]["node"].run_tp(
                        task.timepoint, **kwargs
                    )
                else:
                    value = task.kind
                values[task.key] = value
                finished.append(task.key)
                return value
            finally:
                for resource in reversed(acquired):
                    resources[resource].release()

    for task in contract.tasks:
        futures[task.key] = asyncio.create_task(run(task))
    results = await asyncio.wait_for(
        asyncio.gather(*futures.values(), return_exceptions=True), timeout=10
    )
    return values, started, finished, results


def test_contract_reuses_existing_graph_and_does_not_mutate_pipeline():
    p = pipeline()
    before = deepcopy({k: v for k, v in p.items() if k != "steps"})
    contract = compile_plan(p)
    tasks = {t.key: t for t in contract.tasks}
    assert tasks["frame/0/step/extract_cell"].dependencies == (
        "frame/0/step/segment_cell",
        "frame/0/step/tile",
    )
    assert tasks["frame/0/step/segment_cell"].dependencies == ("frame/0/step/tile",)
    assert tasks["frame/1/step/tile"].dependencies == ("frame/0/commit",)
    assert set(tasks["frame/0/commit"].dependencies) == {
        f"frame/0/step/{n}" for n in p["steps"]
    }
    assert tasks["publish_release"].dependencies == ("validate_release",)
    assert contract.to_dict()["executable"] is False
    assert {k: v for k, v in p.items() if k != "steps"} == before
    assert all(t.retry == "never" for t in contract.tasks)
    assert contract.failure_policy == "abort_site_restart_from_zero"


@pytest.mark.parametrize("backend", ["sequential", "concurrent"])
def test_reference_async_contract_matches_existing_successful_results(backend):
    p = pipeline()
    state = run_pipeline_return_state(
        p,
        None,
        lambda name, params, other: params["node"],
        backend=backend,
        max_workers=4,
    )
    q = pipeline()
    values, _, _, results = asyncio.run(reference_async(compile_plan(q), q))
    assert not any(isinstance(r, BaseException) for r in results)
    for name in ("segment_cell", "extract_cell", "embed"):
        assert [values[f"frame/{t}/step/{name}"] for t in range(3)] == state["data"][
            name
        ]
    assert values["publish_release"] == "publish_release"


def test_async_sibling_overlap_and_early_descendant_without_cross_frame_race():
    async def check():
        p = pipeline(2)
        embed_started = asyncio.Event()
        extract_started = asyncio.Event()

        async def hook(task):
            if task.timepoint == 0 and task.step == "embed":
                embed_started.set()
                await extract_started.wait()
            elif task.timepoint == 0 and task.step == "extract_cell":
                await embed_started.wait()
                extract_started.set()

        _, started, finished, results = await reference_async(compile_plan(p), p, hook)
        assert not any(isinstance(r, BaseException) for r in results)
        # Either ready task can enter first. The rendezvous proves overlap;
        # asserting one start order would test scheduler luck, not the contract.
        assert embed_started.is_set() and extract_started.is_set()
        assert finished.index("frame/0/step/extract_cell") < finished.index(
            "frame/0/step/embed"
        )
        assert finished.index("frame/0/commit") < finished.index("frame/1/step/tile")
        assert p["steps"]["segment_cell"]["node"].seen == [0, 1]

    asyncio.run(check())


def test_failed_stateful_call_never_retries_advances_or_publishes():
    p = pipeline(3, fail_at=1)
    _, started, _, results = asyncio.run(reference_async(compile_plan(p), p))
    assert any(isinstance(r, RuntimeError) for r in results)
    assert p["steps"]["segment_cell"]["node"].seen == [0, 1]
    assert "frame/1/step/extract_cell" not in started
    assert "frame/1/commit" not in started
    assert not any(s.startswith("frame/2/") for s in started)
    assert "validate_release" not in started
    assert "publish_release" not in started
    assert max(Counter(started).values()) == 1


@pytest.mark.parametrize("failure", ["frame/0/commit", "validate_release"])
def test_write_or_validation_failure_prevents_publication(failure):
    async def hook(task):
        if task.key == failure:
            raise RuntimeError("validation/commit failed")

    p = pipeline()
    _, started, _, _ = asyncio.run(reference_async(compile_plan(p), p, hook))
    assert "publish_release" not in started
    if failure == "frame/0/commit":
        assert not any(s.startswith("frame/1/") for s in started)


@pytest.mark.parametrize("failure_type", [TimeoutError, asyncio.CancelledError])
def test_timeout_or_cancellation_cannot_publish_or_continue_a_session(failure_type):
    async def hook(task):
        if task.key == "frame/1/step/segment_cell":
            raise failure_type("remote outcome uncertain")

    p = pipeline()
    _, started, _, results = asyncio.run(reference_async(compile_plan(p), p, hook))
    assert any(isinstance(r, failure_type) for r in results)
    assert "publish_release" not in started
    assert not any(s.startswith("frame/2/") for s in started)
    assert Counter(started)["frame/1/step/segment_cell"] == 1


def test_session_scopes_are_site_attempt_and_step_specific():
    p = pipeline()
    first = compile_plan(p)
    second = compile_execution_contract(
        p,
        decision(),
        attempt_id="different-attempt",
        pipeline_identity="sha256:frozen-config-and-code",
    )
    a = [t.session_key for t in first.tasks if t.step == "segment_cell"]
    b = [t.session_key for t in second.tasks if t.step == "segment_cell"]
    assert len(set(a)) == 1 and a[0] != b[0]
    assert json.loads(a[0]) == ["site-a", "fresh-attempt", "segment_cell"]
    assert all(t.session_key is None for t in first.tasks if t.step == "embed")


def test_global_phase_waits_for_complete_series_and_validation_waits_for_globals():
    p = pipeline()
    p["global_steps"] = {"trackastra": {}, "summary": {}}
    p["global_passed_data"] = {
        "trackastra_result": ("segment_cell",),
        "summary_result": ("extract_cell",),
    }
    contract = compile_plan(p)
    tasks = {t.key: t for t in contract.tasks}
    assert tasks["global/trackastra"].dependencies == ("frame/2/commit",)
    assert tasks["global/summary"].dependencies == ("global/trackastra",)
    assert tasks["validate_release"].dependencies == ("global/summary",)


def test_resource_capacity_and_remote_requirements_are_carried_forward():
    p = pipeline()
    p["steps"]["segment_cell"]["segmenter_kwargs"] = {
        "kind": "nahual_baby",
        "address": "http://127.0.0.1:1234",
    }
    p["step_resources"] = {"embed": {"gpu": 1}}
    with pytest.raises(ValueError, match="no limit"):
        compile_plan(p)
    contract = compile_plan(p, resource_limits={"gpu": 1, "cpu": 2})
    tasks = {t.key: t for t in contract.tasks}
    assert (
        dict(tasks["frame/0/step/segment_cell"].resources)[
            "remote:http://127.0.0.1:1234"
        ]
        == 1
    )
    assert dict(contract.resource_limits)["remote:http://127.0.0.1:1234"] == 1
    assert dict(tasks["frame/0/step/embed"].resources)["gpu"] == 1


def test_baby_hidden_tiler_dependency_is_derived_from_existing_compiler():
    p = pipeline()
    p["steps"]["segment_cell"]["segmenter_kwargs"] = {"kind": "nahual_baby"}
    del p["passed_data"]["segment_cell"]
    contract = compile_plan(p)
    task = next(t for t in contract.tasks if t.key == "frame/0/step/segment_cell")
    assert task.dependencies == ("frame/0/step/tile",)
    p["steps"]["tile_other"] = {}
    with pytest.raises(ValueError, match="unambiguous"):
        compile_plan(p)


@pytest.mark.parametrize(
    "problem", ["cycle", "missing", "stateful", "ntps", "object", "bad_decision"]
)
def test_invalid_contracts_are_rejected_before_node_initialization(problem):
    p = pipeline()
    kwargs = {}
    d = decision()
    if problem == "cycle":
        p["passed_data"]["tile"] = [("pixels", "segment_cell")]
    elif problem == "missing":
        p["passed_data"]["embed"] = [("pixels", "absent")]
    elif problem == "stateful":
        kwargs["stateful_steps"] = ("absent",)
    elif problem == "ntps":
        p["ntps"] = 4
    elif problem == "object":
        d = decide_assay(
            replace(
                d.request,
                segmentations=(
                    SegmentationRequest("absent", (ChannelInput(0, (0,)),)),
                ),
            ),
            d.models,
        )
    else:
        d = replace(d, selected_models=())
    with pytest.raises((ValueError, CycleError)):
        compile_execution_contract(
            p, d, attempt_id="a", pipeline_identity="frozen", **kwargs
        )
    assert all(not params["node"].seen for params in p["steps"].values())


def test_contract_fingerprint_binds_decision_attempt_and_runtime_identity():
    p = pipeline()
    a = compile_plan(p)
    assert a == compile_plan(p)
    assert len(a.fingerprint) == 64
    b = compile_execution_contract(
        p,
        decision(),
        attempt_id="fresh-attempt",
        pipeline_identity="different-code",
        max_workers=4,
    )
    assert a.fingerprint != b.fingerprint
    assert a.assay_fingerprint == decision().fingerprint
    assert json.loads(json.dumps(a.to_dict()))["executable"] is False
