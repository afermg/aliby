from __future__ import annotations

import threading
import time
from graphlib import CycleError

import pytest

from aliby.executor import PipelineStepError, compile_pipeline_graph
from aliby.pipe_core import run_pipeline_return_state


def _init_callable(step_name, parameters, _other_steps):
    return parameters["callable"]


def _arithmetic_pipeline(events=None):
    events = events if events is not None else []

    def source():
        events.append("source")
        return 3

    def double(value):
        events.append("double")
        return value * 2

    def increment(value):
        events.append("increment")
        return value + 1

    def combine(left, right):
        events.append("combine")
        return left + right

    return {
        # Deliberately not topologically ordered: graph compilation, rather
        # than a second workflow language, determines execution order.
        "steps": {
            "combine": {"callable": combine},
            "double": {"callable": double},
            "increment": {"callable": increment},
            "source": {"callable": source},
        },
        "passed_data": {
            "double": [("value", "source")],
            "increment": [("value", "source")],
            "combine": [("left", "double"), ("right", "increment")],
        },
        "passed_methods": {},
        "save": [],
    }


def test_compile_pipeline_graph_uses_existing_wiring():
    pipeline = _arithmetic_pipeline()
    pipeline["global_steps"] = {"global_summary": {}}
    pipeline["global_passed_data"] = {
        "global_summary_result": ("combine",),
    }

    graph = compile_pipeline_graph(pipeline)

    assert graph.dependencies == {
        "combine": ("double", "increment"),
        "double": ("source",),
        "increment": ("source",),
        "source": (),
    }
    assert graph.step_order.index("source") < graph.step_order.index("double")
    assert graph.step_order.index("source") < graph.step_order.index("increment")
    assert graph.step_order[-1] == "combine"
    assert graph.global_dependencies == {"global_summary": ("combine",)}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("passed_data", {"missing": [("x", "source")]}, "target step 'missing'"),
        ("passed_data", {"double": [("x", "missing")]}, "data from 'missing'"),
        ("passed_methods", {"double": ("missing", "method")}, "method from 'missing'"),
    ],
)
def test_compile_pipeline_graph_rejects_missing_dependencies(field, value, match):
    pipeline = _arithmetic_pipeline()
    pipeline[field] = value

    with pytest.raises(ValueError, match=match):
        compile_pipeline_graph(pipeline)


def test_cycle_is_rejected_before_initialisation():
    pipeline = {
        "steps": {"a": {}, "b": {}},
        "passed_data": {"a": [("value", "b")], "b": [("value", "a")]},
        "passed_methods": {},
        "save": [],
    }
    initialised = []

    def init(step_name, _parameters, _other_steps):
        initialised.append(step_name)
        return lambda value: value

    with pytest.raises(CycleError):
        run_pipeline_return_state(pipeline, None, init)
    assert initialised == []


def test_sequential_and_concurrent_results_are_equivalent():
    sequential = run_pipeline_return_state(
        _arithmetic_pipeline(), None, _init_callable, backend="sequential"
    )
    concurrent = run_pipeline_return_state(
        _arithmetic_pipeline(),
        None,
        _init_callable,
        backend="concurrent",
        max_workers=4,
        resource_limits={"cpu": 4},
    )

    assert sequential["data"] == concurrent["data"]
    assert concurrent["data"]["combine"] == [10]
    assert list(sequential["data"]) == list(concurrent["data"])


def test_independent_nodes_overlap_in_concurrent_backend():
    barrier = threading.Barrier(2, timeout=2)
    overlapped = []

    def source():
        return 1

    def branch(value):
        barrier.wait()
        overlapped.append(value)
        return value

    pipeline = {
        "steps": {
            "source": {"callable": source},
            "left": {"callable": branch},
            "right": {"callable": branch},
        },
        "passed_data": {
            "left": [("value", "source")],
            "right": [("value", "source")],
        },
        "passed_methods": {},
        "save": [],
    }

    state = run_pipeline_return_state(
        pipeline,
        None,
        _init_callable,
        backend="concurrent",
        max_workers=2,
        resource_limits={"cpu": 2},
    )

    assert sorted(overlapped) == [1, 1]
    assert state["data"]["left"] == [1]
    assert state["data"]["right"] == [1]


def test_explicit_gpu_resource_limit_is_respected():
    lock = threading.Lock()
    active = 0
    high_watermark = 0

    def gpu_work():
        nonlocal active, high_watermark
        with lock:
            active += 1
            high_watermark = max(high_watermark, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return "done"

    pipeline = {
        "steps": {
            "gpu_a": {"callable": gpu_work},
            "gpu_b": {"callable": gpu_work},
        },
        "passed_data": {},
        "passed_methods": {},
        "step_resources": {"gpu_a": {"gpu": 1}, "gpu_b": {"gpu": 1}},
        "save": [],
    }

    run_pipeline_return_state(
        pipeline,
        None,
        _init_callable,
        backend="concurrent",
        max_workers=2,
        resource_limits={"cpu": 2, "gpu": 1},
    )

    assert high_watermark == 1


def test_inferred_nahual_endpoint_is_bounded_to_one_slot():
    lock = threading.Lock()
    active = 0
    high_watermark = 0

    def remote_work():
        nonlocal active, high_watermark
        with lock:
            active += 1
            high_watermark = max(high_watermark, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return "done"

    address = "ipc:///tmp/nahual-test.ipc"
    pipeline = {
        "steps": {
            "remote_a": {"callable": remote_work, "address": address},
            "remote_b": {
                "callable": remote_work,
                "segmenter_kwargs": {"address": address},
            },
        },
        "passed_data": {},
        "passed_methods": {},
        "save": [],
    }

    run_pipeline_return_state(
        pipeline,
        None,
        _init_callable,
        backend="concurrent",
        max_workers=2,
        resource_limits={"cpu": 2},
    )

    assert high_watermark == 1


def test_concurrent_failure_names_step_and_timepoint():
    def explode():
        raise LookupError("synthetic failure")

    pipeline = {
        "steps": {"explode": {"callable": explode}},
        "passed_data": {},
        "passed_methods": {},
        "save": [],
    }

    with pytest.raises(
        PipelineStepError,
        match="Pipeline step 'explode' failed at timepoint 0: synthetic failure",
    ) as error:
        run_pipeline_return_state(
            pipeline,
            None,
            _init_callable,
            backend="concurrent",
            max_workers=1,
            resource_limits={"cpu": 1},
        )

    assert isinstance(error.value.__cause__, LookupError)
