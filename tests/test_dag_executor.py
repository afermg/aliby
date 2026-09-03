from __future__ import annotations

import threading
import time
from graphlib import CycleError

import numpy as np
import pytest

import aliby.pipe_core as pipe_core
from aliby.executor import (
    PipelineStepError,
    compile_pipeline_graph,
    compile_resource_requirements,
)
from aliby.pipe_core import get_step_output, run_pipeline_return_state

WAIT_TIMEOUT = 10


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
    assert graph.global_outputs == {"global_summary": ("global_summary_result",)}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("passed_data", {"missing": [("x", "source")]}, "target step 'missing'"),
        ("passed_data", {"double": [("x", "missing")]}, "data from 'missing'"),
        (
            "passed_methods",
            {"double": ("missing", "method")},
            "only supported for segment steps",
        ),
    ],
)
def test_compile_pipeline_graph_rejects_missing_dependencies(field, value, match):
    pipeline = _arithmetic_pipeline()
    pipeline[field] = value

    with pytest.raises(ValueError, match=match):
        compile_pipeline_graph(pipeline)


def test_compile_pipeline_graph_rejects_missing_method_source():
    pipeline = _arithmetic_pipeline()
    pipeline["steps"]["segment_test"] = {"callable": lambda: None}
    pipeline["passed_methods"] = {"segment_test": ("missing", "method")}

    with pytest.raises(ValueError, match="method from 'missing'"):
        compile_pipeline_graph(pipeline)


def test_passed_methods_rejects_non_segment_false_cycle_before_sorting():
    pipeline = {
        "steps": {"analyze": {}, "source": {}},
        "passed_data": {"source": [("value", "analyze")]},
        "passed_methods": {"analyze": ("source", "get_value")},
        "save": [],
    }

    with pytest.raises(ValueError, match="only supported for segment steps"):
        compile_pipeline_graph(pipeline)


def test_global_output_names_must_match_exactly_one_global_step():
    pipeline = _arithmetic_pipeline()
    pipeline["global_steps"] = {"summary": {}, "summary_long": {}}
    pipeline["global_passed_data"] = {"summary_long_result": ("combine",)}

    with pytest.raises(ValueError, match="ambiguously matches"):
        compile_pipeline_graph(pipeline)

    pipeline["global_steps"] = {"a": {}}
    pipeline["global_passed_data"] = {"analysis_result": ("combine",)}
    with pytest.raises(ValueError, match="does not match"):
        compile_pipeline_graph(pipeline)


def test_global_dependency_must_reference_a_local_step():
    pipeline = _arithmetic_pipeline()
    pipeline["global_steps"] = {"summary": {}}
    pipeline["global_passed_data"] = {"summary_result": ("missing",)}

    with pytest.raises(ValueError, match="expects data from 'missing'"):
        compile_pipeline_graph(pipeline)


@pytest.mark.parametrize("backend", ["sequential", "concurrent"])
def test_baby_tile_initializer_dependency_ignores_insertion_order(backend):
    tile_ran = threading.Event()

    def tile():
        tile_ran.set()
        return "tile"

    def segment():
        return "segment"

    pipeline = {
        "steps": {
            "segment_cell": {
                "callable": segment,
                "segmenter_kwargs": {"kind": "nahual_baby"},
            },
            "tile": {"callable": tile},
        },
        "passed_data": {},
        "passed_methods": {},
        "save": [],
    }

    def init(step_name, parameters, other_steps):
        if step_name == "segment_cell":
            assert "tile" in other_steps
            assert tile_ran.is_set()
        return parameters["callable"]

    state = run_pipeline_return_state(
        pipeline,
        None,
        init,
        backend=backend,
        max_workers=2,
        resource_limits={"cpu": 2},
    )

    assert state["data"]["segment_cell"] == ["segment"]


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

    with pytest.raises(CycleError) as error:
        run_pipeline_return_state(pipeline, None, init)
    assert {"a", "b"}.issubset(error.value.args[1])
    assert initialised == []


def test_sequential_preserves_valid_legacy_insertion_order():
    events = []

    def record(name, result):
        def run(**_kwargs):
            events.append(name)
            return result

        return run

    pipeline = {
        "steps": {
            "source": {"callable": record("source", 1)},
            "branch": {"callable": record("branch", 2)},
            "branch_child": {"callable": record("branch_child", 3)},
            "independent": {"callable": record("independent", 4)},
        },
        "passed_data": {
            "branch": [("value", "source")],
            "branch_child": [("value", "branch")],
            "independent": [("value", "source")],
        },
        "passed_methods": {},
        "save": [],
    }

    run_pipeline_return_state(pipeline, None, _init_callable)

    assert events == ["source", "branch", "branch_child", "independent"]


def test_default_backend_remains_the_sequential_reference():
    default = run_pipeline_return_state(_arithmetic_pipeline(), None, _init_callable)
    explicit = run_pipeline_return_state(
        _arithmetic_pipeline(), None, _init_callable, backend="sequential"
    )

    assert default["data"] == explicit["data"]


@pytest.mark.parametrize(
    ("save", "save_interval", "ntps", "match"),
    [
        ([], 1, 1, "not listed in 'save'"),
        (["segment"], 2, 1, "save_interval must be 1"),
        (["segment"], 1, 0, "ntps must be a positive int"),
    ],
)
def test_from_disk_global_input_must_be_produced_before_local_work(
    save, save_interval, ntps, match
):
    initialised = []
    pipeline = {
        "steps": {"segment": {"callable": lambda: [[1]]}},
        "passed_data": {},
        "passed_methods": {},
        "global_steps": {"summary": {}},
        "global_passed_data": {"summary_result": ("from_disk:segment",)},
        "save": save,
        "save_interval": save_interval,
        "ntps": ntps,
    }

    def init(step_name, parameters, _other_steps):
        initialised.append(step_name)
        return parameters["callable"]

    with pytest.raises(ValueError, match=match):
        run_pipeline_return_state(pipeline, None, init)
    assert initialised == []


def test_from_disk_loads_only_current_run_timepoints(tmp_path):
    step_dir = tmp_path / "segment"
    step_dir.mkdir()
    for tp, value in enumerate((10, 11, 99)):
        np.savez_compressed(
            step_dir / f"{tp:04d}.npz",
            np.full((1, 2, 2), value, dtype=np.uint16),
        )

    result = get_step_output(
        {},
        ("from_disk:segment",),
        steps_dir=tmp_path,
        expected_ntps=2,
    )

    assert result.shape == (1, 2, 2, 2)
    assert result[0, 0].tolist() == [[10, 10], [10, 10]]
    assert result[0, 1].tolist() == [[11, 11], [11, 11]]
    assert 99 not in result


def test_global_from_disk_ignores_stale_trailing_timepoint(tmp_path):
    stale_dir = tmp_path / "steps" / "synthetic" / "segment"
    stale_dir.mkdir(parents=True)
    np.savez_compressed(
        stale_dir / "0002.npz",
        np.full((1, 2, 2), 99, dtype=np.uint16),
    )

    def init(_step_name, parameters, _other_steps=None):
        return parameters["callable"]

    pipeline = {
        "ntps": 2,
        "steps": {
            "segment": {"callable": lambda: np.full((1, 2, 2), 7, dtype=np.uint16)},
        },
        "passed_data": {},
        "passed_methods": {},
        "global_steps": {
            "summary": {"callable": lambda input_data: input_data},
        },
        "global_passed_data": {
            "summary_result": ("from_disk:segment",),
        },
        "save": ["segment"],
        "save_interval": 1,
    }

    _profiles, post_results = pipe_core._run_pipeline_and_post_impl(
        pipeline,
        "synthetic",
        tmp_path,
        init_step_fn=init,
    )

    assert post_results is not None
    result = post_results["summary_result"]
    assert result.shape == (1, 2, 2, 2)
    assert np.all(result == 7)


def test_from_disk_rejects_missing_current_run_timepoint(tmp_path):
    step_dir = tmp_path / "segment"
    step_dir.mkdir()
    np.savez_compressed(
        step_dir / "0000.npz",
        np.zeros((1, 2, 2), dtype=np.uint16),
    )
    np.savez_compressed(
        step_dir / "0002.npz",
        np.full((1, 2, 2), 99, dtype=np.uint16),
    )

    with pytest.raises(FileNotFoundError, match="0001.npz"):
        get_step_output(
            {},
            ("from_disk:segment",),
            steps_dir=tmp_path,
            expected_ntps=2,
        )


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


def test_concurrent_initialises_dependents_after_predecessors_run():
    source_ran = threading.Event()

    def source():
        source_ran.set()
        return 1

    def consume(value):
        return value

    pipeline = {
        "steps": {
            "source": {"callable": source},
            "consume": {"callable": consume},
        },
        "passed_data": {"consume": [("value", "source")]},
        "passed_methods": {},
        "save": [],
    }

    def init(step_name, parameters, _other_steps):
        if step_name == "consume":
            assert source_ran.is_set()
        return parameters["callable"]

    state = run_pipeline_return_state(
        pipeline,
        None,
        init,
        backend="concurrent",
        max_workers=2,
        resource_limits={"cpu": 2},
    )

    assert state["data"]["consume"] == [1]


def test_independent_nodes_overlap_in_concurrent_backend():
    barrier = threading.Barrier(2, timeout=WAIT_TIMEOUT)
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


def test_ready_descendant_overlaps_an_unfinished_independent_branch():
    descendant_started = threading.Event()

    def source():
        return 1

    def short(value):
        return value

    def descendant(value):
        descendant_started.set()
        return value

    def long(value):
        assert descendant_started.wait(timeout=WAIT_TIMEOUT)
        return value

    pipeline = {
        "steps": {
            "source": {"callable": source},
            "short": {"callable": short},
            "descendant": {"callable": descendant},
            "long": {"callable": long},
        },
        "passed_data": {
            "short": [("value", "source")],
            "descendant": [("value", "short")],
            "long": [("value", "source")],
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

    assert state["data"]["descendant"] == [1]
    assert state["data"]["long"] == [1]


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


def test_endpoint_inference_is_disabled_when_ambiguous_or_explicit():
    pipeline = {
        "steps": {
            "ambiguous": {
                "callable": lambda: None,
                "address": "ipc:///tmp/a.ipc",
                "segmenter_kwargs": {"address": "ipc:///tmp/b.ipc"},
            },
            "explicit": {
                "callable": lambda: None,
                "address": "ipc:///tmp/c.ipc",
            },
        },
        "passed_data": {},
        "passed_methods": {},
        "step_resources": {"explicit": {}},
        "save": [],
    }
    graph = compile_pipeline_graph(pipeline)

    requirements, inferred = compile_resource_requirements(pipeline, graph)

    assert requirements == {
        "ambiguous": {"cpu": 1},
        "explicit": {"cpu": 1},
    }
    assert inferred == set()


def test_missing_explicit_resource_limit_fails_before_initialisation():
    initialised = []
    pipeline = {
        "steps": {"gpu": {"callable": lambda: None}},
        "passed_data": {},
        "passed_methods": {},
        "step_resources": {"gpu": {"gpu": 1}},
        "save": [],
    }

    def init(step_name, parameters, _other_steps):
        initialised.append(step_name)
        return parameters["callable"]

    with pytest.raises(ValueError, match="no limit was provided"):
        run_pipeline_return_state(
            pipeline,
            None,
            init,
            backend="concurrent",
            max_workers=1,
            resource_limits={"cpu": 1},
        )

    assert initialised == []


@pytest.mark.parametrize(
    ("max_workers", "resource_limits", "match"),
    [
        (0, {"cpu": 1}, "max_workers must be a positive int"),
        (1, {"cpu": 0}, "resource limit 'cpu' must be a positive int"),
    ],
)
def test_invalid_executor_capacity_fails_before_initialisation(
    max_workers, resource_limits, match
):
    initialised = []
    pipeline = {
        "steps": {"source": {"callable": lambda: None}},
        "passed_data": {},
        "passed_methods": {},
        "save": [],
    }

    def init(step_name, parameters, _other_steps):
        initialised.append(step_name)
        return parameters["callable"]

    with pytest.raises(ValueError, match=match):
        run_pipeline_return_state(
            pipeline,
            None,
            init,
            backend="concurrent",
            max_workers=max_workers,
            resource_limits=resource_limits,
        )
    assert initialised == []


def test_concurrent_commits_outputs_in_deterministic_graph_order(monkeypatch):
    later_finished = threading.Event()
    completion_order = []
    save_order = []

    def earlier():
        assert later_finished.wait(timeout=WAIT_TIMEOUT)
        completion_order.append("earlier")
        return "earlier"

    def later():
        completion_order.append("later")
        later_finished.set()
        return "later"

    def dispatch_write_fn(_step_name):
        def write(_result, *, subpath, **_kwargs):
            save_order.append(subpath)

        return write

    monkeypatch.setattr(pipe_core, "dispatch_write_fn", dispatch_write_fn)
    pipeline = {
        "steps": {
            "earlier": {"callable": earlier},
            "later": {"callable": later},
        },
        "passed_data": {},
        "passed_methods": {},
        "save": ["earlier", "later"],
    }

    run_pipeline_return_state(
        pipeline,
        None,
        _init_callable,
        backend="concurrent",
        max_workers=2,
        resource_limits={"cpu": 2},
    )

    assert completion_order == ["later", "earlier"]
    assert save_order == ["earlier", "later"]


def test_concurrent_failure_selection_is_deterministic():
    later_finished = threading.Event()

    def earlier_failure():
        assert later_finished.wait(timeout=WAIT_TIMEOUT)
        raise LookupError("earlier failure")

    def later_failure():
        later_finished.set()
        raise LookupError("later failure")

    pipeline = {
        "steps": {
            "earlier_failure": {"callable": earlier_failure},
            "later_failure": {"callable": later_failure},
        },
        "passed_data": {},
        "passed_methods": {},
        "save": [],
    }

    with pytest.raises(PipelineStepError, match="earlier failure") as error:
        run_pipeline_return_state(
            pipeline,
            None,
            _init_callable,
            backend="concurrent",
            max_workers=2,
            resource_limits={"cpu": 2},
        )

    assert error.value.step_name == "earlier_failure"


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


def test_global_outputs_are_saved_with_their_own_results(tmp_path, monkeypatch):
    written = {}

    def init(step_name, parameters, _other_steps=None):
        return parameters["callable"]

    def write(result, _output_path, *, subpath, **_kwargs):
        written[subpath] = result.tolist()

    monkeypatch.setattr(pipe_core, "dispatch_write_fn", lambda _name: write)
    pipeline = {
        "steps": {
            "source_a": {"callable": lambda: [1]},
            "source_b": {"callable": lambda: [2]},
        },
        "passed_data": {},
        "passed_methods": {},
        "global_steps": {
            "global_summary": {"callable": lambda input_data: input_data},
        },
        "global_passed_data": {
            "global_summary_a": ("source_a",),
            "global_summary_b": ("source_b",),
        },
        "save": ["global_summary"],
    }

    _profiles, post_results = pipe_core._run_pipeline_and_post_impl(
        pipeline,
        "synthetic",
        tmp_path,
        init_step_fn=init,
    )

    assert post_results is not None
    assert np.array_equal(post_results["global_summary_a"], [[1]])
    assert np.array_equal(post_results["global_summary_b"], [[2]])
    assert written == {
        "global_summary_a": [[1]],
        "global_summary_b": [[2]],
    }


def test_global_steps_cannot_claim_local_executor_resources():
    pipeline = _arithmetic_pipeline()
    pipeline["global_steps"] = {"global_summary": {}}
    pipeline["global_passed_data"] = {"global_summary_result": ("combine",)}
    pipeline["step_resources"] = {"global_summary": {"gpu": 1}}
    graph = compile_pipeline_graph(pipeline)

    with pytest.raises(ValueError, match="global steps run sequentially"):
        compile_resource_requirements(pipeline, graph)
