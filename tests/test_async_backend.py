import asyncio
from contextvars import ContextVar
from copy import deepcopy
import threading

import pytest

from aliby.executor import PipelineStepError
import aliby.pipe_core as core


def graph(ntps=3):
    return {
        "ntps": ntps,
        "steps": {n: {} for n in ("sum", "left", "right", "source")},
        "passed_data": {
            "left": [("x", "source")],
            "right": [("x", "source")],
            "sum": [("a", "left"), ("b", "right")],
        },
        "passed_methods": {},
        "save": [],
    }


class Arithmetic:
    def __init__(self, name):
        self.name = name

    def run_tp(self, tp, **kw):
        return {
            "source": lambda: tp + 1,
            "left": lambda: kw["x"] * 2,
            "right": lambda: kw["x"] + 1,
            "sum": lambda: kw["a"] + kw["b"],
        }[self.name]()


def init(name, parameters, others):
    return Arithmetic(name)


@pytest.mark.parametrize("backend", ["sequential", "concurrent"])
def test_async_matches_reference_across_timepoints_without_mutating_config(backend):
    p = graph()
    before = deepcopy(p)
    reference = core.run_pipeline_return_state(
        p, None, init, backend=backend, max_workers=3
    )
    actual = core.run_pipeline_return_state(
        p, None, init, backend="async", max_workers=3
    )
    assert actual["data"] == reference["data"]
    assert actual["tps"] == reference["tps"]
    assert p == before


def test_native_async_initializers_steps_and_early_descendants_overlap():
    async def check():
        p = graph(2)
        long_started = asyncio.Event()
        child_ran = asyncio.Event()
        events = []

        class Native:
            def __init__(self, name):
                self.name = name

            async def run_tp(self, tp, **kw):
                events.append((tp, self.name, "start"))
                if tp == 0 and self.name == "right":
                    long_started.set()
                    await child_ran.wait()
                if tp == 0 and self.name == "child":
                    await long_started.wait()
                    child_ran.set()
                if tp == 1 and self.name == "source":
                    assert (0, "right", "end") in events
                    assert (0, "sum", "end") in events
                events.append((tp, self.name, "end"))
                return 1

        async def initialize(name, params, other):
            return Native(name)

        p["steps"]["child"] = {}
        p["passed_data"]["child"] = [("x", "left")]
        state = await asyncio.wait_for(
            core.run_pipeline_return_state_async(p, None, initialize, max_workers=3), 5
        )
        assert state["data"]["sum"] == [1, 1]
        assert events.index((0, "child", "end")) < events.index((0, "right", "end"))

    asyncio.run(check())


@pytest.mark.parametrize("backend", ["sequential", "concurrent"])
def test_native_async_initializer_requires_explicit_async_backend(backend):
    async def initialize(*args):
        return lambda: 1

    with pytest.raises(ValueError, match="backend='async'"):
        core.run_pipeline_return_state(graph(1), None, initialize, backend=backend)


def test_contextvars_propagate_to_blocking_adapters():
    identity = ContextVar("test_attempt")
    identity.set("scoped-attempt")
    p = {"steps": {"source": {}}, "passed_data": {}, "save": []}
    state = core.run_pipeline_return_state(
        p, None, lambda *a: lambda: identity.get(), backend="async"
    )
    assert state["data"]["source"] == ["scoped-attempt"]


def test_resource_slots_are_reserved_before_initialization():
    events = []
    p = {
        "steps": {"a": {}, "b": {}},
        "passed_data": {},
        "save": [],
        "step_resources": {"a": {"remote:test": 1}, "b": {"remote:test": 1}},
    }

    def initialize(name, params, others):
        if name == "b":
            assert "run-a" in events
        events.append(f"init-{name}")

        def run():
            events.append(f"run-{name}")
            return name

        return run

    core.run_pipeline_return_state(
        p,
        None,
        initialize,
        backend="async",
        max_workers=2,
        resource_limits={"remote:test": 1},
    )
    assert events == ["init-a", "run-a", "init-b", "run-b"]


def test_failure_never_retries_commits_or_advances_frames(monkeypatch):
    events = []
    commits = []
    p = graph(3)
    monkeypatch.setattr(
        core, "_write_timepoint_step", lambda *args: commits.append(args)
    )

    def initialize(name, params, others):
        if name == "source":

            def fail():
                events.append(name)
                raise RuntimeError("indeterminate POST")

            return fail
        raise AssertionError("dependent initialization must not run")

    with pytest.raises(PipelineStepError, match="indeterminate POST") as error:
        core.run_pipeline_return_state(p, None, initialize, backend="async")
    assert error.value.tp == 0 and error.value.step_name == "source"
    assert events == ["source"]
    assert not commits


@pytest.mark.parametrize("phase", ["prepare", "run", "publish", "commit"])
def test_each_callback_failure_stops_the_site(phase, monkeypatch):
    events = []
    p = {"ntps": 2, "steps": {"source": {}}, "passed_data": {}, "save": []}

    def fail(*args):
        raise RuntimeError(f"{phase} failure")

    def initialize(*args):
        events.append("init")
        if phase == "prepare":
            fail()
        return fail if phase == "run" else lambda: 1

    if phase == "publish":
        monkeypatch.setattr(core, "_publish_timepoint_step", fail)
    if phase == "commit":
        monkeypatch.setattr(core, "_write_timepoint_step", fail)
    with pytest.raises(PipelineStepError, match=f"{phase} failure"):
        core.run_pipeline_return_state(p, None, initialize, backend="async")
    assert events == ["init"]


def test_native_timeout_does_not_retry_or_commit(monkeypatch):
    commits = []
    monkeypatch.setattr(core, "_write_timepoint_step", lambda *a: commits.append(a))
    calls = []

    async def blocked():
        calls.append("run")
        await asyncio.Event().wait()

    p = {"ntps": 2, "steps": {"source": {}}, "passed_data": {}, "save": []}
    with pytest.raises(PipelineStepError) as error:
        core.run_pipeline_return_state(
            p, None, lambda *a: blocked, backend="async", step_timeout=0.03
        )
    assert isinstance(error.value.cause, TimeoutError)
    assert calls == ["run"] and not commits


def test_cancellation_drains_blocking_work_even_when_cancelled_twice(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()
    commits = []
    monkeypatch.setattr(core, "_write_timepoint_step", lambda *a: commits.append(a))

    def blocked():
        started.set()
        try:
            assert release.wait(5)
            return 1
        finally:
            ended.set()

    p = {"ntps": 2, "steps": {"source": {}}, "passed_data": {}, "save": []}

    async def check():
        task = asyncio.create_task(
            core.run_pipeline_return_state_async(p, None, lambda *a: blocked)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), (
                "must not release ownership while a blocking worker still runs"
            )
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert ended.is_set()
        assert not commits

    asyncio.run(check())


def test_no_new_initialization_after_an_observed_failure():
    calls = []
    p = {"steps": {"a": {}, "b": {}}, "passed_data": {}, "save": []}

    def initialize(name, *args):
        calls.append(name)

        def fail():
            raise RuntimeError("stop admission")

        return fail

    with pytest.raises(PipelineStepError):
        core.run_pipeline_return_state(
            p, None, initialize, backend="async", max_workers=1
        )
    assert calls == ["a"]


def test_sync_async_entry_rejects_running_event_loop_before_initializing():
    calls = []

    async def check():
        with pytest.raises(RuntimeError, match="await"):
            core.run_pipeline_return_state(
                graph(), None, lambda *a: calls.append(a), backend="async"
            )

    asyncio.run(check())
    assert not calls


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_bad_timeout_rejected_before_work(timeout):
    with pytest.raises(ValueError, match="step_timeout"):
        core.run_pipeline_return_state(
            graph(), None, init, backend="async", step_timeout=timeout
        )


@pytest.mark.parametrize("phase", ["lock", "pipeline"])
def test_baby_lock_is_not_released_before_cancelled_work_drains(
    tmp_path, monkeypatch, phase
):
    import aliby.pipe_baby as baby
    from aliby.async_executor import AsyncCalls

    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    released = []

    class Lock:
        lexical_output_path = tmp_path
        descriptor_output_path = tmp_path

        def verify(self):
            pass

        def release(self):
            assert ended.is_set()
            released.append(True)

    def wait():
        started.set()
        assert release.wait(5)
        ended.set()

    def acquire(path):
        if phase == "lock":
            wait()
        return Lock()

    async def core_stub(*args, **kwargs):
        assert phase == "pipeline"
        calls = AsyncCalls(1)
        try:
            await calls.call(wait)
        finally:
            await calls.close()

    monkeypatch.setattr(baby, "acquire_provenance_run_lock", acquire)
    monkeypatch.setattr(baby, "validate_provenance_preflight", lambda *a: None)
    monkeypatch.setattr(baby, "_run_pipeline_and_post_async_impl", core_stub)
    p = {
        "publish_roi_provenance": True,
        "steps": {
            "tile": {},
            "segment_cell": {"segmenter_kwargs": {"kind": "nahual_baby"}},
        },
        "passed_data": {},
        "save": [],
    }

    async def check():
        task = asyncio.create_task(
            baby.run_pipeline_and_post_async(p, "site", tmp_path)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not released
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert released == [True]

    asyncio.run(check())


def test_async_global_timeout_aborts_postprocessing(tmp_path):
    p = {
        "steps": {"source": {}},
        "passed_data": {},
        "save": [],
        "global_steps": {"summary": {}},
        "global_passed_data": {"summary_result": (lambda data: [1],)},
    }

    async def blocked(input_data):
        await asyncio.Event().wait()

    def initialize(name, *args):
        return blocked if name == "summary" else lambda: 1

    with pytest.raises(TimeoutError):
        core._run_pipeline_and_post_impl(
            p,
            "site",
            tmp_path,
            init_step_fn=initialize,
            backend="async",
            step_timeout=0.03,
        )


def test_async_global_steps_and_post_hook_are_awaited(tmp_path):
    events = []
    p = {
        "steps": {"source": {}},
        "passed_data": {},
        "save": [],
        "global_steps": {"summary": {}},
        "global_passed_data": {"summary_result": (lambda data: [1],)},
    }

    async def global_step(input_data):
        events.append("global")
        return 42

    def initialize(name, *args):
        return global_step if name == "summary" else lambda: 1

    async def hook(*args):
        events.append("hook")

    profiles, results = core._run_pipeline_and_post_impl(
        p,
        "site",
        tmp_path,
        init_step_fn=initialize,
        post_state_hook=hook,
        backend="async",
    )
    assert events == ["hook", "global"]
    assert results == {"summary_result": 42}
    assert len(profiles) == 0
