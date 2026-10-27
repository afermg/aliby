"""Opt-in asyncio scheduling of the existing per-frame DAG.

Blocking adapters run in a private bounded thread pool. Cancellation cannot kill
such a call: all admitted calls are drained before returning or releasing an
owner's run lock. Adapters must therefore have finite transport timeouts.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
import inspect
import math

from aliby.executor import PipelineStepError, resolve_resources


def resolve_async_resources(pipeline, graph, max_workers=None, resource_limits=None):
    """Conservative opt-in default; this does not alter the thread backend."""
    return resolve_resources(
        pipeline, graph, 4 if max_workers is None else max_workers, resource_limits
    )


class AsyncCalls:
    """Invoke native async or blocking callbacks without abandoning threads."""

    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aliby")
        self.pending = set()

    async def call(self, function, *args, **kwargs):
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if not callable(function):
            raise TypeError("Async adapter callback must be callable")
        if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
            type(function).__call__
        ):
            return await function(*args, **kwargs)
        future = asyncio.get_running_loop().run_in_executor(
            self.pool, partial(copy_context().run, partial(function, *args, **kwargs))
        )
        self.pending.add(future)
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            # A synchronous wrapper might return a coroutine after cancellation.
            # Consume/close it rather than starting work after the abort.
            def consume(done):
                if not done.cancelled() and done.exception() is None:
                    value = done.result()
                    if inspect.iscoroutine(value):
                        value.close()

            future.add_done_callback(consume)
            raise
        else:
            self.pending.discard(future)
            return await result if inspect.isawaitable(result) else result

    async def timed_call(self, timeout, function, *args, **kwargs):
        validate_timeout(timeout)
        operation = self.call(function, *args, **kwargs)
        if timeout is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=timeout)

    async def close(self):
        if self.pending:
            waiter = asyncio.gather(*self.pending, return_exceptions=True)
            cancelled = False
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    cancelled = True
            self.pending.clear()
            self.pool.shutdown(wait=True, cancel_futures=True)
            if cancelled:
                raise asyncio.CancelledError
        else:
            self.pool.shutdown(wait=True, cancel_futures=True)


def run_sync(coroutine_function, *args, **kwargs):
    """Reject nested event loops before constructing an unawaited coroutine."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_function(*args, **kwargs))
    raise RuntimeError("An event loop is running; await the corresponding *_async API.")


def validate_timeout(value):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("step_timeout must be finite and positive, or None")


async def run_async_dag(
    graph,
    run_node,
    publish_node,
    commit_node,
    *,
    prepare_node=None,
    max_workers,
    resource_limits,
    requirements,
    tp,
    step_timeout=None,
):
    """Run one frame, stop admitting work on failure, and never retry.

    Slots are reserved before initialization (including model/session setup).
    Initializers and publications are serialized; ready computations overlap.
    Writes begin in stable graph order only after every node succeeds.
    """
    validate_timeout(step_timeout)
    calls = AsyncCalls(max_workers)
    rank = {name: i for i, name in enumerate(graph.step_order)}
    pending = list(graph.step_order)
    available = dict(resource_limits)
    completed = set()
    running = {}
    results = {}
    failures = {}

    async def invoke(function, *args):
        return await calls.timed_call(step_timeout, function, *args)

    def reserve(name, sign):
        for resource, amount in requirements[name].items():
            available[resource] -= sign * amount

    try:
        while pending or running:
            # Reap already-failed tasks before admitting another initializer.
            failed = [
                task
                for task in running
                if task.done() and (task.cancelled() or task.exception() is not None)
            ]
            if not failed:
                for name in tuple(pending):
                    if len(running) >= max_workers:
                        break
                    if not set(graph.dependencies[name]).issubset(completed):
                        continue
                    if not all(
                        available[r] >= n for r, n in requirements[name].items()
                    ):
                        continue
                    reserve(name, 1)
                    pending.remove(name)
                    try:
                        if prepare_node is not None:
                            await invoke(prepare_node, name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        failures[name] = exc
                        break
                    # Other work may have failed while initialization awaited.
                    if any(
                        t.done() and (t.cancelled() or t.exception() is not None)
                        for t in running
                    ):
                        break
                    task = asyncio.create_task(
                        invoke(run_node, name), name=f"aliby:{tp}:{name}"
                    )
                    running[task] = name
            if failures:
                break
            if not running:
                if pending:
                    raise RuntimeError(
                        "Async DAG cannot progress with the declared dependencies/resources"
                    )
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            # Inspect every completed failure before publishing any success.
            for task in sorted(done, key=lambda t: rank[running[t]]):
                name = running[task]
                if task.cancelled():
                    raise asyncio.CancelledError
                if task.exception() is not None:
                    failures[name] = task.exception()
            if failures:
                break
            for task in sorted(done, key=lambda t: rank[running[t]]):
                name = running.pop(task)
                reserve(name, -1)
                results[name] = task.result()
                try:
                    await invoke(publish_node, name, results[name])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failures[name] = exc
                    break
                completed.add(name)
            if failures:
                break
        if failures:
            name = min(failures, key=rank.__getitem__)
            raise PipelineStepError(name, tp, failures[name]) from failures[name]
        for name in graph.step_order:
            try:
                await invoke(commit_node, name, results[name])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise PipelineStepError(name, tp, exc) from exc
    finally:
        # Cancellation of async calls is cooperative. Shielding the drain also
        # handles repeated caller cancellation without releasing ownership early.
        for task in running:
            if not task.done():
                task.cancel()
        waiter = asyncio.gather(*running, return_exceptions=True)
        cancelled = False
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                cancelled = True
        try:
            await calls.close()
        finally:
            if cancelled:
                raise asyncio.CancelledError
