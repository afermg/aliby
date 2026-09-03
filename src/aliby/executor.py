"""Compile and execute ALIBY's existing pipeline dictionaries as small DAGs.

The graph is derived from ``steps``, ``passed_data``, ``passed_methods`` and
``global_passed_data``.  This module deliberately does not define another
workflow format.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Callable, Mapping


@dataclass(frozen=True)
class PipelineGraph:
    """Compiled dependency graph for per-timepoint and global pipeline steps."""

    dependencies: dict[str, tuple[str, ...]]
    step_order: tuple[str, ...]
    global_dependencies: dict[str, tuple[str, ...]]


class PipelineStepError(RuntimeError):
    """A per-timepoint step failed in the concurrent backend."""

    def __init__(self, step_name: str, tp: int, cause: BaseException):
        super().__init__(
            f"Pipeline step '{step_name}' failed at timepoint {tp}: {cause}"
        )
        self.step_name = step_name
        self.tp = tp
        self.cause = cause


def _ordered_add(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def compile_pipeline_graph(pipeline: dict) -> PipelineGraph:
    """Compile existing pipeline wiring and reject missing nodes and cycles.

    Global nodes are compiled for validation and introspection, but execution of
    them remains in ALIBY's existing post-time-series phase.
    """
    if not isinstance(pipeline, dict):
        raise TypeError("Pipeline configuration must be a dictionary.")
    steps = pipeline.get("steps")
    if not isinstance(steps, dict):
        raise ValueError(
            "Pipeline must contain a 'steps' dictionary mapping step names to parameters."
        )

    dependencies: dict[str, list[str]] = {name: [] for name in steps}
    passed_data = pipeline.get("passed_data")
    if not isinstance(passed_data, dict):
        raise ValueError("Pipeline must contain a 'passed_data' dictionary.")
    for target, specs in passed_data.items():
        if target not in steps:
            raise ValueError(
                f"'passed_data' references target step '{target}', but it is not defined in 'steps'."
            )
        if not isinstance(specs, (list, tuple)):
            raise TypeError(
                f"'passed_data' dependencies for step '{target}' must be a sequence."
            )
        for spec in specs:
            if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                raise ValueError(
                    f"Invalid dependency format in 'passed_data' for '{target}': {spec}"
                )
            source = spec[1]
            if source not in steps:
                raise ValueError(
                    f"Step '{target}' expects data from '{source}', but '{source}' is not defined in 'steps'."
                )
            _ordered_add(dependencies[target], source)

    passed_methods = pipeline.get("passed_methods", {})
    if not isinstance(passed_methods, dict):
        raise TypeError("'passed_methods' must be a dictionary.")
    for target, spec in passed_methods.items():
        if target not in steps:
            raise ValueError(
                f"'passed_methods' references target step '{target}', but it is not defined in 'steps'."
            )
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
            raise ValueError(f"Invalid method dependency format for '{target}': {spec}")
        source = spec[0]
        if source not in steps:
            raise ValueError(
                f"Step '{target}' expects a method from '{source}', but '{source}' is not defined in 'steps'."
            )
        _ordered_add(dependencies[target], source)

    frozen_dependencies = {
        name: tuple(predecessors) for name, predecessors in dependencies.items()
    }
    # static_order both validates cycles and gives the serial reference backend
    # a stable dependency-respecting order. Dict insertion order breaks ties.
    step_order = tuple(TopologicalSorter(frozen_dependencies).static_order())

    global_steps = pipeline.get("global_steps", {})
    if not isinstance(global_steps, dict):
        raise TypeError("'global_steps' must be a dictionary.")
    overlap = set(steps).intersection(global_steps)
    if overlap:
        name = next(name for name in steps if name in overlap)
        raise ValueError(
            f"Step '{name}' is defined in both 'steps' and 'global_steps'."
        )

    global_dependencies: dict[str, list[str]] = {name: [] for name in global_steps}
    global_passed_data = pipeline.get("global_passed_data", {})
    if not isinstance(global_passed_data, dict):
        raise TypeError("'global_passed_data' must be a dictionary.")
    unmatched_outputs = set(global_passed_data)
    for global_name in global_steps:
        output_names = [
            name for name in global_passed_data if name.startswith(global_name)
        ]
        if not output_names:
            raise ValueError(
                f"Global step '{global_name}' has no entry in 'global_passed_data'."
            )
        unmatched_outputs.difference_update(output_names)
        for output_name in output_names:
            fetchers = global_passed_data[output_name]
            if not isinstance(fetchers, (list, tuple)):
                raise TypeError(
                    f"'global_passed_data' fetchers for '{output_name}' must be a sequence."
                )
            for fetcher in fetchers:
                if isinstance(fetcher, str):
                    source = fetcher.removeprefix("from_disk:")
                    if source not in steps:
                        raise ValueError(
                            f"Global step '{global_name}' expects data from '{source}', but '{source}' is not defined in 'steps'."
                        )
                    _ordered_add(global_dependencies[global_name], source)
                elif callable(fetcher):
                    # A callable can inspect arbitrary state. The global phase is
                    # already a barrier, so represent that conservatively.
                    for source in steps:
                        _ordered_add(global_dependencies[global_name], source)
                else:
                    raise TypeError(
                        f"Invalid global data fetcher for '{output_name}': {fetcher!r}"
                    )
    if unmatched_outputs:
        output_name = next(
            name for name in global_passed_data if name in unmatched_outputs
        )
        raise ValueError(
            f"'global_passed_data' entry '{output_name}' does not match a global step."
        )

    frozen_global = {
        name: tuple(predecessors) for name, predecessors in global_dependencies.items()
    }
    combined = dict(frozen_dependencies)
    combined.update(frozen_global)
    # Global dependencies cannot currently point to other global nodes, but run
    # graphlib over the complete compiled graph to keep cycle validation local.
    tuple(TopologicalSorter(combined).static_order())

    return PipelineGraph(
        dependencies=frozen_dependencies,
        step_order=step_order,
        global_dependencies=frozen_global,
    )


def _positive_int(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{description} must be a positive int, got {value!r}.")
    return value


def _resource_amount(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{description} must be a non-negative int, got {value!r}.")
    return value


def _remote_address(parameters: dict) -> str | None:
    addresses = []
    address = parameters.get("address")
    if isinstance(address, str):
        addresses.append(address)
    segmenter_kwargs = parameters.get("segmenter_kwargs", {})
    if isinstance(segmenter_kwargs, dict):
        address = segmenter_kwargs.get("address")
        if isinstance(address, str):
            addresses.append(address)
    unique = tuple(dict.fromkeys(addresses))
    return unique[0] if len(unique) == 1 else None


def compile_resource_requirements(
    pipeline: dict,
    graph: PipelineGraph,
) -> tuple[dict[str, dict[str, int]], set[str]]:
    """Return per-step slot requirements and resources inferred as remote.

    Every local step requires one CPU slot unless explicitly set to zero.
    An explicit ``step_resources`` entry suppresses endpoint inference for that
    step. Otherwise an unambiguous existing Nahual address adds one
    ``remote:<address>`` slot.
    """
    configured = pipeline.get("step_resources", {})
    if not isinstance(configured, dict):
        raise TypeError("'step_resources' must be a dictionary.")
    known_steps = set(graph.dependencies) | set(graph.global_dependencies)
    for name, explicit in configured.items():
        if name not in known_steps:
            raise ValueError(
                f"'step_resources' references step '{name}', but it is not defined."
            )
        if not isinstance(explicit, dict):
            raise TypeError(f"'step_resources[{name}]' must be a dictionary.")
        for resource, amount in explicit.items():
            if not isinstance(resource, str) or not resource:
                raise ValueError(
                    f"Resource names for step '{name}' must be non-empty strings."
                )
            _resource_amount(amount, f"'step_resources[{name}][{resource}]'")

    requirements: dict[str, dict[str, int]] = {}
    inferred_resources: set[str] = set()
    for name in graph.dependencies:
        explicit = configured.get(name)
        if explicit is not None and not isinstance(explicit, dict):
            raise TypeError(f"'step_resources[{name}]' must be a dictionary.")
        step_requirements = {"cpu": 1}
        if explicit is not None:
            for resource, amount in explicit.items():
                if not isinstance(resource, str) or not resource:
                    raise ValueError(
                        f"Resource names for step '{name}' must be non-empty strings."
                    )
                step_requirements[resource] = _resource_amount(
                    amount, f"'step_resources[{name}][{resource}]'"
                )
        else:
            address = _remote_address(pipeline["steps"][name])
            if address is not None:
                resource = f"remote:{address}"
                step_requirements[resource] = 1
                inferred_resources.add(resource)
        requirements[name] = {
            resource: amount for resource, amount in step_requirements.items() if amount
        }
    return requirements, inferred_resources


def resolve_resources(
    pipeline: dict,
    graph: PipelineGraph,
    max_workers: int | None,
    resource_limits: Mapping[str, int] | None,
) -> tuple[int, dict[str, int], dict[str, dict[str, int]]]:
    """Validate and resolve worker count, capacities, and step requirements."""
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if resource_limits is not None and not isinstance(resource_limits, Mapping):
        raise TypeError("resource_limits must be a mapping.")

    limits = {}
    for resource, amount in (resource_limits or {}).items():
        if not isinstance(resource, str) or not resource:
            raise ValueError("Resource names must be non-empty strings.")
        limits[resource] = _positive_int(amount, f"resource limit '{resource}'")

    workers = max_workers or limits.get("cpu") or min(32, (os.cpu_count() or 1) + 4)
    limits.setdefault("cpu", workers)
    requirements, inferred_resources = compile_resource_requirements(pipeline, graph)
    for resource in inferred_resources:
        limits.setdefault(resource, 1)

    for step_name, step_requirements in requirements.items():
        for resource, amount in step_requirements.items():
            if resource not in limits:
                raise ValueError(
                    f"Step '{step_name}' requires resource '{resource}', but no limit was provided."
                )
            if amount > limits[resource]:
                raise ValueError(
                    f"Step '{step_name}' requires {amount} slots of '{resource}', "
                    f"but its limit is {limits[resource]}."
                )
    return workers, limits, requirements


def run_concurrent_dag(
    graph: PipelineGraph,
    run_node: Callable[[str], object],
    commit_node: Callable[[str, object], None],
    *,
    max_workers: int,
    resource_limits: Mapping[str, int],
    requirements: Mapping[str, Mapping[str, int]],
    tp: int,
) -> None:
    """Execute one timepoint locally with threads and bounded resource slots."""
    sorter = TopologicalSorter(graph.dependencies)
    sorter.prepare()
    rank = {name: index for index, name in enumerate(graph.step_order)}
    available = dict(resource_limits)
    ready: list[str] = []
    futures: dict[Future, str] = {}

    def fits(step_name: str) -> bool:
        return all(
            available[resource] >= amount
            for resource, amount in requirements[step_name].items()
        )

    def reserve(step_name: str, direction: int) -> None:
        for resource, amount in requirements[step_name].items():
            available[resource] -= direction * amount

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        while sorter.is_active():
            ready.extend(sorter.get_ready())
            ready.sort(key=rank.__getitem__)

            index = 0
            while index < len(ready) and len(futures) < max_workers:
                step_name = ready[index]
                if fits(step_name):
                    reserve(step_name, 1)
                    futures[executor.submit(run_node, step_name)] = step_name
                    ready.pop(index)
                else:
                    index += 1

            if not futures:
                raise RuntimeError(
                    "Concurrent scheduler could not make progress with the configured resources."
                )

            completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: rank[futures[item]]):
                step_name = futures.pop(future)
                reserve(step_name, -1)
                try:
                    result = future.result()
                except Exception as exc:
                    raise PipelineStepError(step_name, tp, exc) from exc
                commit_node(step_name, result)
                sorter.done(step_name)
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
