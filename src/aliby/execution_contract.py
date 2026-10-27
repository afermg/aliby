"""Scheduler-independent *target* contract derived from existing pipeline wiring.

This is not an executor or a second user-authored workflow format. It records
conservative temporal/resource/publication constraints for a future async
adapter. Existing sequential/concurrent runners do not consume this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from graphlib import TopologicalSorter
import hashlib
import json

from aliby.assay import AssayDecision
from aliby.executor import compile_pipeline_graph, resolve_resources


@dataclass(frozen=True)
class TaskContract:
    key: str
    kind: str
    dependencies: tuple[str, ...]
    step: str | None = None
    timepoint: int | None = None
    session_key: str | None = None
    resources: tuple[tuple[str, int], ...] = (("cpu", 1),)
    retry: str = "never"


@dataclass(frozen=True)
class ExecutionContract:
    site_id: str
    attempt_id: str
    pipeline_identity: str
    assay_fingerprint: str
    tasks: tuple[TaskContract, ...]
    resource_limits: tuple[tuple[str, int], ...]
    max_workers: int
    failure_policy: str = "abort_site_restart_from_zero"

    def to_dict(self):
        return {
            "schema": "aliby.execution-contract.v1",
            "executable": False,
            **asdict(self),
        }

    @property
    def fingerprint(self):
        value = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(value.encode()).hexdigest()


def compile_execution_contract(
    pipeline: dict,
    decision: AssayDecision,
    *,
    attempt_id: str,
    pipeline_identity: str,
    max_workers: int = 1,
    resource_limits: dict[str, int] | None = None,
    stateful_steps: tuple[str, ...] = (),
) -> ExecutionContract:
    """Expand the current graph into frame barriers and a release gate.

    ``pipeline_identity`` identifies the frozen configuration/code binding;
    it is caller-supplied, not a hash inferred from arbitrary Python callables.
    Channel/model/Z/feature adapters must still be implemented and validated
    before this target contract can be executed. No legacy dictionary is
    modified and no declared assay input is silently injected into a backend.
    """
    decision.require_ready()
    for name, value in (
        ("attempt_id", attempt_id),
        ("pipeline_identity", pipeline_identity),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be explicit and nonempty")
    ntps = pipeline.get("ntps", 1)
    if type(ntps) is not int or ntps != decision.request.ntps:
        raise ValueError("pipeline ntps must equal the declared assay interval")
    graph = compile_pipeline_graph(pipeline)
    if not graph.step_order:
        raise ValueError("execution contract requires per-frame steps")
    workers, limits, requirements = resolve_resources(
        pipeline, graph, max_workers, resource_limits
    )
    stateful = set(stateful_steps)
    if not stateful.issubset(graph.dependencies):
        raise ValueError("stateful_steps references an unknown per-frame step")
    selected = dict(decision.selected_models)
    capabilities = {m.model_id: m for m in decision.models}
    for seg in decision.request.segmentations:
        name = f"segment_{seg.object_name}"
        if name not in graph.dependencies:
            raise ValueError(f"assay object has no corresponding pipeline step: {name}")
        if capabilities[selected[seg.object_name]].stateful:
            stateful.add(name)
    # Legacy tiling/drift and trackers carry mutable history. BABY advances a
    # remote crawler inside segment, even when no separate track step exists.
    for name, parameters in pipeline["steps"].items():
        if name.startswith(("tile", "track")):
            stateful.add(name)
        if (
            isinstance(parameters, dict)
            and parameters.get("segmenter_kwargs", {}).get("kind") == "nahual_baby"
        ):
            if len([n for n in graph.dependencies if n.startswith("tile")]) != 1:
                raise ValueError("BABY requires an unambiguous single tiler binding")
            stateful.add(name)

    tasks = []
    previous_barrier = None
    for tp in range(ntps):
        keys = {name: f"frame/{tp}/step/{name}" for name in graph.step_order}
        for name in graph.step_order:
            dependencies = tuple(keys[p] for p in graph.dependencies[name])
            if not dependencies and previous_barrier is not None:
                dependencies = (previous_barrier,)
            session = None
            if name in stateful:
                session = json.dumps(
                    [decision.request.site_id, attempt_id, name], separators=(",", ":")
                )
            tasks.append(
                TaskContract(
                    keys[name],
                    "step",
                    dependencies,
                    name,
                    tp,
                    session,
                    tuple(sorted(requirements[name].items())),
                )
            )
        # Includes ordered writes, receipt commit, and retention/tiler cleanup.
        # Future frame work cannot race a method call into a mutable tiler.
        previous_barrier = f"frame/{tp}/commit"
        tasks.append(
            TaskContract(
                previous_barrier, "commit_frame", tuple(keys.values()), timepoint=tp
            )
        )

    # Current global fetchers can inspect arbitrary history; retain a complete
    # time-series barrier and conservative serial global execution.
    tail = previous_barrier
    for name in graph.global_dependencies:
        key = f"global/{name}"
        tasks.append(TaskContract(key, "global_step", (tail,), step=name))
        tail = key
    tasks.append(TaskContract("validate_release", "validate_release", (tail,)))
    tasks.append(
        TaskContract("publish_release", "publish_release", ("validate_release",))
    )
    dependencies = {t.key: t.dependencies for t in tasks}
    if len(dependencies) != len(tasks) or any(
        d not in dependencies for t in tasks for d in t.dependencies
    ):
        raise ValueError("invalid generated task keys or dependencies")
    tuple(TopologicalSorter(dependencies).static_order())
    return ExecutionContract(
        decision.request.site_id,
        attempt_id,
        pipeline_identity,
        decision.fingerprint,
        tuple(tasks),
        tuple(sorted(limits.items())),
        workers,
    )
