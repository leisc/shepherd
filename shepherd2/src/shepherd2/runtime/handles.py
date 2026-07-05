"""Minimal synchronous runtime facade over the trace substrate."""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ..kernel.facts import (
    TRUSTED_APPEND_CONTEXT,
    TRUSTED_READ_CONTEXT,
    AppendBatch,
    AppendGroup,
    Fact,
    FactDraft,
)
from ..schemas.execution import (
    complete_execution_batch,
    create_execution_batch,
    execution_id_for,
    fail_execution_batch,
    project_execution,
    publish_execution_frontier,
)
from ..schemas.relations import create_execution_relation_batch, execution_relation_from_fact, relation_id_for
from ..trace_store import SQLiteTraceStore, TraceStoreError

if TYPE_CHECKING:
    from ..kernel.facts import FactId, OwnerCutoff, TraceStore
    from ..schemas.execution import Execution, ExecutionId
    from ..schemas.relations import ExecutionRelation, RelationId

T = TypeVar("T")


@dataclass(frozen=True)
class Run:
    """Live capability-bearing handle for one root execution."""

    store: TraceStore
    execution_id: ExecutionId
    frontier_id: str

    def wait(self) -> Execution:
        """Return the terminal projected execution."""
        cutoff = self.store.read_owner_cutoff(self.frontier_id)
        trace_slice = self.store.resolve_frontier(TRUSTED_READ_CONTEXT, cutoff.frontier_id)
        return project_execution(trace_slice, cutoff.target_trace_owner_id, cutoff=cutoff)

    def snapshot(self) -> Execution:
        """Return the current projected execution.

        The first runtime slice executes synchronously, so snapshot and wait
        resolve the same retained terminal cutoff.
        """
        return self.wait()

    @property
    def cutoff(self) -> OwnerCutoff:
        """Return the durable terminal read receipt for this run."""
        return self.store.read_owner_cutoff(self.frontier_id)


@dataclass(frozen=True)
class ChildHandle:
    """Live handle for a child execution related to a parent."""

    store: TraceStore
    execution_id: ExecutionId
    frontier_id: str
    relation_id: RelationId
    relation: ExecutionRelation

    def wait(self) -> Execution:
        """Return the terminal projected child execution."""
        cutoff = self.store.read_owner_cutoff(self.frontier_id)
        trace_slice = self.store.resolve_frontier(TRUSTED_READ_CONTEXT, cutoff.frontier_id)
        return project_execution(trace_slice, cutoff.target_trace_owner_id, cutoff=cutoff)

    def snapshot(self) -> Execution:
        """Return the current projected child execution."""
        return self.wait()

    @property
    def cutoff(self) -> OwnerCutoff:
        """Return the child's retained terminal cutoff."""
        return self.store.read_owner_cutoff(self.frontier_id)


class TaskControl:
    """Parent-owned control surface available during task execution."""

    def __init__(
        self,
        *,
        store: TraceStore,
        execution_id: ExecutionId,
        run_id: str,
        causal_tail: FactId,
    ) -> None:
        self.store = store
        self.execution_id = execution_id
        self.run_id = run_id
        self._causal_tail = causal_tail
        self._child_index = 0
        self._relation_index = 0
        self._publish_index = 0
        self._frontiers_by_execution: dict[ExecutionId, str] = {}

    @property
    def causal_tail(self) -> FactId:
        """Return the latest parent-visible fact for subsequent appends."""
        return self._causal_tail

    def publish(self, kind: str, data: dict[str, Any] | None = None) -> Fact:
        """Publish a parent-owned fact into the current execution trace."""
        self._publish_index += 1
        receipt = self.store.append(
            TRUSTED_APPEND_CONTEXT,
            AppendBatch(
                append_intent_id=f"{self.run_id}:publish:{self._publish_index}",
                groups=(
                    AppendGroup(
                        trace_owner_id=self.execution_id,
                        causal_parents=(self._causal_tail,),
                        fact_drafts=(
                            FactDraft(
                                mode="capture",
                                schema_ref="shepherd2.runtime.published_fact.v1",
                                kind_label="fact_published",
                                payload={"kind": kind, "data": dict(data or {})},
                            ),
                        ),
                    ),
                ),
            ),
        )
        self._causal_tail = receipt.fact_ids[-1]
        fact = self.store.read_fact(TRUSTED_READ_CONTEXT, receipt.fact_ids[-1])
        if not isinstance(fact, Fact):
            raise TypeError("runtime publish expected payload-visible fact")
        return fact

    def spawn(
        self,
        task_cls: type[T],
        *,
        inputs: dict[str, Any] | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> ChildHandle:
        """Spawn and synchronously run a child task."""
        self._child_index += 1
        child_inputs = dict(inputs or {})
        child_inputs.update(kwargs)
        child_run_id = run_id or f"{self.run_id}:child:{self._child_index}"
        child_create_intent = f"{child_run_id}:create"
        child_execution_id = execution_id_for(child_create_intent)
        child_frontier_id = f"frontier:{child_run_id}:terminal"
        relation_intent = f"{child_run_id}:relation:spawned"
        relation_id = relation_id_for(relation_intent)
        relation_receipt = self.store.append(
            TRUSTED_APPEND_CONTEXT,
            create_execution_relation_batch(
                append_intent_id=relation_intent,
                relation_id=relation_id,
                relation_kind="spawned",
                parent_execution_id=self.execution_id,
                child_execution_id=child_execution_id,
                child_frontier_id=child_frontier_id,
                caused_by=(self._causal_tail,),
            ),
        )
        relation_fact = self.store.read_fact(TRUSTED_READ_CONTEXT, relation_receipt.fact_ids[0])
        if not isinstance(relation_fact, Fact):
            raise TypeError("runtime relation projection expected payload-visible fact")
        relation = execution_relation_from_fact(relation_fact)
        self._causal_tail = relation.created_fact_id

        child_run = _run_task_sync(
            task_cls,
            store=self.store,
            run_id=child_run_id,
            inputs=child_inputs,
            parent_execution_id=self.execution_id,
            create_caused_by=(relation.created_fact_id,),
            frontier_publisher_execution_id=self.execution_id,
            frontier_caused_by=(relation.created_fact_id,),
        )
        self._frontiers_by_execution[child_run.execution_id] = child_run.frontier_id
        self._advance_to_frontier_fact(child_run.cutoff)
        return ChildHandle(
            store=self.store,
            execution_id=child_run.execution_id,
            frontier_id=child_run.frontier_id,
            relation_id=relation_id,
            relation=relation,
        )

    def await_terminal(self, handle: ChildHandle) -> Execution:
        """Observe a child terminal cutoff and return the projected execution."""
        self._frontiers_by_execution[handle.execution_id] = handle.frontier_id
        self._advance_to_frontier_fact(handle.cutoff)
        return handle.wait()

    def read_execution(self, execution_id: ExecutionId) -> Execution:
        """Read a known child execution from its retained frontier."""
        frontier_id = self._frontiers_by_execution.get(execution_id)
        if frontier_id is None:
            raise KeyError(f"execution is not known to this control: {execution_id}")
        cutoff = self.store.read_owner_cutoff(frontier_id)
        trace_slice = self.store.resolve_frontier(TRUSTED_READ_CONTEXT, cutoff.frontier_id)
        return project_execution(trace_slice, cutoff.target_trace_owner_id, cutoff=cutoff)

    def adopt(
        self,
        *,
        execution_id: ExecutionId,
        frontier_id: str,
        relation_id: RelationId | None = None,
    ) -> ChildHandle:
        """Record a parent-owned adopted relation to an existing execution."""
        self._relation_index += 1
        relation_intent = f"{self.run_id}:relation:adopted:{self._relation_index}"
        stable_relation_id = relation_id or relation_id_for(relation_intent)
        relation_receipt = self.store.append(
            TRUSTED_APPEND_CONTEXT,
            create_execution_relation_batch(
                append_intent_id=relation_intent,
                relation_id=stable_relation_id,
                relation_kind="adopted",
                parent_execution_id=self.execution_id,
                child_execution_id=execution_id,
                child_frontier_id=frontier_id,
                caused_by=(self._causal_tail,),
            ),
        )
        relation_fact = self.store.read_fact(TRUSTED_READ_CONTEXT, relation_receipt.fact_ids[0])
        if not isinstance(relation_fact, Fact):
            raise TypeError("runtime relation projection expected payload-visible fact")
        relation = execution_relation_from_fact(relation_fact)
        self._frontiers_by_execution[execution_id] = frontier_id
        self._causal_tail = relation.created_fact_id
        return ChildHandle(
            store=self.store,
            execution_id=execution_id,
            frontier_id=frontier_id,
            relation_id=stable_relation_id,
            relation=relation,
        )

    def abandon(self, handle: ChildHandle) -> ExecutionRelation:
        """Record that a related child is no longer effective for this parent."""
        self._relation_index += 1
        caused_by = tuple(
            dict.fromkeys((self._causal_tail, handle.cutoff.created_by_fact_id or handle.cutoff.through_fact_id))
        )
        relation_receipt = self.store.append(
            TRUSTED_APPEND_CONTEXT,
            create_execution_relation_batch(
                append_intent_id=f"{self.run_id}:relation:abandoned:{self._relation_index}",
                relation_id=handle.relation_id,
                relation_kind="abandoned",
                parent_execution_id=self.execution_id,
                child_execution_id=handle.execution_id,
                child_frontier_id=handle.frontier_id,
                caused_by=caused_by,
            ),
        )
        relation_fact = self.store.read_fact(TRUSTED_READ_CONTEXT, relation_receipt.fact_ids[0])
        if not isinstance(relation_fact, Fact):
            raise TypeError("runtime relation projection expected payload-visible fact")
        relation = execution_relation_from_fact(relation_fact)
        self._causal_tail = relation.created_fact_id
        return relation

    def _advance_to_frontier_fact(self, cutoff: OwnerCutoff) -> None:
        self._causal_tail = cutoff.created_by_fact_id or cutoff.through_fact_id


def task(cls: type[T]) -> type[T]:
    """Decorate a simple programmatic task class with `start`."""

    def start(
        task_cls: type[T],
        /,
        *,
        store: TraceStore | None = None,
        run_id: str | None = None,
        **inputs: Any,
    ) -> Run:
        return _start_task(task_cls, store=store, run_id=run_id, inputs=inputs)

    cls.start = classmethod(start)  # type: ignore[attr-defined]
    return cls


def _start_task(
    task_cls: type[T],
    *,
    store: TraceStore | None,
    run_id: str | None,
    inputs: dict[str, Any],
) -> Run:
    return _run_task_sync(
        task_cls,
        store=SQLiteTraceStore() if store is None else store,
        run_id=run_id or f"run:{uuid.uuid4().hex}",
        inputs=inputs,
        parent_execution_id=None,
        create_caused_by=(),
        frontier_publisher_execution_id=None,
        frontier_caused_by=(),
    )


def _run_task_sync(
    task_cls: type[T],
    *,
    store: TraceStore,
    run_id: str,
    inputs: dict[str, Any],
    parent_execution_id: ExecutionId | None,
    create_caused_by: tuple[FactId, ...],
    frontier_publisher_execution_id: ExecutionId | None,
    frontier_caused_by: tuple[FactId, ...],
) -> Run:
    trace_store = store
    stable_run_id = run_id
    create_intent = f"{stable_run_id}:create"
    complete_intent = f"{stable_run_id}:complete"
    fail_intent = f"{stable_run_id}:fail"
    frontier_id = f"frontier:{stable_run_id}:terminal"
    execution_id = execution_id_for(create_intent)
    task_ref = f"{task_cls.__module__}.{task_cls.__qualname__}"

    if _terminal_frontier_exists(trace_store, frontier_id):
        return Run(store=trace_store, execution_id=execution_id, frontier_id=frontier_id)

    create_receipt = trace_store.append(
        TRUSTED_APPEND_CONTEXT,
        create_execution_batch(
            append_intent_id=create_intent,
            execution_id=execution_id,
            task_ref=task_ref,
            inputs=inputs,
            parent_execution_id=parent_execution_id,
            caused_by=create_caused_by,
        ),
    )
    control = TaskControl(
        store=trace_store,
        execution_id=execution_id,
        run_id=stable_run_id,
        causal_tail=create_receipt.fact_ids[-1],
    )

    try:
        instance = task_cls(**inputs)
        result = _call_execute(instance, control)
    except Exception as exc:  # noqa: BLE001
        terminal_receipt = trace_store.append(
            TRUSTED_APPEND_CONTEXT,
            fail_execution_batch(
                append_intent_id=fail_intent,
                execution_id=execution_id,
                error=f"{type(exc).__name__}: {exc}",
                caused_by=(control.causal_tail,),
            ),
        )
    else:
        terminal_receipt = trace_store.append(
            TRUSTED_APPEND_CONTEXT,
            complete_execution_batch(
                append_intent_id=complete_intent,
                execution_id=execution_id,
                outputs=_coerce_outputs(result),
                caused_by=(control.causal_tail,),
            ),
        )

    publish_execution_frontier(
        trace_store,
        TRUSTED_APPEND_CONTEXT,
        frontier_id=frontier_id,
        target_execution_id=execution_id,
        through_fact_id=terminal_receipt.fact_ids[-1],
        publisher_execution_id=frontier_publisher_execution_id,
        caused_by=frontier_caused_by,
    )
    return Run(store=trace_store, execution_id=execution_id, frontier_id=frontier_id)


def _terminal_frontier_exists(store: TraceStore, frontier_id: str) -> bool:
    try:
        store.read_owner_cutoff(frontier_id)
    except TraceStoreError:
        return False
    return True


def _call_execute(instance: object, control: TaskControl) -> Any:
    execute = getattr(instance, "execute", None)
    if not callable(execute):
        raise TypeError(f"{type(instance).__name__} must define execute()")
    signature = inspect.signature(execute)
    parameters = signature.parameters
    if not parameters:
        return execute()
    if "control" in parameters:
        return execute(control=control)
    if "task_control" in parameters:
        return execute(task_control=control)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return execute(control=control)
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional) == 1 and positional[0].default is inspect.Parameter.empty:
        return execute(control)
    return execute()


def _coerce_outputs(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "model_dump"):
        model_dump = result.model_dump
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {"result": result}
