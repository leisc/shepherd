"""Storage-neutral projections into the Trace Viewer ViewModel."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shepherd_trace_viewer.model import TraceEdge, TraceLane, TraceNode, TraceResource, TraceRun, TraceSource, TraceView

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shepherd2 import TraceSlice


class TraceProjectionError(RuntimeError):
    """Raised when a trace slice cannot be projected faithfully."""


@dataclass(frozen=True)
class TraceSliceSelection:
    """Viewer metadata about how a TraceSlice was selected."""

    selector: str
    selector_value: str
    store_path: str | None = None


@dataclass(frozen=True)
class _ProjectedOccurrence:
    """Viewer-local occurrence DTO for older TraceSlice shapes."""

    trace_owner_id: str
    owner_ordinal: int
    record_id: str
    retained_context_ref: str = ""
    kind_label: str = ""


def project_trace_slice_to_view(trace_slice: TraceSlice, *, selection: TraceSliceSelection) -> TraceView:
    """Project a storage-neutral ``TraceSlice`` into the viewer transport model."""
    occurrences = _occurrences(trace_slice)
    occurrence_ids = [_occurrence_id(occurrence) for occurrence in occurrences]
    occurrence_ids_by_record: dict[str, list[str]] = defaultdict(list)
    occurrences_by_id = {}
    for occurrence, node_id in zip(occurrences, occurrence_ids, strict=True):
        occurrence_ids_by_record[str(occurrence.record_id)].append(node_id)
        occurrences_by_id[node_id] = occurrence

    lanes = _lanes(occurrences)
    nodes = _record_nodes(trace_slice, occurrences)
    node_ids = {node.id for node in nodes}
    extra_nodes, extra_edges, resources = _support_nodes_and_edges(trace_slice, occurrences, node_ids)
    edges = (
        tuple(_owner_edges(lanes))
        + tuple(_causal_edges(trace_slice, occurrence_ids_by_record, resources))
        + tuple(extra_edges)
    )
    all_nodes = nodes + tuple(extra_nodes)
    all_nodes, lanes, edges = _compact_execution_created_nodes(all_nodes, lanes, edges)
    run = _run(trace_slice, all_nodes, edges, selection)
    source = _source(trace_slice, selection)
    return TraceView(
        source=source, run=run, lanes=lanes, nodes=all_nodes, edges=edges, resources=tuple(resources.values())
    )


def _compact_execution_created_nodes(
    nodes: tuple[TraceNode, ...],
    lanes: tuple[TraceLane, ...],
    edges: tuple[TraceEdge, ...],
) -> tuple[tuple[TraceNode, ...], tuple[TraceLane, ...], tuple[TraceEdge, ...]]:
    """Hide execution-created bookkeeping nodes and preserve the graph.

    Execution-created records mostly exist to allocate an execution owner. In an
    event-level trace they show up as a detached-looking first node in every
    lane, while the useful story starts at ``execution_started`` and then flows
    into provider/model/tool/workspace records. The stored facts remain in the
    trace store; this is only a viewer projection compaction.
    """
    hidden = {node.id for node in nodes if node.kind == "execution_created"}
    if not hidden:
        return nodes, lanes, edges

    replacement: dict[str, str] = {}
    compacted_lanes: list[TraceLane] = []
    for lane in lanes:
        kept = tuple(node_id for node_id in lane.node_ids if node_id not in hidden)
        compacted_lanes.append(TraceLane(id=lane.id, label=lane.label, node_ids=kept))
        for index, node_id in enumerate(lane.node_ids):
            if node_id not in hidden:
                continue
            replacement[node_id] = next(
                (candidate for candidate in lane.node_ids[index + 1 :] if candidate not in hidden), ""
            )

    compacted_nodes = tuple(node for node in nodes if node.id not in hidden)
    visible_ids = {node.id for node in compacted_nodes}
    compacted_edges: list[TraceEdge] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        source = replacement.get(edge.source, edge.source)
        target = replacement.get(edge.target, edge.target)
        if not source or not target or source == target or source not in visible_ids or target not in visible_ids:
            continue
        key = (edge.kind, source, target, edge.label)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        compacted_edges.append(
            TraceEdge(
                id=f"{edge.kind}:{source}->{target}",
                kind=edge.kind,
                source=source,
                target=target,
                label=edge.label,
            )
        )
    return compacted_nodes, tuple(compacted_lanes), tuple(compacted_edges)


def _occurrences(trace_slice: TraceSlice) -> tuple[Any, ...]:
    occurrences = tuple(getattr(trace_slice, "occurrences", ()) or ())
    if occurrences:
        return occurrences

    flattened = [record_id for path in trace_slice.owner_paths.values() for record_id in path]
    duplicates = {record_id for record_id, count in Counter(flattened).items() if count > 1}
    if duplicates:
        raise TraceProjectionError(
            "TraceSlice does not expose occurrences, and owner_paths contain repeated record ids: "
            + ", ".join(sorted(map(str, duplicates)))
        )

    # Compatibility fallback for older slices. It is only safe when each selected
    # record id appears once, because the slice otherwise lacks path ordinals.
    projected = []
    for owner, path in trace_slice.owner_paths.items():
        for index, record_id in enumerate(path):
            fact = trace_slice.facts_by_id.get(record_id)
            view = getattr(fact, "view", None)
            projected.append(
                _ProjectedOccurrence(
                    trace_owner_id=owner,
                    owner_ordinal=getattr(view, "owner_ordinal", index),
                    record_id=record_id,
                    retained_context_ref=getattr(view, "retained_context_ref", ""),
                    kind_label=getattr(view, "kind_label", ""),
                )
            )
    return tuple(projected)


def _record_nodes(trace_slice: TraceSlice, occurrences: tuple[Any, ...]) -> tuple[TraceNode, ...]:
    nodes: list[TraceNode] = []
    for sequence, occurrence in enumerate(occurrences):
        fact = trace_slice.facts_by_id.get(occurrence.record_id)
        if fact is None:
            continue
        envelope = fact.envelope
        is_shape = fact.__class__.__name__.endswith("Shape")
        kind = occurrence.kind_label or getattr(fact, "fact_kind", "") or envelope.schema_ref
        body = dict(getattr(getattr(fact, "body", None), "payload", {}) or {})
        nodes.append(
            TraceNode(
                id=_occurrence_id(occurrence),
                kind=kind,
                family=_family(kind, envelope.schema_ref),
                role="record_shape" if is_shape else "record",
                lane_ids=(occurrence.trace_owner_id,),
                sequence=occurrence.owner_ordinal if occurrence.owner_ordinal >= 0 else sequence,
                label=_record_label(kind, body, occurrence.record_id),
                record_digest=envelope.digest,
                body=body,
                payload={
                    "record_id": occurrence.record_id,
                    "schema_ref": envelope.schema_ref,
                    "mode": envelope.mode,
                    "witness_ref": envelope.witness_ref,
                    "retained_context_ref": occurrence.retained_context_ref,
                    "trace_owner_id": occurrence.trace_owner_id,
                    "owner_ordinal": occurrence.owner_ordinal,
                    "visibility_profile": trace_slice.visibility_profile,
                    "hidden_reason": getattr(fact, "hidden_reason", None),
                },
            )
        )
    return tuple(nodes)


def _support_nodes_and_edges(
    trace_slice: TraceSlice,
    occurrences: tuple[Any, ...],
    node_ids: set[str],
) -> tuple[list[TraceNode], list[TraceEdge], dict[str, TraceResource]]:
    nodes: list[TraceNode] = []
    edges: list[TraceEdge] = []
    resources: dict[str, TraceResource] = {}

    for context_id, context in trace_slice.contexts_by_id.items():
        resources[f"context:{context_id}"] = TraceResource(
            id=f"context:{context_id}",
            kind="context",
            label=_short_ref(context_id, "context"),
            detail={"context_id": context_id, **_public_attrs(context)},
        )
    for anchor in trace_slice.context_anchors:
        resources[f"context-anchor:{anchor.context_id}"] = TraceResource(
            id=f"context-anchor:{anchor.context_id}",
            kind="context_anchor",
            label=_short_ref(anchor.context_id, "context"),
            detail={
                "context_id": anchor.context_id,
                "hidden_reason": anchor.hidden_reason,
                "visible_shape": dict(anchor.visible_shape),
            },
        )

    for witness_ref, witness in trace_slice.visible_witnesses_by_id.items():
        resources[f"witness:{_record_prefix(witness_ref)}"] = TraceResource(
            id=f"witness:{_record_prefix(witness_ref)}",
            kind="witness",
            label=_record_prefix(witness_ref),
            detail={
                "record_id": witness_ref,
                "schema_ref": witness.envelope.schema_ref,
                "mode": witness.envelope.mode,
                "witness_ref": witness.envelope.witness_ref,
                "digest": witness.envelope.digest,
                "body": dict(getattr(getattr(witness, "body", None), "payload", {}) or {}),
            },
        )
    for anchor in trace_slice.witness_anchors:
        resources[f"witness-anchor:{_record_prefix(anchor.witness_ref)}"] = TraceResource(
            id=f"witness-anchor:{_record_prefix(anchor.witness_ref)}",
            kind="witness_anchor",
            label=_record_prefix(anchor.witness_ref),
            detail={
                "record_id": anchor.witness_ref,
                "hidden_reason": anchor.hidden_reason,
                "visible_shape": dict(anchor.visible_shape),
            },
        )

    target_lanes_by_external_ref: dict[str, set[str]] = defaultdict(set)
    target_sequence_by_external_ref: dict[str, int] = {}
    for occurrence in occurrences:
        fact = trace_slice.facts_by_id.get(occurrence.record_id)
        if fact is None:
            continue
        for parent in fact.envelope.caused_by_fact_ids:
            target_lanes_by_external_ref[parent].add(occurrence.trace_owner_id)
            target_sequence_by_external_ref.setdefault(parent, max(0, occurrence.owner_ordinal - 1))

    for anchor in trace_slice.external_anchors:
        node_id = f"external:{_record_prefix(anchor.ref)}"
        nodes.append(
            TraceNode(
                id=node_id,
                kind=anchor.anchor_kind,
                family="external",
                role="external_anchor",
                lane_ids=tuple(sorted(target_lanes_by_external_ref.get(anchor.ref, ()))),
                sequence=target_sequence_by_external_ref.get(anchor.ref),
                label=_external_anchor_label(anchor),
                payload={
                    "record_id": anchor.ref,
                    "hidden_reason": anchor.hidden_reason,
                    "visible_shape": dict(anchor.visible_shape),
                },
            )
        )

    for occurrence in occurrences:
        node_id = _occurrence_id(occurrence)
        fact = trace_slice.facts_by_id.get(occurrence.record_id)
        if fact is None:
            continue
        for anchor in trace_slice.external_anchors:
            if anchor.ref not in fact.envelope.caused_by_fact_ids:
                continue
            external_id = f"external:{_record_prefix(anchor.ref)}"
            if external_id in node_ids or any(node.id == external_id for node in nodes):
                edges.append(
                    TraceEdge(
                        id=f"external:{external_id}->{node_id}",
                        kind="external_causal_anchor",
                        source=external_id,
                        target=node_id,
                        label="external",
                    )
                )

    return nodes, edges, resources


def _lanes(occurrences: tuple[Any, ...]) -> tuple[TraceLane, ...]:
    node_ids_by_lane: dict[str, list[str]] = defaultdict(list)
    for occurrence in occurrences:
        node_ids_by_lane[occurrence.trace_owner_id].append(_occurrence_id(occurrence))
    return tuple(
        TraceLane(id=lane_id, label=lane_id, node_ids=tuple(node_ids)) for lane_id, node_ids in node_ids_by_lane.items()
    )


def _owner_edges(lanes: tuple[TraceLane, ...]) -> Iterable[TraceEdge]:
    for lane in lanes:
        for index, (source, target) in enumerate(zip(lane.node_ids, lane.node_ids[1:], strict=False)):
            yield TraceEdge(
                id=f"owner:{lane.id}:{index}:{source}->{target}",
                kind="owner_path",
                source=source,
                target=target,
                label=lane.label,
            )


def _causal_edges(
    trace_slice: TraceSlice,
    occurrence_ids_by_record: dict[str, list[str]],
    resources: dict[str, TraceResource],
) -> Iterable[TraceEdge]:
    for index, (source_record, target_record) in enumerate(trace_slice.causal_edges):
        source_ids = occurrence_ids_by_record.get(str(source_record), [])
        target_ids = occurrence_ids_by_record.get(str(target_record), [])
        if len(source_ids) == 1 and len(target_ids) == 1:
            kind, label = _causal_edge_kind(trace_slice, str(source_record), str(target_record))
            yield TraceEdge(
                id=f"causal:{index}:{source_ids[0]}->{target_ids[0]}",
                kind=kind,
                source=source_ids[0],
                target=target_ids[0],
                label=label,
            )
            continue
        resource_id = f"causal-ambiguous:{index}"
        resources[resource_id] = TraceResource(
            id=resource_id,
            kind="causal_ambiguous",
            label="ambiguous causal edge",
            detail={
                "source_record_id": source_record,
                "target_record_id": target_record,
                "source_occurrence_ids": source_ids,
                "target_occurrence_ids": target_ids,
            },
        )


def _source(trace_slice: TraceSlice, selection: TraceSliceSelection) -> TraceSource:
    frontier = trace_slice.frontier
    trace_owner_id = frontier.target_trace_owner_id if frontier is not None else selection.selector_value
    frontier_id = frontier.frontier_id if frontier is not None else ""
    return TraceSource(
        trace_runtime="shepherd2.TraceStore",
        trace_owner_id=trace_owner_id,
        frontier_id=frontier_id,
        source_kind="trace_store_slice",
        visibility_profile=trace_slice.visibility_profile,
        mode_filter=trace_slice.mode_filter,
        schema="shepherd2.TraceSlice",
        kind=selection.selector,
    )


def _run(
    trace_slice: TraceSlice,
    nodes: tuple[TraceNode, ...],
    edges: tuple[TraceEdge, ...],
    selection: TraceSliceSelection,
) -> TraceRun:
    roles = Counter(node.role for node in nodes)
    kinds = Counter(node.kind for node in nodes)
    return TraceRun(
        id=trace_slice.frontier.frontier_id if trace_slice.frontier is not None else selection.selector_value,
        summary={
            "events": len([node for node in nodes if node.role in {"record", "record_shape"}]),
            "lanes": len(trace_slice.owner_paths),
            "edges": len(edges),
            "roles": dict(sorted(roles.items())),
            "kinds": dict(sorted(kinds.items())),
        },
        detail={
            "selector": selection.selector,
            "selector_value": selection.selector_value,
            "store_path": selection.store_path,
            "visibility_profile": trace_slice.visibility_profile,
            "mode_filter": trace_slice.mode_filter,
        },
    )


def _occurrence_id(occurrence: Any) -> str:
    return f"occ:{occurrence.trace_owner_id}:{occurrence.owner_ordinal}:{_record_prefix(occurrence.record_id)}"


def _record_prefix(record_id: str) -> str:
    value = str(record_id)
    return value.removeprefix("sha256:")[:16] or value[:16]


def _record_label(kind: str, body: dict[str, Any], record_id: str) -> str:
    if kind == "execution_created":
        task_ref = str(body.get("task_ref") or "")
        return f"create {_short_task_ref(task_ref)}" if task_ref else "create execution"
    if kind == "execution_started":
        return "started"
    if kind == "execution_completed":
        return "completed"
    if kind == "execution_failed":
        return "failed"
    if kind in {"execution_relation_created", "relation_created"}:
        relation_kind = str(body.get("relation_kind") or "relation")
        return "replay" if relation_kind == "replayed" else relation_kind
    if kind == "frontier_published":
        return "publish frontier"
    if kind in {"provider.invocation.started", "provider.invocation.completed", "provider.invocation.failed"}:
        provider = str(body.get("provider_id") or "provider")
        state = kind.rsplit(".", 1)[-1]
        return f"provider {state} · {provider}"
    if kind == "model.call":
        model = str(body.get("model") or "model")
        return f"model call · {model}"
    if kind == "model.turn":
        model = str(body.get("model") or "model")
        return f"model turn · {model}"
    if kind == "tool.call":
        tool = _tool_label(body)
        return f"tool start · {tool}" if tool else "tool start"
    if kind == "tool.result":
        tool = _tool_label(body)
        status = str(body.get("status") or "").strip()
        state = "ok" if status in {"", "ok"} else status
        return f"tool {state} · {tool}" if tool else f"tool {state}"
    if kind == "workspace.write":
        path = str(body.get("path") or "workspace")
        return f"write {path}"
    if kind == "fact_published":
        published_kind = str(body.get("kind") or "fact")
        if published_kind == "checkpoint":
            return "checkpoint"
        if published_kind == "attempt_started":
            data = body.get("data")
            strategy = data.get("strategy") if isinstance(data, dict) else None
            return f"try {strategy}" if isinstance(strategy, str) and strategy else "attempt started"
        if published_kind == "attempt_failed":
            return "failed"
        if published_kind == "revert_requested":
            return "revert"
        if published_kind == "replay_observed":
            return "replay observed"
        return f"publish {published_kind}"
    for key in ("name", "task_ref", "execution_id", "kind"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return kind or _record_prefix(record_id)


def _tool_label(body: dict[str, Any]) -> str:
    payload = body.get("payload")
    if isinstance(payload, dict):
        for key in ("tool_name", "canonical_tool_name"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    value = body.get("tool_call_id")
    return str(value) if isinstance(value, str) and value else ""


def _family(kind: str, schema_ref: str) -> str:
    value = kind or schema_ref
    if "." in value:
        return value.split(".", 1)[0]
    if ":" in value:
        return value.split(":", 1)[0]
    if "_" in value:
        return value.split("_", 1)[0]
    return value or "record"


def _short_task_ref(task_ref: str) -> str:
    if not task_ref:
        return "task"
    return task_ref.rsplit(".", 1)[-1].rsplit("<locals>.", 1)[-1]


def _short_ref(value: str, prefix: str) -> str:
    if not value:
        return prefix
    return f"{prefix}:{_record_prefix(value)}"


def _external_anchor_label(anchor: Any) -> str:
    shape = getattr(anchor, "visible_shape", {}) or {}
    kind_label = str(shape.get("kind_label") or shape.get("schema_ref") or "").strip()
    if kind_label:
        return f"external {_record_label(kind_label, {}, str(anchor.ref))}"
    return f"external {_record_prefix(anchor.ref)}"


def _causal_edge_kind(trace_slice: TraceSlice, source_record: str, target_record: str) -> tuple[str, str]:
    target = trace_slice.facts_by_id.get(target_record)
    body = dict(getattr(getattr(target, "body", None), "payload", {}) or {})
    if body.get("relation_kind") != "replayed":
        return "causal", "causal"
    if body.get("replay_control_fact_id") == source_record:
        return "replay_control", "replay"
    if body.get("replay_basis_fact_id") == source_record:
        return "replay_basis", "basis"
    return "causal", "causal"


def _public_attrs(value: Any) -> dict[str, Any]:
    return {
        key: attr
        for key in dir(value)
        if not key.startswith("_") and isinstance((attr := getattr(value, key)), str | int | float | bool | tuple)
    }
