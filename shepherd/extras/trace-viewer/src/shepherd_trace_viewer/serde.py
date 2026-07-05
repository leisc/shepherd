"""JSON serialization for the durable trace-viewer ViewModel."""

from __future__ import annotations

from typing import Any

from shepherd_trace_viewer.model import (
    SCHEMA_VERSION,
    TraceEdge,
    TraceLane,
    TraceNode,
    TraceResource,
    TraceRun,
    TraceSource,
    TraceView,
)

LEGACY_SCHEMA_VERSIONS = frozenset({"shepherd.trace-view.v2"})


class SchemaVersionError(ValueError):
    """Raised when a ViewModel carries an unknown ``schema_version``."""

    def __init__(self, found: str) -> None:
        self.found = found
        super().__init__(f"unknown trace-view schema_version {found!r}; this viewer understands {SCHEMA_VERSION!r}")


def to_json(tv: TraceView) -> dict[str, Any]:
    """Serialize a ``TraceView`` to a JSON-compatible dict."""
    return {
        "schema_version": tv.schema_version,
        "source": _source_to_json(tv.source),
        "run": _run_to_json(tv.run),
        "lanes": [_lane_to_json(lane) for lane in tv.lanes],
        "nodes": [_node_to_json(n) for n in tv.nodes],
        "edges": [_edge_to_json(e) for e in tv.edges],
        "resources": [_resource_to_json(r) for r in tv.resources],
    }


def from_json(data: dict[str, Any]) -> TraceView:
    """Reconstruct a ``TraceView``; raise ``SchemaVersionError`` on mismatch."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION and version not in LEGACY_SCHEMA_VERSIONS:
        raise SchemaVersionError(str(version))
    return TraceView(
        schema_version=SCHEMA_VERSION,
        source=_source_from_json(data["source"]),
        run=_run_from_json(data.get("run", {})),
        lanes=tuple(_lane_from_json(lane) for lane in data.get("lanes", [])),
        nodes=tuple(_node_from_json(n) for n in data.get("nodes", [])),
        edges=tuple(_edge_from_json(e) for e in data.get("edges", [])),
        resources=tuple(_resource_from_json(r) for r in data.get("resources", [])),
    )


def _source_to_json(s: TraceSource) -> dict[str, Any]:
    return {
        "trace_runtime": s.trace_runtime,
        "trace_owner_id": s.trace_owner_id,
        "frontier_id": s.frontier_id,
        "source_kind": s.source_kind,
        "visibility_profile": s.visibility_profile,
        "mode_filter": s.mode_filter,
        "identity_domain": s.identity_domain,
        "schema": s.schema,
        "kind": s.kind,
    }


def _source_from_json(d: dict[str, Any]) -> TraceSource:
    return TraceSource(
        trace_runtime=d["trace_runtime"],
        trace_owner_id=d["trace_owner_id"],
        frontier_id=d["frontier_id"],
        source_kind=d.get("source_kind", "hybrid_revision"),
        visibility_profile=d.get("visibility_profile"),
        mode_filter=d.get("mode_filter"),
        identity_domain=d.get("identity_domain"),
        schema=d.get("schema"),
        kind=d.get("kind"),
    )


def _run_to_json(r: TraceRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "terminal_status": r.terminal_status,
        "transition": r.transition,
        "summary": r.summary,
        "detail": r.detail,
    }


def _run_from_json(d: dict[str, Any]) -> TraceRun:
    return TraceRun(
        id=d.get("id"),
        terminal_status=d.get("terminal_status"),
        transition=d.get("transition"),
        summary=dict(d.get("summary", {})),
        detail=dict(d.get("detail", {})),
    )


def _lane_to_json(lane: TraceLane) -> dict[str, Any]:
    return {"id": lane.id, "label": lane.label, "node_ids": list(lane.node_ids)}


def _lane_from_json(d: dict[str, Any]) -> TraceLane:
    return TraceLane(id=d["id"], label=d["label"], node_ids=tuple(d.get("node_ids", [])))


def _node_to_json(n: TraceNode) -> dict[str, Any]:
    return {
        "id": n.id,
        "kind": n.kind,
        "family": n.family,
        "role": n.role,
        "lane_ids": list(n.lane_ids),
        "sequence": n.sequence,
        "timestamp": n.timestamp,
        "label": n.label,
        "identity_domain": n.identity_domain,
        "record_digest": n.record_digest,
        "body": n.body,
        "payload": n.payload,
    }


def _node_from_json(d: dict[str, Any]) -> TraceNode:
    return TraceNode(
        id=d["id"],
        kind=d["kind"],
        family=d["family"],
        role=d["role"],
        lane_ids=tuple(d.get("lane_ids", [])),
        sequence=d.get("sequence"),
        timestamp=d.get("timestamp"),
        label=d.get("label", ""),
        identity_domain=d.get("identity_domain"),
        record_digest=d.get("record_digest"),
        body=dict(d.get("body", {})),
        payload=dict(d.get("payload", {})),
    )


def _edge_to_json(e: TraceEdge) -> dict[str, Any]:
    return {
        "id": e.id,
        "kind": e.kind,
        "source": e.source,
        "target": e.target,
        "label": e.label,
    }


def _edge_from_json(d: dict[str, Any]) -> TraceEdge:
    return TraceEdge(
        id=d["id"],
        kind=d["kind"],
        source=d["source"],
        target=d["target"],
        label=d.get("label", ""),
    )


def _resource_to_json(r: TraceResource) -> dict[str, Any]:
    return {"id": r.id, "kind": r.kind, "label": r.label, "detail": r.detail}


def _resource_from_json(d: dict[str, Any]) -> TraceResource:
    return TraceResource(
        id=d["id"],
        kind=d["kind"],
        label=d.get("label", ""),
        detail=dict(d.get("detail", {})),
    )
