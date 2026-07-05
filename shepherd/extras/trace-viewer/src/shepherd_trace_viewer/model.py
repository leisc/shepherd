"""ViewModel dataclasses for the trace viewer schema.

The viewer consumes provider-neutral durable task-trace revisions and projected
``shepherd2.TraceSlice`` reads. Storage-specific readers normalize those sources
into this UI transport contract before the browser sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = "shepherd.trace-view.v3"

SOURCE_KINDS = ("hybrid_revision", "trace_store_slice")
NODE_ROLES = (
    "record",
    "record_shape",
    "pointer",
    "external_anchor",
    "context",
    "context_anchor",
    "witness",
    "witness_anchor",
)
EDGE_KINDS = (
    "owner_path",
    "causal",
    "external_causal_anchor",
    "context",
    "witness",
    "replay_control",
    "replay_basis",
)


@dataclass(frozen=True)
class TraceSource:
    """Durable trace revision identity and runtime metadata."""

    trace_runtime: str
    trace_owner_id: str
    frontier_id: str
    source_kind: str = "hybrid_revision"
    visibility_profile: str | None = None
    mode_filter: str | None = None
    identity_domain: str | None = None
    schema: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class TraceRun:
    """Run-level summary derived from durable trace events."""

    id: str | None = None
    terminal_status: str | None = None
    transition: str | None = None
    summary: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TraceLane:
    """Ordered owner path through trace events."""

    id: str
    label: str
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceNode:
    """One durable trace event projected for display."""

    id: str
    kind: str
    family: str
    role: str
    lane_ids: tuple[str, ...] = ()
    sequence: int | None = None
    timestamp: float | None = None
    label: str = ""
    identity_domain: str | None = None
    record_digest: str | None = None
    body: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEdge:
    """A typed edge between two trace events."""

    id: str
    kind: str
    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class TraceResource:
    """External durable resource cited by the trace, such as a world oid."""

    id: str
    kind: str
    label: str = ""
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TraceView:
    """Top-level ViewModel served at ``/api/trace``."""

    source: TraceSource
    run: TraceRun
    lanes: tuple[TraceLane, ...] = ()
    nodes: tuple[TraceNode, ...] = ()
    edges: tuple[TraceEdge, ...] = ()
    resources: tuple[TraceResource, ...] = ()
    schema_version: str = SCHEMA_VERSION
