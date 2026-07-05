"""Schema-library boundary for record builders and pure projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from ..kernel.facts import TraceSlice

ProjectionModeRequirement = Literal["declarations_only", "captures_only", "both", "any"]


class ProjectionModeError(ValueError):
    """Raised when a projection receives a slice with an incompatible mode filter."""


@dataclass(frozen=True)
class ProjectionSpec:
    """Declared requirements for a pure projection over a trace slice."""

    name: str
    mode_requirement: ProjectionModeRequirement = "both"
    requires_payload: bool = True
    accepts_anchors: bool = True


class SchemaLibrary(Protocol):
    """Minimal contract for schema-defined records and projections."""

    name: str
    schema_refs: frozenset[str]
    projection_specs: frozenset[ProjectionSpec]


@dataclass(frozen=True)
class StaticSchemaLibrary:
    """Simple immutable schema-library descriptor."""

    name: str
    schema_refs: frozenset[str]
    projection_specs: frozenset[ProjectionSpec] = frozenset()


def ensure_projection_compatible(trace_slice: TraceSlice, spec: ProjectionSpec) -> None:
    """Validate that a slice satisfies a projection's declared kernel requirements."""
    if spec.mode_requirement not in ("any", trace_slice.mode_filter):
        raise ProjectionModeError(
            f"projection {spec.name!r} requires mode_filter={spec.mode_requirement!r}; got {trace_slice.mode_filter!r}"
        )
