"""Internal workflow/run launch context for flow metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

JsonObject = dict[str, object]

FLOW_SCHEMA = "shepherd.workspace_control.flow.v1"
FLOW_RUN_SCHEMA = "shepherd.workspace_control.flow_run.v1"
FLOW_TRACE_SCHEMA = "shepherd.workspace_control.flow_trace.v1"


@dataclass(frozen=True)
class FlowRunContext:
    """Validated workflow metadata carried into run start."""

    flow_id: str
    name: str
    sequence: int
    after: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.flow_id, field_name="flow_id")
        _require_name(self.name, field_name="flow run name")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("flow run sequence must be a non-negative integer")
        for item in self.after:
            _require_name(item, field_name="flow run after ref")

    def to_record(self, *, run_ref: str, created_at: str) -> JsonObject:
        """Return the durable flow/run edge for a just-started run."""
        _require_name(run_ref, field_name="run_ref")
        _require_name(created_at, field_name="created_at")
        return {
            "schema": FLOW_RUN_SCHEMA,
            "flow_id": self.flow_id,
            "run_ref": run_ref,
            "name": self.name,
            "sequence": self.sequence,
            "after": list(self.after),
            "metadata": dict(self.metadata),
            "created_at": created_at,
        }


def _require_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
