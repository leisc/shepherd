"""Durable task-run launch envelope and local sidecar normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from shepherd_dialect.runtime_options import RuntimeOptions, parse_runtime_options

JsonObject = dict[str, object]
TASK_RUN_ENVELOPE_SCHEMA = "shepherd.task_run_envelope.v1"
LaunchSurface = Literal["python", "operator", "provider"]

_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "args",
        "may",
        "runtime",
        "parent_run_ref",
        "caused_by",
        "launch_surface",
        "settlement_policy",
    }
)
_CONFLICTING_TOP_LEVEL_FIELDS = frozenset({"task_id", "args", "may", "runtime"})


class TaskRunEnvelopeError(ValueError):
    """Raised when a task-run envelope is malformed."""


@dataclass(frozen=True)
class TaskRunEnvelope:
    """Durable, serializable intent for launching one task run."""

    task_id: str
    args: JsonObject = field(default_factory=dict)
    may: str | None = None
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)
    parent_run_ref: str | None = None
    caused_by: str | None = None
    launch_surface: LaunchSurface = "python"
    settlement_policy: JsonObject | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskRunEnvelopeError("envelope.task_id must be a non-empty string")
        if not isinstance(self.args, Mapping):
            raise TaskRunEnvelopeError("envelope.args must be an object")
        if self.may is not None and (not isinstance(self.may, str) or not self.may.strip()):
            raise TaskRunEnvelopeError("envelope.may must be a non-empty string when supplied")
        if not isinstance(self.runtime, RuntimeOptions):
            raise TaskRunEnvelopeError("envelope.runtime must be RuntimeOptions")
        for field_name, value in (
            ("parent_run_ref", self.parent_run_ref),
            ("caused_by", self.caused_by),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskRunEnvelopeError(f"envelope.{field_name} must be a non-empty string when supplied")
        if self.launch_surface not in ("python", "operator", "provider"):
            raise TaskRunEnvelopeError(f"unsupported envelope.launch_surface: {self.launch_surface!r}")
        if self.settlement_policy is not None and not isinstance(self.settlement_policy, Mapping):
            raise TaskRunEnvelopeError("envelope.settlement_policy must be an object when supplied")

    def to_payload(self) -> JsonObject:
        payload: JsonObject = {
            "schema": TASK_RUN_ENVELOPE_SCHEMA,
            "task_id": self.task_id,
            "args": dict(self.args),
            "runtime": self.runtime.to_payload(),
            "launch_surface": self.launch_surface,
        }
        if self.may is not None:
            payload["may"] = self.may
        if self.parent_run_ref is not None:
            payload["parent_run_ref"] = self.parent_run_ref
        if self.caused_by is not None:
            payload["caused_by"] = self.caused_by
        if self.settlement_policy is not None:
            payload["settlement_policy"] = dict(self.settlement_policy)
        return payload


@dataclass(frozen=True)
class TaskRunSidecars:
    """Process-local run launch authority and Python objects."""

    task_body: Any | None = None
    provider: Any | None = None
    substrate_handlers: Sequence[Mapping[type, Any]] = ()
    supervisor_handlers: Sequence[Mapping[type, Any]] = ()


@dataclass(frozen=True)
class NormalizedTaskRun:
    """Canonical run launch shape: durable envelope plus local sidecars."""

    envelope: TaskRunEnvelope
    sidecars: TaskRunSidecars = field(default_factory=TaskRunSidecars)


def parse_task_run_envelope(value: object) -> TaskRunEnvelope:
    """Parse a durable task-run envelope payload."""
    if isinstance(value, TaskRunEnvelope):
        return value
    if not isinstance(value, Mapping):
        raise TaskRunEnvelopeError(f"envelope must be an object, got {type(value).__name__}")
    fields = set(value)
    unknown = sorted(fields - _ENVELOPE_FIELDS)
    if unknown:
        raise TaskRunEnvelopeError(f"unknown envelope field(s): {', '.join(unknown)}")
    schema = value.get("schema")
    if schema != TASK_RUN_ENVELOPE_SCHEMA:
        raise TaskRunEnvelopeError(f"unsupported envelope schema: {schema!r}")
    task_id = value.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskRunEnvelopeError("envelope.task_id must be a non-empty string")
    args = value.get("args", {})
    if not isinstance(args, Mapping):
        raise TaskRunEnvelopeError("envelope.args must be an object")
    may = value.get("may")
    if may is not None and not isinstance(may, str):
        raise TaskRunEnvelopeError("envelope.may must be a string when supplied")
    parent_run_ref = _optional_str(value, "parent_run_ref")
    caused_by = _optional_str(value, "caused_by")
    launch_surface = value.get("launch_surface", "python")
    if launch_surface not in ("python", "operator", "provider"):
        raise TaskRunEnvelopeError(f"unsupported envelope.launch_surface: {launch_surface!r}")
    settlement_policy = value.get("settlement_policy")
    if settlement_policy is not None and not isinstance(settlement_policy, Mapping):
        raise TaskRunEnvelopeError("envelope.settlement_policy must be an object when supplied")
    return TaskRunEnvelope(
        task_id=task_id,
        args=dict(args),
        may=may,
        runtime=parse_runtime_options(value.get("runtime")),
        parent_run_ref=parent_run_ref,
        caused_by=caused_by,
        launch_surface=launch_surface,
        settlement_policy=dict(settlement_policy) if isinstance(settlement_policy, Mapping) else None,
    )


def normalize_task_run_params(params: Mapping[str, Any]) -> NormalizedTaskRun:
    """Normalize legacy runtime.run params or an explicit envelope into one shape."""
    envelope_value = params.get("envelope")
    sidecars = TaskRunSidecars(
        task_body=params.get("task_body"),
        provider=params.get("provider"),
        substrate_handlers=params.get("substrate_handlers", ()),
        supervisor_handlers=params.get("supervisor_handlers", ()),
    )
    if envelope_value is not None:
        supplied_conflicts = sorted(field for field in _CONFLICTING_TOP_LEVEL_FIELDS if field in params)
        if supplied_conflicts:
            raise TaskRunEnvelopeError(
                "envelope cannot be combined with top-level run field(s): " + ", ".join(supplied_conflicts)
            )
        return NormalizedTaskRun(envelope=parse_task_run_envelope(envelope_value), sidecars=sidecars)

    task_body = params.get("task_body")
    task_id = params.get("task_id")
    if (task_body is None) == (task_id is None):
        if task_body is None:
            raise TaskRunEnvelopeError("exactly one of 'envelope' / 'task_body' / 'task_id' must be supplied")
        raise TaskRunEnvelopeError("accepts only one of: task_body, task_id")
    if task_id is None:
        task_id = f"{getattr(task_body, '__module__', '?')}:{getattr(task_body, '__qualname__', '?')}"
    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskRunEnvelopeError("'task_id' must be a non-empty string")
    args = params.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise TaskRunEnvelopeError("'args' must be an object")
    may = params.get("may")
    if may is not None and not isinstance(may, str):
        raise TaskRunEnvelopeError("'may' must be a string when supplied")
    return NormalizedTaskRun(
        envelope=TaskRunEnvelope(
            task_id=task_id,
            args=dict(args),
            may=may,
            runtime=parse_runtime_options(params.get("runtime")),
        ),
        sidecars=sidecars,
    )


def task_run_envelope_payload(envelope: TaskRunEnvelope | Mapping[str, object]) -> JsonObject:
    """Return a canonical task-run envelope payload."""
    return parse_task_run_envelope(envelope).to_payload()


def _optional_str(value: Mapping[str, object], field_name: str) -> str | None:
    raw = value.get(field_name)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise TaskRunEnvelopeError(f"envelope.{field_name} must be a non-empty string when supplied")
    return raw
