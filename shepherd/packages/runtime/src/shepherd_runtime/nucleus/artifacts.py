"""Function-form ``emit_artifact`` verb.

Replacement for class-form ``Artifact(str)`` field markers. Per
CONTRACTS A1 (Workspace handlers) and the day-1 minimal cut, the
verb produces a durable artifact during task execution; the artifact
is collected on ``Run[T].artifacts`` after the task completes.

Pinned by `docs/design/proposed/260505-plans/CONTRACTS.md` A1
(seven-verb day-1 surface) + DECISIONS D5 (class-form Artifact
markers are deletion targets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shepherd_runtime.sync import run_sync

from .delivery import active_task_run
from .types import NoActiveTaskRun

__all__ = ["Artifact", "emit_artifact"]


@dataclass(frozen=True)
class Artifact:
    """Durable artifact produced during a task run.

    Frozen dataclass per DECISIONS D17 (cross-plan trace boundary
    types are frozen dataclasses with JSON-compatible leaves). The
    ``content`` accepts ``str`` or ``bytes`` for text vs. binary
    artifacts; ``metadata`` is a JSON-serializable dict for
    provider-specific extras.
    """

    kind: str
    name: str
    content: str | bytes
    metadata: dict[str, Any] = field(default_factory=dict)


def emit_artifact(
    *,
    content: str | bytes,
    kind: str,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Emit a durable artifact during task execution.

    The artifact appears on ``Run[T].artifacts`` after the task
    completes (in emission order). Returns the constructed
    ``Artifact`` so callers can cite it locally; the framework owns
    the persistence path.

    Sync/async dispatch (CONTRACTS A8): inside an async ``@task``
    body, returns a coroutine that callers ``await``. Inside a sync
    ``@task`` body, blocks via the same ``run_sync`` helper that
    ``deliver()`` uses and returns the ``Artifact`` directly.

    Raises:
        NoActiveTaskRun: called outside a ``@task``-decorated body.
    """
    context = active_task_run()
    if context is None:
        raise NoActiveTaskRun("emit_artifact() requires an active task run")
    coroutine = _emit_artifact_async(content=content, kind=kind, name=name, metadata=metadata)
    if context.is_async:
        return coroutine  # type: ignore[return-value]
    return run_sync(coroutine)


async def _emit_artifact_async(
    *,
    content: str | bytes,
    kind: str,
    name: str,
    metadata: dict[str, Any] | None,
) -> Artifact:
    context = active_task_run()
    if context is None:
        raise NoActiveTaskRun("emit_artifact() requires an active task run")
    artifact = Artifact(
        kind=kind,
        name=name,
        content=content,
        metadata=dict(metadata) if metadata is not None else {},
    )
    context.artifacts.append(artifact)
    context.trace_recorder.record_artifact_emitted(
        artifact_kind=artifact.kind,
        name=artifact.name,
        metadata_summary=_artifact_trace_summary(artifact),
    )
    return artifact


def _artifact_trace_summary(artifact: Artifact) -> dict[str, object]:
    """Return content-safe structural facts for artifact trace records."""
    return {
        "content_type": type(artifact.content).__name__,
        "content_length": len(artifact.content),
        "metadata_keys": sorted(str(key) for key in artifact.metadata),
    }
