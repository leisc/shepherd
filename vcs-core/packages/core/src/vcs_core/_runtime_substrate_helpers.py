"""Private runtime-substrate composition helpers.

These helpers are not part of the launch public surface. They remain in
vcs-core as private support for execution-driver experiments while the public
extension contract lives under :mod:`vcs_core.spi`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vcs_core.spi import DriverContext


class UnhandledAsk(VcsCoreError, LookupError):  # noqa: N818
    """No handler in the composed stack handles a performed effect."""


class TaskIdResolutionError(VcsCoreError, RuntimeError):
    """Raised when a Tier-A ``task_id`` cannot resolve to a callable."""


def resolve_task_id(task_id: str) -> Callable[..., Any]:
    """Resolve ``pkg.module:attr`` or ``pkg.module.attr`` to a callable."""
    from importlib import import_module

    module_name, sep, attr_name = task_id.partition(":")
    if not sep:
        module_name, _, attr_name = task_id.rpartition(".")
        if not module_name:
            raise TaskIdResolutionError(
                f"task_id {task_id!r} is not a fully-qualified import path ('pkg.module:attr' or 'pkg.module.attr')"
            )
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise TaskIdResolutionError(f"task_id {task_id!r}: cannot import {module_name!r}: {exc}") from exc
    try:
        body = getattr(module, attr_name)
    except AttributeError as exc:
        raise TaskIdResolutionError(f"task_id {task_id!r}: {module_name!r} has no attribute {attr_name!r}") from exc
    if not callable(body):
        raise TaskIdResolutionError(f"task_id {task_id!r} resolved to a non-callable {type(body).__name__}")
    return cast("Callable[..., Any]", body)


@dataclass(frozen=True)
class FileCreate:
    path: str
    content: bytes


@dataclass(frozen=True)
class FilePatch:
    path: str
    content: bytes


@dataclass(frozen=True)
class TraceAppend:
    kind: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class SubstrateOperationProposed:
    binding: str
    effect: object


@dataclass(frozen=True)
class SubstrateOperationCommitted:
    binding: str
    effect: object


class HandlerStack:
    """LIFO-ordered effect-handler stack composed by a run command."""

    def __init__(self) -> None:
        self._frames: list[Mapping[type, Callable[..., Any]]] = []

    def push(self, frame: Mapping[type, Callable[..., Any]]) -> None:
        self._frames.append(frame)

    def dispatch(self, effect: Any) -> Any:
        for frame in reversed(self._frames):
            for effect_type, handler in frame.items():
                if isinstance(effect, effect_type):
                    return handler(effect)
        raise UnhandledAsk(f"no handler for {type(effect).__name__}")


@runtime_checkable
class ExecutionProvider(Protocol):
    """Drives a task body through a composed handler stack."""

    provider_id: str

    def execute(
        self,
        task_body: Callable[..., Any] | None,
        stack: HandlerStack,
        context: DriverContext,
        args: Mapping[str, Any],
        *,
        execution: Any = None,
        confinement: Any = None,
    ) -> Mapping[str, Any]: ...


class InProcessExecutionProvider:
    """Default dev-tier provider: run the body in-process."""

    provider_id = "in-process"

    def execute(
        self,
        task_body: Callable[..., Any] | None,
        stack: HandlerStack,
        context: DriverContext,
        args: Mapping[str, Any],
        *,
        execution: Any = None,
        confinement: Any = None,
    ) -> Mapping[str, Any]:
        del context, execution, confinement
        if task_body is None:
            raise TaskIdResolutionError("in-process provider requires a resolved task body")
        result = task_body(stack, **dict(args))
        return {"status": "ok", "provider": self.provider_id, "result": result}
