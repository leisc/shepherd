"""Provider-backed delivery primitive for the syntax nucleus."""

from __future__ import annotations

import time
from asyncio import CancelledError, current_task
from collections.abc import Awaitable, Sequence  # noqa: TC003 - runtime get_type_hints tests resolve these names.
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin
from uuid import uuid4

from shepherd_core.errors import StepOutputError
from shepherd_core.output import coerce_output_value, parse_tuple_output
from shepherd_core.schema import SINGLE_OUTPUT_KEY

from shepherd_runtime.effects._handler_stack import HandlerBinding, invoke_handler, resolve_handler
from shepherd_runtime.provider_boundary.summaries import (
    summarize_model_failure,
    summarize_model_request,
    summarize_model_response,
)
from shepherd_runtime.sync import run_sync

from .types import (
    DeliveryFailed,
    DeliveryLimits,
    DeliveryStopped,
    Failed,
    NoActiveTaskRun,
    Run,
    RunRef,
    WorkspaceNotConfigured,
)
from .workspace import current_workspace

if TYPE_CHECKING:
    from contextvars import Token

    from shepherd_runtime.provider_boundary.payloads import ModelRequest, ModelResponse
    from shepherd_runtime.trace import Trace
    from shepherd_runtime.trace.runtime import RuntimeTraceRecorder

    from .workspace import Workspace

T = TypeVar("T")


class _InvalidModelCallResponseError(TypeError):
    """Internal marker for handlers that do not return a ModelResponse."""

    def __init__(self, message: str, *, actual_type: str) -> None:
        super().__init__(message)
        self.actual_type = actual_type


@dataclass(frozen=True)
class _FailureDiagnostics:
    error_type: str
    message: str
    reason: str
    detail_summary: dict[str, object]


@dataclass(frozen=True)
class TaskRunContext:
    """Process-local active task-run context.

    The ``artifacts`` field is a mutable list under a frozen-dataclass
    parent: the field reference is frozen (callers cannot rebind it)
    but the list itself is appended to as ``emit_artifact(...)`` runs
    inside the task body. Run construction at task-completion time
    snapshots the list into the immutable ``Run.artifacts`` tuple.
    """

    ref: RunRef
    task_name: str
    workspace: Workspace
    is_async: bool
    artifacts: list[Any] = dataclass_field(default_factory=list)
    trace_recorder: RuntimeTraceRecorder = dataclass_field(init=False)

    def __post_init__(self) -> None:
        from shepherd_runtime.trace.runtime import RuntimeTraceRecorder

        object.__setattr__(self, "trace_recorder", RuntimeTraceRecorder(self.ref))


_active_task_runs: ContextVar[tuple[TaskRunContext, ...]] = ContextVar(
    "shepherd_nucleus_task_runs",
    default=(),
)


def active_task_run() -> TaskRunContext | None:
    """Return the innermost active task run, if any."""
    stack = _active_task_runs.get()
    return stack[-1] if stack else None


def push_task_run(context: TaskRunContext) -> Token[tuple[TaskRunContext, ...]]:
    """Push a task run onto the active context stack."""
    return _active_task_runs.set((*_active_task_runs.get(), context))


def pop_task_run(token: Token[tuple[TaskRunContext, ...]]) -> None:
    """Restore the task-run stack to a previous token."""
    _active_task_runs.reset(token)


def build_task_trace(context: TaskRunContext) -> Trace:
    """Return the immutable runtime trace snapshot for a task run."""
    return context.trace_recorder.to_trace()


def make_task_run_context(*, task_name: str, is_async: bool) -> TaskRunContext:
    """Create a task-run context for the current workspace."""
    workspace = current_workspace()
    if workspace is None:
        raise WorkspaceNotConfigured(
            "Task execution requires an active workspace. Wrap the task call in "
            "`with workspace(model=..., root=...):` or call `workspace(model=...)` "
            "before invoking the task."
        )
    return TaskRunContext(
        ref=RunRef(id=f"run-{uuid4().hex}"),
        task_name=task_name,
        workspace=workspace,
        is_async=is_async,
    )


def deliver(
    result_type: type[T],
    *,
    goal: str,
    evidence: Sequence[object] = (),
    constraints: Sequence[str] = (),
    limits: DeliveryLimits | None = None,
) -> T | Awaitable[T]:
    """Deliver a typed result from inside an active function-form task.

    In a sync task body this runs the model-call coroutine to completion and
    returns ``T``. In an async task body it returns an awaitable for ``T``; call
    sites should use ``await deliver(...)``.
    """
    context = active_task_run()
    if context is None:
        raise NoActiveTaskRun(_no_active_task_run_message())

    coroutine = _deliver_async(
        result_type,
        goal=goal,
        evidence=tuple(evidence),
        constraints=tuple(constraints),
        limits=limits,
    )
    if context.is_async:
        return coroutine
    return run_sync(coroutine)


async def _deliver_async(
    result_type: type[T],
    *,
    goal: str,
    evidence: tuple[object, ...],
    constraints: tuple[str, ...],
    limits: DeliveryLimits | None,
) -> T:
    context = active_task_run()
    if context is None:
        raise NoActiveTaskRun(_no_active_task_run_message())

    delivery_limits = limits or DeliveryLimits()
    start = time.perf_counter()
    prompt = _build_prompt(goal=goal, evidence=evidence, constraints=constraints, limits=delivery_limits)
    request = _build_model_request(context, prompt)
    provider_id = _provider_id(context.workspace.model)
    handler = resolve_handler("model.call")

    if handler is None:
        message = _unhandled_model_call_message(context, result_type, request, provider_id)
        context.trace_recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="failed",
            detail_summary={
                "error_type": "UnhandledModelCall",
                "reason": "no_model_call_handler",
                "task_name": context.task_name,
                "model_id": request.settings.model,
                "provider_id": provider_id,
                "next_step": 'install handle("model.call", ...) around the task run',
            },
        )
        raise _delivery_failed(
            context,
            start,
            Failed(error_type="UnhandledModelCall", message=message),
        )

    recorder = context.trace_recorder
    declaration_ref = recorder.record_provider_call_requested(
        request_summary=summarize_model_request(request, provider_id=provider_id)
    )
    selection_ref = recorder.record_handler_selected(declaration_ref, handler_key=handler.handler_id)

    try:
        response = await _perform_handled_model_call(context, handler, request, result_type)
    except CancelledError as exc:
        diagnostics = _model_call_cancelled_diagnostics(
            context,
            result_type,
            handler,
            request,
            provider_id,
            externally_cancelled=_current_task_is_cancelling(),
        )
        cancellation_summary = _model_call_cancellation_summary(
            exc,
            request=request,
            provider_id=provider_id,
            diagnostics=diagnostics,
        )
        capture_ref = recorder.record_provider_call_completed(
            selection_ref,
            status="cancelled",
            response_summary=cancellation_summary,
        )
        recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="stopped",
            citing=(capture_ref,),
            detail_summary=_with_model_call_lifecycle(
                diagnostics.detail_summary,
                request_ref=declaration_ref,
                selection_ref=selection_ref,
                completion_ref=capture_ref,
                completion_status="cancelled",
                delivery_status="stopped",
            ),
        )
        if diagnostics.reason == "task_cancelled":
            raise
        raise DeliveryStopped(diagnostics.message) from exc
    except Exception as exc:
        diagnostics = _model_call_failure_diagnostics(
            context,
            result_type,
            handler,
            request,
            provider_id,
            exc,
        )
        failure_summary = _model_call_failure_summary(
            exc,
            request=request,
            provider_id=provider_id,
            diagnostics=diagnostics,
        )
        capture_ref = recorder.record_provider_call_completed(
            selection_ref,
            status="raised",
            response_summary=failure_summary,
        )
        recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="failed",
            citing=(capture_ref,),
            detail_summary=_with_model_call_lifecycle(
                diagnostics.detail_summary,
                request_ref=declaration_ref,
                selection_ref=selection_ref,
                completion_ref=capture_ref,
                completion_status="raised",
                delivery_status="failed",
            ),
        )
        raise _delivery_failed(
            context,
            start,
            Failed(error_type=diagnostics.error_type, message=diagnostics.message),
        ) from exc

    capture_ref = recorder.record_provider_call_completed(
        selection_ref,
        status="returned",
        response_summary=summarize_model_response(
            response,
            provider_id=provider_id,
            model_id=request.settings.model,
        ),
    )
    is_tuple_result = get_origin(result_type) is tuple
    structured = response.structured_output
    if structured is None:
        message = _missing_structured_output_message(context, result_type, response)
        recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="failed",
            citing=(capture_ref,),
            detail_summary=_structured_output_detail(
                error_type="MissingStructuredOutput",
                reason="response_without_structured_output",
                context=context,
                result_type=result_type,
                request=request,
                provider_id=provider_id,
                handler=handler,
                response=response,
                structured=None,
                request_ref=declaration_ref,
                selection_ref=selection_ref,
                completion_ref=capture_ref,
                completion_status="returned",
                delivery_status="failed",
            ),
        )
        raise _delivery_failed(
            context,
            start,
            Failed(
                error_type="MissingStructuredOutput",
                message=message,
            ),
        )

    if not is_tuple_result and SINGLE_OUTPUT_KEY not in structured:
        message = _missing_single_output_key_message(context, result_type, structured)
        recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="failed",
            citing=(capture_ref,),
            detail_summary=_structured_output_detail(
                error_type="MissingStructuredOutput",
                reason="missing_single_output_key",
                context=context,
                result_type=result_type,
                request=request,
                provider_id=provider_id,
                handler=handler,
                response=response,
                structured=structured,
                request_ref=declaration_ref,
                selection_ref=selection_ref,
                completion_ref=capture_ref,
                completion_status="returned",
                delivery_status="failed",
            ),
        )
        raise _delivery_failed(
            context,
            start,
            Failed(
                error_type="MissingStructuredOutput",
                message=message,
            ),
        )

    try:
        if is_tuple_result:
            value = parse_tuple_output(structured, get_args(result_type), context.task_name)
        else:
            value = coerce_output_value(
                structured[SINGLE_OUTPUT_KEY],
                result_type,
                context.task_name,
                SINGLE_OUTPUT_KEY,
            )
    except Exception as exc:
        diagnostics = _structured_output_failure_diagnostics(
            context,
            result_type,
            request,
            provider_id,
            handler,
            response,
            structured,
            exc,
        )
        recorder.record_delivery_completed(
            result_type=_result_type_name(result_type),
            status="failed",
            citing=(capture_ref,),
            detail_summary=_with_model_call_lifecycle(
                diagnostics.detail_summary,
                request_ref=declaration_ref,
                selection_ref=selection_ref,
                completion_ref=capture_ref,
                completion_status="returned",
                delivery_status="failed",
            ),
        )
        raise _delivery_failed(
            context,
            start,
            Failed(error_type=diagnostics.error_type, message=diagnostics.message),
        ) from exc

    recorder.record_delivery_completed(
        result_type=_result_type_name(result_type),
        status="completed",
        citing=(capture_ref,),
        detail_summary=_successful_delivery_detail(
            context=context,
            result_type=result_type,
            request=request,
            provider_id=provider_id,
            handler=handler,
            response=response,
            structured=structured,
            request_ref=declaration_ref,
            selection_ref=selection_ref,
            completion_ref=capture_ref,
        ),
    )
    return value


async def _perform_handled_model_call(
    context: TaskRunContext,
    handler: HandlerBinding,
    request: ModelRequest,
    result_type: object,
) -> ModelResponse:
    from shepherd_runtime.provider_boundary.interposition import BypassInterposition
    from shepherd_runtime.provider_boundary.payloads import ModelResponse
    from shepherd_runtime.provider_boundary.runtime import StubProviderRuntime
    from shepherd_runtime.provider_boundary.tools import StubToolHandler

    async def responder(model_request: ModelRequest) -> ModelResponse:
        result = await invoke_handler(handler, model_request)
        if not isinstance(result, ModelResponse):
            raise _InvalidModelCallResponseError(
                _invalid_model_call_response_message(
                    context,
                    result_type,
                    handler,
                    model_request,
                    result,
                ),
                actual_type=type(result).__name__,
            )
        return result

    interposition = BypassInterposition(responder, handler_id=handler.handler_id)
    return await interposition.perform_model_call(
        request,
        StubProviderRuntime(
            run_ref=context.ref,
            task_name=f"{context.task_name}.deliver",
        ),
        StubToolHandler(),
    )


def _delivery_failed(context: TaskRunContext, start: float, outcome: Failed) -> DeliveryFailed:
    run: Run[Any] = Run(
        outcome=outcome,
        effects=(),
        artifacts=tuple(context.artifacts),
        usage=None,
        duration=time.perf_counter() - start,
        trace=build_task_trace(context),
        ref=context.ref,
    )
    return DeliveryFailed(outcome.message, run=run)


def _no_active_task_run_message() -> str:
    return (
        "deliver(...) requires an active task run. Call deliver(...) only from inside a function decorated with "
        "@task while that task is executing; start a run with task(...), await task(...), or task.detailed(...)."
    )


def _delivery_context(context: TaskRunContext, result_type: object) -> str:
    return f"deliver({_result_type_name(result_type)}) in task {context.task_name!r}"


def _unhandled_model_call_message(
    context: TaskRunContext,
    result_type: object,
    request: ModelRequest,
    provider_id: str | None,
) -> str:
    provider = f", provider_id={provider_id!r}" if provider_id is not None else ""
    return (
        f"{_delivery_context(context, result_type)} cannot perform model.call because no handler is installed "
        f"for model {request.settings.model!r}{provider}. The Phase 1 callable spine uses handled provider "
        'calls; wrap the task run with `with handle("model.call", ...):` and return a ModelResponse.'
    )


def _model_call_failure_diagnostics(
    context: TaskRunContext,
    result_type: object,
    handler: HandlerBinding,
    request: ModelRequest,
    provider_id: str | None,
    exc: Exception,
) -> _FailureDiagnostics:
    if isinstance(exc, _InvalidModelCallResponseError):
        error_type = "InvalidModelResponse"
        reason = "handler_returned_invalid_response"
        message = str(exc)
    else:
        error_type = type(exc).__name__
        reason = "handler_failed"
        message = (
            f"{_delivery_context(context, result_type)} failed because model.call handler "
            f"{handler.handler_id!r} raised {type(exc).__name__}: {exc}"
        )

    detail_summary = {
        **_model_call_detail_base(
            context=context,
            result_type=result_type,
            request=request,
            provider_id=provider_id,
            handler=handler,
        ),
        "error_type": error_type,
        "exception_type": type(exc).__name__,
        "reason": reason,
    }
    if isinstance(exc, _InvalidModelCallResponseError):
        detail_summary["handler_result_type"] = exc.actual_type

    return _FailureDiagnostics(
        error_type=error_type,
        message=message,
        reason=reason,
        detail_summary=detail_summary,
    )


def _model_call_failure_summary(
    exc: Exception,
    *,
    request: ModelRequest,
    provider_id: str | None,
    diagnostics: _FailureDiagnostics,
) -> dict[str, object]:
    summary = summarize_model_failure(exc, request=request, provider_id=provider_id)
    summary["reason"] = diagnostics.reason
    summary["delivery_error_type"] = diagnostics.error_type
    return summary


def _model_call_cancelled_diagnostics(
    context: TaskRunContext,
    result_type: object,
    handler: HandlerBinding,
    request: ModelRequest,
    provider_id: str | None,
    *,
    externally_cancelled: bool = False,
) -> _FailureDiagnostics:
    error_type = "CancelledError"
    reason = "task_cancelled" if externally_cancelled else "handler_cancelled"
    cancelled_actor = (
        "the surrounding task was cancelled"
        if externally_cancelled
        else (f"model.call handler {handler.handler_id!r} was cancelled")
    )
    return _FailureDiagnostics(
        error_type=error_type,
        message=(f"{_delivery_context(context, result_type)} stopped because {cancelled_actor}."),
        reason=reason,
        detail_summary={
            **_model_call_detail_base(
                context=context,
                result_type=result_type,
                request=request,
                provider_id=provider_id,
                handler=handler,
            ),
            "error_type": error_type,
            "exception_type": error_type,
            "reason": reason,
        },
    )


def _model_call_cancellation_summary(
    exc: BaseException,
    *,
    request: ModelRequest,
    provider_id: str | None,
    diagnostics: _FailureDiagnostics,
) -> dict[str, object]:
    summary = summarize_model_failure(
        exc,
        request=request,
        provider_id=provider_id,
        status="cancelled",
    )
    summary["reason"] = diagnostics.reason
    summary["delivery_error_type"] = diagnostics.error_type
    return summary


def _current_task_is_cancelling() -> bool:
    task = current_task()
    return task is not None and task.cancelling() > 0


def _invalid_model_call_response_message(
    context: TaskRunContext,
    result_type: object,
    handler: HandlerBinding,
    request: ModelRequest,
    result: object,
) -> str:
    return (
        f"{_delivery_context(context, result_type)} received an invalid model.call response from handler "
        f"{handler.handler_id!r} for model {request.settings.model!r}: expected ModelResponse, got "
        f"{type(result).__name__}. Return ModelResponse(structured_output={{{SINGLE_OUTPUT_KEY!r}: ...}}) "
        "for typed delivery."
    )


def _missing_structured_output_message(
    context: TaskRunContext,
    result_type: object,
    response: ModelResponse,
) -> str:
    return (
        f"{_delivery_context(context, result_type)} received a {_response_shape(response)} ModelResponse from "
        f"model.call, but typed delivery requires structured_output. Return "
        f"ModelResponse(structured_output={{{SINGLE_OUTPUT_KEY!r}: ...}}) for single-value results or "
        "output_N keys for tuple results."
    )


def _missing_single_output_key_message(
    context: TaskRunContext,
    result_type: object,
    structured: dict[str, object],
) -> str:
    return (
        f"{_delivery_context(context, result_type)} received structured output (structured_output) with "
        f"{len(structured)} key(s), but single-value delivery requires the {SINGLE_OUTPUT_KEY!r} key. Return "
        f"ModelResponse(structured_output={{{SINGLE_OUTPUT_KEY!r}: <value>}})."
    )


def _structured_output_failure_diagnostics(
    context: TaskRunContext,
    result_type: object,
    request: ModelRequest,
    provider_id: str | None,
    handler: HandlerBinding,
    response: ModelResponse,
    structured: dict[str, object],
    exc: Exception,
) -> _FailureDiagnostics:
    return _FailureDiagnostics(
        error_type=type(exc).__name__,
        message=(
            f"{_delivery_context(context, result_type)} could not coerce model.call structured_output into "
            f"{_result_type_name(result_type)}: {type(exc).__name__}{_safe_exception_detail(exc)}"
        ),
        detail_summary=_structured_output_detail(
            error_type=type(exc).__name__,
            reason="structured_output_coercion_failed",
            context=context,
            result_type=result_type,
            request=request,
            provider_id=provider_id,
            handler=handler,
            response=response,
            structured=structured,
            exception_type=type(exc).__name__,
        ),
        reason="structured_output_coercion_failed",
    )


def _safe_exception_detail(exc: Exception) -> str:
    if isinstance(exc, StepOutputError) and exc.reason.startswith("Missing required keys:"):
        return f": {exc.reason}"
    return ""


def _structured_output_detail(
    *,
    error_type: str,
    reason: str,
    context: TaskRunContext,
    result_type: object,
    request: ModelRequest,
    provider_id: str | None,
    handler: HandlerBinding,
    response: ModelResponse,
    structured: dict[str, object] | None,
    exception_type: str | None = None,
    request_ref: str | None = None,
    selection_ref: str | None = None,
    completion_ref: str | None = None,
    completion_status: str | None = None,
    delivery_status: str | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {
        **_model_call_detail_base(
            context=context,
            result_type=result_type,
            request=request,
            provider_id=provider_id,
            handler=handler,
        ),
        "error_type": error_type,
        "reason": reason,
        "response_shape": _response_shape(response),
        "structured_key_count": len(structured) if structured is not None else 0,
    }
    if exception_type is not None:
        detail["exception_type"] = exception_type
    if request_ref is not None and selection_ref is not None and completion_ref is not None:
        if completion_status is None or delivery_status is None:
            raise ValueError("completion_status and delivery_status are required with lifecycle refs")
        return _with_model_call_lifecycle(
            detail,
            request_ref=request_ref,
            selection_ref=selection_ref,
            completion_ref=completion_ref,
            completion_status=completion_status,
            delivery_status=delivery_status,
        )
    return detail


def _successful_delivery_detail(
    *,
    context: TaskRunContext,
    result_type: object,
    request: ModelRequest,
    provider_id: str | None,
    handler: HandlerBinding,
    response: ModelResponse,
    structured: dict[str, object],
    request_ref: str,
    selection_ref: str,
    completion_ref: str,
) -> dict[str, object]:
    detail = {
        **_model_call_detail_base(
            context=context,
            result_type=result_type,
            request=request,
            provider_id=provider_id,
            handler=handler,
        ),
        "reason": "structured_output_coerced",
        "response_shape": _response_shape(response),
        "structured_key_count": len(structured),
    }
    return _with_model_call_lifecycle(
        detail,
        request_ref=request_ref,
        selection_ref=selection_ref,
        completion_ref=completion_ref,
        completion_status="returned",
        delivery_status="completed",
    )


def _model_call_detail_base(
    *,
    context: TaskRunContext,
    result_type: object,
    request: ModelRequest,
    provider_id: str | None,
    handler: HandlerBinding | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {
        "task_name": context.task_name,
        "result_type": _result_type_name(result_type),
        "model_id": request.settings.model,
        "provider_id": provider_id,
    }
    if handler is not None:
        detail["handler_id"] = handler.handler_id
    return detail


def _with_model_call_lifecycle(
    detail: dict[str, object],
    *,
    request_ref: str,
    selection_ref: str,
    completion_ref: str,
    completion_status: str,
    delivery_status: str,
) -> dict[str, object]:
    return {
        **detail,
        "provider_call_lifecycle": {
            "request_ref": request_ref,
            "request_status": "requested",
            "selection_ref": selection_ref,
            "selection_status": "selected",
            "completion_ref": completion_ref,
            "completion_status": completion_status,
            "delivery_status": delivery_status,
        },
    }


def _response_shape(response: ModelResponse) -> str:
    if response.structured_output is not None:
        return "structured_output"
    if response.text is not None:
        return "text"
    if response.tool_calls:
        return "tool_calls"
    return "empty"


def _build_prompt(
    *,
    goal: str,
    evidence: tuple[object, ...],
    constraints: tuple[str, ...],
    limits: DeliveryLimits,
) -> str:
    lines = [goal]
    if evidence:
        lines.append("\nEvidence:")
        lines.extend(f"- {item!r}" for item in evidence)
    if constraints:
        lines.append("\nConstraints:")
        lines.extend(f"- {constraint}" for constraint in constraints)
    if limits.max_turns is not None:
        lines.append(f"\nDelivery limit: max_turns={limits.max_turns}")
    return "\n".join(lines)


def _build_model_request(context: TaskRunContext, prompt: str) -> ModelRequest:
    from shepherd_runtime.provider_boundary.payloads import ModelRequest, ProviderMessage, ProviderSettings

    return ModelRequest(
        messages=(ProviderMessage(role="user", content=prompt),),
        tools=(),
        settings=ProviderSettings(model=_model_name(context.workspace.model)),
    )


def _model_name(model: object) -> str:
    if isinstance(model, str) and model:
        return model
    for attr in ("name", "model", "model_name", "provider_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _provider_id(model: object) -> str | None:
    value = getattr(model, "provider_id", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(model, "provider", None)
    if isinstance(value, str) and value:
        return value
    return None


def _result_type_name(result_type: object) -> str:
    return getattr(result_type, "__name__", repr(result_type))


__all__ = [
    "TaskRunContext",
    "active_task_run",
    "build_task_trace",
    "deliver",
    "make_task_run_context",
    "pop_task_run",
    "push_task_run",
]
