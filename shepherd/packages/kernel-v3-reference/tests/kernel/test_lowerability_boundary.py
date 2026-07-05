import pytest

from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.source.eval_direct import run
from shepherd_kernel_v3_reference.source.handlers import (
    DynamicHandlerInstall,
    HandlerEnv,
    StaticHandlerInstall,
)
from shepherd_kernel_v3_reference.source.outcomes import Completed
from shepherd_kernel_v3_reference.source.syntax import (
    Abort,
    Handle,
    Let,
    Lit,
    Perform,
    Resume,
    Return,
    Var,
)
from shepherd_kernel_v3_reference.source.wellformed import SourceFormError
from shepherd_kernel_v3_reference.trace.machine import run_trace
from shepherd_kernel_v3_reference.trace.records import EffectCapture, SelectionClosed
from shepherd_kernel_v3_reference.trace.validate import (
    TraceValidationError,
    validate_generated_trace_against_program,
    validate_runtime_trace,
)


def _static_install(effect_kind: str, body) -> StaticHandlerInstall:
    return StaticHandlerInstall(
        effect_kind=effect_kind,
        handler_id="h.v1",
        handled_result_schema=AnySchema(),
        payload_name="_payload",
        body=body,
    )


def test_static_fragment_trace_is_reference_validatable() -> None:
    program = Handle(
        Let("answer", Perform("eff.lookup", Lit({"id": 7})), Return(Var("answer"))),
        HandlerEnv(
            (
                _static_install(
                    "eff.lookup",
                    Let("result", Resume(Lit({"title": "static"})), Return(Var("result"))),
                ),
            )
        ),
    )

    kernel = elaborate(program)
    result = run_trace(kernel)

    assert result.outcome == Completed({"title": "static"})
    validate_generated_trace_against_program(kernel, result.trace)


def test_dynamic_python_handler_is_runtime_only_not_lowerable() -> None:
    def provider_sdk_call(payload: object) -> object:
        return {"provider": "example", "payload": payload}

    program = Handle(
        Perform("model.call", Lit({"prompt": "summarize"})),
        HandlerEnv(
            (
                DynamicHandlerInstall(
                    effect_kind="model.call",
                    handler_id="provider.python",
                    handled_result_schema=AnySchema(),
                    body=lambda payload: Return(Lit(provider_sdk_call(payload))),
                ),
            )
        ),
    )

    assert run(program) == Completed({"provider": "example", "payload": {"prompt": "summarize"}})
    with pytest.raises(SourceFormError, match="requires static handler bodies"):
        elaborate(program)


def test_runtime_operational_trace_is_not_generated_trace_validatable() -> None:
    program = Handle(
        Perform("eff.a", Lit(None)),
        HandlerEnv((_static_install("eff.a", Abort(Lit({"reason": "host-exception"}))),)),
    )
    kernel = elaborate(program)
    result = run_trace(kernel)
    capture = next(record for record in result.trace if isinstance(record, EffectCapture))
    runtime_trace = tuple(result.trace) + (
        SelectionClosed(
            ref="selection-closed:runtime-failure",
            selection_ref=capture.selection_ref,
            selection_path_ref=capture.selection_path_ref,
            branch_ref=capture.branch_ref,
            reason="runtime_failure",
            caused_by_ref=capture.ref,
            caused_by_record_type="EffectCapture",
            closed_by_selection_ref=capture.selection_ref,
            closed_by_selection_path_ref=capture.selection_path_ref,
            branch_scope_ref=capture.branch_scope_ref,
        ),
    )

    validate_runtime_trace(runtime_trace)
    with pytest.raises(TraceValidationError, match="runtime-operational|program-generated"):
        validate_generated_trace_against_program(kernel, runtime_trace)
