"""Per-construct source-AST admission validator for `-lite`.

Per `260521-0600-kernel.md` §"Canonical Value and Schema Profile" /
"`core-reference-v0-lite` source admission" and 2026-05-23 §"Handler-body
admission shape: Resume-or-Abort".
"""

from __future__ import annotations

import pytest

from shepherd_kernel_v3_reference.envelope import KernelRejection
from shepherd_kernel_v3_reference.profile_admission import (
    ProfileAdmissionError,
    validate_profile_admission,
)
from shepherd_kernel_v3_reference.profiles import (
    CORE_A,
)
from shepherd_kernel_v3_reference.schemas import (
    AnySchema,
    IntSchema,
    LiteralSchema,
    NullSchema,
    TaggedRecordSchema,
    TypeSchema,
)
from shepherd_kernel_v3_reference.source.handlers import (
    DynamicHandlerInstall,
    HandlerEnv,
    StaticHandlerInstall,
)
from shepherd_kernel_v3_reference.source.syntax import (
    Abort,
    Handle,
    Let,
    Lit,
    Perform,
    RecordExpr,
    Resume,
    Return,
    Var,
)


def _resume_handler_install(handler_value: int = 42) -> StaticHandlerInstall:
    return StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        body=Let("r", Resume(Lit(handler_value)), Return(Var("r"))),
    )


def _abort_handler_install(abort_value: int = 0) -> StaticHandlerInstall:
    return StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=NullSchema(),
        payload_name="_p",
        body=Abort(Lit(abort_value)),
    )


def _resume_shape_program(handler_value: int = 42):
    return Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((_resume_handler_install(handler_value),)),
    )


def _abort_shape_program(abort_value: int = 0):
    return Handle(
        Perform("ask", Lit(None)),
        HandlerEnv((_abort_handler_install(abort_value),)),
    )


# --- Positive cases -----------------------------------------------------


def test_pure_let_chain_admits() -> None:
    program = Let("x", Return(Lit(1)), Return(Var("x")))
    validate_profile_admission(program)


def test_resume_shape_handler_admits() -> None:
    validate_profile_admission(_resume_shape_program())


def test_abort_shape_handler_admits() -> None:
    validate_profile_admission(_abort_shape_program())


def test_nested_handlers_both_resume_shape_admit() -> None:
    inner = StaticHandlerInstall(
        effect_kind="inner",
        handler_id="inner.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        body=Let("r", Resume(Lit(7)), Return(Var("r"))),
    )
    outer = StaticHandlerInstall(
        effect_kind="outer",
        handler_id="outer.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        body=Let("r", Resume(Lit(3)), Return(Var("r"))),
    )
    program = Handle(
        Handle(
            Let("a", Perform("inner", Lit(None)), Let("b", Perform("outer", Lit(None)), Return(Var("a")))),
            HandlerEnv((inner,)),
        ),
        HandlerEnv((outer,)),
    )
    validate_profile_admission(program)


def test_null_value_admits() -> None:
    validate_profile_admission(Let("x", Return(Lit(None)), Return(Var("x"))))


def test_literal_schema_int_admits() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=LiteralSchema(42),
        payload_name="_p",
        body=Let("r", Resume(Lit(42)), Return(Var("r"))),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    validate_profile_admission(program)


# --- Source-syntax rejections -------------------------------------------


def test_record_expr_in_payload_rejects() -> None:
    program = Let(
        "y",
        Perform("ask", RecordExpr(fields=(("k", Lit(1)),))),
        Return(Var("y")),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "RecordExpr"
    assert exc.value.rejection.kind == "profile-admission"


def test_record_expr_in_return_rejects() -> None:
    program = Return(RecordExpr(fields=(("k", Lit(1)),)))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "RecordExpr"


def test_dynamic_handler_install_rejects() -> None:
    install = DynamicHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=IntSchema(),
        body=lambda _payload: Return(Lit(42)),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "DynamicHandlerInstall"


# --- Handler-body shape rejections --------------------------------------


def test_no_resume_no_abort_body_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=NullSchema(),
        payload_name="_p",
        body=Return(Lit(None)),
    )
    program = Handle(
        Perform("ask", Lit(None)),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "no-resume-no-abort handler body"


def test_multi_shot_resume_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        body=Let(
            "a",
            Resume(Lit(1)),
            Let("b", Resume(Lit(2)), Return(Var("a"))),
        ),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert "multi-shot" in exc.value.rejection.construct.lower()


def test_mixed_resume_abort_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        body=Let("r", Resume(Lit(1)), Abort(Lit(2))),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert "mixed" in exc.value.rejection.construct.lower() or "Resume+Abort" in exc.value.rejection.construct


def test_abort_shape_with_trailing_return_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=IntSchema(),
        payload_name="_p",
        # An Abort with trailing computation is Resume-shape territory but
        # has no Resume — actually with no Resume it's only an Abort, the
        # Return after is unreachable. _terminates_in_abort walks the let-tail
        # so Abort in the middle fails the Abort-shape rule.
        body=Let("r", Abort(Lit(0)), Return(Var("r"))),
    )
    program = Handle(
        Perform("ask", Lit(None)),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert "Abort-shape" in exc.value.rejection.construct


# --- Value rejections ----------------------------------------------------


def test_bool_value_rejects() -> None:
    program = Let("x", Return(Lit(True)), Return(Var("x")))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "bool"


def test_string_value_rejects() -> None:
    program = Let("x", Return(Lit("hello")), Return(Var("x")))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "str"


def test_list_value_rejects() -> None:
    program = Let("x", Return(Lit([1, 2, 3])), Return(Var("x")))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "list"


def test_float_value_rejects() -> None:
    program = Let("x", Return(Lit(1.5)), Return(Var("x")))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "float"


# --- Schema rejections ---------------------------------------------------


def test_any_schema_in_handler_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=AnySchema(),
        payload_name="_p",
        body=Let("r", Resume(Lit(42)), Return(Var("r"))),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "AnySchema"


def test_type_schema_in_handler_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=TypeSchema(int),
        payload_name="_p",
        body=Let("r", Resume(Lit(42)), Return(Var("r"))),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "TypeSchema"


def test_tagged_record_schema_in_handler_rejects() -> None:
    install = StaticHandlerInstall(
        effect_kind="ask",
        handler_id="ask.v1",
        handled_result_schema=TaggedRecordSchema("Section"),
        payload_name="_p",
        body=Let("r", Resume(Lit(42)), Return(Var("r"))),
    )
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv((install,)),
    )
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    assert exc.value.rejection.construct == "TaggedRecordSchema"


# --- Rejection metadata -------------------------------------------------


def test_rejection_carries_kernel_rejection_payload() -> None:
    program = Let("x", Return(Lit("bad")), Return(Var("x")))
    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(program)
    rejection = exc.value.rejection
    assert isinstance(rejection, KernelRejection)
    assert rejection.kind == "profile-admission"
    assert rejection.construct == "str"
    assert rejection.diagnostic
    assert rejection.source_location is not None
    assert rejection.source_location.construct_path


# --- Non-lite profiles are no-ops ---------------------------------------


def test_non_lite_profile_admits_wider_constructs() -> None:
    """CORE_A profile is a no-op for validate_profile_admission; the wider
    operational corpus is admitted at the structural-validation layer, not
    here."""
    program = Let("x", Return(Lit("hello")), Return(Var("x")))
    validate_profile_admission(program, profile=CORE_A)
