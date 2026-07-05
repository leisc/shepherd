"""KernelResultEnvelope / WireResult / KernelRejection invariants.

Per 260521-0600-kernel.md §"Kernel Result Envelope" and 2026-05-24
§"Settled Design Decisions" entries "Post-#72 design pass" (item D/E)
and "Pre-#73 micro-design pass" (KernelRejection field shape).
"""

from __future__ import annotations

import pytest

from shepherd_kernel_v3_reference.envelope import (
    CompletedResult,
    KernelRejection,
    KernelResultEnvelope,
    SourceLocation,
    WireResult,
)
from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.replay import (
    ExternalEffectRequest,
    start_kernel_replay,
)
from shepherd_kernel_v3_reference.profiles import CORE_A
from shepherd_kernel_v3_reference.projection import (
    project_envelope_to_wire,
    validate_semantic_batch,
)
from shepherd_kernel_v3_reference.semantic import ProfileRejected, SemanticTransitionBatch
from shepherd_kernel_v3_reference.source.syntax import Let, Lit, Perform, Return, Var

# --- CompletedResult ----------------------------------------------------


def test_completed_result_requires_program_ref() -> None:
    with pytest.raises(ValueError, match="program_ref"):
        CompletedResult(program_ref="", value=42)


# --- KernelRejection: positive constructors per kind --------------------


def test_kernel_rejection_profile_admission_valid() -> None:
    r = KernelRejection(
        kind="profile-admission",
        diagnostic="RecordExpr is not admitted by -lite",
        construct="RecordExpr",
        source_location=SourceLocation(construct_path="Handle.body.Let[1].body"),
    )
    assert r.kind == "profile-admission"
    assert r.construct == "RecordExpr"
    assert r.program_ref is None


def test_kernel_rejection_kernel_admission_valid() -> None:
    r = KernelRejection(
        kind="kernel-admission",
        diagnostic="binder reference cycle",
        program_ref="program:sha256:abc",
    )
    assert r.kind == "kernel-admission"
    assert r.program_ref == "program:sha256:abc"


def test_kernel_rejection_execution_failure_valid() -> None:
    r = KernelRejection(
        kind="execution-failure",
        diagnostic="resume returned malformed value",
        program_ref="program:sha256:abc",
        partial_records=(),
    )
    assert r.kind == "execution-failure"


def test_kernel_rejection_observation_admission_valid() -> None:
    r = KernelRejection(
        kind="observation-admission",
        diagnostic="frontier mismatch",
        program_ref="program:sha256:abc",
        rejection_index=2,
        rejection_class="frontier-prefix",
    )
    assert r.kind == "observation-admission"
    assert r.rejection_index == 2


# --- KernelRejection: __post_init__ invariants --------------------------


def test_kernel_rejection_requires_diagnostic() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        KernelRejection(kind="kernel-admission", diagnostic="", program_ref="program:sha256:abc")


def test_profile_admission_must_carry_construct() -> None:
    with pytest.raises(ValueError, match="construct"):
        KernelRejection(kind="profile-admission", diagnostic="boom")


def test_profile_admission_must_not_carry_program_ref() -> None:
    with pytest.raises(ValueError, match="profile-admission"):
        KernelRejection(
            kind="profile-admission",
            diagnostic="boom",
            construct="RecordExpr",
            program_ref="program:sha256:abc",
        )


def test_non_profile_admission_requires_program_ref() -> None:
    with pytest.raises(ValueError, match="program_ref"):
        KernelRejection(kind="kernel-admission", diagnostic="boom")


def test_kernel_admission_forbids_construct_field() -> None:
    with pytest.raises(ValueError, match="construct"):
        KernelRejection(
            kind="kernel-admission",
            diagnostic="boom",
            program_ref="program:sha256:abc",
            construct="RecordExpr",
        )


def test_observation_admission_requires_index_and_class() -> None:
    with pytest.raises(ValueError, match="rejection_index"):
        KernelRejection(
            kind="observation-admission",
            diagnostic="boom",
            program_ref="program:sha256:abc",
        )


def test_kernel_admission_forbids_observation_fields() -> None:
    with pytest.raises(ValueError, match="rejection_index"):
        KernelRejection(
            kind="kernel-admission",
            diagnostic="boom",
            program_ref="program:sha256:abc",
            rejection_index=0,
            rejection_class="state-level",
        )


def test_non_execution_failure_forbids_partial_records() -> None:
    from shepherd_kernel_v3_reference.trace.records import EffectDeclaration

    fake_decl = EffectDeclaration(
        ref="declaration:0",
        program_ref="program:sha256:abc",
        effect_kind="ask",
        payload=None,
        full_continuation_ref="continuation:runtime:0",
        branch_ref="branch:root",
        payload_schema_ref=None,
        operation_result_schema_ref=None,
    )
    with pytest.raises(ValueError, match="partial_records"):
        KernelRejection(
            kind="kernel-admission",
            diagnostic="boom",
            program_ref="program:sha256:abc",
            partial_records=(fake_decl,),
        )


# --- KernelResultEnvelope: status-conditioned transition presence -------


def test_envelope_profile_rejected_forbids_transition() -> None:
    rejection = KernelRejection(
        kind="profile-admission",
        diagnostic="bad",
        construct="RecordExpr",
    )
    # Build a transition just to pass an object in
    _state, transition = start_kernel_replay(elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y")))))
    with pytest.raises(ValueError, match="must not carry a transition"):
        KernelResultEnvelope(
            profile=CORE_A,
            status="profile-rejected",
            payload=rejection,
            transition=transition,
        )


def test_envelope_completed_requires_transition() -> None:
    payload = CompletedResult(program_ref="program:sha256:abc", value=42)
    with pytest.raises(ValueError, match="requires a transition"):
        KernelResultEnvelope(
            profile=CORE_A,
            status="completed",
            payload=payload,
            transition=None,
        )


def test_envelope_completed_requires_completed_payload() -> None:
    _state, transition = start_kernel_replay(elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y")))))
    # ExternalEffectRequest payload with status='completed' should reject
    with pytest.raises(TypeError, match="CompletedResult"):
        KernelResultEnvelope(
            profile=CORE_A,
            status="completed",
            payload=transition.payload,
            transition=transition,
        )


def test_envelope_external_effect_request_pairs_with_request_payload() -> None:
    _state, transition = start_kernel_replay(elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y")))))
    request = transition.payload
    assert isinstance(request, ExternalEffectRequest)
    env = KernelResultEnvelope(
        profile=CORE_A,
        status="external-effect-request",
        payload=request,
        transition=transition,
    )
    assert env.status == "external-effect-request"
    assert env.transition is transition


def test_envelope_rejected_forbids_profile_admission_payload() -> None:
    rejection = KernelRejection(
        kind="profile-admission",
        diagnostic="bad",
        construct="RecordExpr",
    )
    _state, transition = start_kernel_replay(elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y")))))
    with pytest.raises(ValueError, match="profile-admission"):
        KernelResultEnvelope(
            profile=CORE_A,
            status="rejected",
            payload=rejection,
            transition=transition,
        )


def test_envelope_profile_rejected_requires_profile_admission_payload() -> None:
    rejection = KernelRejection(
        kind="kernel-admission",
        diagnostic="boom",
        program_ref="program:sha256:abc",
    )
    with pytest.raises(ValueError, match="profile-admission"):
        KernelResultEnvelope(
            profile=CORE_A,
            status="profile-rejected",
            payload=rejection,
            transition=None,
        )


# --- project_envelope_to_wire: end-to-end -------------------------------


def test_project_envelope_to_wire_external_effect_request() -> None:
    """Suspended program produces an external-effect-request transition;
    envelope wraps it and projects to a SemanticTransitionBatch.

    Uses KernelReplaySession to get a live evaluator whose continuation_objects
    catalog can be fed to the projection; the executable transition's payload
    is ExternalEffectRequest (compact request refs are not the envelope shape).
    """
    from shepherd_kernel_v3_reference.kernel.replay import (
        KernelReplaySession,
        start_replayable_kernel_transition,
    )

    program = elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y"))))
    # Live session gives us the catalog to feed the projection
    session, _session_transition = KernelReplaySession.start(program)
    # Independent executable transition gives us the ExternalEffectRequest form
    # for the envelope payload
    transition = start_replayable_kernel_transition(program)
    request = transition.payload
    assert isinstance(request, ExternalEffectRequest)
    envelope = KernelResultEnvelope(
        profile=CORE_A,
        status="external-effect-request",
        payload=request,
        transition=transition,
    )
    catalog = dict(session._evaluator.continuation_objects)
    wire = project_envelope_to_wire(envelope, session.state, catalog)
    assert isinstance(wire, WireResult)
    assert isinstance(wire.batch, SemanticTransitionBatch)
    validate_semantic_batch(wire.batch)


def test_project_envelope_to_wire_profile_rejected_synthesizes_batch() -> None:
    rejection = KernelRejection(
        kind="profile-admission",
        diagnostic="RecordExpr is not admitted by -lite",
        construct="RecordExpr",
    )
    envelope = KernelResultEnvelope(
        profile=CORE_A,
        status="profile-rejected",
        payload=rejection,
        transition=None,
    )
    wire = project_envelope_to_wire(envelope, None, {})
    assert isinstance(wire, WireResult)
    assert isinstance(wire.batch, ProfileRejected)
    assert wire.batch.rejection_reason == rejection.diagnostic
    assert wire.batch.partial_records == ()
    assert wire.batch.consumed_source_keys == ()
    # Synthetic transition_id is content-addressed over the rejection facts
    assert wire.batch.transition_id.startswith("profile-rejected:sha256:")
