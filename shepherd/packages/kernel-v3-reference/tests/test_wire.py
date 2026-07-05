"""Unit tests for the canonical wire serializers (`wire.py`).

These pin the normative wire encoding the `-lite` corpus freezes and Lean
Phase 9 consumes: profile-as-name, deterministic canonical bytes,
round-trip-stable JSON, transition-as-id on the envelope, and explicit-null
per-kind fields on KernelRejection.
"""

from __future__ import annotations

import json

from shepherd_kernel_v3_reference.envelope import (
    CompletedResult,
    KernelRejection,
    KernelResultEnvelope,
    SourceLocation,
    WireResult,
)
from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.program_admission import ensure_prepared_kernel_program
from shepherd_kernel_v3_reference.kernel.refs import canonical_json, content_ref
from shepherd_kernel_v3_reference.kernel.replay import KernelReplaySession
from shepherd_kernel_v3_reference.profiles import CORE_REFERENCE_V0_LITE
from shepherd_kernel_v3_reference.projection import semantic_batch_from_transition
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.semantic import SemanticTransitionBatch
from shepherd_kernel_v3_reference.source.handlers import HandlerEnv, StaticHandlerInstall
from shepherd_kernel_v3_reference.source.syntax import (
    Handle,
    Let,
    Lit,
    Perform,
    Resume,
    Return,
    Var,
)
from shepherd_kernel_v3_reference.wire import (
    completed_result_to_wire,
    kernel_rejection_to_wire,
    kernel_result_envelope_to_wire,
    semantic_batch_to_wire,
    wire_result_to_wire,
)


def _handled_resume_batch() -> SemanticTransitionBatch:
    """Project a single handled-resume transition to a batch (proven pattern
    from tests/test_projection_corpus.py)."""

    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ask.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(42)), Return(Var("r"))),
                ),
            )
        ),
    )
    prepared = ensure_prepared_kernel_program(elaborate(program))
    session, transition = KernelReplaySession.start(prepared)
    catalog = dict(session._evaluator.continuation_objects)
    batch = semantic_batch_from_transition(transition, session.state, catalog)
    assert isinstance(batch, SemanticTransitionBatch)
    return batch


# --- canonical_json -----------------------------------------------------


def test_canonical_json_sorts_keys_and_drops_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_deterministic_and_roundtrips() -> None:
    value = {"z": [3, 2, 1], "a": {"nested": True, "k": None}}
    first = canonical_json(value)
    second = canonical_json(value)
    assert first == second
    assert json.loads(first) == value


def test_content_ref_bytes_unchanged_after_canonical_json_refactor() -> None:
    # content_ref now delegates to canonical_json; its output must be stable.
    assert content_ref("k", {"a": 1, "b": [1, 2]}).startswith("k:sha256:")
    assert content_ref("k", {"a": 1}) == content_ref("k", {"a": 1})


# --- semantic_batch_to_wire --------------------------------------------


def test_batch_wire_encodes_profile_as_name() -> None:
    batch = _handled_resume_batch()
    wire = semantic_batch_to_wire(batch)
    # Profile is the name string, not the {name, version, validated} record.
    assert wire["profile"] == batch.profile.name
    assert isinstance(wire["profile"], str)


def test_batch_wire_is_canonical_and_deterministic() -> None:
    batch = _handled_resume_batch()
    a = canonical_json(semantic_batch_to_wire(batch))
    b = canonical_json(semantic_batch_to_wire(batch))
    assert a == b
    # Two independent projections of the same program are byte-identical.
    batch2 = _handled_resume_batch()
    assert canonical_json(semantic_batch_to_wire(batch2)) == a


def test_batch_wire_admission_basis_profile_is_name_when_present() -> None:
    batch = _handled_resume_batch()
    wire = semantic_batch_to_wire(batch)
    basis = wire.get("admission_basis")
    if basis is not None:
        assert isinstance(basis["profile"], str)


# --- kernel_rejection_to_wire ------------------------------------------


def test_rejection_wire_emits_all_per_kind_fields_with_explicit_nulls() -> None:
    rejection = KernelRejection(
        kind="kernel-admission",
        diagnostic="structural failure",
        program_ref="program:sha256:abc",
    )
    wire = kernel_rejection_to_wire(rejection)
    # Per-kind fields present with explicit null (not omitted).
    for key in (
        "construct",
        "source_location",
        "rejection_index",
        "rejection_class",
    ):
        assert key in wire, f"{key} must be present (explicit-null policy)"
        assert wire[key] is None
    assert wire["partial_records"] == []


def test_rejection_wire_profile_admission_construct_and_location() -> None:
    rejection = KernelRejection(
        kind="profile-admission",
        diagnostic="RecordExpr not admitted",
        construct="RecordExpr",
        source_location=SourceLocation(construct_path="Handle.body"),
    )
    wire = kernel_rejection_to_wire(rejection)
    assert wire["construct"] == "RecordExpr"
    assert wire["program_ref"] is None
    assert wire["source_location"] == {
        "construct_path": "Handle.body",
        "line": None,
        "column": None,
    }


# --- envelope + wire_result --------------------------------------------


def test_envelope_wire_encodes_transition_as_id_reference() -> None:
    batch = _handled_resume_batch()
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ask.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(42)), Return(Var("r"))),
                ),
            )
        ),
    )
    prepared = ensure_prepared_kernel_program(elaborate(program))
    session, transition = KernelReplaySession.start(prepared)
    envelope = KernelResultEnvelope(
        profile=session.state.profile,
        status="completed",
        payload=CompletedResult(program_ref=transition.program_ref, value=42),
        transition=transition,
    )
    wire = kernel_result_envelope_to_wire(envelope)
    # The full transition is NOT inlined; only its id.
    assert wire["transition_id"] == transition.transition_id
    assert wire["status"] == "completed"
    assert wire["profile"] == session.state.profile.name
    assert wire["payload"]["payload_type"] == "completed"
    assert wire["payload"]["value"] == 42

    wire_result = WireResult(envelope=envelope, batch=batch)
    composed = wire_result_to_wire(wire_result)
    assert composed["envelope"]["transition_id"] == transition.transition_id
    assert composed["batch"]["batch_type"] == "semantic-transition-batch"
    # Composed result is canonical-encodable.
    assert canonical_json(composed)


def test_completed_result_wire_shape() -> None:
    result = CompletedResult(program_ref="program:sha256:x", value=7)
    assert completed_result_to_wire(result) == {
        "program_ref": "program:sha256:x",
        "value": 7,
    }


def test_lite_profile_name_is_stable_wire_token() -> None:
    # The wire token Lean keys on must be the documented profile name.
    assert CORE_REFERENCE_V0_LITE.name == "core_reference_v0_lite"
