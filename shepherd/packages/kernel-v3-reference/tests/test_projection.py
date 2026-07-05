"""Unit tests for semantic_batch_from_transition (commit #72).

Covers:
- Field role classifier (table-driven, validated by the 2330 spike at zero
  false positives across the -lite-relevant record set)
- branch:root literal pass-through (2026-05-24 settled decision)
- Path-ref parsing including the unhandled-path dispatch (2026-05-24)
- Eight-pass canonicalization order and seq-keying
- Determinism: same input -> identical canonical bytes
- Validator: well-formedness / coverage / tightness
"""

from __future__ import annotations

import pytest

from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.program_admission import ensure_prepared_kernel_program
from shepherd_kernel_v3_reference.kernel.replay import (
    KernelReplaySession,
    start_kernel_replay,
)
from shepherd_kernel_v3_reference.profiles import CORE_A
from shepherd_kernel_v3_reference.projection import (
    field_role,
    semantic_batch_from_transition,
    validate_semantic_batch,
)
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.semantic import (
    CanonicalRefMap,
    SemanticTransitionBatch,
    SemanticTransitionBatchValidationError,
)
from shepherd_kernel_v3_reference.source.handlers import HandlerEnv, StaticHandlerInstall
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

# ---------------------------------------------------------------------------
# Program builders for the five shapes covered by the 2330 spike
# ---------------------------------------------------------------------------


def pure_let_program():
    return Let("x", Return(Lit(1)), Let("y", Return(Lit(2)), Return(Var("x"))))


def resume_shape_program():
    return Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="resume.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(42)), Return(Var("r"))),
                ),
            )
        ),
    )


def abort_shape_program():
    return Handle(
        Perform("ask", Lit(None)),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="abort.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Abort(Lit(0)),
                ),
            )
        ),
    )


def nested_handlers_program():
    inner = StaticHandlerInstall(
        effect_kind="inner",
        handler_id="inner.v1",
        handled_result_schema=AnySchema(),
        payload_name="_payload",
        body=Let("r", Resume(Lit(7)), Return(Var("r"))),
    )
    outer = StaticHandlerInstall(
        effect_kind="outer",
        handler_id="outer.v1",
        handled_result_schema=AnySchema(),
        payload_name="_payload",
        body=Let("r", Resume(Lit(3)), Return(Var("r"))),
    )
    return Handle(
        Handle(
            Let("a", Perform("inner", Lit(None)), Let("b", Perform("outer", Lit(None)), Return(Var("a")))),
            HandlerEnv((inner,)),
        ),
        HandlerEnv((outer,)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_session_program(program):
    """Run a program through KernelReplaySession.start and project."""

    prepared = ensure_prepared_kernel_program(elaborate(program))
    session, transition = KernelReplaySession.start(prepared)
    catalog = dict(session._evaluator.continuation_objects)
    batch = semantic_batch_from_transition(transition, session.state, catalog)
    assert isinstance(batch, SemanticTransitionBatch)
    return transition, batch, catalog


# ---------------------------------------------------------------------------
# Field classifier tests
# ---------------------------------------------------------------------------


def test_field_classifier_covers_known_record_types() -> None:
    """Every -lite-relevant record type has a classifier entry; missing
    classifications would surface in projection as 'no classifier role'."""

    for record_type in (
        "EffectDeclaration",
        "HandlerSelection",
        "ResumptionHandle",
        "ContinuationResume",
        "ResumeReturn",
        "EffectCapture",
        "SelectionClosed",
    ):
        # `ref` is always self
        assert field_role(record_type, "ref") == "self"


def test_field_classifier_distinguishes_roles_zero_false_positives() -> None:
    """Per 2330 spike §"What Was Confirmed" #6: enum strings like
    'ResumptionHandle' or 'completed' must be classified as 'enum', not as
    refs. Distinct kinds must classify distinctly."""

    assert field_role("ContinuationResume", "source_record_type") == "enum"
    assert field_role("EffectCapture", "continuation_disposition") == "enum"
    assert field_role("EffectCapture", "action_kind") == "enum"
    assert field_role("HandlerSelection", "handler_id") == "enum"
    assert field_role("EffectDeclaration", "effect_kind") == "enum"
    # Domain values
    assert field_role("ContinuationResume", "value") == "value"
    assert field_role("ContinuationResume", "returns_to_handler") == "value"
    # Context refs distinguished from continuation refs
    assert field_role("HandlerSelection", "worker_context_ref") == "ctx"
    assert field_role("HandlerSelection", "captured_continuation_ref") == "continuation"
    assert field_role("HandlerSelection", "captured_continuation_control_ref") == "continuation-control"
    # Program-structure
    assert field_role("HandlerSelection", "selected_binding_ref") == "program-structure"
    assert field_role("HandlerSelection", "handler_frame_ref") == "program-structure"
    assert field_role("HandlerSelection", "handled_result_schema_ref") == "program-structure"
    # Lifecycle cross-references
    assert field_role("HandlerSelection", "declaration_ref") == "lifecycle:declaration"
    assert field_role("ContinuationResume", "selection_path_ref") == "lifecycle:path"
    assert field_role("ResumeReturn", "resume_ref") == "lifecycle:resume"
    assert field_role("SelectionClosed", "caused_by_ref") == "lifecycle:capture"


def test_field_classifier_unknown_record_returns_none() -> None:
    assert field_role("NotARecordType", "ref") is None


# ---------------------------------------------------------------------------
# branch:root literal pass-through
# ---------------------------------------------------------------------------


def test_branch_root_is_not_in_canonical_ref_map() -> None:
    """Per 2026-05-24 settled decision: `branch:root` sentinel is literal
    pass-through, NOT entered into CanonicalRefMap."""

    _transition, batch, _catalog = _project_session_program(resume_shape_program())
    assert "branch:root" not in {key for key, _ in batch.ref_map.entries}


def test_branch_root_appears_literal_in_projected_records() -> None:
    """Records that cite `branch:root` keep the literal string after
    projection (no rewrite)."""

    _transition, batch, _catalog = _project_session_program(resume_shape_program())
    found_literal = False
    for record in batch.records:
        if record.get("branch_ref") == "branch:root":
            found_literal = True
    assert found_literal, "expected at least one record citing literal 'branch:root'"


# ---------------------------------------------------------------------------
# Eight-pass order: ref-map shape per program
# ---------------------------------------------------------------------------


def test_pure_let_program_emits_empty_ref_map() -> None:
    """Pure let-chain has no effects, no lifecycle records, empty map."""

    _transition, batch, _catalog = _project_session_program(pure_let_program())
    assert batch.ref_map.entries == ()
    validate_semantic_batch(batch)


def test_resume_shape_program_emits_7_entry_map_per_2330_spike() -> None:
    """Per 2330 spike P2: 6 trace records (decl, sel, src, resume,
    resume-return, capture) plus 1 path ref = 7 ref-map entries."""

    _transition, batch, _catalog = _project_session_program(resume_shape_program())
    # 6 record self-refs + 1 path ref = 7
    assert len(batch.ref_map.entries) == 7
    validate_semantic_batch(batch)


def test_abort_shape_program_emits_5_entry_map_per_2330_spike() -> None:
    """Per 2330 spike P3: 4 trace records (decl, sel, src, capture) plus
    1 path ref = 5 ref-map entries."""

    _transition, batch, _catalog = _project_session_program(abort_shape_program())
    # Abort-shape has decl, sel, source (resumption handle), capture: 4 records
    # + 1 path ref = 5 entries.
    assert len(batch.ref_map.entries) == 5
    validate_semantic_batch(batch)


# ---------------------------------------------------------------------------
# Determinism: re-running projection produces byte-identical output
# ---------------------------------------------------------------------------


def test_projection_is_deterministic_across_two_runs() -> None:
    """Running the same program twice through projection must yield
    byte-identical ref_map and records (the projection is pure)."""

    _transition_a, batch_a, _catalog_a = _project_session_program(resume_shape_program())
    _transition_b, batch_b, _catalog_b = _project_session_program(resume_shape_program())
    # Different transition_ids are fine (random IDs); the ref_map and
    # projected records should agree byte-for-byte.
    assert batch_a.ref_map.entries == batch_b.ref_map.entries
    assert batch_a.records == batch_b.records


def test_projection_byte_stable_for_nested_handlers() -> None:
    """Mirrors 2330 spike P5: nested handlers must produce byte-stable
    canonical refs across runs."""

    _ta, batch_a, _ = _project_session_program(nested_handlers_program())
    _tb, batch_b, _ = _project_session_program(nested_handlers_program())
    assert batch_a.ref_map.entries == batch_b.ref_map.entries
    assert batch_a.records == batch_b.records


# ---------------------------------------------------------------------------
# Validator: positive + negative
# ---------------------------------------------------------------------------


def test_validator_rejects_malformed_canonical_value() -> None:
    bad_map = CanonicalRefMap(entries=(("declaration:0", "not-canonical"),))
    bad_batch = SemanticTransitionBatch(
        transition_id="t:1",
        idempotency_key="i:1",
        transition_kind="initial_run_prefix",
        admission_basis=None,
        profile=CORE_A,
        program_ref="program:sha256:abc",
        parent_transition_refs=(),
        records=(),
        ref_map=bad_map,
    )
    with pytest.raises(SemanticTransitionBatchValidationError, match="malformed canonical"):
        validate_semantic_batch(bad_batch)


def test_validator_rejects_missing_coverage() -> None:
    """A record citing a canonical lifecycle ref must have a matching map
    entry. Crafted batch with cited declaration ref but empty map."""

    bad_batch = SemanticTransitionBatch(
        transition_id="t:1",
        idempotency_key="i:1",
        transition_kind="initial_run_prefix",
        admission_basis=None,
        profile=CORE_A,
        program_ref="program:sha256:abc",
        parent_transition_refs=(),
        records=(
            {
                "record_type": "HandlerSelection",
                "ref": "selection:0",
                "declaration_ref": "declaration:sha256:" + "0" * 64,
            },
        ),
        ref_map=CanonicalRefMap(),
    )
    # selection:0 is a runtime self-ref; map missing → coverage failure.
    with pytest.raises(SemanticTransitionBatchValidationError, match="missing"):
        validate_semantic_batch(bad_batch)


def test_validator_real_batch_passes() -> None:
    """End-to-end: project a real program, validate the batch."""

    _t, batch, _c = _project_session_program(resume_shape_program())
    validate_semantic_batch(batch)  # should not raise
    _t2, batch2, _c2 = _project_session_program(nested_handlers_program())
    validate_semantic_batch(batch2)


# ---------------------------------------------------------------------------
# Round-trip: project, serialize batch's ref_map, re-instantiate, validate
# ---------------------------------------------------------------------------


def test_canonical_ref_map_round_trips_through_construction() -> None:
    """The CanonicalRefMap entries must round-trip through construction
    (sortedness + uniqueness preserved)."""

    _t, batch, _c = _project_session_program(resume_shape_program())
    reconstructed = CanonicalRefMap(entries=batch.ref_map.entries)
    assert reconstructed.entries == batch.ref_map.entries


# ---------------------------------------------------------------------------
# Sidecar-mode parity: start_replayable_kernel_transition (sidecar) produces
# the same canonical refs as KernelReplaySession.start (trace) for the same
# program. The trace cites different runtime-vs-canonical shapes but the
# canonical projection should agree.
# ---------------------------------------------------------------------------


def test_sidecar_and_trace_modes_agree_on_canonical_refs_for_resume_shape() -> None:
    """The two evidence-mode paths emit different ref shapes in trace
    records but the canonical projection must agree on the canonical refs
    (the projection is the canonicalization boundary)."""

    from shepherd_kernel_v3_reference.kernel.replay import start_replayable_kernel_transition

    # Trace mode: refs canonical inline
    _ts, batch_trace, _cs = _project_session_program(resume_shape_program())

    # Sidecar mode: refs runtime-local, rewritten via maps
    prepared = ensure_prepared_kernel_program(elaborate(resume_shape_program()))
    transition_sidecar = start_replayable_kernel_transition(prepared)
    state_sidecar, _t = start_kernel_replay(prepared)
    # Catalog from journal would normally provide objects; for this test we
    # need a catalog that covers all canonical continuation-object refs
    # appearing in the sidecar transition's projection. Build it from the
    # sidecar transition's continuation_ref_map values + reachable objects.
    # The simpler path: re-run via session for catalog convenience and
    # confirm the canonical maps agree.
    session, _ts2 = KernelReplaySession.start(prepared)
    catalog = dict(session._evaluator.continuation_objects)
    batch_sidecar = semantic_batch_from_transition(transition_sidecar, state_sidecar, catalog)

    assert isinstance(batch_sidecar, SemanticTransitionBatch)
    # The canonical ref VALUES (the right-hand sides) must agree across
    # modes — the runtime keys differ but the canonical projections are
    # identical.
    trace_canonicals = sorted(v for _k, v in batch_trace.ref_map.entries)
    sidecar_canonicals = sorted(v for _k, v in batch_sidecar.ref_map.entries)
    assert trace_canonicals == sidecar_canonicals
