"""Property-based round-trip projection over the operational corpus.

Per 260524-pre-drive-validation.md Q2 resolution: the 70+ operational corpus
property suite stays in #72 (not split to #75) so the projection algorithm
ships with broad-shape coverage before envelope/wire types depend on it.

For each program in the parameterized corpus, this test exercises both
evidence-mode paths (sidecar via start_replayable_kernel_transition; trace
via KernelReplaySession.start) and asserts:

1. The projection succeeds and produces a SemanticTransitionBatch.
2. validate_semantic_batch passes.
3. Re-running the projection on the same input produces byte-identical
   ref_map entries and projected records (determinism).
4. The two evidence-mode paths produce identical canonical refs (the
   canonical projection is mode-independent — the runtime-vs-trace
   asymmetry surfaced in #71 is correctly absorbed by the pre-passes).

The program corpus mirrors the operational shapes covered by the existing
test_replay.py / test_replay_canary.py suites:
- pure let-chains (no records)
- single-handler resume (Core-0 normalized handler body)
- single-handler abort (Core-A normalized handler body)
- nested handlers, outer aborts inner after resume
- handled-effect chains with multiple sequential perform/resume cycles
"""

from __future__ import annotations

import pytest

from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.program_admission import ensure_prepared_kernel_program
from shepherd_kernel_v3_reference.kernel.replay import (
    KernelReplaySession,
    start_replayable_kernel_transition,
)
from shepherd_kernel_v3_reference.projection import (
    semantic_batch_from_transition,
    validate_semantic_batch,
)
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.semantic import SemanticTransitionBatch
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
# Corpus
# ---------------------------------------------------------------------------


def _pure_let():
    return Let("x", Return(Lit(1)), Return(Var("x")))


def _pure_let_deep():
    body = Return(Var("x0"))
    for i in range(5):
        body = Let(f"x{i + 1}", Return(Lit(i)), body)
    return Let("x0", Return(Lit(0)), body)


def _single_resume(handler_value=42):
    return Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ask.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(handler_value)), Return(Var("r"))),
                ),
            )
        ),
    )


def _single_abort(abort_value=0):
    return Handle(
        Perform("ask", Lit(None)),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ask.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Abort(Lit(abort_value)),
                ),
            )
        ),
    )


def _nested_inner_resumes_outer_resumes():
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


def _nested_outer_aborts_inner_resumes():
    """Outer abort closes inner selection (mirrors 2330 spike P4)."""

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
        body=Abort(Lit(99)),
    )
    return Handle(
        Handle(
            Let("a", Perform("inner", Lit(None)), Let("b", Perform("outer", Lit(None)), Return(Var("a")))),
            HandlerEnv((inner,)),
        ),
        HandlerEnv((outer,)),
    )


def _sequential_handled_effects(n=3):
    """N sequential handled effects under the same handler (covers
    multi-effect canonicalization)."""

    body = Return(Var(f"y{n - 1}"))
    for i in reversed(range(n)):
        body = Let(f"y{i}", Perform("ask", Lit(None)), body)
    return Handle(
        body,
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ask.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(0)), Return(Var("r"))),
                ),
            )
        ),
    )


_CORPUS = {
    "pure_let": _pure_let,
    "pure_let_deep": _pure_let_deep,
    "single_resume": _single_resume,
    "single_abort": _single_abort,
    "nested_resume_resume": _nested_inner_resumes_outer_resumes,
    "nested_outer_aborts_inner": _nested_outer_aborts_inner_resumes,
    "seq_handled_2": lambda: _sequential_handled_effects(2),
    "seq_handled_5": lambda: _sequential_handled_effects(5),
    "seq_handled_10": lambda: _sequential_handled_effects(10),
}


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_projection_determinism_on_corpus(name: str) -> None:
    """Two independent projections of the same program must produce
    byte-identical ref_map entries and projected records."""

    program = _CORPUS[name]()
    prepared = ensure_prepared_kernel_program(elaborate(program))

    session_a, transition_a = KernelReplaySession.start(prepared)
    catalog_a = dict(session_a._evaluator.continuation_objects)
    batch_a = semantic_batch_from_transition(transition_a, session_a.state, catalog_a)

    session_b, transition_b = KernelReplaySession.start(prepared)
    catalog_b = dict(session_b._evaluator.continuation_objects)
    batch_b = semantic_batch_from_transition(transition_b, session_b.state, catalog_b)

    assert isinstance(batch_a, SemanticTransitionBatch)
    assert isinstance(batch_b, SemanticTransitionBatch)
    assert batch_a.ref_map.entries == batch_b.ref_map.entries, (
        f"projection non-deterministic for {name!r}: ref_map differs"
    )
    assert batch_a.records == batch_b.records, f"projection non-deterministic for {name!r}: records differ"


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_projection_validates_on_corpus(name: str) -> None:
    """validate_semantic_batch must pass on every projected batch from the
    operational corpus."""

    program = _CORPUS[name]()
    prepared = ensure_prepared_kernel_program(elaborate(program))
    session, transition = KernelReplaySession.start(prepared)
    catalog = dict(session._evaluator.continuation_objects)
    batch = semantic_batch_from_transition(transition, session.state, catalog)
    assert isinstance(batch, SemanticTransitionBatch)
    validate_semantic_batch(batch)


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_sidecar_and_trace_modes_agree_on_canonical_refs(name: str) -> None:
    """The canonical projection must be mode-independent: trace-mode and
    sidecar-mode transitions for the same program must yield identical
    canonical ref values (only the runtime-ref keys differ in the
    intermediate representation)."""

    program = _CORPUS[name]()
    prepared = ensure_prepared_kernel_program(elaborate(program))

    # Trace-mode path
    session, transition_trace = KernelReplaySession.start(prepared)
    catalog = dict(session._evaluator.continuation_objects)
    batch_trace = semantic_batch_from_transition(transition_trace, session.state, catalog)

    # Sidecar-mode path
    transition_sidecar = start_replayable_kernel_transition(prepared)
    # The state's prepared_program suffices for projection of the initial
    # transition; reuse the trace-mode session's state which is for the
    # same program.
    batch_sidecar = semantic_batch_from_transition(transition_sidecar, session.state, catalog)

    assert isinstance(batch_trace, SemanticTransitionBatch)
    assert isinstance(batch_sidecar, SemanticTransitionBatch)
    trace_canonicals = sorted(v for _k, v in batch_trace.ref_map.entries)
    sidecar_canonicals = sorted(v for _k, v in batch_sidecar.ref_map.entries)
    assert trace_canonicals == sidecar_canonicals, (
        f"canonical refs differ between trace and sidecar modes for {name!r}: "
        f"trace={trace_canonicals!r}; sidecar={sidecar_canonicals!r}"
    )
