"""Sidecar-coupling acceptance test for ReplayableKernelTransition.context_ref_map.

Per 260521-0600-kernel.md §"Risks" #9 and §"Settled Design Decisions"
2026-05-23 — "Context catalog plumbing":

  Commit #71 must therefore include an explicit acceptance test that any
  transition emerging from start_replayable_kernel_transition(...) or
  KernelReplaySession carries a context_ref_map populated for every distinct
  ctx:runtime:N cited by its trace_delta records. This codifies the
  sidecar-mode obligation as a test rather than leaving it as unowned prose.

A future runtime change that disables sidecar-evidence mode on a
transition-producing path will fail this test at its source.
"""

import dataclasses

from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.replay import (
    KernelReplaySession,
    ReplayableKernelTransition,
    start_replayable_kernel_transition,
)
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.source.handlers import HandlerEnv, StaticHandlerInstall
from shepherd_kernel_v3_reference.source.syntax import Handle, Let, Lit, Perform, Resume, Return, Var


def _cited_refs_with_prefix(transition: ReplayableKernelTransition, prefix: str) -> set[str]:
    """Walk trace_delta records and collect every distinct ref matching prefix."""

    cited: set[str] = set()
    for record in transition.trace_delta:
        for f in dataclasses.fields(record):
            value = getattr(record, f.name)
            if isinstance(value, str) and value.startswith(prefix):
                cited.add(value)
    return cited


def _cited_ctx_refs(transition: ReplayableKernelTransition) -> set[str]:
    return _cited_refs_with_prefix(transition, "ctx:runtime:")


def _handled_effect_program():
    return Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="ctx-sidecar.handler.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(42)), Return(Var("r"))),
                ),
            )
        ),
    )


def _nested_handler_program():
    inner_handler = StaticHandlerInstall(
        effect_kind="inner",
        handler_id="ctx-sidecar.inner.v1",
        handled_result_schema=AnySchema(),
        payload_name="_payload",
        body=Let("r", Resume(Lit(7)), Return(Var("r"))),
    )
    outer_handler = StaticHandlerInstall(
        effect_kind="outer",
        handler_id="ctx-sidecar.outer.v1",
        handled_result_schema=AnySchema(),
        payload_name="_payload",
        body=Let("r", Resume(Lit(3)), Return(Var("r"))),
    )
    return Handle(
        Handle(
            Let(
                "a",
                Perform("inner", Lit(None)),
                Let("b", Perform("outer", Lit(None)), Return(Var("a"))),
            ),
            HandlerEnv((inner_handler,)),
        ),
        HandlerEnv((outer_handler,)),
    )


def _assert_sidecar_covers_cited(transition: ReplayableKernelTransition) -> None:
    # Programs that cite no runtime refs (e.g. pure-let or trace mode)
    # acceptably carry empty maps; the rule is coverage, not non-emptiness.
    #
    # Canonical prefixes: ctx refs canonicalize to `ctx:sha256:...`;
    # continuation/continuation-control runtime refs canonicalize to
    # `continuation-object:sha256:...` (catalog-keyed) via the
    # continuation-object catalog.
    for runtime_prefix, attr_name, expected_canonical_prefixes in (
        ("ctx:runtime:", "context_ref_map", ("ctx:sha256:",)),
        (
            "continuation:runtime:",
            "continuation_ref_map",
            ("continuation:sha256:", "continuation-object:sha256:"),
        ),
        (
            "continuation-control:runtime:",
            "continuation_control_ref_map",
            ("continuation-control:sha256:", "continuation-object:sha256:"),
        ),
    ):
        cited = _cited_refs_with_prefix(transition, runtime_prefix)
        sidecar = getattr(transition, attr_name)
        missing = cited - set(sidecar)
        assert not missing, (
            f"{attr_name} missing entries for refs cited by trace_delta: "
            f"{sorted(missing)!r}; cited={sorted(cited)!r}; "
            f"map_keys={sorted(sidecar)!r}"
        )
        for runtime_ref, canonical_ref in sidecar.items():
            assert runtime_ref.startswith(runtime_prefix), (
                f"{attr_name} key {runtime_ref!r} does not match runtime prefix {runtime_prefix!r}"
            )
            assert any(canonical_ref.startswith(p) for p in expected_canonical_prefixes), (
                f"{attr_name}[{runtime_ref!r}] canonical {canonical_ref!r} "
                f"does not match any expected prefix {expected_canonical_prefixes!r}"
            )


def test_start_replayable_kernel_transition_populates_context_ref_map() -> None:
    """start_replayable_kernel_transition runs under evidence_mode='sidecar':
    trace records cite runtime-local ctx:runtime:N refs; context_ref_map must
    cover every cited ref. This is the canary path the projection function
    consumes."""

    transition = start_replayable_kernel_transition(elaborate(_handled_effect_program()))
    cited = _cited_ctx_refs(transition)
    assert cited, (
        "handled-effect program under sidecar mode must cite at least one "
        "ctx:runtime ref; regression in sidecar trace evidence emission?"
    )
    _assert_sidecar_covers_cited(transition)


def test_kernel_replay_session_satisfies_coverage_invariant() -> None:
    """KernelReplaySession.start runs under evidence_mode='trace': trace records
    cite content-addressed ctx:sha256:HEX directly, so no ctx:runtime:N refs
    appear. The coverage invariant is vacuously satisfied (empty cited set ⊆
    any map). The projection function's ctx pre-pass becomes a no-op for these
    transitions, which is consistent — canonical-inline ctx refs need no
    rewrite. This test enforces that the trace-mode path does not silently
    leak runtime-local ctx refs without a backing map."""

    _session, transition = KernelReplaySession.start(elaborate(_handled_effect_program()))
    _assert_sidecar_covers_cited(transition)


def test_nested_handler_program_carries_all_distinct_ctx_refs() -> None:
    """Nested handlers produce multiple distinct ctx refs; the sidecar must
    cover every one. Mirrors the 2330 projection spike's P5 shape."""

    transition = start_replayable_kernel_transition(elaborate(_nested_handler_program()))
    cited = _cited_ctx_refs(transition)
    assert len(cited) >= 2, f"nested-handler program should cite multiple distinct ctx refs; got {sorted(cited)!r}"
    _assert_sidecar_covers_cited(transition)


def test_context_ref_map_round_trips_through_json() -> None:
    """The sidecar map must survive JSON round-trip on both standalone and
    journal-embedded transition serialization."""

    from shepherd_kernel_v3_reference.kernel.replay import (
        replayable_kernel_transition_from_json,
        replayable_kernel_transition_to_json,
    )

    transition = start_replayable_kernel_transition(elaborate(_handled_effect_program()))
    assert transition.context_ref_map, "test premise: sidecar mode should populate the map"

    round_tripped = replayable_kernel_transition_from_json(replayable_kernel_transition_to_json(transition))
    assert dict(round_tripped.context_ref_map) == dict(transition.context_ref_map)
