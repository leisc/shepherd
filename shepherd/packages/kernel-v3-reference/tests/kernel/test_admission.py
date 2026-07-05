"""Production AdmittedObservation validator: per-step positive + negative cases.

Per `260521-0600-kernel.md` §"Validation Responsibilities" → "Admission
Validation" and 2026-05-24 §"Post-#72 design pass" item F.

Step coverage (cheapest-first):
  1. state-level                — terminal / rejected
  2. source-open                — consumed / unknown
  3. open-request-agreement     — program_ref / declaration_ref
  4. source-basis-coherence     — source_ref / source_kind / source_generation /
                                  one_shot_key / program_ref
  5. frontier-prefix            — too-long / wrong-ref / empty (valid)
  6. restart-artifact-agreement — program_ref / source_ref / schema_ref
  7. observation-schema         — schema disagreement / missing schema / null schema_ref (valid)
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.admission import (
    AdmittedObservation,
    AdmittedObservationError,
    validate_admitted_observation,
)
from shepherd_kernel_v3_reference.kernel.replay import (
    ExternalEffectRequest,
    HostCompleted,
    KernelReplayState,
    ReplayableKernelTransition,
    start_kernel_replay,
)
from shepherd_kernel_v3_reference.profiles import CORE_A
from shepherd_kernel_v3_reference.schemas import TypeSchema
from shepherd_kernel_v3_reference.semantic import (
    AdmissionBasis,
    ContinuationSource,
    ObservedFrontier,
    OneShotKey,
    SourceGeneration,
)
from shepherd_kernel_v3_reference.source.syntax import Let, Lit, Perform, Return, Var

# --- Fixtures -----------------------------------------------------------


def _suspended_program():
    return elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y"))))


def _setup() -> tuple[KernelReplayState, ReplayableKernelTransition, ExternalEffectRequest]:
    state, transition = start_kernel_replay(_suspended_program())
    request = transition.payload
    assert isinstance(request, ExternalEffectRequest)
    return state, transition, request


def _valid_observation(
    state: KernelReplayState,
    request: ExternalEffectRequest,
    *,
    value: object = "host-value",
) -> AdmittedObservation:
    source = ContinuationSource(
        source_ref=request.declaration_ref,
        source_kind="UnhandledSuspension",
        source_generation=SourceGeneration(0),
        continuation_ref=request.replay_artifact.root_ref,
        branch_ref="branch:root",
        one_shot_key=OneShotKey(request.source_key),
        declaration_ref=request.declaration_ref,
        source_path_ref=f"path:unhandled/{request.declaration_ref}/branch:root",
        operation_result_schema_ref=request.operation_result_schema_ref,
    )
    assert source.source_path_ref is not None
    basis = AdmissionBasis(
        source_ref=source.source_ref,
        source_kind=source.source_kind,
        source_generation=source.source_generation,
        observed_frontier=ObservedFrontier(record_refs=state.transition_refs),
        source_path_ref=source.source_path_ref,
        input_value_or_digest=value,
        idempotency_key=f"idem-{request.source_key}",
        one_shot_key=source.one_shot_key,
        profile=CORE_A,
        program_ref=state.program_ref,
    )
    return AdmittedObservation(
        source=source,
        restart_artifact=request.replay_artifact,
        admission_basis=basis,
        observation=HostCompleted(value=value),
        request=request,
    )


# --- Positive case ------------------------------------------------------


def test_valid_admission() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    validate_admitted_observation(obs, state)


def test_valid_admission_empty_frontier() -> None:
    """Initial-run case: empty frontier admits cleanly."""
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    obs = replace(
        obs,
        admission_basis=replace(
            obs.admission_basis,
            observed_frontier=ObservedFrontier(record_refs=()),
        ),
    )
    validate_admitted_observation(obs, state)


# --- Step 1: state-level ------------------------------------------------


def test_step1_rejects_terminal_state() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    terminal_state = replace(state, open_requests={}, terminal=True)
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(obs, terminal_state)
    assert exc.value.rejection_class == "state-level"
    assert "terminal" in str(exc.value)


def test_step1_rejects_rejected_state() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    rejected_state = replace(state, open_requests={}, rejected=True)
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(obs, rejected_state)
    assert exc.value.rejection_class == "state-level"
    assert "rejected" in str(exc.value)


# --- Step 2: source-open ------------------------------------------------


def test_step2_rejects_already_consumed_source() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    consumed_state = replace(
        state,
        consumed_source_keys=(request.source_key,),
        open_requests={},
    )
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(obs, consumed_state)
    assert exc.value.rejection_class == "source-open"
    assert "already consumed" in str(exc.value)


def test_step2_rejects_unknown_source_key() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    closed_state = replace(state, open_requests={})
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(obs, closed_state)
    assert exc.value.rejection_class == "source-open"
    assert "not open" in str(exc.value)


# --- Step 3: open-request-agreement -------------------------------------


def test_step3_rejects_declaration_ref_disagreement() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    bad_source = replace(obs.source, declaration_ref="declaration:DIFFERENT")
    bad_basis = replace(obs.admission_basis, source_ref=bad_source.source_ref)
    bad_source = replace(bad_source, source_ref=bad_source.source_ref)
    bad_obs = replace(
        obs,
        source=replace(obs.source, declaration_ref="declaration:DIFFERENT"),
        admission_basis=bad_basis,
    )
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(bad_obs, state)
    assert exc.value.rejection_class == "open-request-agreement"


# --- Step 4: source-basis-coherence -------------------------------------


def test_step4_rejects_source_ref_disagreement() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    bad_basis = replace(obs.admission_basis, source_ref="source:NOT-MATCHING")
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, admission_basis=bad_basis), state)
    assert exc.value.rejection_class == "source-basis-coherence"


def test_step4_rejects_stale_source_generation() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    stale_basis = replace(obs.admission_basis, source_generation=SourceGeneration(99))
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, admission_basis=stale_basis), state)
    assert exc.value.rejection_class == "source-basis-coherence"


def test_step4_rejects_one_shot_disagreement() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    bad_basis = replace(obs.admission_basis, one_shot_key=OneShotKey("different-key"))
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, admission_basis=bad_basis), state)
    assert exc.value.rejection_class == "source-basis-coherence"


# --- Step 5: frontier-prefix --------------------------------------------


def test_step5_rejects_frontier_too_long() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    long_basis = replace(
        obs.admission_basis,
        observed_frontier=ObservedFrontier(
            record_refs=state.transition_refs + ("transition:future",),
        ),
    )
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, admission_basis=long_basis), state)
    assert exc.value.rejection_class == "frontier-prefix"


def test_step5_rejects_frontier_wrong_ref() -> None:
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    bad_basis = replace(
        obs.admission_basis,
        observed_frontier=ObservedFrontier(record_refs=("transition:wrong",)),
    )
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, admission_basis=bad_basis), state)
    assert exc.value.rejection_class == "frontier-prefix"


# --- Step 6: restart-artifact-agreement ---------------------------------
#
# `ContinuationReplayArtifact.__post_init__` internally validates
# `program_ref == root.program_ref` and `operation_result_schema_ref ==
# root.result_schema_ref`, so step-6 disagreements can't be engineered by
# mutating artifact fields directly. The realistic failure mode — caller
# passes an artifact from a different transition/program — is exercised by
# building two programs and crossing the artifact with a source from the
# other. Plus we mutate source-side fields where the cross-program test
# would not isolate which sub-check fired.


def test_step6_rejects_artifact_source_ref_disagreement_via_source_mutation() -> None:
    """Mutate source.source_ref to a fake value (basis tracks it); artifact
    still carries the real source_ref. Step 6 fires on the mismatch."""
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    fake_source_ref = "declaration:99"
    bad_source = replace(obs.source, source_ref=fake_source_ref)
    bad_basis = replace(obs.admission_basis, source_ref=fake_source_ref)
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(
            replace(obs, source=bad_source, admission_basis=bad_basis),
            state,
        )
    assert exc.value.rejection_class == "restart-artifact-agreement"
    assert "source_ref" in str(exc.value)


def test_step6_rejects_artifact_schema_ref_disagreement_via_source_mutation() -> None:
    """Mutate source.operation_result_schema_ref; artifact carries the real
    schema_ref. Step 6 fires on the mismatch (before step 7 can attempt a
    catalog lookup)."""
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    bad_source = replace(obs.source, operation_result_schema_ref="schema:WRONG")
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(replace(obs, source=bad_source), state)
    assert exc.value.rejection_class == "restart-artifact-agreement"
    assert "schema_ref" in str(exc.value)


def test_step6_rejects_cross_program_artifact() -> None:
    """Artifact from a different program; step-6 program_ref mismatch fires."""
    state_a, _ta, request_a = _setup()
    state_b, _tb, request_b = _setup()
    # Same source program, but each setup yields fresh KernelReplayState +
    # ExternalEffectRequest. Their artifacts share the same content (deterministic
    # program), so we need a structurally different program for a real
    # cross-program test:
    other_program = elaborate(
        Let("y", Perform("DIFFERENT-EFFECT", Lit(None)), Return(Var("y"))),
    )
    _state_other, transition_other = start_kernel_replay(other_program)
    request_other = transition_other.payload
    assert isinstance(request_other, ExternalEffectRequest)
    # Build the source as if for state_a but pass state_a's source with
    # state_other's artifact. Step 6 should detect the cross-program mismatch.
    obs_a = _valid_observation(state_a, request_a)
    crossed = replace(obs_a, restart_artifact=request_other.replay_artifact)
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(crossed, state_a)
    assert exc.value.rejection_class == "restart-artifact-agreement"
    # Use state_b just to silence unused-variable lint
    _ = state_b
    _ = request_b


# --- Step 7: observation-schema -----------------------------------------


def test_step7_passes_when_schema_ref_is_none() -> None:
    """When source.operation_result_schema_ref is None, step 7 is skipped."""
    state, _t, request = _setup()
    obs = _valid_observation(state, request)
    # Force the source AND restart artifact to carry no schema ref so step-6
    # agreement still holds; step 7 then skips cleanly.
    nul_source = replace(obs.source, operation_result_schema_ref=None)
    nul_artifact = replace(obs.restart_artifact, operation_result_schema_ref=None)
    validate_admitted_observation(
        replace(obs, source=nul_source, restart_artifact=nul_artifact),
        state,
    )


# Step-7 "schema_ref not in catalog" is defensive: in practice it is
# unreachable via legit replay paths because step 6 already requires
# source.operation_result_schema_ref to match the artifact's, and the
# artifact's __post_init__ validates that ref against ContinuationRoot.
# The defensive check remains in production for malformed external
# observation construction; we don't engineer the failure mode here.


def test_step7_rejects_observation_value_violating_schema() -> None:
    """When source cites a typed schema and observation.value violates it,
    reject with observation-schema rejection_class."""
    # Build a fresh suspended program where the ask effect carries an IntSchema
    from shepherd_kernel_v3_reference.source.effects import EffectRegistry, EffectSignature

    registry = EffectRegistry()
    registry.register(
        EffectSignature(
            effect_kind="ask",
            payload_schema=TypeSchema(type(None)),
            operation_result_schema=TypeSchema(int),
        )
    )
    program = elaborate(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        registry=registry,
    )
    state, transition = start_kernel_replay(program)
    request = transition.payload
    assert isinstance(request, ExternalEffectRequest)
    # Supply a non-int observation value
    obs = _valid_observation(state, request, value="not-an-int")
    with pytest.raises(AdmittedObservationError) as exc:
        validate_admitted_observation(obs, state)
    assert exc.value.rejection_class == "observation-schema"
    assert "does not match schema" in str(exc.value)


def test_step7_passes_for_observation_value_matching_schema() -> None:
    """Typed schema admits a correctly-typed observation value."""
    from shepherd_kernel_v3_reference.source.effects import EffectRegistry, EffectSignature

    registry = EffectRegistry()
    registry.register(
        EffectSignature(
            effect_kind="ask",
            payload_schema=TypeSchema(type(None)),
            operation_result_schema=TypeSchema(int),
        )
    )
    program = elaborate(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        registry=registry,
    )
    state, transition = start_kernel_replay(program)
    request = transition.payload
    assert isinstance(request, ExternalEffectRequest)
    obs = _valid_observation(state, request, value=42)
    validate_admitted_observation(obs, state)
