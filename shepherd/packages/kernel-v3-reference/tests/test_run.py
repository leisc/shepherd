"""Normative API: start_kernel_run, resume_kernel_run, validate_observation_stream.

Per `260521-0600-kernel.md` §"API Shape" and
`260524-observation-stream-spike.md`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from shepherd_kernel_v3_reference.envelope import (
    CompletedResult,
    KernelRejection,
)
from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.admission import (
    AdmittedObservation,
    AdmittedObservationError,
)
from shepherd_kernel_v3_reference.kernel.replay import (
    ExternalEffectRequest,
    HostCompleted,
    KernelReplayState,
)
from shepherd_kernel_v3_reference.profiles import CORE_A
from shepherd_kernel_v3_reference.run import (
    resume_kernel_run,
    start_kernel_run,
    validate_observation_stream,
)
from shepherd_kernel_v3_reference.semantic import (
    AdmissionBasis,
    ContinuationSource,
    ObservedFrontier,
    OneShotKey,
    SourceGeneration,
)
from shepherd_kernel_v3_reference.source.syntax import Let, Lit, Perform, Return, Var

# --- Helpers -----------------------------------------------------------


def _observation_for(
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


def _suspended_program():
    return elaborate(Let("y", Perform("ask", Lit(None)), Return(Var("y"))))


def _two_suspend_program():
    return elaborate(
        Let("a", Perform("ask1", Lit(None)), Let("b", Perform("ask2", Lit(None)), Return(Var("a")))),
    )


# --- start_kernel_run --------------------------------------------------


def test_start_kernel_run_suspended() -> None:
    state, envelope = start_kernel_run(_suspended_program())
    assert envelope.status == "external-effect-request"
    assert isinstance(envelope.payload, ExternalEffectRequest)
    assert envelope.transition is not None
    assert envelope.profile is state.profile


def test_start_kernel_run_carries_profile() -> None:
    _state, envelope = start_kernel_run(_suspended_program())
    assert envelope.profile is CORE_A  # via ensure_prepared_kernel_program shim default


# --- resume_kernel_run -------------------------------------------------


def test_resume_kernel_run_advances() -> None:
    state, envelope = start_kernel_run(_suspended_program())
    request = envelope.payload
    assert isinstance(request, ExternalEffectRequest)
    obs = _observation_for(state, request, value="answer")
    new_state, new_envelope = resume_kernel_run(state, obs)
    assert new_envelope.status == "completed"
    assert isinstance(new_envelope.payload, CompletedResult)
    assert new_envelope.payload.value == "answer"
    assert new_state.terminal


def test_resume_kernel_run_raises_on_admission_failure() -> None:
    """Per the design: resume_kernel_run raises AdmittedObservationError on
    admission failure (no new transition is constructed)."""
    state, envelope = start_kernel_run(_suspended_program())
    request = envelope.payload
    assert isinstance(request, ExternalEffectRequest)
    obs = _observation_for(state, request)
    # Corrupt the bundle so admission fails (stale generation)
    bad_obs = replace(
        obs,
        admission_basis=replace(obs.admission_basis, source_generation=SourceGeneration(99)),
    )
    with pytest.raises(AdmittedObservationError) as exc:
        resume_kernel_run(state, bad_obs)
    assert exc.value.rejection_class == "source-basis-coherence"


# --- validate_observation_stream ---------------------------------------


def test_observation_stream_completes_two_step() -> None:
    """2-step stream completes terminally."""
    from shepherd_kernel_v3_reference.kernel.replay import (
        resume_kernel_replay,
        start_kernel_replay,
    )

    program = _two_suspend_program()
    # Build observations against fresh sequential states (mirroring spike)
    state, t1 = start_kernel_replay(program)
    req1 = t1.payload
    assert isinstance(req1, ExternalEffectRequest)
    obs1 = _observation_for(state, req1, value="first")
    state2, t2 = resume_kernel_replay(state, req1, obs1.observation)
    req2 = t2.payload
    assert isinstance(req2, ExternalEffectRequest)
    obs2 = _observation_for(state2, req2, value="second")

    final_state, envelope = validate_observation_stream(program, (obs1, obs2))
    assert envelope.status == "completed"
    assert isinstance(envelope.payload, CompletedResult)
    assert final_state.terminal


def test_observation_stream_fails_fast_on_admission_error() -> None:
    """Stale-frontier observation triggers admission rejection at index 1."""
    from shepherd_kernel_v3_reference.kernel.replay import (
        resume_kernel_replay,
        start_kernel_replay,
    )

    program = _two_suspend_program()
    state, t1 = start_kernel_replay(program)
    req1 = t1.payload
    assert isinstance(req1, ExternalEffectRequest)
    obs1 = _observation_for(state, req1, value="first")
    state2, t2 = resume_kernel_replay(state, req1, obs1.observation)
    req2 = t2.payload
    assert isinstance(req2, ExternalEffectRequest)
    obs2 = _observation_for(state2, req2, value="second")
    bad_obs2 = replace(
        obs2,
        admission_basis=replace(
            obs2.admission_basis,
            observed_frontier=ObservedFrontier(record_refs=("transition:WRONG",)),
        ),
    )

    _state, envelope = validate_observation_stream(program, (obs1, bad_obs2))
    assert envelope.status == "rejected"
    assert isinstance(envelope.payload, KernelRejection)
    assert envelope.payload.kind == "observation-admission"
    assert envelope.payload.rejection_index == 1
    assert envelope.payload.rejection_class == "frontier-prefix"


def test_observation_stream_detects_stream_too_long() -> None:
    """3 observations on a 2-suspend program: surplus rejection at index 2."""
    from shepherd_kernel_v3_reference.kernel.replay import (
        resume_kernel_replay,
        start_kernel_replay,
    )

    program = _two_suspend_program()
    state, t1 = start_kernel_replay(program)
    req1 = t1.payload
    assert isinstance(req1, ExternalEffectRequest)
    obs1 = _observation_for(state, req1, value="a")
    state2, t2 = resume_kernel_replay(state, req1, obs1.observation)
    req2 = t2.payload
    assert isinstance(req2, ExternalEffectRequest)
    obs2 = _observation_for(state2, req2, value="b")
    # Third observation: stream exceeds program needs
    _state, envelope = validate_observation_stream(program, (obs1, obs2, obs2))
    assert envelope.status == "rejected"
    assert isinstance(envelope.payload, KernelRejection)
    assert envelope.payload.rejection_index == 2
    assert envelope.payload.rejection_class == "state-level"
    assert "terminal" in envelope.payload.diagnostic


def test_observation_stream_no_observations_returns_initial_envelope() -> None:
    """Empty stream returns the start_kernel_run envelope unchanged."""
    program = _suspended_program()
    state, envelope = validate_observation_stream(program, ())
    assert envelope.status == "external-effect-request"
    assert state.transition_refs


def test_envelope_is_wireresult_compatible() -> None:
    """KernelResultEnvelope from start/resume can be projected to WireResult."""
    from shepherd_kernel_v3_reference.envelope import WireResult
    from shepherd_kernel_v3_reference.kernel.replay import KernelReplaySession
    from shepherd_kernel_v3_reference.projection import project_envelope_to_wire

    program = _suspended_program()
    # Use a session to get a catalog
    session, _t = KernelReplaySession.start(program)
    state, envelope = start_kernel_run(program)
    catalog = dict(session._evaluator.continuation_objects)
    wire = project_envelope_to_wire(envelope, state, catalog)
    assert isinstance(wire, WireResult)
