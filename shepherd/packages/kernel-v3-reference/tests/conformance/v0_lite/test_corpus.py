"""End-to-end exercise of the `-lite` differential corpus.

Per `260524-post-72-design-pass.md` §"Item B" and `260521-0600-kernel.md`
§"Phase 6".

For positive fixtures, runs the program through `start_kernel_run(...)`
and validates the envelope shape matches `expected.envelope_status`
(plus completed value where applicable).

Negative fixtures are handled per `kind`: `negative-profile-admission`
runs `validate_profile_admission(...)` and asserts the rejection
construct/substring; `negative-observation-admission` runs
`validate_observation_stream(...)` and asserts the rejection
index/class/substring. The remaining negative kinds
(`negative-kernel-admission`, `negative-runtime-rejection`,
`negative-ref-map`) are reserved for later corpus expansions; see
README §"Coverage status and deliberate omissions" for why
`negative-ref-map` is deferred to Phase 8.

Positive fixtures also freeze `expected.batch` — the canonical wire
encoding of the initial-run-prefix transition — and assert it
byte-for-byte (`test_positive_fixture_batch_is_byte_stable`). That is
the Python-side shape-lock and the Lean Phase 9 differential input.
Per-stream resume-transition batches are not yet frozen (blocked on
`semantic_batch_from_transition` supporting non-initial transitions);
see README §"Coverage status". Regenerate frozen batches with
`uv run python tests/conformance/v0_lite/regenerate.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from shepherd_kernel_v3_reference.envelope import CompletedResult, KernelRejection
from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.kernel.admission import AdmittedObservation
from shepherd_kernel_v3_reference.kernel.program_admission import admit_and_prepare
from shepherd_kernel_v3_reference.kernel.refs import canonical_json
from shepherd_kernel_v3_reference.kernel.replay import (
    ExternalEffectRequest,
    HostCompleted,
    KernelReplaySession,
    KernelReplayState,
    ReplayableExternalEffectRequest,
    ReplayableKernelTransition,
    external_effect_request_from_json,
    external_effect_request_to_json,
)
from shepherd_kernel_v3_reference.profile_admission import (
    ProfileAdmissionError,
    validate_profile_admission,
)
from shepherd_kernel_v3_reference.profiles import CORE_A, CORE_REFERENCE_V0_LITE
from shepherd_kernel_v3_reference.projection import (
    semantic_batch_from_transition,
    validate_semantic_batch,
)
from shepherd_kernel_v3_reference.run import start_kernel_run, validate_observation_stream
from shepherd_kernel_v3_reference.semantic import (
    AdmissionBasis,
    ContinuationSource,
    ObservedFrontier,
    OneShotKey,
    SemanticTransitionBatch,
    SourceGeneration,
)
from shepherd_kernel_v3_reference.wire import (
    profile_rejected_to_wire,
    semantic_batch_to_wire,
)

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from loader import (
    Fixture,
    iter_fixtures,
)

CORPUS_ROOT = Path(__file__).parent


def _positive_fixtures() -> list[Fixture]:
    return iter_fixtures(CORPUS_ROOT, "positive")


def _negative_fixtures() -> list[Fixture]:
    return iter_fixtures(CORPUS_ROOT, "negative")


# --- Observation-stream helpers ----------------------------------------
#
# Lifted from tests/test_run.py and the observation_stream_validation
# spike. The fixture JSON encodes observation specs (e.g.
# ``{"value": 42}`` or ``{"reuse_index": 0}``); this helper threads
# state through sequential resume_kernel_run / validate_observation_stream
# calls and constructs the full AdmittedObservation bundle from the spec
# against the live request.


def _observation_for(
    state: KernelReplayState,
    request: ExternalEffectRequest,
    *,
    value: object,
) -> AdmittedObservation:
    """Build an AdmittedObservation bundle from current state + open request.

    Used by the observation-stream tests (negative observation-admission
    fixtures) under the operational `CORE_A` profile. The positive corpus's
    batch projection uses `_lite_admission_basis(...)` instead, which needs
    only the open request's `declaration_ref` / `source_key` and stamps the
    `-lite` profile.
    """

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


def _build_observation_stream(
    fixture: Fixture,
) -> tuple[KernelReplayState, tuple[AdmittedObservation, ...]]:
    """Walk the fixture's observation specs and construct AdmittedObservations.

    Threads state forward via start_kernel_run / resume_kernel_run between
    specs so each AdmittedObservation is built against the live open
    request. ``{"reuse_index": N}`` returns the observation from earlier
    index N (used to exercise one-shot reuse).
    """

    program = elaborate(fixture.program, registry=fixture.registry)
    state, envelope = start_kernel_run(program, registry=fixture.registry)
    built: list[AdmittedObservation] = []

    for i, spec in enumerate(fixture.observations):
        if "reuse_index" in spec:
            reuse_index = spec["reuse_index"]
            if reuse_index >= len(built):
                raise ValueError(
                    f"{fixture.case}: observations[{i}].reuse_index={reuse_index} refers to a not-yet-built observation"
                )
            built.append(built[reuse_index])
            continue

        request = envelope.payload
        if not isinstance(request, ExternalEffectRequest):
            raise ValueError(
                f"{fixture.case}: observations[{i}] requires an open "
                f"ExternalEffectRequest but envelope.payload is "
                f"{type(request).__name__}"
            )

        if fixture.restart_via_serialized:
            # Prove closure-free replay: rebuild the request (and its
            # ContinuationReplayArtifact) from its JSON form, so no live
            # Python continuation closure survives into resume. The rebuilt
            # request must still content-ref-match the open request in state
            # (resume_kernel_replay validates this), which is exactly the
            # closure-free property. Closure-free replay acceptance criteria
            # (#2/#3) in 260521-0600-kernel.md.
            request = external_effect_request_from_json(external_effect_request_to_json(request))

        obs = _observation_for(state, request, value=spec["value"])
        built.append(obs)

        # Advance state for the next spec (unless this is the last one;
        # validate_observation_stream will re-run from start in the test).
        if i + 1 < len(fixture.observations) and "reuse_index" not in fixture.observations[i + 1]:
            # Step forward to expose the next open request, but only if
            # the next spec needs to be built against a fresh state.
            from shepherd_kernel_v3_reference.run import resume_kernel_run

            state, envelope = resume_kernel_run(state, obs)

    return state, tuple(built)


# --- Byte-stable batch projection --------------------------------------
#
# `expected.batches` freezes the canonical wire encoding of the fixture's
# full transition sequence: the initial-run-prefix transition plus one
# resume transition per observation. This is the differential gate Lean
# Phase 9 consumes — it diffs the whole observation stream, not just the
# initial prefix.
#
# Batches are produced through the proven projection path
# (`KernelReplaySession.start` / `.resume` -> `semantic_batch_from_transition`
# -> `wire.semantic_batch_to_wire`), under the `-lite` profile so each batch
# carries `profile="core_reference_v0_lite"`. A single live session keeps the
# evaluator/continuation-object catalog live across resumes; each resume's
# AdmissionBasis (#78) is derived from the live open request and carries the
# `-lite` profile so `build_admitted_transition_batch` profile-agreement holds.


def _project_transition_to_wire(
    transition: ReplayableKernelTransition,
    session: KernelReplaySession,
    *,
    admission_basis: AdmissionBasis | None,
) -> dict:
    catalog = dict(session._evaluator.continuation_objects)
    batch = semantic_batch_from_transition(transition, session.state, catalog, admission_basis=admission_basis)
    if isinstance(batch, SemanticTransitionBatch):
        validate_semantic_batch(batch)
        return semantic_batch_to_wire(batch)
    return profile_rejected_to_wire(batch)


def _lite_admission_basis(
    state: KernelReplayState,
    request: ReplayableExternalEffectRequest,
    *,
    value: object,
) -> AdmissionBasis:
    """Build the `-lite` AdmissionBasis for a resume against an open request.

    Reads only `declaration_ref` and `source_key`, both exposed by the full
    `ExternalEffectRequest` and the compact `ExternalEffectRequestRef` the
    live session may hand back. The projection needs the basis (not the
    source/restart artifact), so the compact ref is sufficient here.
    """

    return AdmissionBasis(
        source_ref=request.declaration_ref,
        source_kind="UnhandledSuspension",
        source_generation=SourceGeneration(0),
        observed_frontier=ObservedFrontier(record_refs=state.transition_refs),
        source_path_ref=f"path:unhandled/{request.declaration_ref}/branch:root",
        input_value_or_digest=value,
        idempotency_key=f"idem-{request.source_key}",
        one_shot_key=OneShotKey(request.source_key),
        profile=CORE_REFERENCE_V0_LITE,
        program_ref=state.program_ref,
    )


def collect_batches_wire(fixture: Fixture) -> list[dict]:
    """Project a fixture's full transition sequence to canonical wire batches.

    Returns one wire batch per transition: the initial-run-prefix transition
    plus one resume transition per observation. Shared by the byte-stability
    test (compares against the committed `expected.batches`) and
    `regenerate.py` (rewrites it). Both must use this one function so
    generation and verification cannot drift.
    """

    prepared = admit_and_prepare(
        fixture.program,
        profile=CORE_REFERENCE_V0_LITE,
        registry=fixture.registry,
    )
    session, transition = KernelReplaySession.start(prepared, registry=fixture.registry)
    batches: list[dict] = [_project_transition_to_wire(transition, session, admission_basis=None)]

    for i, spec in enumerate(fixture.observations):
        if "reuse_index" in spec:
            raise ValueError(
                f"{fixture.case}: collect_batches_wire does not support reuse_index "
                f"specs — one-shot-reuse streams are negative fixtures, not positive corpus"
            )
        request = session.current_request()
        if request is None:
            raise ValueError(f"{fixture.case}: observations[{i}] has no open request to resume")
        basis = _lite_admission_basis(session.state, request, value=spec["value"])
        transition = session.resume(request, HostCompleted(value=spec["value"]), registry=fixture.registry)
        batches.append(_project_transition_to_wire(transition, session, admission_basis=basis))
    return batches


# --- Corpus-level invariants -------------------------------------------


def test_corpus_has_positive_and_negative_fixtures() -> None:
    assert _positive_fixtures(), "expected at least one positive fixture"
    assert _negative_fixtures(), "expected at least one negative fixture"


def test_every_fixture_has_unique_case_name() -> None:
    fixtures = _positive_fixtures() + _negative_fixtures()
    case_names = [f.case for f in fixtures]
    assert len(case_names) == len(set(case_names)), (
        f"duplicate case names: {[n for n in case_names if case_names.count(n) > 1]!r}"
    )


def test_every_negative_fixture_kind_is_a_negative_discriminator() -> None:
    for f in _negative_fixtures():
        assert f.kind.startswith("negative-"), f"{f.case}: negative fixture has non-negative kind {f.kind!r}"


# --- Positive fixtures: envelope-shape stability ------------------------


@pytest.mark.parametrize(
    "fixture",
    _positive_fixtures(),
    ids=lambda f: f.case,
)
def test_positive_fixture_runs_to_expected_envelope(fixture: Fixture) -> None:
    """Positive fixture: profile-admit + elaborate + run; envelope matches
    ``expected.envelope_status`` (and ``completed_value`` if applicable).

    Dispatches on ``fixture.observations``: empty → ``start_kernel_run``
    only (single-transition); non-empty → ``validate_observation_stream``
    (multi-step driver). Both paths verify the same envelope-shape
    assertions; only the entry point differs.
    """

    # Profile admission first
    validate_profile_admission(fixture.program, profile=CORE_REFERENCE_V0_LITE)

    if not fixture.observations:
        program = elaborate(fixture.program, registry=fixture.registry)
        _state, envelope = start_kernel_run(program, registry=fixture.registry)
    else:
        _state, observations = _build_observation_stream(fixture)
        program = elaborate(fixture.program, registry=fixture.registry)
        _state, envelope = validate_observation_stream(
            program,
            observations,
            registry=fixture.registry,
        )

    expected_status = fixture.expected["envelope_status"]
    assert envelope.status == expected_status, (
        f"{fixture.case}: expected status {expected_status!r}, got {envelope.status!r}"
    )

    if expected_status == "completed":
        expected_value = fixture.expected["completed_value"]
        assert isinstance(envelope.payload, CompletedResult)
        assert envelope.payload.value == expected_value, (
            f"{fixture.case}: expected value {expected_value!r}, got {envelope.payload.value!r}"
        )
    elif expected_status == "external-effect-request":
        expected_effect_kind = fixture.expected.get("open_request_effect_kind")
        assert isinstance(envelope.payload, ExternalEffectRequest)
        if expected_effect_kind is not None:
            assert envelope.payload.declaration.effect_kind == expected_effect_kind


@pytest.mark.parametrize(
    "fixture",
    _positive_fixtures(),
    ids=lambda f: f.case,
)
def test_positive_fixture_batches_are_byte_stable(fixture: Fixture) -> None:
    """The full transition-sequence batches must match the committed
    ``expected.batches`` byte-for-byte (after canonical encoding).

    This is the Python-side shape-lock and the Lean Phase 9 differential
    input: a projection/serializer change that alters the canonical bytes
    fails here, forcing a deliberate regenerate-and-review
    (``uv run python tests/conformance/v0_lite/regenerate.py``) rather than
    silent drift. Comparison is via ``canonical_json`` so the stored form
    may be pretty-printed for readability.
    """

    if "batches" not in fixture.expected:
        pytest.fail(
            f"{fixture.case}: missing expected.batches; run regenerate.py to "
            f"freeze the transition-sequence wire batches"
        )

    produced = collect_batches_wire(fixture)
    expected = fixture.expected["batches"]
    assert canonical_json(produced) == canonical_json(expected), (
        f"{fixture.case}: transition-sequence batches drifted from the committed "
        f"expected.batches. If this change is intentional, regenerate with "
        f"`uv run python tests/conformance/v0_lite/regenerate.py` and review the diff."
    )
    # Batch 0 is the initial run prefix; every batch carries the -lite profile.
    assert produced, f"{fixture.case}: expected at least one batch"
    assert produced[0].get("transition_kind") == "initial_run_prefix"
    assert len(produced) == 1 + len(fixture.observations)
    for batch in produced:
        assert batch.get("profile") == CORE_REFERENCE_V0_LITE.name


# --- Negative fixtures: profile-admission rejection ---------------------


@pytest.mark.parametrize(
    "fixture",
    [f for f in _negative_fixtures() if f.kind == "negative-profile-admission"],
    ids=lambda f: f.case,
)
def test_negative_profile_admission_fixture_rejects(fixture: Fixture) -> None:
    """profile-admission negative fixtures: validate_profile_admission must
    reject with the expected construct and substring."""

    expected = fixture.expected["rejection"]
    assert expected["rejection_kind"] == "profile-admission", (
        f"{fixture.case}: this loader path only handles profile-admission rejections"
    )

    with pytest.raises(ProfileAdmissionError) as exc:
        validate_profile_admission(fixture.program, profile=CORE_REFERENCE_V0_LITE)

    rejection = exc.value.rejection
    assert rejection.kind == "profile-admission"
    assert rejection.construct == expected["construct"], (
        f"{fixture.case}: expected construct {expected['construct']!r}, got {rejection.construct!r}"
    )
    assert expected["message_substring"] in rejection.diagnostic, (
        f"{fixture.case}: expected substring {expected['message_substring']!r} in diagnostic {rejection.diagnostic!r}"
    )
    if "source_location" in expected:
        assert rejection.source_location is not None
        expected_loc = expected["source_location"]
        if "construct_path" in expected_loc:
            assert rejection.source_location.construct_path == expected_loc["construct_path"]


# --- Negative fixtures: observation-admission rejection -----------------


@pytest.mark.parametrize(
    "fixture",
    [f for f in _negative_fixtures() if f.kind == "negative-observation-admission"],
    ids=lambda f: f.case,
)
def test_negative_observation_admission_fixture_rejects(fixture: Fixture) -> None:
    """observation-admission negative fixtures: profile-admit passes;
    validate_observation_stream rejects a specific observation in the
    stream with the expected rejection_index, rejection_class, and
    message substring."""

    expected = fixture.expected["rejection"]
    assert expected["rejection_kind"] == "observation-admission", (
        f"{fixture.case}: this loader path only handles observation-admission rejections"
    )

    # Profile admission must pass (this is a stream-level rejection,
    # not a source-construct rejection).
    validate_profile_admission(fixture.program, profile=CORE_REFERENCE_V0_LITE)

    _state, observations = _build_observation_stream(fixture)
    program = elaborate(fixture.program, registry=fixture.registry)
    _state, envelope = validate_observation_stream(
        program,
        observations,
        registry=fixture.registry,
    )

    assert envelope.status == "rejected", f"{fixture.case}: expected status='rejected', got {envelope.status!r}"
    assert isinstance(envelope.payload, KernelRejection)
    rejection = envelope.payload
    assert rejection.kind == "observation-admission", (
        f"{fixture.case}: expected kind='observation-admission', got {rejection.kind!r}"
    )
    assert rejection.rejection_index == expected["rejection_index"], (
        f"{fixture.case}: expected rejection_index={expected['rejection_index']}, got {rejection.rejection_index}"
    )
    assert rejection.rejection_class == expected["rejection_class"], (
        f"{fixture.case}: expected rejection_class={expected['rejection_class']!r}, got {rejection.rejection_class!r}"
    )
    assert expected["message_substring"] in rejection.diagnostic, (
        f"{fixture.case}: expected substring {expected['message_substring']!r} in diagnostic {rejection.diagnostic!r}"
    )
