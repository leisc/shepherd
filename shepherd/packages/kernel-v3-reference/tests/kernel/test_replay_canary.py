"""Runtime-shaped canaries for serialized continuation replay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest

from shepherd_kernel_v3_reference.kernel import elaborate, elaborate_publication_experimental
from shepherd_kernel_v3_reference.kernel import replay as replay_module
from shepherd_kernel_v3_reference.kernel.continuation_objects import ContinuationRoot
from shepherd_kernel_v3_reference.kernel.refs import content_ref
from shepherd_kernel_v3_reference.kernel.replay import (
    ContinuationReplayArtifact,
    ContinuationReplayError,
    ContinuationReplayLedger,
    ContinuationReplaySerializationError,
    ExternalEffectRequest,
    ExternalEffectRequestDescriptor,
    ExternalEffectRequestRef,
    HostCompleted,
    KernelReplayJournal,
    KernelReplayRejected,
    KernelReplaySession,
    KernelReplayState,
    OpenReplayRequest,
    ReplayableCompleted,
    ReplayableKernelTransition,
    ReplayableRejected,
    ReplayArtifactCatalog,
    continuation_replay_artifact_from_json,
    continuation_replay_artifact_from_objects,
    continuation_replay_artifact_to_json,
    external_effect_request_from_json,
    external_effect_request_to_json,
    host_completed_from_json,
    host_completed_to_json,
    kernel_replay_journal_current_request,
    kernel_replay_journal_current_request_descriptor,
    kernel_replay_journal_from_json,
    kernel_replay_journal_to_json,
    kernel_replay_state_from_journal,
    kernel_replay_state_from_json,
    kernel_replay_state_to_json,
    replayable_kernel_transition_from_json,
    replayable_kernel_transition_to_json,
    resume_continuation,
    resume_external_effect_request,
    resume_kernel_replay,
    resume_kernel_replay_from_journal,
    resume_replayable_kernel_transition,
    start_kernel_replay,
    start_replayable_kernel_run,
    start_replayable_kernel_transition,
)
from shepherd_kernel_v3_reference.schemas import AnySchema, TypeSchema, ValidationError
from shepherd_kernel_v3_reference.source.effects import EffectRegistry, EffectSignature
from shepherd_kernel_v3_reference.source.experimental import TerminalDelay, TerminalFork
from shepherd_kernel_v3_reference.source.handlers import HandlerEnv, StaticHandlerInstall
from shepherd_kernel_v3_reference.source.outcomes import Completed, ResumptionUsed, SourceOutcome
from shepherd_kernel_v3_reference.source.syntax import Handle, Let, Lit, Perform, Resume, Return, Var
from shepherd_kernel_v3_reference.trace.machine import TraceResult, run_trace
from shepherd_kernel_v3_reference.trace.records import ResumptionHandle
from shepherd_kernel_v3_reference.trace.serde import trace_record_to_json


@dataclass(frozen=True)
class ReplayCanaryMetrics:
    source_record_type: Literal["EffectDeclaration", "ResumptionHandle"]
    source_key: str
    effect_kind: str | None
    root_continuation_kind: str
    artifact_bytes: int
    continuation_object_count: int
    trace_record_count: int
    ledger_consumed_source_keys: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_record_type": self.source_record_type,
            "source_key": self.source_key,
            "effect_kind": self.effect_kind,
            "root_continuation_kind": self.root_continuation_kind,
            "artifact_bytes": self.artifact_bytes,
            "continuation_object_count": self.continuation_object_count,
            "trace_record_count": self.trace_record_count,
            "ledger_consumed_source_keys": list(self.ledger_consumed_source_keys),
        }


@dataclass(frozen=True)
class ReplayCanaryResult:
    outcome: SourceOutcome
    metrics: ReplayCanaryMetrics


class KernelReplayCanaryRuntime:
    """Small host-loop canary over the public serialized replay boundary."""

    def __init__(self) -> None:
        self.ledger = ContinuationReplayLedger()

    def run_effect_request(
        self,
        program: Any,
        effect_kind: str,
        host_value: Any,
        *,
        registry: EffectRegistry | None = None,
    ) -> ReplayCanaryResult:
        transition = start_replayable_kernel_transition(program, registry=registry)
        request = transition.payload
        assert isinstance(request, ExternalEffectRequest)
        assert request.effect_kind == effect_kind
        return self._round_trip_request_and_resume(
            program,
            request,
            host_value,
            registry=registry,
            parent_transition_refs=(transition.transition_id,),
        )

    def run_resumption_handle(
        self,
        program: Any,
        host_value: Any,
        *,
        registry: EffectRegistry | None = None,
    ) -> ReplayCanaryResult:
        result = run_trace(program, registry=registry, include_debug_evidence=True)
        handle = next(record for record in result.trace if isinstance(record, ResumptionHandle))
        artifact = _artifact_from_resumption_handle(result, handle)
        return self._round_trip_and_resume(
            program,
            artifact,
            host_value,
            registry=registry,
            trace_record_count=len(result.trace),
        )

    def _round_trip_request_and_resume(
        self,
        program: Any,
        request: ExternalEffectRequest,
        host_value: Any,
        *,
        registry: EffectRegistry | None,
        parent_transition_refs: tuple[str, ...],
    ) -> ReplayCanaryResult:
        encoded_request = external_effect_request_to_json(request)
        request_bytes = len(json.dumps(encoded_request, sort_keys=True))
        decoded_request = external_effect_request_from_json(json.loads(json.dumps(encoded_request)))
        encoded_observation = host_completed_to_json(HostCompleted(host_value))
        decoded_observation = host_completed_from_json(json.loads(json.dumps(encoded_observation)))

        result = resume_external_effect_request(
            program,
            decoded_request,
            decoded_observation,
            registry=registry,
            ledger=self.ledger,
            parent_transition_refs=parent_transition_refs,
        )
        root = decoded_request.replay_artifact.continuation_objects[decoded_request.replay_artifact.root_ref]
        assert isinstance(root, ContinuationRoot)
        assert decoded_request.replay_artifact.source_record_type is not None
        return ReplayCanaryResult(
            outcome=result.outcome,
            metrics=ReplayCanaryMetrics(
                source_record_type=decoded_request.replay_artifact.source_record_type,
                source_key=decoded_request.source_key,
                effect_kind=decoded_request.effect_kind,
                root_continuation_kind=root.continuation_kind,
                artifact_bytes=request_bytes,
                continuation_object_count=len(decoded_request.replay_artifact.continuation_objects),
                trace_record_count=len(decoded_request.trace_prefix),
                ledger_consumed_source_keys=self.ledger.consumed_source_keys,
            ),
        )

    def _round_trip_and_resume(
        self,
        program: Any,
        artifact: ContinuationReplayArtifact,
        host_value: Any,
        *,
        registry: EffectRegistry | None,
        trace_record_count: int,
    ) -> ReplayCanaryResult:
        encoded = continuation_replay_artifact_to_json(artifact)
        artifact_bytes = len(json.dumps(encoded, sort_keys=True))
        decoded = continuation_replay_artifact_from_json(json.loads(json.dumps(encoded)))
        root = decoded.continuation_objects[decoded.root_ref]
        assert isinstance(root, ContinuationRoot)
        assert decoded.source_key is not None
        assert decoded.source_record_type is not None

        outcome = resume_continuation(program, decoded, host_value, registry=registry, ledger=self.ledger)
        return ReplayCanaryResult(
            outcome=outcome,
            metrics=ReplayCanaryMetrics(
                source_record_type=decoded.source_record_type,
                source_key=decoded.source_key,
                effect_kind=decoded.effect_kind,
                root_continuation_kind=root.continuation_kind,
                artifact_bytes=artifact_bytes,
                continuation_object_count=len(decoded.continuation_objects),
                trace_record_count=trace_record_count,
                ledger_consumed_source_keys=self.ledger.consumed_source_keys,
            ),
        )


def test_replayable_kernel_run_completed_result() -> None:
    result = start_replayable_kernel_run(elaborate(Return(Lit("done"))))

    assert isinstance(result, ReplayableCompleted)
    assert result.outcome == Completed("done")
    assert result.value == "done"
    assert result.trace == ()


def test_replayable_kernel_transition_completed_result_round_trips() -> None:
    transition = start_replayable_kernel_transition(elaborate(Return(Lit("done"))))
    decoded = replayable_kernel_transition_from_json(
        json.loads(json.dumps(replayable_kernel_transition_to_json(transition)))
    )

    assert isinstance(decoded, ReplayableKernelTransition)
    assert decoded == transition
    assert decoded.status == "completed"
    assert isinstance(decoded.payload, ReplayableCompleted)
    assert decoded.payload.program_ref == decoded.program_ref
    assert decoded.payload.value == "done"
    assert decoded.trace_delta == ()


def test_compact_replay_transition_requires_journal_serialization() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, transition = KernelReplaySession.start(program)
    assert isinstance(transition.payload, ExternalEffectRequestRef)

    with pytest.raises(ContinuationReplaySerializationError, match="KernelReplayJournal"):
        replayable_kernel_transition_to_json(transition)

    encoded_journal = kernel_replay_journal_to_json(session.to_journal())
    assert encoded_journal["transitions"][0]["payload"]["payload_type"] == "ExternalEffectRequestRef"
    decoded_journal = kernel_replay_journal_from_json(json.loads(json.dumps(encoded_journal)))
    assert decoded_journal.transitions[0] == transition


def test_completed_replayable_kernel_transition_rejects_program_ref_tampering() -> None:
    transition = start_replayable_kernel_transition(elaborate(Return(Lit("done"))))
    assert transition.status == "completed"
    assert isinstance(transition.payload, ReplayableCompleted)
    encoded = _transition_json(transition)

    bad_payload = json.loads(json.dumps(encoded))
    bad_payload["payload"]["program_ref"] = "program:sha256:tampered"
    with pytest.raises(ContinuationReplayError, match="program_ref|transition_id"):
        replayable_kernel_transition_from_json(bad_payload)

    bad_top_level = json.loads(json.dumps(encoded))
    bad_top_level["program_ref"] = "program:sha256:tampered"
    bad_top_level["transition_id"] = _canonical_transition_ref_from_json(bad_top_level)
    with pytest.raises(ContinuationReplayError, match="program_ref"):
        replayable_kernel_transition_from_json(bad_top_level)


def test_replayable_kernel_transition_json_rejects_tampering() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    request_transition = start_replayable_kernel_transition(program)
    assert request_transition.status == "external-effect-request"
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    completed_transition = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted"),
        parent_transition_refs=(request_transition.transition_id,),
    )

    bad_status = _transition_json(request_transition)
    bad_status["status"] = "bogus"
    with pytest.raises(ContinuationReplayError, match="status"):
        replayable_kernel_transition_from_json(bad_status)

    wrong_payload_for_status = _transition_json(request_transition)
    wrong_payload_for_status["status"] = "completed"
    with pytest.raises(ContinuationReplayError, match="completed replay transition"):
        replayable_kernel_transition_from_json(wrong_payload_for_status)

    bad_id = _transition_json(request_transition)
    bad_id["transition_id"] = "kernel-replay-transition:sha256:tampered"
    with pytest.raises(ContinuationReplayError, match="transition_id"):
        replayable_kernel_transition_from_json(bad_id)

    bad_program = _transition_json(request_transition)
    bad_program["program_ref"] = "program:sha256:tampered"
    with pytest.raises(ContinuationReplayError, match="program_ref"):
        replayable_kernel_transition_from_json(bad_program)

    bad_trace_delta = _transition_json(request_transition)
    bad_trace_delta["trace_delta"] = []
    with pytest.raises(ContinuationReplayError, match="trace_delta"):
        replayable_kernel_transition_from_json(bad_trace_delta)

    bad_parent = _transition_json(completed_transition)
    bad_parent["parent_transition_refs"] = ["kernel-replay-transition:sha256:tampered"]
    with pytest.raises(ContinuationReplayError, match="transition_id"):
        replayable_kernel_transition_from_json(bad_parent)


def test_replay_canary_runtime_resumes_unhandled_provider_effect() -> None:
    program = elaborate(
        Let(
            "request_id",
            Return(Lit("req-1")),
            Let(
                "draft",
                Perform("provider.llm.generate", Lit({"prompt": "draft"})),
                Return(Var("draft")),
            ),
        )
    )

    result = KernelReplayCanaryRuntime().run_effect_request(
        program,
        "provider.llm.generate",
        {"text": "provider result"},
    )

    assert result.outcome == Completed({"text": "provider result"})
    assert result.metrics.source_record_type == "EffectDeclaration"
    assert result.metrics.effect_kind == "provider.llm.generate"
    assert result.metrics.root_continuation_kind != "empty-terminal"
    assert result.metrics.artifact_bytes > 0
    assert result.metrics.continuation_object_count > 1
    assert result.metrics.trace_record_count == 1
    assert result.metrics.ledger_consumed_source_keys == (result.metrics.source_key,)
    json.dumps(result.metrics.to_metadata())


def test_replay_canary_runtime_resumes_provider_effect_inside_handler_body() -> None:
    program = elaborate(
        Handle(
            Let("x", Perform("eff.worker", Lit({"input": "draft"})), Return(Var("x"))),
            HandlerEnv(
                (
                    StaticHandlerInstall(
                        effect_kind="eff.worker",
                        handler_id="replay-canary.handler.v1",
                        handled_result_schema=AnySchema(),
                        payload_name="_payload",
                        body=Let(
                            "approved",
                            Perform("provider.llm.generate", Lit({"prompt": "approve"})),
                            Resume(Var("approved")),
                        ),
                    ),
                )
            ),
        )
    )

    result = KernelReplayCanaryRuntime().run_effect_request(
        program,
        "provider.llm.generate",
        "accepted",
    )

    assert result.outcome == Completed("accepted")
    assert result.metrics.source_record_type == "EffectDeclaration"
    assert result.metrics.effect_kind == "provider.llm.generate"
    assert result.metrics.root_continuation_kind != "empty-terminal"
    assert result.metrics.trace_record_count > 1


def test_replayable_external_effect_resume_emits_trace_delta() -> None:
    program = elaborate(
        Handle(
            Let("x", Perform("eff.worker", Lit({"input": "draft"})), Return(Var("x"))),
            HandlerEnv(
                (
                    StaticHandlerInstall(
                        effect_kind="eff.worker",
                        handler_id="replay-canary-trace-delta.handler.v1",
                        handled_result_schema=AnySchema(),
                        payload_name="_payload",
                        body=Let(
                            "approved",
                            Perform("provider.llm.generate", Lit({"prompt": "approve"})),
                            Resume(Var("approved")),
                        ),
                    ),
                )
            ),
        )
    )
    request_transition = start_replayable_kernel_transition(program)
    assert request_transition.status == "external-effect-request"
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    completed_transition = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted"),
        parent_transition_refs=(request_transition.transition_id,),
    )

    assert completed_transition.status == "completed"
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "accepted"
    assert completed_transition.payload.trace == completed_transition.trace_delta
    assert completed_transition.trace_delta
    assert completed_transition.parent_transition_refs == (request_transition.transition_id,)
    assert completed_transition.resume_observation_ref is not None


def test_replayable_external_effect_resume_binds_host_observation() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    request_transition = start_replayable_kernel_transition(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    without_evidence = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted"),
        parent_transition_refs=(request_transition.transition_id,),
    )
    with_evidence = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted", evidence_refs=("host-proof:1",)),
        parent_transition_refs=(request_transition.transition_id,),
    )

    assert without_evidence.resume_observation_ref is not None
    assert with_evidence.resume_observation_ref is not None
    assert without_evidence.resume_observation_ref != with_evidence.resume_observation_ref
    assert without_evidence.transition_id != with_evidence.transition_id

    tampered_ref = _transition_json(without_evidence)
    tampered_ref["resume_observation_ref"] = with_evidence.resume_observation_ref
    with pytest.raises(ContinuationReplayError, match="transition_id"):
        replayable_kernel_transition_from_json(tampered_ref)

    missing_ref = _transition_json(without_evidence)
    missing_ref["resume_observation_ref"] = None
    missing_ref["transition_id"] = _canonical_transition_ref_from_json(missing_ref)
    with pytest.raises(ContinuationReplayError, match="resume_observation_ref"):
        replayable_kernel_transition_from_json(missing_ref)


def test_replay_canary_runtime_replays_resumption_handle() -> None:
    program = elaborate(
        Handle(
            Let("x", Perform("eff.worker", Lit({"input": "draft"})), Return(Var("x"))),
            HandlerEnv(
                (
                    StaticHandlerInstall(
                        effect_kind="eff.worker",
                        handler_id="replay-canary-resumption.handler.v1",
                        handled_result_schema=AnySchema(),
                        payload_name="_payload",
                        body=Let("r", Resume(Lit("worker-value")), Return(Var("r"))),
                    ),
                )
            ),
        )
    )

    result = KernelReplayCanaryRuntime().run_resumption_handle(program, "direct-worker-resume")

    assert result.outcome == Completed("direct-worker-resume")
    assert result.metrics.source_record_type == "ResumptionHandle"
    assert result.metrics.effect_kind is None
    assert result.metrics.root_continuation_kind != "empty-terminal"
    assert result.metrics.ledger_consumed_source_keys == (result.metrics.source_key,)


def test_replay_canary_runtime_bad_provider_value_consumes_ledger() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    request_transition = start_replayable_kernel_transition(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    ledger = ContinuationReplayLedger()

    rejected = resume_external_effect_request(
        program,
        request_transition.payload,
        HostCompleted("not-an-int"),
        registry=registry,
        ledger=ledger,
        parent_transition_refs=(request_transition.transition_id,),
    )

    assert isinstance(rejected, ReplayableRejected)
    assert rejected.reason_type == "ValidationError"
    assert "expected int" in rejected.reason_message
    consumed = ledger.consumed_source_keys
    assert len(consumed) == 1
    with pytest.raises(ResumptionUsed, match="already consumed"):
        resume_external_effect_request(
            program,
            request_transition.payload,
            HostCompleted(7),
            registry=registry,
            ledger=ledger,
            parent_transition_refs=(request_transition.transition_id,),
        )


def test_host_completed_rejects_non_json_values() -> None:
    HostCompleted({"items": [1, True, None, ("nested",)]})

    with pytest.raises(TypeError, match="non-JSON-compatible"):
        HostCompleted(object())
    with pytest.raises(TypeError, match="non-string mapping key"):
        HostCompleted({1: "bad"})
    with pytest.raises(TypeError, match="non-finite float"):
        HostCompleted(math.inf)


def test_external_effect_request_rejects_non_json_payload() -> None:
    request = start_replayable_kernel_run(
        elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    )
    assert isinstance(request, ExternalEffectRequest)
    bad_declaration = replace(request.declaration, payload=object())

    with pytest.raises(TypeError, match="non-JSON-compatible"):
        ExternalEffectRequest(declaration=bad_declaration, replay_artifact=request.replay_artifact)


def test_replay_canary_runtime_ledger_keys_do_not_collide_across_programs() -> None:
    first_program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "first"})), Return(Var("draft")))
    )
    second_program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "second"})), Return(Var("draft")))
    )
    first_request = start_replayable_kernel_run(first_program)
    second_request = start_replayable_kernel_run(second_program)
    assert isinstance(first_request, ExternalEffectRequest)
    assert isinstance(second_request, ExternalEffectRequest)
    assert first_request.declaration.ref == second_request.declaration.ref
    assert first_request.declaration.full_continuation_ref == second_request.declaration.full_continuation_ref
    assert first_request.source_key != second_request.source_key
    ledger = ContinuationReplayLedger()

    assert (
        resume_external_effect_request(first_program, first_request, HostCompleted("first"), ledger=ledger).value
        == "first"
    )
    assert (
        resume_external_effect_request(second_program, second_request, HostCompleted("second"), ledger=ledger).value
        == "second"
    )


def test_replayable_external_effect_request_json_rejects_malformed_inputs() -> None:
    request = start_replayable_kernel_run(
        elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    )
    assert isinstance(request, ExternalEffectRequest)

    bad_version = external_effect_request_to_json(request)
    bad_version["request_schema_version"] = "wrong"
    with pytest.raises(ContinuationReplayError, match="request_schema_version"):
        external_effect_request_from_json(bad_version)

    handle_program = elaborate(
        Handle(
            Let("x", Perform("eff.worker", Lit({"input": "draft"})), Return(Var("x"))),
            HandlerEnv(
                (
                    StaticHandlerInstall(
                        effect_kind="eff.worker",
                        handler_id="replay-canary-json.handler.v1",
                        handled_result_schema=AnySchema(),
                        payload_name="_payload",
                        body=Let("r", Resume(Lit("worker-value")), Return(Var("r"))),
                    ),
                )
            ),
        )
    )
    handle_trace = run_trace(handle_program, include_debug_evidence=True)
    handle = next(record for record in handle_trace.trace if isinstance(record, ResumptionHandle))
    wrong_declaration = external_effect_request_to_json(request)
    wrong_declaration["declaration"] = trace_record_to_json(handle)
    with pytest.raises(ContinuationReplaySerializationError, match="EffectDeclaration"):
        external_effect_request_from_json(wrong_declaration)

    bad_artifact = replace(request.replay_artifact, effect_kind="provider.llm.other", source_key=None)
    with pytest.raises(ContinuationReplayError, match="effect_kind"):
        ExternalEffectRequest(declaration=request.declaration, replay_artifact=bad_artifact)


def test_replayable_external_effect_resume_can_emit_next_request() -> None:
    program = elaborate(
        Let(
            "first",
            Perform("provider.llm.generate", Lit({"prompt": "first"})),
            Let(
                "second",
                Perform("provider.llm.generate", Lit({"prompt": "second"})),
                Return(Var("second")),
            ),
        )
    )
    first_transition = start_replayable_kernel_transition(program)
    assert first_transition.status == "external-effect-request"
    assert isinstance(first_transition.payload, ExternalEffectRequest)

    second_transition = resume_replayable_kernel_transition(
        program,
        first_transition.payload,
        HostCompleted("first-result"),
        parent_transition_refs=(first_transition.transition_id,),
    )
    assert second_transition.status == "external-effect-request"
    assert isinstance(second_transition.payload, ExternalEffectRequest)
    assert second_transition.payload.payload == {"prompt": "second"}
    assert second_transition.trace_delta
    assert second_transition.resume_observation_ref is not None

    completed_transition = resume_replayable_kernel_transition(
        program,
        second_transition.payload,
        HostCompleted("second-result"),
        parent_transition_refs=(second_transition.transition_id,),
    )
    assert completed_transition.status == "completed"
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "second-result"
    assert completed_transition.resume_observation_ref is not None


def test_kernel_replay_state_resumes_one_external_request_to_completion() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))

    state, request_transition = start_kernel_replay(program)
    assert isinstance(state, KernelReplayState)
    assert state.terminal is False
    assert state.rejected is False
    assert state.transition_refs == (request_transition.transition_id,)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    assert state.open_source_keys == (request_transition.payload.source_key,)
    assert isinstance(state.open_requests[request_transition.payload.source_key], OpenReplayRequest)

    completed_state, completed_transition = resume_kernel_replay(
        state,
        request_transition.payload,
        HostCompleted("accepted"),
    )

    assert completed_state.terminal is True
    assert completed_state.open_source_keys == ()
    assert completed_state.consumed_source_keys == (request_transition.payload.source_key,)
    assert completed_state.transition_refs == (request_transition.transition_id, completed_transition.transition_id)
    assert completed_state.trace == request_transition.trace_delta + completed_transition.trace_delta
    assert completed_transition.parent_transition_refs == (request_transition.transition_id,)
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "accepted"


def test_kernel_replay_state_advances_across_sequential_external_requests() -> None:
    program = elaborate(
        Let(
            "first",
            Perform("provider.llm.generate", Lit({"prompt": "first"})),
            Let(
                "second",
                Perform("provider.llm.generate", Lit({"prompt": "second"})),
                Return(Var("second")),
            ),
        )
    )

    state, first_transition = start_kernel_replay(program)
    assert isinstance(first_transition.payload, ExternalEffectRequest)
    first_source_key = first_transition.payload.source_key

    state, second_transition = resume_kernel_replay(state, first_transition.payload, HostCompleted("first-result"))
    assert isinstance(second_transition.payload, ExternalEffectRequest)
    assert state.open_source_keys == (second_transition.payload.source_key,)
    assert state.consumed_source_keys == (first_source_key,)
    assert second_transition.parent_transition_refs == (first_transition.transition_id,)

    state, completed_transition = resume_kernel_replay(state, second_transition.payload, HostCompleted("second-result"))
    assert state.terminal is True
    assert state.open_source_keys == ()
    assert state.consumed_source_keys == (first_source_key, second_transition.payload.source_key)
    assert state.transition_refs == (
        first_transition.transition_id,
        second_transition.transition_id,
        completed_transition.transition_id,
    )
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "second-result"


def test_kernel_replay_session_direct_state_construction_is_not_supported() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))

    state, _request_transition = start_kernel_replay(program)

    with pytest.raises(ContinuationReplayError, match="requires live evaluator state"):
        KernelReplaySession(state)


def test_kernel_replay_session_partial_live_construction_is_not_supported() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))

    state, request_transition = start_kernel_replay(program)

    with pytest.raises(ContinuationReplayError, match="requires live evaluator state"):
        KernelReplaySession(state, transition=request_transition, transitions=(request_transition,))


def test_kernel_replay_session_rejects_non_current_resume_request() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    other_program = elaborate(
        Let("other", Perform("provider.llm.generate", Lit({"prompt": "other"})), Return(Var("other")))
    )

    session, _request_transition = KernelReplaySession.start(program)
    _other_state, other_transition = start_kernel_replay(other_program)
    assert isinstance(other_transition.payload, ExternalEffectRequest)
    state_before = session.state
    transitions_before = session.transitions

    with pytest.raises(ContinuationReplayError, match="does not match current live request"):
        session.resume(other_transition.payload, HostCompleted("accepted"))

    assert session.state is state_before
    assert session.transitions == transitions_before


def test_kernel_replay_journal_round_trips_and_derives_state() -> None:
    program = elaborate(
        Let(
            "first",
            Perform("provider.llm.generate", Lit({"prompt": "first"})),
            Let(
                "second",
                Perform("provider.llm.generate", Lit({"prompt": "second"})),
                Return(Var("second")),
            ),
        )
    )

    state, first_transition = start_kernel_replay(program)
    assert isinstance(first_transition.payload, ExternalEffectRequest)
    state, second_transition = resume_kernel_replay(state, first_transition.payload, HostCompleted("first-result"))
    assert isinstance(second_transition.payload, ExternalEffectRequest)
    state, completed_transition = resume_kernel_replay(
        state,
        second_transition.payload,
        HostCompleted("second-result"),
    )

    journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(first_transition, second_transition, completed_transition),
    )
    decoded_journal = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(journal))))
    derived_state = kernel_replay_state_from_journal(program, decoded_journal)

    assert derived_state.terminal is True
    assert derived_state.rejected is False
    assert derived_state.open_source_keys == ()
    assert derived_state.consumed_source_keys == state.consumed_source_keys
    assert derived_state.transition_refs == state.transition_refs
    assert derived_state.trace == state.trace


def test_kernel_replay_journal_serializes_external_requests_through_catalog_refs() -> None:
    program = elaborate(
        Let(
            "first",
            Perform("provider.llm.generate", Lit({"prompt": "first"})),
            Let(
                "second",
                Perform("provider.llm.generate", Lit({"prompt": "second"})),
                Return(Var("second")),
            ),
        )
    )
    state, first_transition = start_kernel_replay(program)
    assert isinstance(first_transition.payload, ExternalEffectRequest)
    state, second_transition = resume_kernel_replay(state, first_transition.payload, HostCompleted("first-result"))
    assert isinstance(second_transition.payload, ExternalEffectRequest)
    state, completed_transition = resume_kernel_replay(
        state,
        second_transition.payload,
        HostCompleted("second-result"),
    )

    journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(first_transition, second_transition, completed_transition),
    )
    encoded = kernel_replay_journal_to_json(journal)

    assert encoded["continuation_objects"]
    assert encoded["artifacts"]
    request_payloads = [
        transition["payload"]
        for transition in encoded["transitions"]
        if transition["payload"]["payload_type"] == "ExternalEffectRequestRef"
    ]
    assert len(request_payloads) == 2
    for payload in request_payloads:
        assert "request_ref" in payload
        assert "artifact_ref" in payload
        assert "replay_artifact" not in payload
        assert "continuation_objects" not in json.dumps(payload)

    decoded = kernel_replay_journal_from_json(json.loads(json.dumps(encoded)))
    assert isinstance(decoded.transitions[0].payload, ExternalEffectRequestRef)
    assert isinstance(decoded.transitions[1].payload, ExternalEffectRequestRef)
    assert kernel_replay_state_from_journal(program, decoded).transition_refs == state.transition_refs


def test_kernel_replay_journal_materializes_current_request_after_round_trip() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    journal = KernelReplayJournal(program_ref=state.program_ref, transitions=(request_transition,))
    decoded = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(journal))))
    assert isinstance(decoded.transitions[0].payload, ExternalEffectRequestRef)

    request = kernel_replay_journal_current_request(decoded)

    assert isinstance(request, ExternalEffectRequest)
    assert request.source_key == request_transition.payload.source_key
    assert request.payload == {"prompt": "draft"}
    completed_state, completed_transition = resume_kernel_replay_from_journal(
        program,
        decoded,
        HostCompleted("accepted"),
    )
    assert completed_state.terminal is True
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "accepted"


def test_kernel_replay_journal_rejects_unclosed_compact_session_transitions() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)

    with pytest.raises(ContinuationReplayError, match="missing root object"):
        KernelReplayJournal(program_ref=session.state.program_ref, transitions=session.transitions)

    closed_journal = session.to_journal()
    decoded = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(closed_journal))))

    assert isinstance(decoded.transitions[0].payload, ExternalEffectRequestRef)
    assert kernel_replay_journal_current_request(decoded) is not None


def test_external_effect_request_descriptors_match_full_and_compact_requests() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", TypeSchema(dict), AnySchema()))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )

    state, full_transition = start_kernel_replay(program, registry=registry)
    full_request = full_transition.payload
    assert isinstance(full_request, ExternalEffectRequest)
    session, compact_transition = KernelReplaySession.start(program, registry=registry)
    compact_request = compact_transition.payload
    assert isinstance(compact_request, ExternalEffectRequestRef)

    assert full_request.payload_schema_ref is not None
    assert compact_request.payload_schema_ref == full_request.payload_schema_ref
    assert full_request.root_ref == compact_request.root_ref
    assert state.program_ref == session.state.program_ref
    assert isinstance(full_request.descriptor, ExternalEffectRequestDescriptor)
    assert compact_request.descriptor.payload == full_request.descriptor.payload
    assert compact_request.descriptor.payload_schema_ref == full_request.descriptor.payload_schema_ref
    assert compact_request.descriptor.root_ref == full_request.descriptor.root_ref
    assert compact_request.descriptor.replay_artifact_ref == full_request.descriptor.replay_artifact_ref
    assert session.current_request_descriptor() == compact_request.descriptor

    journal = KernelReplayJournal(program_ref=state.program_ref, transitions=(full_transition,))
    decoded = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(journal))))
    decoded_request = decoded.transitions[0].payload
    assert isinstance(decoded_request, ExternalEffectRequestRef)
    assert decoded_request.descriptor == full_request.descriptor


def test_kernel_replay_journal_descriptor_does_not_materialize_executable_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)
    journal = session.to_journal()

    def fail_materialization(*_args: object, **_kwargs: object) -> ContinuationReplayArtifact:
        raise AssertionError("descriptor lookup should not materialize replay artifacts")

    monkeypatch.setattr(replay_module, "continuation_replay_artifact_from_record", fail_materialization)

    descriptor = kernel_replay_journal_current_request_descriptor(journal)

    assert descriptor == request_transition.payload.descriptor
    assert descriptor is not None
    assert descriptor.payload == {"prompt": "draft"}


def test_external_effect_request_descriptor_payload_is_not_live_request_payload() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, _request_transition = KernelReplaySession.start(program)
    descriptor = session.current_request_descriptor()
    assert descriptor is not None

    descriptor.payload["prompt"] = "mutated"
    request = session.current_request()

    assert request is not None
    assert request.payload == {"prompt": "draft"}
    transition = session.resume_current(HostCompleted("accepted"))
    assert isinstance(transition.payload, ReplayableCompleted)
    assert transition.payload.value == "accepted"


def test_replay_artifact_catalog_materializes_closed_compact_request() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)
    journal = session.to_journal()

    request = journal.catalog.materialize(request_transition.payload)

    assert isinstance(journal.catalog, ReplayArtifactCatalog)
    assert isinstance(request, ExternalEffectRequest)
    assert request.descriptor == request_transition.payload.descriptor


def test_kernel_replay_journal_reuses_validated_catalog() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, _request_transition = KernelReplaySession.start(program)
    journal = session.to_journal()

    assert journal.catalog is journal.catalog


def test_kernel_replay_journal_current_request_materializes_request_local_closure() -> None:
    body: Any = Return(Lit("done"))
    for index in reversed(range(5)):
        body = Let(f"external{index}", Perform("provider.llm.generate", Lit({"i": index})), body)
    session, transition = KernelReplaySession.start(elaborate(body))

    for index in range(4):
        assert isinstance(transition.payload, ExternalEffectRequestRef)
        transition = session.resume_current(HostCompleted(f"value:{index}"))

    journal = session.to_journal()
    request = kernel_replay_journal_current_request(journal)

    assert isinstance(request, ExternalEffectRequest)
    assert len(request.replay_artifact.continuation_objects) < len(journal.continuation_objects)
    assert request.descriptor == transition.payload.descriptor


def test_kernel_replay_journal_rejects_missing_non_root_closure_object() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)
    journal = session.to_journal()
    continuation_objects = dict(journal.continuation_objects)
    missing_ref = next(ref for ref in continuation_objects if ref != request_transition.payload.root_ref)
    del continuation_objects[missing_ref]

    with pytest.raises(ContinuationReplayError, match="missing continuation object"):
        KernelReplayJournal(
            program_ref=journal.program_ref,
            transitions=journal.transitions,
            continuation_objects=continuation_objects,
            artifacts=journal.artifacts,
        )


def test_kernel_replay_journal_json_rejects_missing_non_root_closure_object() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)
    journal = session.to_journal()
    encoded = kernel_replay_journal_to_json(journal)
    missing_ref = next(ref for ref in journal.continuation_objects if ref != request_transition.payload.root_ref)
    encoded["continuation_objects"] = [
        entry for entry in encoded["continuation_objects"] if entry["ref"] != missing_ref
    ]

    with pytest.raises(ContinuationReplayError, match="missing continuation object"):
        kernel_replay_journal_from_json(json.loads(json.dumps(encoded)))


def test_kernel_replay_direct_resume_rejects_compact_request_without_consumption() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    journal = KernelReplayJournal(program_ref=state.program_ref, transitions=(request_transition,))
    decoded = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(journal))))
    compact_request = decoded.transitions[0].payload
    assert isinstance(compact_request, ExternalEffectRequestRef)
    decoded_state = kernel_replay_state_from_journal(program, decoded)

    with pytest.raises(ContinuationReplayError, match="executable ExternalEffectRequest"):
        resume_kernel_replay(decoded_state, compact_request, HostCompleted("accepted"))  # type: ignore[arg-type]

    assert decoded_state.consumed_source_keys == ()
    assert decoded_state.open_source_keys == (request_transition.payload.source_key,)


def test_kernel_replay_journal_current_request_returns_none_for_terminal_journal() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    state, completed_transition = resume_kernel_replay(state, request_transition.payload, HostCompleted("accepted"))
    journal = KernelReplayJournal(program_ref=state.program_ref, transitions=(request_transition, completed_transition))

    assert kernel_replay_journal_current_request(journal) is None
    with pytest.raises(ContinuationReplayError, match="no open external effect request"):
        resume_kernel_replay_from_journal(program, journal, HostCompleted("again"))


def test_kernel_replay_journal_is_trusted_local_not_semantic_admission() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    _state, completed_transition = resume_kernel_replay(state, request_transition.payload, HostCompleted("real"))

    forged_completed_json = replayable_kernel_transition_to_json(completed_transition)
    forged_completed_json["payload"]["value"] = "forged"
    forged_completed_json["resume_observation_ref"] = "host-completed:sha256:forged"
    forged_completed_json["transition_id"] = _canonical_transition_ref_from_json(forged_completed_json)
    forged_completed_transition = replayable_kernel_transition_from_json(forged_completed_json)

    trusted_local_journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(request_transition, forged_completed_transition),
    )
    derived_state = kernel_replay_state_from_journal(program, trusted_local_journal)

    assert derived_state.terminal is True
    assert derived_state.transition_refs == (
        request_transition.transition_id,
        forged_completed_transition.transition_id,
    )


def test_kernel_replay_journal_rejects_non_sequential_transitions() -> None:
    program = elaborate(
        Let(
            "first",
            Perform("provider.llm.generate", Lit({"prompt": "first"})),
            Let(
                "second",
                Perform("provider.llm.generate", Lit({"prompt": "second"})),
                Return(Var("second")),
            ),
        )
    )

    state, first_transition = start_kernel_replay(program)
    assert isinstance(first_transition.payload, ExternalEffectRequest)
    state, second_transition = resume_kernel_replay(state, first_transition.payload, HostCompleted("first-result"))
    assert isinstance(second_transition.payload, ExternalEffectRequest)
    state, completed_transition = resume_kernel_replay(
        state,
        second_transition.payload,
        HostCompleted("second-result"),
    )

    with pytest.raises(ContinuationReplayError, match="unique"):
        KernelReplayJournal(program_ref=state.program_ref, transitions=(first_transition, first_transition))

    bad_completed_json = replayable_kernel_transition_to_json(completed_transition)
    bad_completed_json["parent_transition_refs"] = [first_transition.transition_id]
    bad_completed_json["transition_id"] = _canonical_transition_ref_from_json(bad_completed_json)
    bad_completed_transition = replayable_kernel_transition_from_json(bad_completed_json)
    with pytest.raises(ContinuationReplayError, match="previous frontier"):
        KernelReplayJournal(
            program_ref=state.program_ref,
            transitions=(first_transition, second_transition, bad_completed_transition),
        )


def test_kernel_replay_state_json_round_trips_runtime_states() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    state, request_transition = start_kernel_replay(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    decoded_open = kernel_replay_state_from_json(
        program,
        json.loads(json.dumps(kernel_replay_state_to_json(state))),
    )
    assert decoded_open.program_ref == state.program_ref
    assert decoded_open.open_source_keys == state.open_source_keys
    assert decoded_open.transition_refs == state.transition_refs
    assert decoded_open.trace == state.trace

    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("not-an-int"), registry=registry)
    decoded_rejected = kernel_replay_state_from_json(
        program,
        json.loads(json.dumps(kernel_replay_state_to_json(rejected.value.state))),
    )
    assert decoded_rejected.rejected is True
    assert decoded_rejected.consumed_source_keys == (request_transition.payload.source_key,)

    completed_state, _completed_transition = resume_kernel_replay(
        state,
        request_transition.payload,
        HostCompleted(7),
        registry=registry,
    )
    decoded_terminal = kernel_replay_state_from_json(
        program,
        json.loads(json.dumps(kernel_replay_state_to_json(completed_state))),
    )
    assert decoded_terminal.terminal is True
    assert decoded_terminal.open_source_keys == ()
    assert decoded_terminal.consumed_source_keys == (request_transition.payload.source_key,)


def test_kernel_replay_state_json_process_style_resume() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    decoded_state = kernel_replay_state_from_json(
        program,
        json.loads(json.dumps(kernel_replay_state_to_json(state))),
    )
    decoded_request = external_effect_request_from_json(
        json.loads(json.dumps(external_effect_request_to_json(request_transition.payload)))
    )

    completed_state, completed_transition = resume_kernel_replay(
        decoded_state,
        decoded_request,
        HostCompleted("accepted"),
    )

    assert completed_state.terminal is True
    assert isinstance(completed_transition.payload, ReplayableCompleted)
    assert completed_transition.payload.value == "accepted"
    assert completed_transition.parent_transition_refs == (request_transition.transition_id,)


def test_kernel_replay_state_trace_is_diagnostic_unless_journal_reconstructed() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    tampered_state_json = json.loads(json.dumps(kernel_replay_state_to_json(state)))
    tampered_state_json["trace"] = []
    decoded_state = kernel_replay_state_from_json(program, tampered_state_json)

    completed_state, completed_transition = resume_kernel_replay(
        decoded_state,
        request_transition.payload,
        HostCompleted("accepted"),
    )
    assert completed_state.terminal is True
    assert completed_state.trace == completed_transition.trace_delta
    assert completed_transition.parent_transition_refs == (request_transition.transition_id,)

    journal_state = kernel_replay_state_from_journal(
        program,
        KernelReplayJournal(
            program_ref=state.program_ref,
            transitions=(request_transition, completed_transition),
        ),
    )
    assert journal_state.trace == request_transition.trace_delta + completed_transition.trace_delta


def test_kernel_replay_state_json_rejects_malformed_state() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    other_program = elaborate(Return(Lit("other")))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    encoded = kernel_replay_state_to_json(state)

    bad_version = json.loads(json.dumps(encoded))
    bad_version["state_schema_version"] = "wrong"
    with pytest.raises(ContinuationReplayError, match="state_schema_version"):
        kernel_replay_state_from_json(program, bad_version)

    with pytest.raises(ContinuationReplayError, match="prepared program"):
        kernel_replay_state_from_json(other_program, encoded)

    consumed_overlap = json.loads(json.dumps(encoded))
    consumed_overlap["consumed_source_keys"] = [request_transition.payload.source_key]
    with pytest.raises(ContinuationReplayError, match="both open and consumed"):
        kernel_replay_state_from_json(program, consumed_overlap)

    terminal_with_open = json.loads(json.dumps(encoded))
    terminal_with_open["terminal"] = True
    with pytest.raises(ContinuationReplayError, match="terminal or rejected"):
        kernel_replay_state_from_json(program, terminal_with_open)

    missing_transition = json.loads(json.dumps(encoded))
    missing_transition["transition_refs"] = []
    with pytest.raises(ContinuationReplayError, match="transition_refs"):
        kernel_replay_state_from_json(program, missing_transition)

    trailing_transition = json.loads(json.dumps(encoded))
    trailing_transition["transition_refs"].append("kernel-replay-transition:sha256:tampered")
    with pytest.raises(ContinuationReplayError, match="current frontier"):
        kernel_replay_state_from_json(program, trailing_transition)

    duplicate_transition = json.loads(json.dumps(encoded))
    duplicate_transition["transition_refs"].append(request_transition.transition_id)
    with pytest.raises(ContinuationReplayError, match="unique"):
        kernel_replay_state_from_json(program, duplicate_transition)

    second_open_request = json.loads(json.dumps(encoded))
    extra_open_request = json.loads(json.dumps(second_open_request["open_requests"][0]))
    extra_open_request["source_key"] = "source:extra"
    second_open_request["open_requests"].append(extra_open_request)
    with pytest.raises(ContinuationReplayError, match="exactly one open request"):
        kernel_replay_state_from_json(program, second_open_request)

    bad_request_ref = json.loads(json.dumps(encoded))
    bad_request_ref["open_requests"][0]["request_ref"] = "external-effect-request:sha256:tampered"
    decoded_bad_request_ref = kernel_replay_state_from_json(program, bad_request_ref)
    with pytest.raises(ContinuationReplayError, match="does not match open request"):
        resume_kernel_replay(decoded_bad_request_ref, request_transition.payload, HostCompleted("accepted"))


def test_kernel_replay_state_rejects_invalid_resume_order() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    other_program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "other"})), Return(Var("draft")))
    )
    state, request_transition = start_kernel_replay(program)
    _other_state, other_transition = start_kernel_replay(other_program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    assert isinstance(other_transition.payload, ExternalEffectRequest)

    with pytest.raises(ContinuationReplayError, match="not open"):
        resume_kernel_replay(state, other_transition.payload, HostCompleted("wrong-request"))

    completed_state, _completed_transition = resume_kernel_replay(
        state,
        request_transition.payload,
        HostCompleted("accepted"),
    )
    with pytest.raises(ContinuationReplayError, match="terminal"):
        resume_kernel_replay(completed_state, request_transition.payload, HostCompleted("again"))
    reopened_state = replace(
        completed_state,
        terminal=False,
    )
    with pytest.raises(ResumptionUsed, match="already consumed"):
        resume_kernel_replay(reopened_state, request_transition.payload, HostCompleted("again"))


def test_kernel_replay_state_rejects_tampered_open_request_before_consumption() -> None:
    program = elaborate(Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))))
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    request = request_transition.payload

    bad_payload_request = ExternalEffectRequest(
        declaration=replace(request.declaration, payload={"prompt": "tampered"}),
        replay_artifact=request.replay_artifact,
        trace_prefix=request.trace_prefix,
    )
    with pytest.raises(ContinuationReplayError, match="does not match open request"):
        resume_kernel_replay(state, bad_payload_request, HostCompleted("accepted"))
    assert state.consumed_source_keys == ()

    bad_trace_request = ExternalEffectRequest(
        declaration=request.declaration,
        replay_artifact=request.replay_artifact,
        trace_prefix=(),
    )
    with pytest.raises(ContinuationReplayError, match="does not match open request"):
        resume_kernel_replay(state, bad_trace_request, HostCompleted("accepted"))
    assert state.consumed_source_keys == ()


def test_kernel_replay_state_schema_rejection_consumes_source() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    state, request_transition = start_kernel_replay(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("not-an-int"), registry=registry)

    rejected_state = rejected.value.state
    rejected_transition = rejected.value.transition
    assert isinstance(rejected.value.reason, ValidationError)
    assert rejected_transition.status == "rejected"
    assert rejected_transition.parent_transition_refs == (request_transition.transition_id,)
    assert rejected_transition.resume_observation_ref is not None
    assert isinstance(rejected_transition.payload, ReplayableRejected)
    assert rejected_transition.payload.source_key == request_transition.payload.source_key
    assert (
        rejected_transition.payload.request_ref
        == state.open_requests[request_transition.payload.source_key].request_ref
    )
    assert rejected_transition.payload.reason_type == "ValidationError"
    assert rejected_state.rejected is True
    assert rejected_state.terminal is False
    assert rejected_state.open_source_keys == ()
    assert rejected_state.consumed_source_keys == (request_transition.payload.source_key,)
    assert rejected_state.transition_refs == (request_transition.transition_id, rejected_transition.transition_id)

    with pytest.raises(ContinuationReplayError, match="rejected"):
        resume_kernel_replay(rejected_state, request_transition.payload, HostCompleted(7), registry=registry)


def test_direct_replay_transition_schema_rejection_returns_rejected_transition() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    state, request_transition = start_kernel_replay(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    direct_transition = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("not-an-int"),
        registry=registry,
        parent_transition_refs=(request_transition.transition_id,),
    )
    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("not-an-int"), registry=registry)

    assert direct_transition == rejected.value.transition
    assert direct_transition.status == "rejected"
    assert isinstance(direct_transition.payload, ReplayableRejected)
    assert direct_transition.payload.reason_type == "ValidationError"
    assert "expected int" in direct_transition.payload.reason_message


def test_direct_kernel_replay_rejection_preserves_emitted_trace_delta() -> None:
    program = elaborate(
        Let(
            "draft",
            Perform("provider.llm.generate", Lit({"prompt": "draft"})),
            Handle(
                Perform("eff.a", Lit("payload")),
                HandlerEnv(
                    (
                        StaticHandlerInstall(
                            effect_kind="eff.a",
                            handler_id="direct-replay-bad-answer-schema",
                            handled_result_schema=TypeSchema(int),
                            payload_name="_payload",
                            body=Return(Lit("not-an-int")),
                        ),
                    )
                ),
            ),
        )
    )
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("accepted"))

    rejected_state = rejected.value.state
    rejected_transition = rejected.value.transition
    direct_transition = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted"),
        parent_transition_refs=(request_transition.transition_id,),
    )
    assert isinstance(rejected_transition.payload, ReplayableRejected)
    assert direct_transition == rejected_transition
    assert isinstance(rejected.value.reason, ValidationError)
    assert tuple(type(record).__name__ for record in rejected_transition.trace_delta) == (
        "EffectDeclaration",
        "HandlerSelection",
        "ResumptionHandle",
    )
    assert rejected_transition.payload.trace == rejected_transition.trace_delta
    assert rejected_state.trace == request_transition.trace_delta + rejected_transition.trace_delta

    journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(request_transition, rejected_transition),
    )
    derived_state = kernel_replay_state_from_journal(program, journal)
    assert derived_state.rejected is True
    assert derived_state.trace == rejected_state.trace


def test_kernel_replay_rejected_transition_round_trips_through_journal() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    state, request_transition = start_kernel_replay(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("not-an-int"), registry=registry)
    rejected_transition = rejected.value.transition

    decoded_transition = replayable_kernel_transition_from_json(
        json.loads(json.dumps(replayable_kernel_transition_to_json(rejected_transition)))
    )
    assert decoded_transition == rejected_transition
    assert isinstance(decoded_transition.payload, ReplayableRejected)
    assert decoded_transition.payload.reason_type == "ValidationError"

    journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(request_transition, rejected_transition),
    )
    decoded_journal = kernel_replay_journal_from_json(json.loads(json.dumps(kernel_replay_journal_to_json(journal))))
    derived_state = kernel_replay_state_from_journal(program, decoded_journal)

    assert derived_state.rejected is True
    assert derived_state.terminal is False
    assert derived_state.open_source_keys == ()
    assert derived_state.consumed_source_keys == (request_transition.payload.source_key,)
    assert derived_state.transition_refs == (request_transition.transition_id, rejected_transition.transition_id)
    assert kernel_replay_journal_current_request(decoded_journal) is None
    assert kernel_replay_journal_current_request_descriptor(decoded_journal) is None
    with pytest.raises(ContinuationReplayError, match="no open external effect request"):
        resume_kernel_replay_from_journal(program, decoded_journal, HostCompleted(7), registry=registry)


def test_rejected_replayable_kernel_transition_json_rejects_tampering() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    state, request_transition = start_kernel_replay(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequest)
    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("not-an-int"), registry=registry)
    rejected_transition = rejected.value.transition

    bad_id = _transition_json(rejected_transition)
    bad_id["transition_id"] = "kernel-replay-transition:sha256:tampered"
    with pytest.raises(ContinuationReplayError, match="transition_id"):
        replayable_kernel_transition_from_json(bad_id)

    missing_parent = _transition_json(rejected_transition)
    missing_parent["parent_transition_refs"] = []
    missing_parent["transition_id"] = _canonical_transition_ref_from_json(missing_parent)
    with pytest.raises(ContinuationReplayError, match="parent transition"):
        replayable_kernel_transition_from_json(missing_parent)

    multiple_parents = _transition_json(rejected_transition)
    multiple_parents["parent_transition_refs"] = [
        request_transition.transition_id,
        "kernel-replay-transition:sha256:extra",
    ]
    multiple_parents["transition_id"] = _canonical_transition_ref_from_json(multiple_parents)
    with pytest.raises(ContinuationReplayError, match="exactly one parent transition"):
        replayable_kernel_transition_from_json(multiple_parents)

    missing_observation = _transition_json(rejected_transition)
    missing_observation["resume_observation_ref"] = None
    missing_observation["transition_id"] = _canonical_transition_ref_from_json(missing_observation)
    with pytest.raises(ContinuationReplayError, match="resume_observation_ref"):
        replayable_kernel_transition_from_json(missing_observation)

    bad_request_ref = _transition_json(rejected_transition)
    bad_request_ref["payload"]["request_ref"] = "external-effect-request:sha256:tampered"
    bad_request_ref["transition_id"] = _canonical_transition_ref_from_json(bad_request_ref)
    bad_transition = replayable_kernel_transition_from_json(bad_request_ref)
    with pytest.raises(ContinuationReplayError, match="request_ref"):
        KernelReplayJournal(program_ref=state.program_ref, transitions=(request_transition, bad_transition))


def test_kernel_replay_session_persists_rejected_state() -> None:
    registry = EffectRegistry()
    registry.register(EffectSignature("provider.llm.generate", AnySchema(), TypeSchema(int)))
    program = elaborate(
        Let("draft", Perform("provider.llm.generate", Lit({"prompt": "draft"})), Return(Var("draft"))),
        registry=registry,
    )
    session, request_transition = KernelReplaySession.start(program, registry=registry)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)

    with pytest.raises(KernelReplayRejected):
        session.resume(request_transition.payload, HostCompleted("not-an-int"), registry=registry)

    assert session.state.rejected is True
    assert session.state.consumed_source_keys == (request_transition.payload.source_key,)
    assert len(session.transitions) == 2
    rejected_transition = session.transitions[-1]
    assert rejected_transition.status == "rejected"
    assert isinstance(rejected_transition.payload, ReplayableRejected)
    decoded_journal = kernel_replay_journal_from_json(
        json.loads(json.dumps(kernel_replay_journal_to_json(session.to_journal())))
    )
    derived_state = kernel_replay_state_from_journal(program, decoded_journal)
    assert derived_state.rejected is True
    assert derived_state.open_source_keys == ()
    assert derived_state.consumed_source_keys == session.state.consumed_source_keys
    assert derived_state.transition_refs == session.state.transition_refs
    with pytest.raises(ContinuationReplayError, match="rejected"):
        session.resume(request_transition.payload, HostCompleted(7), registry=registry)


@pytest.mark.parametrize(
    ("handler_body", "outcome_name"),
    [
        (TerminalDelay(Lit("waiting")), "Delayed"),
        (TerminalFork((("branch:A", Lit("value-A")),)), "Forked"),
    ],
)
def test_direct_kernel_replay_unsupported_publication_resume_rejects_state(
    handler_body: Any,
    outcome_name: str,
) -> None:
    program = elaborate_publication_experimental(
        Let(
            "draft",
            Perform("provider.llm.generate", Lit({"prompt": "draft"})),
            Handle(
                Perform("eff.a", Lit("payload")),
                HandlerEnv(
                    (
                        StaticHandlerInstall(
                            effect_kind="eff.a",
                            handler_id="h.experimental.direct",
                            handled_result_schema=AnySchema(),
                            payload_name="_payload",
                            body=handler_body,
                        ),
                    )
                ),
            ),
        )
    )
    state, request_transition = start_kernel_replay(program)
    assert isinstance(request_transition.payload, ExternalEffectRequest)

    direct_transition = resume_replayable_kernel_transition(
        program,
        request_transition.payload,
        HostCompleted("accepted"),
        parent_transition_refs=(request_transition.transition_id,),
    )
    with pytest.raises(KernelReplayRejected) as rejected:
        resume_kernel_replay(state, request_transition.payload, HostCompleted("accepted"))

    rejected_state = rejected.value.state
    rejected_transition = rejected.value.transition
    assert isinstance(rejected.value.reason, ContinuationReplayError)
    assert outcome_name in str(rejected.value.reason)
    assert rejected_state.rejected is True
    assert rejected_state.terminal is False
    assert rejected_state.open_source_keys == ()
    assert rejected_state.consumed_source_keys == (request_transition.payload.source_key,)
    assert rejected_state.transition_refs == (request_transition.transition_id, rejected_transition.transition_id)
    assert rejected_transition.status == "rejected"
    assert direct_transition == rejected_transition
    assert rejected_transition.parent_transition_refs == (request_transition.transition_id,)
    assert isinstance(rejected_transition.payload, ReplayableRejected)
    assert rejected_transition.payload.trace == rejected_transition.trace_delta

    journal = KernelReplayJournal(
        program_ref=state.program_ref,
        transitions=(request_transition, rejected_transition),
    )
    derived_state = kernel_replay_state_from_journal(program, journal)
    assert derived_state.rejected is True
    assert derived_state.open_source_keys == ()
    assert derived_state.consumed_source_keys == rejected_state.consumed_source_keys
    assert derived_state.transition_refs == rejected_state.transition_refs


@pytest.mark.parametrize(
    ("handler_body", "outcome_name"),
    [
        (TerminalDelay(Lit("waiting")), "Delayed"),
        (TerminalFork((("branch:A", Lit("value-A")),)), "Forked"),
    ],
)
def test_kernel_replay_session_unsupported_publication_resume_rejects_state(
    handler_body: Any,
    outcome_name: str,
) -> None:
    program = elaborate_publication_experimental(
        Let(
            "draft",
            Perform("provider.llm.generate", Lit({"prompt": "draft"})),
            Handle(
                Perform("eff.a", Lit("payload")),
                HandlerEnv(
                    (
                        StaticHandlerInstall(
                            effect_kind="eff.a",
                            handler_id="h.experimental",
                            handled_result_schema=AnySchema(),
                            payload_name="_payload",
                            body=handler_body,
                        ),
                    )
                ),
            ),
        )
    )
    session, request_transition = KernelReplaySession.start(program)
    assert isinstance(request_transition.payload, ExternalEffectRequestRef)

    with pytest.raises(KernelReplayRejected) as rejected:
        session.resume_current(HostCompleted("accepted"))

    assert isinstance(rejected.value.reason, ContinuationReplayError)
    assert outcome_name in str(rejected.value.reason)
    assert session.state is rejected.value.state
    assert session.state.rejected is True
    assert session.state.terminal is False
    assert session.state.open_source_keys == ()
    assert session.state.consumed_source_keys == (request_transition.payload.source_key,)
    assert len(session.transitions) == 2
    assert session.transitions[0] == request_transition
    assert session.transitions[1].status == "rejected"
    assert isinstance(session.transitions[1].payload, ReplayableRejected)

    with pytest.raises(ContinuationReplayError, match="rejected"):
        session.resume_current(HostCompleted("accepted-again"))


def test_kernel_replay_session_uses_compact_requests_without_hot_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_materializations = 0
    original_from_objects = replay_module.continuation_replay_artifact_from_objects

    def counted_from_objects(*args: Any, **kwargs: Any) -> ContinuationReplayArtifact:
        nonlocal artifact_materializations
        artifact_materializations += 1
        return original_from_objects(*args, **kwargs)

    monkeypatch.setattr(replay_module, "continuation_replay_artifact_from_objects", counted_from_objects)

    body: Any = Return(Lit("done"))
    for index in reversed(range(20)):
        body = Let(f"external{index}", Perform("provider.llm.generate", Lit({"i": index})), body)
    session, transition = KernelReplaySession.start(elaborate(body))
    transitions = [transition]

    for index in range(20):
        request = session.current_request()
        assert isinstance(request, ExternalEffectRequestRef)
        assert request.payload == {"i": index}
        transition = session.resume_current(HostCompleted(f"value:{index}"))
        transitions.append(transition)

    assert session.state.terminal is True
    assert session.state.consumed_source_keys == tuple(
        transition.payload.source_key
        for transition in transitions[:-1]
        if isinstance(transition.payload, ExternalEffectRequestRef)
    )
    assert artifact_materializations == 0


def test_kernel_replay_session_journal_closure_walks_shared_objects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body: Any = Return(Lit("done"))
    for index in reversed(range(40)):
        body = Let(f"external{index}", Perform("provider.llm.generate", Lit({"i": index})), body)
    session, transition = KernelReplaySession.start(elaborate(body))

    for index in range(40):
        assert isinstance(transition.payload, ExternalEffectRequestRef)
        transition = session.resume_current(HostCompleted(f"value:{index}"))

    object_gets = 0
    original_get = session._evaluator.get_continuation_object

    def counted_get(ref: str) -> Any:
        nonlocal object_gets
        object_gets += 1
        return original_get(ref)

    monkeypatch.setattr(session._evaluator, "get_continuation_object", counted_get)

    journal = session.to_journal()

    assert object_gets == len(journal.continuation_objects)
    assert len(journal.artifacts) == 40


def test_kernel_replay_journal_validation_walks_shared_objects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body: Any = Return(Lit("done"))
    for index in reversed(range(40)):
        body = Let(f"external{index}", Perform("provider.llm.generate", Lit({"i": index})), body)
    session, transition = KernelReplaySession.start(elaborate(body))
    for index in range(40):
        assert isinstance(transition.payload, ExternalEffectRequestRef)
        transition = session.resume_current(HostCompleted(f"value:{index}"))
    journal = session.to_journal()

    child_walks = 0
    original_child_refs = replay_module.continuation_object_child_refs

    def counted_child_refs(obj: Any) -> Any:
        nonlocal child_walks
        child_walks += 1
        return original_child_refs(obj)

    monkeypatch.setattr(replay_module, "continuation_object_child_refs", counted_child_refs)

    KernelReplayJournal(
        program_ref=journal.program_ref,
        transitions=journal.transitions,
        continuation_objects=journal.continuation_objects,
        artifacts=journal.artifacts,
    )

    assert child_walks == len(journal.continuation_objects)


def _artifact_from_resumption_handle(
    result: TraceResult,
    handle: ResumptionHandle,
) -> ContinuationReplayArtifact:
    evidence = result.require_debug_evidence()
    root_ref = evidence.continuation_ref_map[handle.continuation_ref]
    return continuation_replay_artifact_from_objects(
        root_ref,
        evidence.continuation_objects,
        program_ref=evidence.program_ref,
        source_ref=handle.ref,
        source_record_type="ResumptionHandle",
        operation_result_schema_ref=handle.operation_result_schema_ref,
    )


def _transition_json(transition: ReplayableKernelTransition) -> dict[str, Any]:
    return json.loads(json.dumps(replayable_kernel_transition_to_json(transition)))


def _canonical_transition_ref_from_json(encoded: dict[str, Any]) -> str:
    return content_ref(
        "kernel-replay-transition",
        {
            "transition_schema_version": encoded["transition_schema_version"],
            "program_ref": encoded["program_ref"],
            "status": encoded["status"],
            "parent_transition_refs": encoded["parent_transition_refs"],
            "resume_observation_ref": encoded["resume_observation_ref"],
            "payload_ref": _payload_ref_from_json(encoded["payload"]),
            "trace_delta_ref": content_ref("trace-prefix", encoded["trace_delta"]),
        },
    )


def _payload_ref_from_json(payload: dict[str, Any]) -> str:
    payload_type = payload["payload_type"]
    if payload_type == "ReplayableCompleted":
        return content_ref("replayable-completed", payload)
    if payload_type == "ReplayableRejected":
        return content_ref("replayable-rejected", payload)
    if payload_type == "ExternalEffectRequest":
        request = payload["request"]
        artifact = request["replay_artifact"]
        artifact_record = {
            "artifact_schema_version": artifact["artifact_schema_version"],
            "root_ref": artifact["root_ref"],
            "program_ref": artifact["program_ref"],
            "source_key": artifact["source_key"],
            "source_ref": artifact["source_ref"],
            "source_record_type": artifact["source_record_type"],
            "effect_kind": artifact["effect_kind"],
            "operation_result_schema_ref": artifact["operation_result_schema_ref"],
        }
        return content_ref(
            "external-effect-request",
            {
                "request_schema_version": request["request_schema_version"],
                "declaration": request["declaration"],
                "artifact_ref": content_ref("continuation-replay-artifact", artifact_record),
                "trace_prefix_ref": content_ref("trace-prefix", request["trace_prefix"]),
            },
        )
    raise AssertionError(f"unexpected payload_type: {payload_type!r}")
