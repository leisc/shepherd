# under-test: vcs_core._substrate_driver
"""Unit tests for the SPI v0.1 typed ingress family.

Covers:

- discriminated union dispatch under ``match request:``;
- per-variant ``ingress_kind`` derivation (Q3);
- per-request validator invariants (Q1 / §Result Shape);
- capability fail-closed pre-flight (UnsupportedRequestError);
- ActiveSurface dual allow/deny polarity (Q5a);
- sink fan-out delivery discipline (Q5b).
"""

from __future__ import annotations

import pytest
from vcs_core import InvalidRepositoryStateError
from vcs_core._substrate_driver import (
    ActiveSurface,
    CapabilitySet,
    CaptureRequest,
    CommandRequest,
    Diagnostic,
    DriverContext,
    DriverIngressResult,
    DriverSchema,
    FanOutSink,
    IngressRequest,
    MergeRequest,
    ObservationDraft,
    ParseResult,
    ReduceRequest,
    ReductionBatch,
    ScanRequest,
    SurfacePolicyError,
    TransitionDraft,
    TupleSink,
    UnsupportedRequestError,
    validate_driver_ingress,
)
from vcs_core._world_transition_coordinator import (
    _check_active_surface_post_dispatch,
    _check_active_surface_pre_dispatch,
    _check_capability_accepts,
)
from vcs_core.spi import SubstrateStoreIdentity


def _store_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_test", kind="test.memory", resource_id="memory:test")


def _ctx(*, active_surface: ActiveSurface | None = None) -> DriverContext:
    return DriverContext(
        operation_id="op-1",
        binding="memory",
        role="test.Role",
        store_identity=_store_identity(),
        active_surface=active_surface,
    )


class _AcceptingDriver:
    """Driver that accepts every IngressRequest variant via match dispatch."""

    driver_id = "test.accepting"
    driver_version = "v1"
    capabilities = CapabilitySet(
        accepts=frozenset({CommandRequest, ScanRequest, CaptureRequest, ReduceRequest, MergeRequest})
    )

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
        )

    def prepare(
        self,
        context: DriverContext,
        request: IngressRequest,
    ) -> DriverIngressResult:
        del context
        match request:
            case CommandRequest():
                return DriverIngressResult(
                    transitions=(
                        TransitionDraft(
                            transition_id="t-cmd",
                            semantic_op=request.command,
                            payload={"schema": "test/cmd"},
                            observation_ids=(),
                        ),
                    )
                )
            case ScanRequest():
                return DriverIngressResult(
                    transitions=(
                        TransitionDraft(
                            transition_id="t-scan",
                            semantic_op=request.scan_kind,
                            payload={"schema": "test/scan"},
                            observation_ids=(),
                        ),
                    )
                )
            case CaptureRequest():
                return DriverIngressResult(observations=tuple(request.observations))
            case ReduceRequest():
                return DriverIngressResult(
                    transitions=(
                        TransitionDraft(
                            transition_id="t-reduce",
                            semantic_op="capture-reduction",
                            payload={"schema": "test/reduce"},
                            observation_ids=(),
                            evidence_citation_ids=tuple(c.citation_id for c in request.evidence_citations.citations),
                        ),
                    )
                )
            case MergeRequest():
                return DriverIngressResult(
                    transitions=(
                        TransitionDraft(
                            transition_id="t-merge",
                            semantic_op="merge",
                            payload={"schema": "test/merge", "other": request.other_head},
                            observation_ids=(),
                        ),
                    )
                )

    def capture_adapters(self, context: DriverContext) -> tuple[()]:
        return ()

    def validate_result(
        self,
        request: IngressRequest,
        result: DriverIngressResult,
    ) -> None:
        return None


class _CommandOnlyDriver:
    """Driver whose capabilities admit only CommandRequest."""

    driver_id = "test.cmd_only"
    driver_version = "v1"
    capabilities = CapabilitySet(accepts=frozenset({CommandRequest}))

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
        )

    def prepare(
        self,
        context: DriverContext,
        request: IngressRequest,
    ) -> DriverIngressResult:
        del context
        if isinstance(request, CommandRequest):
            return DriverIngressResult(
                transitions=(
                    TransitionDraft(
                        transition_id="t-legacy",
                        semantic_op=request.command,
                        payload={"schema": "test/legacy"},
                        observation_ids=(),
                    ),
                )
            )
        from vcs_core._substrate_driver import UnsupportedRequestError

        raise UnsupportedRequestError(driver_id=self.driver_id, request_type=type(request))

    def capture_adapters(self, context: DriverContext) -> tuple[()]:
        return ()

    def validate_result(
        self,
        request: IngressRequest,
        result: DriverIngressResult,
    ) -> None:
        return None


def test_ingress_kind_derives_from_request_type() -> None:
    assert CommandRequest(command="x").ingress_kind == "command"
    assert ScanRequest(scan_kind="x").ingress_kind == "scan"
    assert CaptureRequest(adapter_id="x").ingress_kind == "capture"
    assert ReduceRequest(evidence_citations=ReductionBatch(citations=())).ingress_kind == "reduce"
    assert MergeRequest(other_head="x").ingress_kind == "merge"


def test_accepting_driver_handles_every_variant() -> None:
    driver = _AcceptingDriver()
    context = _ctx()
    for request in (
        CommandRequest(command="put"),
        ScanRequest(scan_kind="ws-scan"),
        CaptureRequest(adapter_id="overlay"),
        ReduceRequest(evidence_citations=ReductionBatch(citations=())),
        MergeRequest(other_head="abc"),
    ):
        result = driver.prepare(context, request)
        validate_driver_ingress(request, result, driver)


def test_command_only_driver_rejects_non_command_requests() -> None:
    """Command-only drivers raise UnsupportedRequestError on non-CommandRequest variants.

    T3-final removed the ``legacy_prepare_via_command`` helper that previously
    centralized this translation. Each driver now implements its own typed
    dispatch with explicit UnsupportedRequestError for variants outside
    ``capabilities.accepts``.
    """
    driver = _CommandOnlyDriver()
    context = _ctx()
    cmd_result = driver.prepare(context, CommandRequest(command="put", params={}))
    assert cmd_result.transitions[0].semantic_op == "put"

    with pytest.raises(UnsupportedRequestError):
        driver.prepare(context, ScanRequest(scan_kind="x"))


def test_capability_check_rejects_unsupported_request_type() -> None:
    driver = _CommandOnlyDriver()
    with pytest.raises(UnsupportedRequestError):
        _check_capability_accepts(driver, ScanRequest(scan_kind="x"))


def test_capture_request_must_produce_no_transitions() -> None:
    driver = _AcceptingDriver()
    request = CaptureRequest(adapter_id="overlay")
    bad_result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="t",
                semantic_op="bogus",
                payload={"schema": "x"},
                observation_ids=(),
            ),
        )
    )
    with pytest.raises(InvalidRepositoryStateError, match="must not contain transitions"):
        validate_driver_ingress(request, bad_result, driver)


def test_reduce_request_may_emit_reduction_proof_observations() -> None:
    # T2c relaxed the per-request invariant: ReduceRequest results MAY
    # contain observations representing reduction outputs (e.g., the
    # ``reduce:reduced-state-proof`` observation that documents the
    # proof linking citations to the produced state manifest).
    # Coordinator-persisted citations live in ``request.evidence_citations``;
    # observations in the result are strictly reduction-output kinds.
    driver = _AcceptingDriver()
    request = ReduceRequest(evidence_citations=ReductionBatch(citations=()))
    result_with_observation = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="reduced-state-proof",
                evidence_kind="reduce:reduced-state-proof",
                stable_observation={"payload_digest": "sha256:" + "f" * 64},
            ),
        ),
    )
    # No exception: observations are now admitted on ReduceRequest results.
    validate_driver_ingress(request, result_with_observation, driver)


def test_reduce_request_transitions_must_cite_only_batch_citations() -> None:
    driver = _AcceptingDriver()
    request = ReduceRequest(evidence_citations=ReductionBatch(citations=()))
    bad_result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="t",
                semantic_op="capture-reduction",
                payload={"schema": "x"},
                observation_ids=(),
                evidence_citation_ids=("c-not-in-batch",),
            ),
        )
    )
    with pytest.raises(InvalidRepositoryStateError, match="not in the request's batch"):
        validate_driver_ingress(request, bad_result, driver)


def test_active_surface_request_type_allow_and_deny() -> None:
    driver = _AcceptingDriver()
    allow_only_command = ActiveSurface(allow_request_types=frozenset({CommandRequest}))
    _check_active_surface_pre_dispatch(driver, allow_only_command, CommandRequest(command="x"))
    with pytest.raises(SurfacePolicyError, match="not in active-surface allow set"):
        _check_active_surface_pre_dispatch(driver, allow_only_command, ScanRequest(scan_kind="x"))

    deny_scan = ActiveSurface(deny_request_types=frozenset({ScanRequest}))
    _check_active_surface_pre_dispatch(driver, deny_scan, CommandRequest(command="x"))
    with pytest.raises(SurfacePolicyError, match="denied by active surface"):
        _check_active_surface_pre_dispatch(driver, deny_scan, ScanRequest(scan_kind="x"))


def test_active_surface_post_dispatch_filters_evidence_kinds_and_semantic_ops() -> None:
    driver = _AcceptingDriver()
    surface = ActiveSurface(deny_evidence_kinds=frozenset({"fs:write"}))
    bad_result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="o",
                evidence_kind="fs:write",
                stable_observation={},
            ),
        )
    )
    with pytest.raises(SurfacePolicyError, match="evidence_kind denied"):
        _check_active_surface_post_dispatch(driver, surface, bad_result)

    surface_ops = ActiveSurface(deny_semantic_ops=frozenset({"write"}))
    bad_op_result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="t",
                semantic_op="write",
                payload={"schema": "x"},
                observation_ids=(),
            ),
        )
    )
    with pytest.raises(SurfacePolicyError, match="semantic_op denied"):
        _check_active_surface_post_dispatch(driver, surface_ops, bad_op_result)


def test_fan_out_sink_atomicity_and_ordering() -> None:
    sinks = [TupleSink(), TupleSink(), TupleSink()]
    fan = FanOutSink(sinks)
    observations = [
        ObservationDraft(
            observation_id=f"o-{i}",
            evidence_kind="x:y",
            stable_observation={"i": i},
        )
        for i in range(4)
    ]
    for o in observations:
        fan.emit(o)

    for sink in sinks:
        assert [obs.observation_id for obs in sink.observations] == [
            "o-0",
            "o-1",
            "o-2",
            "o-3",
        ]


def test_fan_out_sink_failure_isolation_records_diagnostic() -> None:
    class _RaisingSink:
        def emit(self, observation: ObservationDraft) -> None:
            if observation.observation_id == "o-1":
                raise RuntimeError("boom")

        def diagnostic(self, diagnostic: Diagnostic) -> None:
            return None

    good_a = TupleSink()
    good_b = TupleSink()
    raising = _RaisingSink()
    fan = FanOutSink([good_a, raising, good_b])

    fan.emit(ObservationDraft(observation_id="o-0", evidence_kind="x", stable_observation={}))
    fan.emit(ObservationDraft(observation_id="o-1", evidence_kind="x", stable_observation={}))
    fan.emit(ObservationDraft(observation_id="o-2", evidence_kind="x", stable_observation={}))

    # Good sinks see every observation.
    assert [o.observation_id for o in good_a.observations] == ["o-0", "o-1", "o-2"]
    assert [o.observation_id for o in good_b.observations] == ["o-0", "o-1", "o-2"]
    # Raising sink's exception is captured as a sink_failure diagnostic; no
    # other sink is affected.
    assert len(fan.failures) == 1
    failure = fan.failures[0]
    assert failure.operation == "emit"
    assert failure.subject_id == "o-1"
    assert "boom" in failure.exception_repr


def test_parse_result_skip_and_complete_constructors() -> None:
    skipped = ParseResult.skip()
    assert skipped.skipped is True
    assert skipped.parsed_count == 0

    done = ParseResult.complete(parsed=3, diagnostics=1)
    assert done.parsed_count == 3
    assert done.diagnostic_count == 1
    assert done.skipped is False
