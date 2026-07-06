# under-test: vcs_core._world_substrate_adapters
"""Unit tests for ``WorkspaceSubstrateDriver``'s typed ingress dispatch.

Covers the migration table from SPI v0.1 §Migration Table for the
workspace driver: each of the 8 legacy string arms is exercised through
its typed ``IngressRequest`` variant via ``prepare(context, request)``.

T2c wired the typed ``ReduceRequest`` handler (re-adding ``ReduceRequest``
to ``capabilities.accepts`` in the same commit per the
capabilities-as-runtime-contract rule). The handler delegates to the same
``_workspace_capture_reduction_from_evidence_ingress_result`` logic that
the legacy ``prepare_command("capture-reduction-from-evidence", ...)``
path uses, accepting the caller-computed reduction state via
``ReduceRequest.reduction_payload`` / ``reduction_proof``.
"""

from __future__ import annotations

import pytest
from vcs_core._overlay_capture_adapter import (
    OVERLAY_ADAPTER_ID,
    OVERLAY_EVIDENCE_KIND_WRITE_CLOSE,
    OVERLAY_EVIDENCE_KINDS,
    OverlayCaptureAdapter,
)
from vcs_core._substrate_driver import ReductionBatch
from vcs_core._world_substrate_adapters import (
    WORKSPACE_REVISION_SCHEMA,
    WorkspaceSubstrateDriver,
)
from vcs_core.runtime_api import CommandRequest, DriverContext, DriverIngressResult
from vcs_core.spi import (
    CaptureRequest,
    DriverSchema,
    MergeRequest,
    ObservationDraft,
    ReduceRequest,
    ScanRequest,
    SubstrateDriver,
    SubstrateStoreIdentity,
    validate_driver_ingress,
)


def _ctx(*, base_heads: tuple[str, ...] = ()) -> DriverContext:
    return DriverContext(
        operation_id="op-1",
        binding="workspace",
        role="shepherd.WorkspaceRef",
        store_identity=SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="ws:test"),
        base_heads=base_heads,
    )


def _payload(label: str = "ws") -> dict[str, object]:
    return {"label": label}


def test_workspace_driver_satisfies_protocol_and_describe_is_complete() -> None:
    driver = WorkspaceSubstrateDriver()
    assert isinstance(driver, SubstrateDriver)
    schema = driver.describe()
    assert isinstance(schema, DriverSchema)
    assert schema.driver_id == "shepherd.workspace_ref"
    assert set(schema.commands.keys()) == {"bootstrap", "import", "create-candidate"}
    assert set(schema.scans.keys()) == {"workspace-scan", "workspace-adoption"}
    assert set(schema.merges.keys()) == {"workspace-overlay-merge"}
    assert len(schema.capture_adapters) == 1
    assert schema.capture_adapters[0].adapter_id == OVERLAY_ADAPTER_ID
    assert set(schema.capture_adapters[0].evidence_kinds) == set(OVERLAY_EVIDENCE_KINDS)
    accepts = driver.capabilities.accepts
    # T2c: ``ReduceRequest`` re-added to ``accepts`` in the same commit that
    # wired the typed reduce handler (per SPI v0.1 §Result Shape
    # "Capabilities are a runtime contract").
    assert accepts == frozenset(
        {
            CommandRequest,
            ScanRequest,
            CaptureRequest,
            ReduceRequest,
            MergeRequest,
        }
    )


def test_workspace_driver_returns_overlay_adapter_as_default() -> None:
    driver = WorkspaceSubstrateDriver()
    adapters = driver.capture_adapters(_ctx())
    assert len(adapters) == 1
    assert isinstance(adapters[0], OverlayCaptureAdapter)


@pytest.mark.parametrize(
    ("command", "expected_semantic_op"),
    [
        ("bootstrap", "bootstrap"),
        ("import", "import"),
        ("create-candidate", "workspace-json-revision"),
    ],
)
def test_command_request_dispatches_to_expected_semantic_op(command: str, expected_semantic_op: str) -> None:
    driver = WorkspaceSubstrateDriver()
    request = CommandRequest(command=command, params={"payload": _payload(command)})
    result = driver.prepare(_ctx(base_heads=("1" * 40,)), request)
    assert isinstance(result, DriverIngressResult)
    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.semantic_op == expected_semantic_op
    assert transition.payload["schema"] == WORKSPACE_REVISION_SCHEMA
    assert transition.payload["label"] == command
    validate_driver_ingress(request, result, driver)


def test_command_request_with_unknown_command_raises_value_error() -> None:
    driver = WorkspaceSubstrateDriver()
    with pytest.raises(ValueError, match="unsupported workspace command"):
        driver.prepare(
            _ctx(),
            CommandRequest(command="not-a-real-command", params={"payload": _payload()}),
        )


@pytest.mark.parametrize(
    ("scan_kind", "expected_semantic_op"),
    [
        ("workspace-scan", "workspace-scan"),
        ("workspace-adoption", "workspace-adoption"),
    ],
)
def test_scan_request_dispatches_to_expected_semantic_op(scan_kind: str, expected_semantic_op: str) -> None:
    driver = WorkspaceSubstrateDriver()
    request = ScanRequest(scan_kind=scan_kind, external_state={"payload": _payload()})
    result = driver.prepare(_ctx(base_heads=("1" * 40,)), request)
    assert len(result.transitions) == 1
    assert result.transitions[0].semantic_op == expected_semantic_op
    validate_driver_ingress(request, result, driver)


def test_scan_request_with_unknown_kind_raises_value_error() -> None:
    driver = WorkspaceSubstrateDriver()
    with pytest.raises(ValueError, match="unsupported workspace scan kind"):
        driver.prepare(_ctx(), ScanRequest(scan_kind="future-scan", external_state={"payload": _payload()}))


def test_merge_request_produces_overlay_merge_semantic_op() -> None:
    driver = WorkspaceSubstrateDriver()
    request = MergeRequest(
        other_head="2" * 40,
        policy={"payload": _payload("merged")},
    )
    result = driver.prepare(_ctx(base_heads=("1" * 40,)), request)
    assert len(result.transitions) == 1
    assert result.transitions[0].semantic_op == "workspace-overlay-merge"
    validate_driver_ingress(request, result, driver)


def test_capture_request_returns_observations_only_no_transitions() -> None:
    driver = WorkspaceSubstrateDriver()
    observation = ObservationDraft(
        observation_id="obs-1",
        evidence_kind=OVERLAY_EVIDENCE_KIND_WRITE_CLOSE,
        stable_observation={"op": "write_close", "path": "src/a.py"},
        mechanism="overlay",
        correlation_id="op-cmd",
    )
    request = CaptureRequest(adapter_id=OVERLAY_ADAPTER_ID, observations=(observation,))
    result = driver.prepare(_ctx(), request)
    assert result.transitions == ()
    assert result.observations == (observation,)
    validate_driver_ingress(request, result, driver)


def test_capture_request_rejects_unknown_adapter_id() -> None:
    driver = WorkspaceSubstrateDriver()
    with pytest.raises(ValueError, match="workspace driver only accepts CaptureRequest"):
        driver.prepare(
            _ctx(),
            CaptureRequest(adapter_id="other.adapter", observations=()),
        )


def test_reduce_request_requires_payload_and_proof_in_v01() -> None:
    # T2c: typed ``ReduceRequest`` handler accepts the request but per the
    # SPI v0.1 ``ReduceRequest.reduction_payload`` / ``reduction_proof``
    # contract, the caller must supply the pre-computed reduction state
    # (v0.1 DriverContext doesn't yet carry a coordinator-supplied evidence
    # resolver). Calling ``prepare`` with ``reduction_payload=None`` raises
    # ``ValueError`` from the workspace driver's handler.
    driver = WorkspaceSubstrateDriver()
    assert ReduceRequest in driver.capabilities.accepts
    with pytest.raises(ValueError, match="reduction_payload and reduction_proof"):
        driver.prepare(
            _ctx(),
            ReduceRequest(evidence_citations=ReductionBatch(citations=())),
        )


def test_reduce_request_requires_at_least_one_citation() -> None:
    driver = WorkspaceSubstrateDriver()
    with pytest.raises(ValueError, match="at least one EvidenceCitation"):
        driver.prepare(
            _ctx(),
            ReduceRequest(
                evidence_citations=ReductionBatch(citations=()),
                reduction_payload={"schema": "x"},
                reduction_proof={"manifest_digest": "y"},
            ),
        )


def test_reduce_request_happy_path_emits_capture_reduction_transition() -> None:
    """T2c happy-path: typed ReduceRequest produces a workspace-capture-reduction transition."""
    from vcs_core import canonical_digest
    from vcs_core._substrate_driver import EvidenceCitation
    from vcs_core._world_substrate_adapters import workspace_state_revision_payload

    driver = WorkspaceSubstrateDriver()
    payload = workspace_state_revision_payload(
        (
            {
                "path": "src/app.py",
                "state": "present",
                "mode": 0o100644,
                "content_digest": "sha256:" + "a" * 64,
            },
        )
    )
    manifest_digest = canonical_digest(payload["state_manifest"])
    proof = {
        "byte_authority": "digest-only",
        "manifest_digest": manifest_digest,
        "command_operation_id": "op-cmd-X",
    }
    citation = EvidenceCitation(
        citation_id="raw-1",
        producer_operation_id="op-cmd-X",
        evidence_ref="refs/vcscore/evidence/raw1",
        evidence_digest="sha256:" + "a" * 64,
        record_digest="sha256:" + "b" * 64,
        payload_digest="sha256:" + "c" * 64,
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
        evidence_kind="python-runtime:write",
    )
    request = ReduceRequest(
        evidence_citations=ReductionBatch(citations=(citation,)),
        reduction_payload=payload,
        reduction_proof=proof,
    )
    result = driver.prepare(_ctx(), request)
    assert request.ingress_kind == "reduce"
    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.semantic_op == "workspace-capture-reduction"
    assert transition.evidence_citation_ids == ("raw-1",)
    # The handler emits a reduced-state-proof observation alongside the
    # transition (per the legacy reduce flow's data shape).
    assert len(result.observations) == 1
    assert result.observations[0].evidence_kind == "reduce:reduced-state-proof"
    validate_driver_ingress(request, result, driver)


def test_reduce_request_requires_uniform_producer_operation_id() -> None:
    from vcs_core._substrate_driver import EvidenceCitation

    driver = WorkspaceSubstrateDriver()
    citations = (
        EvidenceCitation(
            citation_id="c1",
            producer_operation_id="op-A",
            evidence_ref="refs/vcscore/evidence/a",
            evidence_digest="sha256:" + "a" * 64,
            record_digest="sha256:" + "b" * 64,
            payload_digest="sha256:" + "c" * 64,
            binding="workspace",
            store_id="store_workspace",
            substrate_kind="filesystem",
            evidence_kind="python-runtime:write",
        ),
        EvidenceCitation(
            citation_id="c2",
            producer_operation_id="op-B",  # different!
            evidence_ref="refs/vcscore/evidence/b",
            evidence_digest="sha256:" + "d" * 64,
            record_digest="sha256:" + "e" * 64,
            payload_digest="sha256:" + "f" * 64,
            binding="workspace",
            store_id="store_workspace",
            substrate_kind="filesystem",
            evidence_kind="python-runtime:write",
        ),
    )
    with pytest.raises(ValueError, match="share one producer_operation_id"):
        driver.prepare(
            _ctx(),
            ReduceRequest(
                evidence_citations=ReductionBatch(citations=citations),
                reduction_payload={"schema": "x"},
                reduction_proof={"manifest_digest": "y"},
            ),
        )


# T3-final: test_legacy_prepare_command_still_handles_all_eight_arms removed.
# The legacy ``prepare_command`` Protocol method and the workspace driver's
# implementation were removed in T3-final; the eight string-dispatch arms
# no longer exist. The typed ``prepare(IngressRequest)`` path is the
# canonical contract — see test_command_request_dispatches_to_typed_handler
# and the per-variant happy-path tests above for the post-T3 coverage.
