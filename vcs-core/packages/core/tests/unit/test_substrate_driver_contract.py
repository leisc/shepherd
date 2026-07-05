"""Unit tests for the private substrate driver draft contract."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._substrate_driver import (
    CapabilitySet,
    ChildWorldSnapshot,
    CommandRequest,
    Diagnostic,
    DriverContext,
    DriverIngressResult,
    DriverSchema,
    DriverSelectionRequirementDraft,
    IngressRequest,
    KeyedJsonPut,
    KeyedJsonTreeDraft,
    ObservationDraft,
    RetentionHint,
    RevisionStorageProfile,
    SubstrateDriver,
    TransitionDraft,
    validate_driver_identity,
    validate_driver_ingress,
    validate_driver_ingress_result,
)
from vcs_core._transition_kernel_records import PayloadDescriptorClaim, ValidatedPayloadDescriptor
from vcs_core._world_types import SubstrateStoreIdentity, canonical_digest


class _ChildWorldResolver:
    def resolve_child_world(
        self,
        world_oid: str,
        *,
        expected_world_store_id: str | None = None,
        expected_snapshot_digest: str | None = None,
    ) -> ChildWorldSnapshot:
        return ChildWorldSnapshot(
            world_store_id=expected_world_store_id or "store_world_test",
            world_oid=world_oid,
            snapshot_digest=expected_snapshot_digest or canonical_digest({"snapshot": "empty"}),
        )


class _MemoryDriver:
    driver_id = "test.memory"
    driver_version = "v1"
    capabilities = CapabilitySet(accepts=frozenset({CommandRequest}), selectable=True)

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
            payload = {"schema": "test/memory", "command": request.command, "params": dict(request.params)}
            observation = ObservationDraft(
                observation_id="observed-command",
                evidence_kind=f"command:{request.command}",
                stable_observation={"command": request.command},
                mechanism=self.driver_id,
            )
            transition = TransitionDraft(
                transition_id="transition",
                semantic_op=request.command,
                payload=payload,
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                observation_ids=(observation.observation_id,),
            )
            return DriverIngressResult(observations=(observation,), transitions=(transition,))
        from vcs_core._substrate_driver import UnsupportedRequestError

        raise UnsupportedRequestError(driver_id=self.driver_id, request_type=type(request))

    def capture_adapters(self, context: DriverContext) -> tuple[()]:
        return ()

    def validate_result(self, request: IngressRequest, result: DriverIngressResult) -> None:
        return None


def _store_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_memory", kind="test.memory", resource_id="memory:test")


def test_driver_context_is_read_only_and_narrow() -> None:
    resolver = _ChildWorldResolver()
    context = DriverContext(
        operation_id="op",
        binding="memory",
        role="shepherd.Memory",
        store_identity=_store_identity(),
        base_heads=("abc123",),
        child_worlds=resolver,
    )

    assert context.child_worlds is resolver
    assert not hasattr(context, "manager")
    assert not hasattr(context, "store")
    assert not hasattr(context, "world_store")
    assert not hasattr(context, "journal")


def test_child_world_resolver_returns_payload_facts_without_publication_refs() -> None:
    snapshot = _ChildWorldResolver().resolve_child_world(
        "1" * 40,
        expected_world_store_id="store_world_test",
        expected_snapshot_digest=canonical_digest({"snapshot": "expected"}),
    )

    assert snapshot.to_payload() == {
        "world_store_id": "store_world_test",
        "world_oid": "1" * 40,
        "snapshot_digest": canonical_digest({"snapshot": "expected"}),
    }
    assert not hasattr(snapshot, "authority_ref")
    assert not hasattr(snapshot, "retention_ref")


def test_driver_protocol_and_capabilities_are_runtime_checkable() -> None:
    driver = _MemoryDriver()

    assert isinstance(driver, SubstrateDriver)
    assert CommandRequest in driver.capabilities.accepts
    assert driver.capabilities.selectable is True


def test_driver_ingress_result_batches_non_authoritative_drafts() -> None:
    context = DriverContext(
        operation_id="op-put",
        binding="memory",
        role="shepherd.Memory",
        store_identity=_store_identity(),
    )

    result = _MemoryDriver().prepare(context, CommandRequest(command="put", params={"key": "k", "value": "v"}))

    assert result.observations[0].observation_id == "observed-command"
    assert result.transitions[0].payload["params"] == {"key": "k", "value": "v"}
    assert result.transitions[0].payload_descriptor_claim == PayloadDescriptorClaim.for_json_payload(
        result.transitions[0].payload
    )
    validate_driver_ingress_result(result)


def test_driver_dtos_do_not_expose_coordinator_authority_slots() -> None:
    banned_slots = {
        "authority_ref",
        "candidate_ref",
        "candidate_refs",
        "evidence_ref",
        "evidence_refs",
        "journal_ref",
        "publication_lease",
        "publication_ref",
        "retention_receipt",
        "selected_head_pin",
        "world_publication_ref",
    }
    driver_dtos = (
        CapabilitySet,
        ChildWorldSnapshot,
        DriverContext,
        DriverIngressResult,
        DriverSelectionRequirementDraft,
        ObservationDraft,
        RetentionHint,
        TransitionDraft,
    )

    for dto in driver_dtos:
        assert is_dataclass(dto)
        assert {field.name for field in fields(dto)}.isdisjoint(banned_slots)


def test_transition_draft_allows_semantic_payload_world_facts() -> None:
    payload = {
        "schema": "vcscore/world-ref-payload/v1",
        "world_store_id": "store_world_test",
        "world_oid": "1" * 40,
        "snapshot_digest": canonical_digest({"snapshot": "child"}),
    }
    draft = TransitionDraft(
        transition_id="link-child",
        semantic_op="world-ref-json-revision",
        payload=payload,
        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
        observation_ids=(),
    )

    assert draft.payload["world_oid"] == "1" * 40
    assert draft.payload_descriptor_claim is not None
    validate_driver_ingress_result(DriverIngressResult(transitions=(draft,)))


def test_driver_validator_rejects_dangling_observation_ids() -> None:
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload={"schema": "test/memory"},
                observation_ids=("missing-observation",),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="unknown observations"):
        validate_driver_ingress_result(result)


def test_driver_validator_rejects_duplicate_evidence_citation_ids() -> None:
    payload = {"schema": "test/memory", "value": 1}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                evidence_citation_ids=("raw-0", "raw-0"),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="duplicates"):
        validate_driver_ingress_result(result)


def test_driver_validator_allows_duplicate_keyed_tree_keys_at_distinct_paths() -> None:
    payload = {"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                content=KeyedJsonTreeDraft(
                    manifest=payload,
                    base_head=None,
                    puts=(
                        KeyedJsonPut(key="run-1", path="runs/by-ref/ru/run-1.json", payload={"kind": "run"}),
                        KeyedJsonPut(
                            key="run-1",
                            path="flow-runs/by-run/ru/run-1.json",
                            payload={"kind": "flow-run"},
                        ),
                    ),
                ),
            ),
        )
    )

    validate_driver_ingress_result(result)


def test_driver_validator_rejects_duplicate_keyed_tree_paths() -> None:
    payload = {"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                content=KeyedJsonTreeDraft(
                    manifest=payload,
                    base_head=None,
                    puts=(
                        KeyedJsonPut(key="left", path="runs/by-ref/ru/run-1.json", payload={"side": "left"}),
                        KeyedJsonPut(key="right", path="runs/by-ref/ru/run-1.json", payload={"side": "right"}),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="duplicate path"):
        validate_driver_ingress_result(result)


class _KeyedTreeDriver(_MemoryDriver):
    driver_id = "test.keyed_tree"

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            storage_profile=RevisionStorageProfile(
                shape="keyed-json-tree",
                authority_role="authority",
                growth_bound="unbounded",
            ),
        )


class _UnsupportedStorageDriver(_MemoryDriver):
    driver_id = "test.unsupported_storage"

    def __init__(self, shape: str) -> None:
        self._shape = shape

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            storage_profile=RevisionStorageProfile(
                shape=self._shape,  # type: ignore[arg-type]
                authority_role="authority",
                growth_bound="unbounded",
            ),
        )


def _keyed_content(payload: dict[str, object]) -> KeyedJsonTreeDraft:
    return KeyedJsonTreeDraft(
        manifest=payload,
        base_head=None,
        puts=(KeyedJsonPut(key="run-1", path="runs/by-ref/ru/run-1.json", payload={"run_ref": "run-1"}),),
    )


def test_driver_validator_rejects_structured_content_for_json_snapshot_driver() -> None:
    payload = {"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                content=_keyed_content(payload),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="json-snapshot storage"):
        validate_driver_ingress(CommandRequest(command="put", params={}), result, _MemoryDriver())


@pytest.mark.parametrize(
    "shape",
    [
        "tree-projection",
        "event-stream",
        "derived-index",
        "object-store",
    ],
)
def test_driver_validator_rejects_transitions_for_unimplemented_storage_shapes(shape: str) -> None:
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload={"schema": "test/unsupported-storage"},
                observation_ids=(),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="unsupported storage shape"):
        validate_driver_ingress(CommandRequest(command="put", params={}), result, _UnsupportedStorageDriver(shape))


def test_driver_validator_requires_keyed_tree_content_for_keyed_tree_driver() -> None:
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload={"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"},
                observation_ids=(),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="not KeyedJsonTreeDraft"):
        validate_driver_ingress(CommandRequest(command="put", params={}), result, _KeyedTreeDriver())


def test_driver_validator_rejects_workspace_tree_oid_for_keyed_tree_driver() -> None:
    payload = {"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                git_tree_oid="1" * 40,
                content=_keyed_content(payload),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="git_tree_oid is set"):
        validate_driver_ingress(CommandRequest(command="put", params={}), result, _KeyedTreeDriver())


def test_driver_validator_accepts_keyed_tree_content_for_keyed_tree_driver() -> None:
    payload = {"schema": "test/keyed-manifest", "storage_shape": "keyed-json-tree"}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                observation_ids=(),
                content=_keyed_content(payload),
            ),
        )
    )

    validate_driver_ingress(CommandRequest(command="put", params={}), result, _KeyedTreeDriver())


def test_driver_validator_rejects_control_plane_authority_refs_but_not_payload_facts() -> None:
    evidence_ref = "refs/vcscore/evidence/op/abc123"
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload={"schema": "test/memory", "ordinary_note": evidence_ref},
                observation_ids=(),
                metadata={"citation": evidence_ref},
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="reserved authority ref"):
        validate_driver_ingress_result(result)

    observation_result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="command:put",
                stable_observation={"evidence_ref": evidence_ref},
            ),
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="reserved authority fields"):
        validate_driver_ingress_result(observation_result)

    keyed_observation_result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="command:put",
                stable_observation={evidence_ref: "smuggled"},
            ),
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="reserved authority ref"):
        validate_driver_ingress_result(keyed_observation_result)

    validate_driver_ingress_result(
        DriverIngressResult(
            transitions=(
                TransitionDraft(
                    transition_id="transition",
                    semantic_op="put",
                    payload={"schema": "test/memory", "ordinary_note": evidence_ref},
                    observation_ids=(),
                ),
            )
        )
    )


def test_driver_validator_rejects_reserved_authority_fields_in_metadata() -> None:
    result = DriverIngressResult(
        diagnostics=(
            Diagnostic(
                code="bad-control-plane",
                message="attempted authority smuggling",
                detail={"evidence_refs": ["refs/vcscore/evidence/op/abc123"]},
            ),
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="reserved authority fields"):
        validate_driver_ingress_result(result)


def test_driver_validator_rejects_free_form_diagnostics() -> None:
    result = DriverIngressResult(
        diagnostics=({"reason": "unsupported"},),  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidRepositoryStateError, match="diagnostics must be Diagnostic"):
        validate_driver_ingress_result(result)


def test_driver_validator_rejects_non_string_control_plane_mapping_keys() -> None:
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="command:put",
                stable_observation={1: "not allowed"},  # type: ignore[dict-item]
            ),
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="non-string key"):
        validate_driver_ingress_result(result)


def test_driver_identity_rejects_reserved_refs_and_unsupported_characters() -> None:
    validate_driver_identity(driver_id="shepherd.workspace_ref", driver_version="v1")

    with pytest.raises(InvalidRepositoryStateError, match="reserved authority ref"):
        validate_driver_identity(driver_id="refs/vcscore/evidence/op/abc123", driver_version="v1")
    with pytest.raises(InvalidRepositoryStateError, match="unsupported characters"):
        validate_driver_identity(driver_id="shepherd workspace", driver_version="v1")
    with pytest.raises(InvalidRepositoryStateError, match="driver_version"):
        validate_driver_identity(driver_id="shepherd.workspace_ref", driver_version="../v1")


def test_driver_validator_rejects_malformed_selection_requirement_fields() -> None:
    selection = DriverSelectionRequirementDraft(
        binding="workspace",
        role="shepherd.WorkspaceRef",
        selection_kind="new-candidate",
    )

    malformed = (
        (replace(selection, binding=""), "binding"),
        (replace(selection, role=""), "role"),
        (replace(selection, selection_kind=""), "selection_kind"),
        (replace(selection, transition_id=""), "transition_id"),
        (replace(selection, binding="refs/vcscore/evidence/op/abc123"), "reserved authority ref"),
        (
            replace(selection, metadata={"refs/vcscore/evidence/op/abc123": "smuggled"}),
            "reserved authority ref",
        ),
    )
    for draft, match in malformed:
        with pytest.raises(InvalidRepositoryStateError, match=match):
            validate_driver_ingress_result(DriverIngressResult(selection_requirements=(draft,)))


def test_driver_validator_rejects_malformed_observation_fields() -> None:
    observation = ObservationDraft(
        observation_id="obs",
        evidence_kind="command:put",
        stable_observation={"command": "put"},
    )

    malformed = (
        (replace(observation, evidence_kind=""), "evidence_kind"),
        (replace(observation, stable_observation=[]), "stable_observation"),  # type: ignore[arg-type]
        (replace(observation, observed_head=""), "observed_head"),
        (replace(observation, observed_at_unix_ns="1"), "observed_at_unix_ns"),  # type: ignore[arg-type]
        (replace(observation, mechanism=""), "mechanism"),
        (replace(observation, correlation_id=""), "correlation_id"),
    )
    for draft, match in malformed:
        with pytest.raises(InvalidRepositoryStateError, match=match):
            validate_driver_ingress_result(DriverIngressResult(observations=(draft,)))


def test_driver_validator_rejects_validated_payload_descriptor_outputs() -> None:
    payload = {"schema": "test/memory", "value": 1}
    descriptor = ValidatedPayloadDescriptor.for_json_payload(payload)
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                payload_descriptor_claim=descriptor,  # type: ignore[arg-type]
                observation_ids=(),
            ),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="coordinator-validated payload descriptor"):
        validate_driver_ingress_result(result)

    descriptor_shape = descriptor.to_json()
    with pytest.raises(InvalidRepositoryStateError, match="validated payload descriptor"):
        validate_driver_ingress_result(
            DriverIngressResult(
                diagnostics=(
                    Diagnostic(
                        code="bad-payload-descriptor",
                        message="attempted descriptor smuggling",
                        detail=descriptor_shape,
                    ),
                )
            )
        )

    with pytest.raises(InvalidRepositoryStateError, match="coordinator-validated payload descriptor"):
        validate_driver_ingress_result(
            DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="observation",
                        evidence_kind="command:put",
                        stable_observation={"command": "put"},
                        evidence_payload_descriptor_claim=descriptor,  # type: ignore[arg-type]
                    ),
                )
            )
        )


def test_driver_validator_allows_untrusted_payload_descriptor_claims() -> None:
    payload = {"schema": "test/memory", "value": 1}
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="observation",
                evidence_kind="command:put",
                stable_observation={"command": "put"},
                evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload({"command": "put"}),
            ),
        ),
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="put",
                payload=payload,
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                observation_ids=("observation",),
            ),
        ),
    )

    validate_driver_ingress_result(result)


def test_driver_validator_rejects_mandatory_retention_hints() -> None:
    result = DriverIngressResult(
        retention_hints=(RetentionHint(kind="selected-head-pin", target="abc123", mandatory=True),)
    )

    with pytest.raises(InvalidRepositoryStateError, match="advisory"):
        validate_driver_ingress_result(result)
