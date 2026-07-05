"""Unit tests for the private v2 WorldStorageManager."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pygit2
import pytest
from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._substrate_driver import (
    DriverIngressResult,
    DriverSelectionRequirementDraft,
    ObservationDraft,
    ReductionBatch,
    RetentionHint,
    TransitionDraft,
)
from vcs_core._substrate_store import SubstrateStore
from vcs_core._transition_kernel import JsonPayloadTransitionDriver
from vcs_core._transition_kernel_records import (
    CandidateCommitRecord,
    EvidenceOnlyEnvelopeRecord,
    EvidenceRef,
    HeadSelectionEvidence,
    PayloadDescriptorClaim,
    RetentionPolicyRequirement,
)
from vcs_core._world_operation_builder import CandidateSelection, OperationFinalBuilder, PreparedWorldOperation
from vcs_core._world_refs import (
    candidate_ref,
    child_world_retention_ref,
    evidence_record_ref,
    scope_ref,
    world_fork_origin_receipt_ref,
    world_pin_ref,
    world_publication_lease_prefix,
    world_retention_receipt_ref,
)
from vcs_core._world_storage_manager import DEFAULT_GROUND_REF, SubstrateStoreSpec, WorldStorageManager
from vcs_core._world_types import (
    WORLD_REF_SUBSTRATE_KIND,
    WORLD_TRANSITION_SCHEMA,
    SubstrateHead,
    SubstrateStoreIdentity,
    WorldRefPayload,
    WorldSnapshot,
    canonical_bytes,
    canonical_digest,
    load_canonical_json,
)
from vcs_core.git_store import create_commit_with_recovery, create_or_update_reference, insert_tree_entry

from .world_vectors_v2_helpers import (
    attach_selection_evidence_ref,
    candidate_outcome_for_commit,
    create_prepared_candidate,
    operation_final_with_head_selections,
    selection_evidence_ref,
)


def _workspace_identity(*, resource_id: str = "fs:repo-main") -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id=resource_id)


def _session_identity(*, resource_id: str = "shepherd-session:child-baseline") -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(
        store_id="store_session",
        kind="shepherd.session_state",
        resource_id=resource_id,
    )


def _world_ref_identity(*, resource_id: str = "world-ref:child-task") -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(
        store_id="store_child_world_ref",
        kind=WORLD_REF_SUBSTRATE_KIND,
        resource_id=resource_id,
    )


def _trace_identity(*, resource_id: str = "shepherd-trace:parent") -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(
        store_id="store_trace",
        kind="shepherd.trace",
        resource_id=resource_id,
    )


def _role_for_binding(binding: str) -> str:
    if binding.startswith("session"):
        return "shepherd.SessionState"
    if binding.startswith("trace"):
        return "shepherd.TraceState"
    if binding.startswith("child"):
        return "vcscore.WorldRef"
    return "shepherd.WorkspaceRef"


def _specs(
    *,
    workspace_identity: SubstrateStoreIdentity | None = None,
    session_identity: SubstrateStoreIdentity | None = None,
    workspace_locator: str = "substrates/workspace.git",
    session_locator: str = "substrates/session.git",
) -> tuple[SubstrateStoreSpec, ...]:
    return (
        SubstrateStoreSpec(
            identity=workspace_identity or _workspace_identity(),
            locator=workspace_locator,
        ),
        SubstrateStoreSpec(
            identity=session_identity or _session_identity(),
            locator=session_locator,
        ),
    )


def _manager(tmp_path, *, stores: tuple[SubstrateStoreSpec, ...] | None = None) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=stores or _specs(),
    )


def _publish_world(
    manager: WorldStorageManager,
    *,
    ref: str,
    world_oid: str,
    expected_oid: str | None,
    allow_same_resource_alias: bool = False,
    authority_refs: tuple[str, ...] | None = None,
) -> bool:
    if expected_oid is None:
        return manager.publish_root_world(
            ref=ref,
            world_oid=world_oid,
            allow_same_resource_alias=allow_same_resource_alias,
            authority_refs=authority_refs,
        )
    return manager.advance_world_ref(
        ref=ref,
        world_oid=world_oid,
        input_world_oid=expected_oid,
        allow_same_resource_alias=allow_same_resource_alias,
        authority_refs=authority_refs,
    )


def _operation_final(
    operation_id: str,
    selected: dict[str, str],
    *,
    outcomes: list[dict[str, object]] | None = None,
    candidate_commits=None,
) -> dict[str, object]:
    return operation_final_with_head_selections(
        operation_id,
        selected,
        outcomes=outcomes,
        candidate_commits=candidate_commits,
    )


def _candidate_outcome(
    manager: WorldStorageManager,
    store_id: str,
    candidate_commit: CandidateCommitRecord,
    *,
    final_operation_id: str,
    outcome: str = "selected",
    producer_world_oid: str | None = None,
) -> dict[str, object]:
    return candidate_outcome_for_commit(
        manager.store(store_id),
        candidate_commit,
        final_operation_id=final_operation_id,
        world_store=manager.world_store,
        outcome=outcome,
        producer_world_oid=producer_world_oid,
    )


def _bootstrap_revision(
    manager: WorldStorageManager,
    store_id: str,
    ref: str,
    payload: dict[str, object],
    *,
    operation_id: str,
    binding: str,
    parents: tuple[str, ...] = (),
) -> str:
    return manager.create_prepared_json_revision(
        store_id,
        ref,
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        parents=parents,
        semantic_op="bootstrap",
    )


def _bootstrap_final(
    manager: WorldStorageManager,
    operation_id: str,
    selected: dict[str, str],
    *,
    stores_by_binding: dict[str, str],
    outcomes: list[dict[str, object]] | None = None,
    candidate_commits=None,
) -> dict[str, object]:
    store_ids: dict[str, str] = {}
    resource_ids: dict[str, str] = {}
    for binding, store_id in stores_by_binding.items():
        identity = manager.store(store_id).identity
        store_ids[binding] = identity.store_id
        resource_ids[binding] = identity.resource_id
    final = operation_final_with_head_selections(
        operation_id,
        selected,
        outcomes=outcomes,
        candidate_commits=candidate_commits,
        store_ids=store_ids,
        resource_ids=resource_ids,
        selection_kinds=dict.fromkeys(selected, "bootstrap"),
    )
    for binding, head in selected.items():
        final = attach_selection_evidence_ref(
            final,
            binding=binding,
            evidence_ref=manager.create_existing_head_selection_evidence(
                operation_id=operation_id,
                head=manager.substrate_head(
                    stores_by_binding[binding],
                    binding=binding,
                    head=head,
                    role=_role_for_binding(binding),
                ),
                selection_kind="bootstrap",
            ),
        )
    return final


def _checkpoint_final(
    manager: WorldStorageManager,
    operation_id: str,
    selected: dict[str, str],
    *,
    unsafe_evidence: bool = False,
) -> dict[str, object]:
    evidence_ref = (
        selection_evidence_ref(
            manager.world_store,
            operation_id=operation_id,
            binding="session",
            store=manager.store("store_session"),
            head=selected["session"],
            evidence_kind="checkpoint",
        )
        if unsafe_evidence
        else manager.create_existing_head_selection_evidence(
            operation_id=operation_id,
            head=manager.substrate_head(
                "store_session",
                binding="session",
                head=selected["session"],
                role="shepherd.SessionState",
            ),
            selection_kind="checkpoint",
        )
    )
    return attach_selection_evidence_ref(
        operation_final_with_head_selections(
            operation_id,
            selected,
            selection_kinds={"session": "checkpoint"},
        ),
        binding="session",
        evidence_ref=evidence_ref,
    )


def _workspace_existing_head_final(
    manager: WorldStorageManager,
    operation_id: str,
    selected_head: str,
    *,
    selection_kind: str,
    selected_from: str | None = None,
    unsafe_evidence: bool = False,
) -> dict[str, object]:
    evidence_ref = (
        selection_evidence_ref(
            manager.world_store,
            operation_id=operation_id,
            binding="workspace",
            store=manager.store("store_workspace"),
            head=selected_head,
            evidence_kind=selection_kind,
            selected_from=selected_from,
        )
        if unsafe_evidence
        else manager.create_existing_head_selection_evidence(
            operation_id=operation_id,
            head=manager.substrate_head(
                "store_workspace",
                binding="workspace",
                head=selected_head,
                role="shepherd.WorkspaceRef",
            ),
            selection_kind=selection_kind,  # type: ignore[arg-type]
            selected_from=selected_from,
        )
    )
    return attach_selection_evidence_ref(
        operation_final_with_head_selections(
            operation_id,
            {"workspace": selected_head},
            selection_kinds={"workspace": selection_kind},
        ),
        binding="workspace",
        evidence_ref=evidence_ref,
        selected_from=selected_from,
    )


def _prepared_candidate(
    manager: WorldStorageManager,
    store_id: str,
    *,
    operation_id: str,
    binding: str,
    payload: dict[str, object],
    parents: tuple[str, ...],
):
    return create_prepared_candidate(
        manager.store(store_id),
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        parents=parents,
        world_store=manager.world_store,
    )


def _prepared_revision(
    manager: WorldStorageManager,
    store_id: str,
    ref: str,
    *,
    operation_id: str,
    binding: str,
    payload: dict[str, object],
    parents: tuple[str, ...] = (),
    semantic_op: str,
) -> str:
    return manager.create_prepared_json_revision(
        store_id,
        ref,
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        parents=parents,
        semantic_op=semantic_op,
    )


def _prepared_session_checkpoint(
    manager: WorldStorageManager,
    ref: str,
    payload: dict[str, object],
    *,
    parents: tuple[str, ...] = (),
    operation_id: str | None = None,
) -> str:
    return _prepared_revision(
        manager,
        "store_session",
        ref,
        operation_id=operation_id or f"op-create-{ref.rsplit('/', 1)[-1]}",
        binding="session",
        payload=payload,
        parents=parents,
        semantic_op="checkpoint",
    )


def _prepared_world_ref_revision(
    manager: WorldStorageManager,
    ref: str,
    payload: dict[str, object],
    *,
    operation_id: str,
    binding: str = "child",
) -> str:
    return _prepared_revision(
        manager,
        "store_child_world_ref",
        ref,
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        semantic_op="import",
    )


def _transition(operation_id: str, *, parents: list[str] | None = None, **extra: object) -> dict[str, object]:
    resolved_parents = parents or []
    if resolved_parents and "input_world" not in extra:
        extra["input_world"] = resolved_parents[0]
    return {
        "schema": WORLD_TRANSITION_SCHEMA,
        "operation_id": operation_id,
        "parent_worlds": resolved_parents,
        **extra,
    }


def _snapshot(*heads: SubstrateHead) -> WorldSnapshot:
    return WorldSnapshot(tuple(heads))


def _prepared_bootstrap_with_forged_selection_evidence(
    manager: WorldStorageManager,
    operation_id: str,
) -> PreparedWorldOperation:
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id=f"{operation_id}-workspace",
        binding="workspace",
    )
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    selection_plan = manager.plan_existing_head_selection(
        operation_id=operation_id,
        head=head,
        selection_kind="bootstrap",
    )
    prepared = (
        OperationFinalBuilder(operation_id)
        .select_existing(plan=selection_plan)
        .build_prepared(
            operation_kind="bootstrap",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=_snapshot(head),
            transition=_transition(operation_id),
        )
    )
    evidence = HeadSelectionEvidence.from_json(dict(prepared.selection_evidence[0]))
    forged_ref = EvidenceRef(
        ref=f"refs/vcscore/evidence/{operation_id}/missing",
        evidence_digest="sha256:" + "1" * 64,
        record_digest="sha256:" + "2" * 64,
        payload_digest="sha256:" + "3" * 64,
    )
    forged_evidence = HeadSelectionEvidence(
        operation_id=evidence.operation_id,
        binding=evidence.binding,
        store_id=evidence.store_id,
        resource_id=evidence.resource_id,
        selected_head=evidence.selected_head,
        selection_digest=evidence.selection_digest,
        revision_preparation_digest=evidence.revision_preparation_digest,
        candidate_commit_digest=evidence.candidate_commit_digest,
        candidate_ref=evidence.candidate_ref,
        producer_operation_id=evidence.producer_operation_id,
        evidence_refs=(forged_ref,),
        retention_policy_requirements=evidence.retention_policy_requirements,
    )
    return PreparedWorldOperation(
        operation_id=prepared.operation_id,
        operation_kind=prepared.operation_kind,
        target_ref=prepared.target_ref,
        input_world_oid=prepared.input_world_oid,
        snapshot=prepared.snapshot,
        transition=prepared.transition,
        candidate_tuples=prepared.candidate_tuples,
        candidate_refs=prepared.candidate_refs,
        candidate_commits=prepared.candidate_commits,
        candidate_outcomes=prepared.candidate_outcomes,
        head_selections=prepared.head_selections,
        selection_evidence=(forged_evidence.to_json(),),
        selected=prepared.selected,
        parents=prepared.parents,
    )


def _published_workspace_world(manager: WorldStorageManager) -> tuple[str, str]:
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-published-workspace",
        binding="workspace",
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-published-workspace"),
        operation_final=_bootstrap_final(
            manager,
            "op-published-workspace",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    return workspace, world_oid


def _workspace_advance_world(
    manager: WorldStorageManager,
    *,
    parent_world_oid: str,
    parent_workspace_head: str,
    operation_id: str,
) -> str:
    candidate, candidate_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id=operation_id,
        binding="workspace",
        payload={"label": f"workspace {operation_id}"},
        parents=(parent_workspace_head,),
    )
    return manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
            )
        ),
        transition=_transition(operation_id, parents=[parent_world_oid]),
        operation_final=_operation_final(
            operation_id,
            {"workspace": candidate.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    candidate_commit,
                    final_operation_id=operation_id,
                )
            ],
            candidate_commits=[candidate_commit],
        ),
        parents=(parent_world_oid,),
    )


def _workspace_root_world(manager: WorldStorageManager, *, workspace_head: str, operation_id: str) -> str:
    return manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace", binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef"
            )
        ),
        transition=_transition(operation_id),
        operation_final=_bootstrap_final(
            manager,
            operation_id,
            {"workspace": workspace_head},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )


def _world_ref_manager(tmp_path) -> WorldStorageManager:
    return _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )


def _task_loop_manager(tmp_path) -> WorldStorageManager:
    return _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
            SubstrateStoreSpec(
                identity=_trace_identity(),
                locator="substrates/trace.git",
            ),
        ),
    )


def _published_candidate_child_world(
    manager: WorldStorageManager,
    *,
    child_ref: str,
) -> tuple[str, str, str]:
    base = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/base-child", {"label": "workspace W0"}
    )
    child_head, child_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-published-child",
        binding="workspace",
        payload={"label": "workspace W1"},
        parents=(base,),
    )
    child_snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=child_head.head, role="shepherd.WorkspaceRef"
        )
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-published-child"),
        operation_final=_operation_final(
            "op-published-child",
            {"workspace": child_head.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    child_commit,
                    final_operation_id="op-published-child",
                )
            ],
            candidate_commits=[child_commit],
        ),
    )
    assert _publish_world(manager, ref=child_ref, world_oid=child_world, expected_oid=None)
    return child_world, child_head.head, child_head.ref


def _world_ref_parent_world(
    manager: WorldStorageManager,
    *,
    child_world: str,
    child_workspace_head: str,
    operation_id: str = "op-parent-import-published-child",
) -> str:
    child_snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace",
            binding="workspace",
            head=child_workspace_head,
            role="shepherd.WorkspaceRef",
        )
    )
    world_ref_head = _prepared_world_ref_revision(
        manager,
        f"refs/heads/{operation_id}",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id=operation_id,
    )
    parent_snapshot = _snapshot(
        manager.substrate_head("store_child_world_ref", binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    retention_requirements = (
        RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
        RetentionPolicyRequirement(
            kind="child-world-retention",
            target=f"world:{child_world}",
            digest=child_snapshot.digest(),
        ),
    )
    final = attach_selection_evidence_ref(
        operation_final_with_head_selections(
            operation_id,
            {"child": world_ref_head},
            store_ids={"child": "store_child_world_ref"},
            resource_ids={"child": "world-ref:child-task"},
            selection_kinds={"child": "import"},
            retention_policy_requirements={"child": retention_requirements},
        ),
        binding="child",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id=operation_id,
            binding="child",
            store=manager.store("store_child_world_ref"),
            head=world_ref_head,
            evidence_kind="import",
        ),
    )
    return manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition(operation_id),
        operation_final=final,
    )


def _read_receipt_payload(manager: WorldStorageManager, ref: str) -> dict[str, object]:
    repo = manager.world_store.repo
    commit = repo.references[ref].peel(pygit2.Commit)
    obj: pygit2.Object = commit.tree
    for component in ["meta", "world-retention-receipt.json"]:
        assert isinstance(obj, pygit2.Tree)
        obj = repo[obj[component].id]
    assert isinstance(obj, pygit2.Blob)
    return load_canonical_json(bytes(obj.data))


def _read_lease_payload(manager: WorldStorageManager, ref: str) -> dict[str, object]:
    repo = manager.world_store.repo
    commit = repo.references[ref].peel(pygit2.Commit)
    obj: pygit2.Object = commit.tree
    for component in ["meta", "world-publication-lease.json"]:
        assert isinstance(obj, pygit2.Tree)
        obj = repo[obj[component].id]
    assert isinstance(obj, pygit2.Blob)
    return load_canonical_json(bytes(obj.data))


def _write_receipt_payload(manager: WorldStorageManager, ref: str, payload: dict[str, object]) -> None:
    repo = manager.world_store.repo
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        "world-retention-receipt.json",
        repo.create_blob(canonical_bytes(payload)),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root_builder = repo.TreeBuilder()
    insert_tree_entry(repo, root_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("vcs-core test", "test@example.invalid")
    oid = create_commit_with_recovery(repo, None, signature, signature, "test receipt", root_builder.write(), [])
    create_or_update_reference(repo, ref, oid, force=True)


def test_world_storage_manager_persists_json_evidence_for_driver_draft(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"label": "workspace"}
    )
    called = False

    class DraftOnlyJsonDriver:
        driver_id = "test.json-draft"
        driver_version = "v1"

        def prepare_candidate(
            self,
            *,
            store,
            operation_id,
            binding,
            payload,
            parents,
            ingress_kind,
            semantic_op,
            relationship_requirements,
        ):
            nonlocal called
            called = True
            return JsonPayloadTransitionDriver(
                driver_id=self.driver_id,
                driver_version=self.driver_version,
            ).prepare_candidate(
                store=store,
                operation_id=operation_id,
                binding=binding,
                payload=payload,
                parents=parents,
                ingress_kind=ingress_kind,
                semantic_op=semantic_op,
                relationship_requirements=relationship_requirements,
            )

    bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-driver-draft",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
        driver=DraftOnlyJsonDriver(),
    )

    assert called
    assert bundle.preparation.evidence_refs
    evidence = manager.world_store.resolve_evidence_ref(
        bundle.preparation.evidence_refs[0],
        expected_operation_id="op-driver-draft",
    )
    assert evidence.evidence_digest() == bundle.preparation.evidence_digests[0]


def test_world_storage_manager_validates_driver_payload_descriptor_claim(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"label": "workspace"}
    )
    payload = {"schema": "example/workspace", "n": 2}

    class BadDescriptorDriver:
        driver_id = "test.bad-json-descriptor"
        driver_version = "v1"

        def __init__(self, claim: PayloadDescriptorClaim) -> None:
            self._claim = claim

        def prepare_candidate(
            self,
            *,
            store,
            operation_id,
            binding,
            payload,
            parents,
            ingress_kind,
            semantic_op,
            relationship_requirements,
        ):
            draft = JsonPayloadTransitionDriver(
                driver_id=self.driver_id,
                driver_version=self.driver_version,
            ).prepare_candidate(
                store=store,
                operation_id=operation_id,
                binding=binding,
                payload=payload,
                parents=parents,
                ingress_kind=ingress_kind,
                semantic_op=semantic_op,
                relationship_requirements=relationship_requirements,
            )
            return replace(draft, payload_descriptor_claim=self._claim)

    bad_digest_claim = PayloadDescriptorClaim(
        codec_id="vcscore.json",
        codec_version="v1",
        authority_mode="coordinator-native",
        payload_digest=canonical_digest({"schema": "example/workspace", "n": 3}),
        canonical_manifest={"payload_format": "canonical-json-v1"},
    )
    with pytest.raises(InvalidRepositoryStateError, match="payload descriptor disagrees with payload"):
        manager.create_prepared_json_candidate_bundle(
            "store_workspace",
            operation_id="op-bad-descriptor-digest",
            binding="workspace",
            payload=payload,
            parents=(parent,),
            driver=BadDescriptorDriver(bad_digest_claim),
        )

    bad_codec_claim = PayloadDescriptorClaim(
        codec_id="test.unregistered-json",
        codec_version="v1",
        authority_mode="coordinator-native",
        payload_digest=canonical_digest(payload),
        canonical_manifest={"payload_format": "canonical-json-v1"},
    )
    with pytest.raises(InvalidRepositoryStateError, match="codec is not registered"):
        manager.create_prepared_json_candidate_bundle(
            "store_workspace",
            operation_id="op-bad-descriptor-codec",
            binding="workspace",
            payload=payload,
            parents=(parent,),
            driver=BadDescriptorDriver(bad_codec_claim),
        )

    bad_manifest_claim = PayloadDescriptorClaim(
        codec_id="vcscore.json",
        codec_version="v1",
        authority_mode="coordinator-native",
        payload_digest=canonical_digest(payload),
        canonical_manifest={"payload_format": "other-json-v1"},
    )
    with pytest.raises(InvalidRepositoryStateError, match="manifest is invalid"):
        manager.create_prepared_json_candidate_bundle(
            "store_workspace",
            operation_id="op-bad-descriptor-manifest",
            binding="workspace",
            payload=payload,
            parents=(parent,),
            driver=BadDescriptorDriver(bad_manifest_claim),
        )

    payload_ref_claim = PayloadDescriptorClaim(
        codec_id="vcscore.json",
        codec_version="v1",
        authority_mode="coordinator-native",
        payload_digest=canonical_digest(payload),
        canonical_manifest={"payload_format": "canonical-json-v1"},
        payload_ref="refs/vcscore/payloads/test",
    )
    with pytest.raises(InvalidRepositoryStateError, match="must not carry payload_ref"):
        manager.create_prepared_json_candidate_bundle(
            "store_workspace",
            operation_id="op-bad-descriptor-payload-ref",
            binding="workspace",
            payload=payload,
            parents=(parent,),
            driver=BadDescriptorDriver(payload_ref_claim),
        )


def test_world_transition_coordinator_lowers_valid_driver_ingress_to_prepared_draft(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"label": "workspace"}
    )
    payload = {"schema": "example/workspace", "n": 2}
    payload_digest = canonical_digest(payload)
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="command:workspace-json-revision",
                stable_observation={"payload_digest": payload_digest},
                mechanism="test.driver",
            ),
        ),
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="workspace-json-revision",
                payload=payload,
                observation_ids=("obs",),
                base_heads=(parent,),
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
            ),
        ),
    )

    prepared = manager._transition_coordinator.lower_driver_ingress_candidate(
        "store_workspace",
        operation_id="op-driver-ingress",
        binding="workspace",
        result=result,
        parents=(parent,),
        driver_id="test.driver",
        driver_version="v1",
    )

    assert prepared.transition.driver == "test.driver"
    assert prepared.transition.semantic_op == "workspace-json-revision"
    assert prepared.transition.evidence_digests == (prepared.evidence_records[0].evidence_digest(),)
    assert prepared.plan.content_digest == payload_digest
    assert prepared.payload_descriptor_claim == PayloadDescriptorClaim.for_json_payload(payload)


def test_world_transition_coordinator_persists_evidence_only_driver_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    stable_observation = {
        "command_operation_id": "op-command",
        "binding": "workspace",
        "path": "src/app.py",
        "op": "write_observed",
        "global_seq": 1,
    }
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="capture-event",
                evidence_kind="capture:filesystem-event",
                stable_observation=stable_observation,
                mechanism="test.driver",
                correlation_id="op-command",
                evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(stable_observation),
            ),
        )
    )

    persisted = manager.persist_driver_evidence_only(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=result,
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
        envelope_id="capture-events",
    )

    assert len(persisted.evidence_refs) == 1
    evidence = manager.world_store.resolve_evidence_ref(
        persisted.evidence_refs[0],
        expected_operation_id="op-command",
    )
    assert evidence.payload_digest == canonical_digest(stable_observation)
    assert evidence.evidence_kind == "capture:filesystem-event"
    envelope = manager.world_store.resolve_evidence_only_envelope(
        persisted.envelope_ref,
        expected_operation_id="op-command",
    )
    assert envelope.evidence_refs == persisted.evidence_refs
    assert envelope.evidence_kinds == ("capture:filesystem-event",)
    assert envelope.ingress_kind == "capture"
    assert not any(ref.startswith("refs/vcscore/candidates/") for ref in manager.world_store.repo.references)


def test_world_transition_coordinator_rejects_evidence_only_ingress_without_payload_descriptor(tmp_path) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(InvalidRepositoryStateError, match="evidence payload descriptor"):
        manager.persist_driver_evidence_only(
            "store_workspace",
            operation_id="op-command",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="capture-event",
                        evidence_kind="capture:filesystem-event",
                        stable_observation={"path": "src/app.py"},
                    ),
                )
            ),
            ingress_kind="capture",
            driver_id="test.driver",
            driver_version="v1",
        )
    assert not any(ref.startswith("refs/vcscore/evidence/") for ref in manager.world_store.repo.references)
    assert not any(ref.startswith("refs/vcscore/evidence-only/") for ref in manager.world_store.repo.references)


def test_world_transition_coordinator_rejects_evidence_only_ingress_with_transition(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}

    with pytest.raises(InvalidRepositoryStateError, match="must not include transition"):
        manager.persist_driver_evidence_only(
            "store_workspace",
            operation_id="op-command",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="obs",
                        evidence_kind="capture:filesystem-event",
                        stable_observation={"path": "src/app.py"},
                        evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(
                            {"path": "src/app.py"}
                        ),
                    ),
                ),
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-json-revision",
                        payload=payload,
                        observation_ids=("obs",),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
            ),
            ingress_kind="capture",
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_lowers_reduction_batch_citations(tmp_path) -> None:
    manager = _manager(tmp_path)
    raw_observation = {
        "command_operation_id": "op-command",
        "binding": "workspace",
        "path": "src/app.py",
        "op": "write_observed",
        "global_seq": 1,
    }
    raw = manager.persist_driver_evidence_only(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="raw",
                    evidence_kind="capture:filesystem-event",
                    stable_observation=raw_observation,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(raw_observation),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )
    batch = manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw")
    payload = {"schema": "example/workspace", "path": "src/app.py", "content_digest": "sha256:" + "1" * 64}
    proof_observation = {"reducer": "test.driver", "content_digest": payload["content_digest"]}

    bundle = manager.create_prepared_driver_candidate_bundle(
        "store_workspace",
        operation_id="op-reduce",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="proof",
                    evidence_kind="reduce:reduced-state-proof",
                    stable_observation=proof_observation,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(proof_observation),
                ),
            ),
            transitions=(
                TransitionDraft(
                    transition_id="transition",
                    semantic_op="workspace-capture-reduction",
                    payload=payload,
                    observation_ids=("proof",),
                    evidence_citation_ids=("raw-0",),
                    payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                ),
            ),
        ),
        ingress_kind="reduce",
        driver_id="test.driver",
        driver_version="v1",
        reduction_batch=batch,
    )

    assert len(bundle.preparation.evidence_refs) == 2
    assert bundle.preparation.cited_evidence_refs == (bundle.preparation.evidence_refs[1],)
    assert bundle.preparation.evidence_refs[-len(bundle.preparation.cited_evidence_refs) :] == (
        bundle.preparation.cited_evidence_refs
    )
    cited_record = manager.world_store.resolve_evidence_ref(bundle.preparation.evidence_refs[1])
    fresh_record = manager.world_store.resolve_evidence_ref(bundle.preparation.evidence_refs[0])
    assert cited_record.operation_id == "op-command"
    assert fresh_record.operation_id == "op-reduce"
    assert sorted(bundle.preparation.evidence_digests) == sorted(bundle.transition.evidence_digests)
    manager.store("store_workspace").validate_prepared_candidate(
        bundle.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )


def test_world_transition_coordinator_rejects_dangling_evidence_citation(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "path": "src/app.py"}
    observation = {"reducer": "test.driver"}

    with pytest.raises(InvalidRepositoryStateError, match="outside reduction batch"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-reduce",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="proof",
                        evidence_kind="reduce:reduced-state-proof",
                        stable_observation=observation,
                        evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(observation),
                    ),
                ),
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-capture-reduction",
                        payload=payload,
                        observation_ids=("proof",),
                        evidence_citation_ids=("raw-0",),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
            ),
            ingress_kind="reduce",
            driver_id="test.driver",
            driver_version="v1",
            reduction_batch=ReductionBatch(citations=()),
        )


def test_world_transition_coordinator_rejects_citation_digest_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    raw_observation = {"command_operation_id": "op-command", "binding": "workspace", "path": "src/app.py"}
    raw = manager.persist_driver_evidence_only(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="raw",
                    evidence_kind="capture:filesystem-event",
                    stable_observation=raw_observation,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(raw_observation),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )
    batch = manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw")
    bad_citation = replace(batch.citations[0], evidence_digest="sha256:" + "0" * 64)
    payload = {"schema": "example/workspace", "path": "src/app.py"}
    observation = {"reducer": "test.driver"}

    with pytest.raises(InvalidRepositoryStateError, match="evidence_digest"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-reduce",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="proof",
                        evidence_kind="reduce:reduced-state-proof",
                        stable_observation=observation,
                        evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(observation),
                    ),
                ),
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-capture-reduction",
                        payload=payload,
                        observation_ids=("proof",),
                        evidence_citation_ids=("raw-0",),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
            ),
            ingress_kind="reduce",
            driver_id="test.driver",
            driver_version="v1",
            reduction_batch=ReductionBatch(citations=(bad_citation,)),
        )


def test_world_transition_coordinator_rejects_duplicate_reduction_batch_citation_ids(tmp_path) -> None:
    manager = _manager(tmp_path)
    raw_observation = {"command_operation_id": "op-command", "binding": "workspace", "path": "src/app.py"}
    raw = manager.persist_driver_evidence_only(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="raw",
                    evidence_kind="capture:filesystem-event",
                    stable_observation=raw_observation,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(raw_observation),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )
    citation = manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw").citations[0]
    payload = {"schema": "example/workspace", "path": "src/app.py"}
    observation = {"reducer": "test.driver"}

    with pytest.raises(InvalidRepositoryStateError, match="duplicate citation_id"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-reduce",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="proof",
                        evidence_kind="reduce:reduced-state-proof",
                        stable_observation=observation,
                        evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(observation),
                    ),
                ),
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-capture-reduction",
                        payload=payload,
                        observation_ids=("proof",),
                        evidence_citation_ids=("raw-0",),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
            ),
            ingress_kind="reduce",
            driver_id="test.driver",
            driver_version="v1",
            reduction_batch=ReductionBatch(citations=(citation, citation)),
        )


def test_world_transition_coordinator_persists_diagnostic_driver_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    diagnostic = {
        "command_operation_id": "op-command",
        "reason": "unsupported_capture_event",
        "path": "src/app.py",
    }

    persisted = manager.persist_driver_diagnostics(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="diagnostic",
                    evidence_kind="diagnostic:capture:unsupported-event",
                    stable_observation=diagnostic,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(diagnostic),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )

    assert len(persisted.evidence_refs) == 1
    record = manager.world_store.resolve_evidence_ref(persisted.evidence_refs[0], expected_operation_id="op-command")
    assert record.evidence_kind == "diagnostic:capture:unsupported-event"
    assert record.payload_digest == canonical_digest(diagnostic)
    envelope = manager.world_store.resolve_evidence_only_envelope(persisted.envelope_ref)
    assert envelope.evidence_kinds == ("diagnostic:capture:unsupported-event",)
    assert not any(ref.startswith("refs/vcscore/candidates/") for ref in manager.world_store.repo.references)


def test_world_store_rejects_evidence_only_envelope_with_missing_evidence_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    missing_ref = EvidenceRef(
        ref=evidence_record_ref("op-command", "sha256:" + "1" * 64),
        evidence_digest="sha256:" + "2" * 64,
        record_digest="sha256:" + "1" * 64,
        payload_digest="sha256:" + "3" * 64,
    )
    envelope = EvidenceOnlyEnvelopeRecord(
        producer_operation_id="op-command",
        envelope_id="missing",
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        substrate_kind="filesystem",
        ingress_kind="capture",
        evidence_refs=(missing_ref,),
        evidence_kinds=("capture:filesystem-event",),
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence ref is missing"):
        manager.world_store.store_evidence_only_envelope(envelope)


def test_world_store_rejects_evidence_only_envelope_kind_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    persisted = manager.persist_driver_diagnostics(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="diagnostic",
                    evidence_kind="diagnostic:capture:unsupported-event",
                    stable_observation={"reason": "unsupported"},
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(
                        {"reason": "unsupported"}
                    ),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )
    forged = replace(
        persisted.envelope,
        envelope_id="forged-kind",
        evidence_kinds=("diagnostic:capture:other",),
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence kind disagrees"):
        manager.world_store.store_evidence_only_envelope(forged)


def test_world_store_rejects_evidence_only_envelope_duplicate_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    diagnostic = {"reason": "unsupported"}
    persisted = manager.persist_driver_diagnostics(
        "store_workspace",
        operation_id="op-command",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="diagnostic",
                    evidence_kind="diagnostic:capture:unsupported-event",
                    stable_observation=diagnostic,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(diagnostic),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id="test.driver",
        driver_version="v1",
    )
    forged = replace(
        persisted.envelope,
        envelope_id="duplicate",
        evidence_refs=(persisted.evidence_refs[0], persisted.evidence_refs[0]),
        evidence_kinds=(
            "diagnostic:capture:unsupported-event",
            "diagnostic:capture:unsupported-event",
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="duplicate evidence ref"):
        manager.world_store.store_evidence_only_envelope(forged)


def test_world_transition_coordinator_rejects_free_form_driver_diagnostics(tmp_path) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(InvalidRepositoryStateError, match="diagnostics must be Diagnostic"):
        manager.persist_driver_diagnostics(
            "store_workspace",
            operation_id="op-command",
            binding="workspace",
            result=DriverIngressResult(diagnostics=({"reason": "unsupported"},)),  # type: ignore[arg-type]
            ingress_kind="capture",
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_non_diagnostic_observation_on_diagnostic_path(tmp_path) -> None:
    manager = _manager(tmp_path)
    observation = {"reason": "unsupported"}

    with pytest.raises(InvalidRepositoryStateError, match="diagnostic evidence kinds"):
        manager.persist_driver_diagnostics(
            "store_workspace",
            operation_id="op-command",
            binding="workspace",
            result=DriverIngressResult(
                observations=(
                    ObservationDraft(
                        observation_id="raw",
                        evidence_kind="capture:filesystem-event",
                        stable_observation=observation,
                        evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(observation),
                    ),
                )
            ),
            ingress_kind="capture",
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_multi_transition_driver_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}
    transition = TransitionDraft(
        transition_id="transition-1",
        semantic_op="workspace-json-revision",
        payload=payload,
        observation_ids=(),
        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
    )

    with pytest.raises(InvalidRepositoryStateError, match="exactly one transition"):
        manager._transition_coordinator.lower_driver_ingress_candidate(
            "store_workspace",
            operation_id="op-driver-ingress",
            binding="workspace",
            result=DriverIngressResult(
                transitions=(
                    transition,
                    replace(transition, transition_id="transition-2"),
                )
            ),
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_driver_retention_hints_until_mapped(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}

    with pytest.raises(InvalidRepositoryStateError, match="retention hints"):
        manager._transition_coordinator.lower_driver_ingress_candidate(
            "store_workspace",
            operation_id="op-driver-ingress",
            binding="workspace",
            result=DriverIngressResult(
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-json-revision",
                        payload=payload,
                        observation_ids=(),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
                retention_hints=(RetentionHint(kind="selected-head-pin", target="abc123"),),
            ),
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_driver_selection_requirements_until_mapped(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}

    with pytest.raises(InvalidRepositoryStateError, match="selection requirements"):
        manager._transition_coordinator.lower_driver_ingress_candidate(
            "store_workspace",
            operation_id="op-driver-ingress",
            binding="workspace",
            result=DriverIngressResult(
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-json-revision",
                        payload=payload,
                        observation_ids=(),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
                selection_requirements=(
                    DriverSelectionRequirementDraft(
                        binding="workspace",
                        role="shepherd.WorkspaceRef",
                        selection_kind="new-candidate",
                        transition_id="transition",
                    ),
                ),
            ),
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_authority_bearing_driver_ingress_before_lowering(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="workspace-json-revision",
                payload=payload,
                observation_ids=(),
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                metadata={"evidence_ref": "refs/vcscore/evidence/op/abc123"},
            ),
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="reserved authority fields"):
        manager._transition_coordinator.lower_driver_ingress_candidate(
            "store_workspace",
            operation_id="op-driver-ingress",
            binding="workspace",
            result=result,
            driver_id="test.driver",
            driver_version="v1",
        )


def test_world_transition_coordinator_rejects_evidence_less_driver_revision_before_write(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}
    ref = "refs/heads/no-evidence"

    with pytest.raises(InvalidRepositoryStateError, match="requires at least one observation"):
        manager.create_prepared_driver_revision_bundle(
            "store_workspace",
            ref,
            operation_id="op-no-evidence",
            binding="workspace",
            result=DriverIngressResult(
                transitions=(
                    TransitionDraft(
                        transition_id="transition",
                        semantic_op="workspace-json-revision",
                        payload=payload,
                        observation_ids=(),
                        payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
                    ),
                ),
            ),
            driver_id="test.driver",
            driver_version="v1",
        )

    assert ref not in manager.store("store_workspace").repo.references
    assert not any(ref.startswith("refs/vcscore/evidence/") for ref in manager.world_store.repo.references)


def test_world_transition_coordinator_rejects_malformed_observation_before_evidence_write(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}
    payload_digest = canonical_digest(payload)
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="",
                stable_observation={"payload_digest": payload_digest},
            ),
        ),
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="workspace-json-revision",
                payload=payload,
                observation_ids=("obs",),
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
            ),
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence_kind"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-bad-observation",
            binding="workspace",
            result=result,
            driver_id="test.driver",
            driver_version="v1",
        )

    assert not any(ref.startswith("refs/vcscore/evidence/") for ref in manager.world_store.repo.references)


def test_world_transition_coordinator_rejects_empty_driver_identity_before_evidence_write(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {"schema": "example/workspace", "n": 2}
    payload_digest = canonical_digest(payload)
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="obs",
                evidence_kind="command:workspace-json-revision",
                stable_observation={"payload_digest": payload_digest},
            ),
        ),
        transitions=(
            TransitionDraft(
                transition_id="transition",
                semantic_op="workspace-json-revision",
                payload=payload,
                observation_ids=("obs",),
                payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(payload),
            ),
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="driver_id"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-empty-driver",
            binding="workspace",
            result=result,
            driver_id="",
            driver_version="v1",
        )
    with pytest.raises(InvalidRepositoryStateError, match="driver_version"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-empty-driver-version",
            binding="workspace",
            result=result,
            driver_id="test.driver",
            driver_version="",
        )
    with pytest.raises(InvalidRepositoryStateError, match="reserved authority ref"):
        manager.create_prepared_driver_candidate_bundle(
            "store_workspace",
            operation_id="op-reserved-driver",
            binding="workspace",
            result=result,
            driver_id="refs/vcscore/evidence/op/abc123",
            driver_version="v1",
        )

    assert not any(ref.startswith("refs/vcscore/evidence/") for ref in manager.world_store.repo.references)


def test_world_storage_manager_creates_existing_head_selection_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    session = _prepared_revision(
        manager,
        "store_session",
        "refs/checkpoints/S7",
        operation_id="op-create-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    head = manager.substrate_head("store_session", binding="session", head=session, role="shepherd.SessionState")

    evidence_ref = manager.create_existing_head_selection_evidence(
        operation_id="op-select-checkpoint",
        head=head,
        selection_kind="checkpoint",
        mechanism="unit-test",
    )
    evidence = manager.world_store.resolve_evidence_ref(
        evidence_ref,
        expected_operation_id="op-select-checkpoint",
    )

    assert evidence.evidence_kind == "checkpoint"
    assert evidence.ingress_kind == "coordinator"
    assert evidence.binding == "session"
    assert evidence.store_id == "store_session"
    assert evidence.substrate_kind == "shepherd.session_state"
    assert evidence.observed_head == session
    assert evidence.mechanism == "unit-test"


def test_world_storage_manager_plans_existing_head_selection(tmp_path) -> None:
    manager = _manager(tmp_path)
    session = _prepared_revision(
        manager,
        "store_session",
        "refs/checkpoints/S7",
        operation_id="op-create-planned-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    head = manager.substrate_head("store_session", binding="session", head=session, role="shepherd.SessionState")

    plan = manager.plan_existing_head_selection(
        operation_id="op-plan-checkpoint",
        head=head,
        selection_kind="checkpoint",
    )

    assert plan.binding == "session"
    assert plan.selected_head == session
    assert plan.selection_kind == "checkpoint"
    assert plan.retention_policy_requirements[0].target == session
    assert plan.evidence_refs
    assert manager.world_store.resolve_evidence_ref(plan.evidence_refs[0]).observed_head == session


def test_world_storage_manager_plans_unchanged_selection_only_from_input_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, input_world = _published_workspace_world(manager)
    workspace_head = manager.substrate_head(
        "store_workspace",
        binding="workspace",
        head=workspace,
        role="shepherd.WorkspaceRef",
    )

    plan = manager.plan_unchanged_selection(
        operation_id="op-plan-unchanged-workspace",
        head=workspace_head,
        input_world_oid=input_world,
    )

    assert plan.selection_kind == "unchanged"
    assert plan.selected_head == workspace
    assert plan.evidence_refs == ()

    moved_workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/moved",
        {"label": "workspace moved"},
        operation_id="op-bootstrap-moved-workspace",
        binding="workspace",
    )
    moved_head = manager.substrate_head(
        "store_workspace",
        binding="workspace",
        head=moved_workspace,
        role="shepherd.WorkspaceRef",
    )
    with pytest.raises(InvalidRepositoryStateError, match="must match input world head"):
        manager.plan_unchanged_selection(
            operation_id="op-plan-unchanged-moved-workspace",
            head=moved_head,
            input_world_oid=input_world,
        )

    session = _prepared_session_checkpoint(
        manager,
        "refs/checkpoints/S7",
        {"label": "session S7"},
        operation_id="op-create-session-not-in-input-world",
    )
    session_head = manager.substrate_head(
        "store_session",
        binding="session",
        head=session,
        role="shepherd.SessionState",
    )
    with pytest.raises(InvalidRepositoryStateError, match="missing from input world"):
        manager.plan_unchanged_selection(
            operation_id="op-plan-unchanged-missing-session",
            head=session_head,
            input_world_oid=input_world,
        )


def test_world_storage_manager_rejects_prepared_unchanged_selection_with_forged_head_identity(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, input_world = _published_workspace_world(manager)
    input_head = manager.read_world(input_world).snapshot.head_for("workspace")
    plan = manager.plan_unchanged_selection(
        operation_id="op-forged-unchanged-identity",
        head=manager.substrate_head(
            "store_workspace",
            binding="workspace",
            head=workspace,
            role="shepherd.WorkspaceRef",
        ),
        input_world_oid=input_world,
    )
    prepared = (
        OperationFinalBuilder("op-forged-unchanged-identity")
        .select_unchanged(plan=plan)
        .build_prepared(
            operation_kind="no-op",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=input_world,
            snapshot=_snapshot(replace(input_head, role="shepherd.OtherWorkspaceRef")),
            transition=_transition("op-forged-unchanged-identity", parents=[input_world]),
            parents=(input_world,),
        )
    )

    with pytest.raises(InvalidRepositoryStateError, match="input world head identity"):
        manager.create_world_from_prepared(prepared)

    manager.open_operation_journal(
        operation_id="op-forged-unchanged-identity",
        operation_kind="no-op",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=input_world,
    )
    with pytest.raises(InvalidRepositoryStateError, match="input world head identity"):
        manager.record_operation_prepared("op-forged-unchanged-identity", prepared=prepared)


def test_world_storage_manager_plans_world_ref_selection_retention(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace child"},
        operation_id="op-bootstrap-child-workspace-for-plan",
        binding="workspace",
    )
    child_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-child-world-for-world-ref-plan"),
        operation_final=_bootstrap_final(
            manager,
            "op-child-world-for-world-ref-plan",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    world_ref_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/child-plan",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id="op-import-child-world-ref-for-plan",
    )
    head = manager.substrate_head(
        "store_child_world_ref",
        binding="child",
        head=world_ref_head,
        role="vcscore.WorldRef",
    )

    plan = manager.plan_existing_head_selection(
        operation_id="op-parent-select-world-ref",
        head=head,
        selection_kind="import",
    )
    retention_by_kind = {requirement.kind: requirement for requirement in plan.retention_policy_requirements}

    assert retention_by_kind["selected-head-pin"].target == world_ref_head
    assert retention_by_kind["child-world-retention"].target == f"world:{child_world}"
    assert retention_by_kind["child-world-retention"].digest == child_snapshot.digest()

    prepared = (
        OperationFinalBuilder("op-parent-select-world-ref")
        .select_existing(plan=plan)
        .build_prepared(
            operation_kind="import-child-world",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=_snapshot(head),
            transition=_transition("op-parent-select-world-ref"),
        )
    )

    parent_world = manager.create_world_from_prepared(prepared)

    unchanged_plan = manager.plan_unchanged_selection(
        operation_id="op-parent-unchanged-world-ref",
        head=head,
        input_world_oid=parent_world,
    )
    unchanged_retention_by_kind = {
        requirement.kind: requirement for requirement in unchanged_plan.retention_policy_requirements
    }

    assert unchanged_retention_by_kind["selected-head-pin"].target == world_ref_head
    assert unchanged_retention_by_kind["child-world-retention"].target == f"world:{child_world}"
    assert unchanged_retention_by_kind["child-world-retention"].digest == child_snapshot.digest()


def test_world_storage_manager_rejects_prepared_world_with_missing_existing_head_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    prepared = _prepared_bootstrap_with_forged_selection_evidence(manager, "op-forged-prepared-bootstrap")

    with pytest.raises(InvalidRepositoryStateError, match="evidence ref"):
        manager.create_world_from_prepared(prepared)


def test_world_storage_manager_rejects_prepared_journal_with_missing_existing_head_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    operation_id = "op-forged-prepared-journal-bootstrap"
    prepared = _prepared_bootstrap_with_forged_selection_evidence(manager, operation_id)
    manager.open_operation_journal(
        operation_id=operation_id,
        operation_kind="bootstrap",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=None,
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence ref"):
        manager.record_operation_prepared(operation_id, prepared=prepared)


def test_world_storage_manager_rejects_cross_operation_existing_head_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    operation_id = "op-select-bootstrap-with-foreign-evidence"
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-create-bootstrap-for-foreign-evidence",
        binding="workspace",
    )
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    foreign_plan = manager.plan_existing_head_selection(
        operation_id="op-foreign-bootstrap-evidence",
        head=head,
        selection_kind="bootstrap",
    )
    plan = manager.plan_existing_head_selection(
        operation_id=operation_id,
        head=head,
        selection_kind="bootstrap",
    )
    prepared = (
        OperationFinalBuilder(operation_id)
        .select_existing(plan=plan)
        .build_prepared(
            operation_kind="bootstrap",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=_snapshot(head),
            transition=_transition(operation_id),
        )
    )
    evidence = HeadSelectionEvidence.from_json(dict(prepared.selection_evidence[0]))
    tampered_evidence = replace(evidence, evidence_refs=foreign_plan.evidence_refs)
    tampered = PreparedWorldOperation(
        operation_id=prepared.operation_id,
        operation_kind=prepared.operation_kind,
        target_ref=prepared.target_ref,
        input_world_oid=prepared.input_world_oid,
        snapshot=prepared.snapshot,
        transition=prepared.transition,
        candidate_tuples=prepared.candidate_tuples,
        candidate_refs=prepared.candidate_refs,
        candidate_commits=prepared.candidate_commits,
        candidate_outcomes=prepared.candidate_outcomes,
        head_selections=prepared.head_selections,
        selection_evidence=(tampered_evidence.to_json(),),
        selected=prepared.selected,
        parents=prepared.parents,
    )

    with pytest.raises(InvalidRepositoryStateError, match="existing-head selection evidence ref is invalid"):
        manager.create_world_from_prepared(tampered)


def test_world_storage_manager_rejects_prepared_world_with_tampered_candidate_outcome(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, parent_world = _published_workspace_world(manager)
    bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-tampered-candidate-outcome",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
    )
    head = manager.substrate_head(
        "store_workspace",
        binding="workspace",
        head=bundle.candidate.head,
        role="shepherd.WorkspaceRef",
    )
    prepared = (
        OperationFinalBuilder("op-tampered-candidate-outcome")
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id="op-tampered-candidate-outcome",
                selection=CandidateSelection.from_bundle(bundle),
                role="shepherd.WorkspaceRef",
            )
        )
        .build_prepared(
            operation_kind="shepherd.task",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=parent_world,
            snapshot=_snapshot(head),
            transition=_transition("op-tampered-candidate-outcome", parents=[parent_world]),
            parents=(parent_world,),
            candidate_refs=(bundle.candidate,),
        )
    )
    tampered = PreparedWorldOperation(
        operation_id=prepared.operation_id,
        operation_kind=prepared.operation_kind,
        target_ref=prepared.target_ref,
        input_world_oid=prepared.input_world_oid,
        snapshot=prepared.snapshot,
        transition=prepared.transition,
        candidate_tuples=prepared.candidate_tuples,
        candidate_refs=prepared.candidate_refs,
        candidate_commits=prepared.candidate_commits,
        candidate_outcomes=(replace(prepared.candidate_outcomes[0], transition_digest="sha256:" + "0" * 64),),
        head_selections=prepared.head_selections,
        selection_evidence=prepared.selection_evidence,
        selected=prepared.selected,
        parents=prepared.parents,
    )

    with pytest.raises(InvalidRepositoryStateError, match="candidate outcome transition_digest"):
        manager.create_world_from_prepared(tampered)


def test_world_storage_manager_existing_head_evidence_rejects_wrong_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _prepared_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        operation_id="op-create-workspace",
        binding="workspace",
        payload={"label": "workspace"},
        semantic_op="workspace-json-revision",
    )
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")

    with pytest.raises(InvalidRepositoryStateError, match="checkpoint selection requires original checkpoint"):
        manager.create_existing_head_selection_evidence(
            operation_id="op-select-workspace-as-checkpoint",
            head=head,
            selection_kind="checkpoint",
        )


def test_world_storage_manager_rejects_checkpoint_selection_without_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    session = manager.create_unsafe_unprepared_json_revision(
        "store_session", "refs/checkpoints/S7", {"label": "session S7"}
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_session", binding="session", head=session, role="shepherd.SessionState"),
        ),
        transition=_transition("op-legacy-checkpoint"),
        operation_final=_checkpoint_final(
            manager,
            "op-legacy-checkpoint",
            {"session": session},
            unsafe_evidence=True,
        ),
    )

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert any(
        "checkpoint selection requires prepared revision provenance" in issue.message for issue in report.issue_details
    )


def test_world_storage_manager_accepts_checkpoint_selection_with_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    session = _prepared_revision(
        manager,
        "store_session",
        "refs/checkpoints/S7",
        operation_id="op-create-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_session", binding="session", head=session, role="shepherd.SessionState"),
        ),
        transition=_transition("op-prepared-checkpoint"),
        operation_final=_checkpoint_final(manager, "op-prepared-checkpoint", {"session": session}),
    )

    assert manager.fsck_world(world_oid).ok


def test_world_storage_manager_rejects_import_selection_without_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/imported", {"label": "workspace import"}
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef"
            ),
        ),
        transition=_transition("op-legacy-import"),
        operation_final=_workspace_existing_head_final(
            manager,
            "op-legacy-import",
            workspace,
            selection_kind="import",
            unsafe_evidence=True,
        ),
    )

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert any(
        "import selection requires prepared revision provenance" in issue.message for issue in report.issue_details
    )


def test_world_storage_manager_accepts_import_selection_with_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _prepared_revision(
        manager,
        "store_workspace",
        "refs/heads/imported",
        operation_id="op-create-import",
        binding="workspace",
        payload={"label": "workspace import"},
        semantic_op="bootstrap",
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef"
            ),
        ),
        transition=_transition("op-prepared-import"),
        operation_final=_workspace_existing_head_final(
            manager,
            "op-prepared-import",
            workspace,
            selection_kind="import",
        ),
    )

    assert manager.fsck_world(world_oid).ok


def test_world_storage_manager_rejects_revert_selection_without_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    reverted = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/reverted", {"label": "workspace W1"}
    )
    selected_from = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/advanced",
        {"label": "workspace W2"},
        parents=(reverted,),
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=reverted, role="shepherd.WorkspaceRef"),
        ),
        transition=_transition("op-legacy-revert"),
        operation_final=_workspace_existing_head_final(
            manager,
            "op-legacy-revert",
            reverted,
            selection_kind="revert",
            selected_from=selected_from,
            unsafe_evidence=True,
        ),
    )

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert any(
        "revert selection requires prepared revision provenance" in issue.message for issue in report.issue_details
    )


def test_world_storage_manager_accepts_revert_selection_with_original_provenance(tmp_path) -> None:
    manager = _manager(tmp_path)
    reverted = _prepared_revision(
        manager,
        "store_workspace",
        "refs/heads/reverted",
        operation_id="op-create-revert-target",
        binding="workspace",
        payload={"label": "workspace W1"},
        semantic_op="revert",
    )
    selected_from = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/advanced",
        {"label": "workspace W2"},
        parents=(reverted,),
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=reverted, role="shepherd.WorkspaceRef"),
        ),
        transition=_transition("op-prepared-revert"),
        operation_final=_workspace_existing_head_final(
            manager,
            "op-prepared-revert",
            reverted,
            selection_kind="revert",
            selected_from=selected_from,
        ),
    )

    assert manager.fsck_world(world_oid).ok


def test_world_storage_manager_runs_shepherd_loop_and_reopens(tmp_path) -> None:
    manager = _manager(tmp_path)

    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-parent-workspace",
        binding="workspace",
    )
    s7 = _prepared_session_checkpoint(manager, "refs/checkpoints/S7", {"label": "session S7"})
    s19 = _bootstrap_revision(
        manager,
        "store_session",
        "refs/heads/parent",
        {"label": "session S19"},
        operation_id="op-bootstrap-parent-session",
        binding="session",
        parents=(s7,),
    )

    p0_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
        manager.substrate_head("store_session", binding="session", head=s19, role="shepherd.SessionState"),
    )
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-parent-initial"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-initial",
            {"workspace": w42, "session": s19},
            stores_by_binding={"workspace": "store_workspace", "session": "store_session"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)

    c0_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
        manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
    )
    c0 = manager.create_unsafe_world(
        snapshot=c0_snapshot,
        transition=_transition("op-child-fork", parents=[p0]),
        operation_final=_checkpoint_final(manager, "op-child-fork", {"workspace": w42, "session": s7}),
        parents=(p0,),
    )
    w43, w43_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-child",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
    )
    s8, s8_commit = _prepared_candidate(
        manager,
        "store_session",
        operation_id="op-child",
        binding="session",
        payload={"label": "session S8", "requires": [{"binding": "workspace", "head": w42}]},
        parents=(s7,),
    )
    c1_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
        manager.substrate_head("store_session", binding="session", head=s8.head, role="shepherd.SessionState"),
    )
    c1 = manager.create_unsafe_world(
        snapshot=c1_snapshot,
        transition=_transition("op-child", parents=[c0]),
        operation_final=_operation_final(
            "op-child",
            {"workspace": w43.head, "session": s8.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    w43_commit,
                    final_operation_id="op-child",
                ),
                _candidate_outcome(
                    manager,
                    "store_session",
                    s8_commit,
                    final_operation_id="op-child",
                ),
            ],
            candidate_commits=[w43_commit, s8_commit],
        ),
        parents=(c0,),
    )

    p1_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
        manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
    )
    p1 = manager.create_unsafe_world(
        snapshot=p1_snapshot,
        transition=_transition(
            "op-parent-merge",
            parents=[p0, c1],
            input_world=p0,
            changes={
                "workspace": {"from": w42, "to": w43.head, "policy": "take-child"},
                "session": {"from": s19, "to": s7, "policy": "pin-prior"},
            },
        ),
        operation_final=attach_selection_evidence_ref(
            operation_final_with_head_selections(
                "op-parent-merge",
                {"workspace": w43.head, "session": s7},
                outcomes=[
                    _candidate_outcome(
                        manager,
                        "store_workspace",
                        w43_commit,
                        final_operation_id="op-parent-merge",
                        producer_world_oid=c1,
                    )
                ],
                candidate_commits=[w43_commit],
                selection_kinds={"session": "checkpoint"},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                manager.world_store,
                operation_id="op-parent-merge",
                binding="session",
                store=manager.store("store_session"),
                head=s7,
                evidence_kind="checkpoint",
            ),
        ),
        parents=(p0, c1),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    ground = manager.read_world(DEFAULT_GROUND_REF)
    report = manager.fsck_world(p1)
    reopened = _manager(tmp_path)
    reopened_ground = reopened.read_world(DEFAULT_GROUND_REF)

    assert report.ok
    assert ground.oid == p1
    assert ground.snapshot.head_for("workspace").head == w43.head
    assert ground.snapshot.head_for("session").head == s7
    assert ground.snapshot.head_for("session").head not in {s8.head, s19}
    assert ground.manifest["locator_hints"] == manager.locator_hints()
    assert "store_locator" not in ground.manifest["snapshot"]["workspace"]
    assert reopened_ground.oid == p1
    assert reopened_ground.snapshot == ground.snapshot


def test_world_storage_manager_records_recursive_task_loop_world_ref_and_trace(tmp_path) -> None:
    manager = _task_loop_manager(tmp_path)
    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-parent-task-workspace",
        binding="workspace",
    )
    s7 = _prepared_session_checkpoint(manager, "refs/checkpoints/S7", {"label": "session S7"})
    s19 = _bootstrap_revision(
        manager,
        "store_session",
        "refs/heads/parent",
        {"label": "session S19"},
        operation_id="op-bootstrap-parent-task-session",
        binding="session",
        parents=(s7,),
    )
    t0 = _bootstrap_revision(
        manager,
        "store_trace",
        "refs/heads/parent",
        {"label": "parent trace T0"},
        operation_id="op-bootstrap-parent-task-trace",
        binding="trace",
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s19, role="shepherd.SessionState"),
            manager.substrate_head("store_trace", binding="trace", head=t0, role="shepherd.TraceState"),
        ),
        transition=_transition("op-parent-task-initial"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-task-initial",
            {"workspace": w42, "session": s19, "trace": t0},
            stores_by_binding={"workspace": "store_workspace", "session": "store_session", "trace": "store_trace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    c0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        ),
        transition=_transition("op-child-task-call", parents=[p0]),
        operation_final=_checkpoint_final(manager, "op-child-task-call", {"workspace": w42, "session": s7}),
        parents=(p0,),
    )
    w43, w43_commit = manager.create_prepared_json_candidate(
        "store_workspace",
        operation_id="op-child-task-iteration",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        semantic_op="child-task-workspace-edit",
    )
    c1_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
        manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
    )
    c1 = manager.create_unsafe_world(
        snapshot=c1_snapshot,
        transition=_transition("op-child-task-iteration", parents=[c0]),
        operation_final=_operation_final(
            "op-child-task-iteration",
            {"workspace": w43.head, "session": s7},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    w43_commit,
                    final_operation_id="op-child-task-iteration",
                )
            ],
            candidate_commits=[w43_commit],
        ),
        parents=(c0,),
    )
    child_call_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/child-call",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=c1,
            snapshot_digest=c1_snapshot.digest(),
        ).to_json(),
        operation_id="op-import-child-call",
        binding="child_call",
    )
    t1, t1_commit = manager.create_prepared_json_candidate(
        "store_trace",
        operation_id="op-parent-record-child",
        binding="trace",
        payload={"label": "parent trace T1", "child_world": c1, "workspace_head": w43.head},
        parents=(t0,),
        semantic_op="task-trace-append",
    )
    child_call_retention = (
        RetentionPolicyRequirement(kind="selected-head-pin", target=child_call_head),
        RetentionPolicyRequirement(kind="child-world-retention", target=f"world:{c1}", digest=c1_snapshot.digest()),
    )
    parent_final = operation_final_with_head_selections(
        "op-parent-record-child",
        {"workspace": w43.head, "session": s7, "trace": t1.head, "child_call": child_call_head},
        outcomes=[
            _candidate_outcome(
                manager,
                "store_workspace",
                w43_commit,
                final_operation_id="op-parent-record-child",
                producer_world_oid=c1,
            ),
            _candidate_outcome(
                manager,
                "store_trace",
                t1_commit,
                final_operation_id="op-parent-record-child",
            ),
        ],
        candidate_commits=[w43_commit, t1_commit],
        store_ids={"child_call": "store_child_world_ref"},
        resource_ids={"child_call": "world-ref:child-task"},
        selection_kinds={"session": "checkpoint", "child_call": "import"},
        retention_policy_requirements={"child_call": child_call_retention},
    )
    parent_final = attach_selection_evidence_ref(
        parent_final,
        binding="session",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id="op-parent-record-child",
            binding="session",
            store=manager.store("store_session"),
            head=s7,
            evidence_kind="checkpoint",
        ),
    )
    parent_final = attach_selection_evidence_ref(
        parent_final,
        binding="child_call",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id="op-parent-record-child",
            binding="child_call",
            store=manager.store("store_child_world_ref"),
            head=child_call_head,
            evidence_kind="import",
        ),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
            manager.substrate_head("store_trace", binding="trace", head=t1.head, role="shepherd.TraceState"),
            manager.substrate_head(
                "store_child_world_ref",
                binding="child_call",
                head=child_call_head,
                role="vcscore.WorldRef",
            ),
        ),
        transition=_transition("op-parent-record-child", parents=[p0, c1], input_world=p0),
        operation_final=parent_final,
        parents=(p0, c1),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    manager.store("store_workspace").repo.references[w43.ref].delete()
    reopened = _task_loop_manager(tmp_path)
    report = reopened.fsck_world(p1)

    assert report.ok
    assert reopened.read_world(DEFAULT_GROUND_REF).snapshot.head_for("workspace").head == w43.head
    assert reopened.read_world(DEFAULT_GROUND_REF).snapshot.head_for("session").head == s7
    assert reopened.read_world(DEFAULT_GROUND_REF).snapshot.head_for("trace").head == t1.head
    assert (
        world_pin_ref(manager.world_store.world_store_id, p1, "child_call")
        in reopened.store("store_child_world_ref").repo.references
    )
    assert child_world_retention_ref(p1, "root.workspace.producer") in reopened.world_store.repo.references


def test_world_storage_manager_preserves_published_parent_world_pins_after_authority_advances(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-parent-workspace",
        binding="workspace",
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-parent-initial"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-initial",
            {"workspace": w42},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")

    w43, w43_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-parent-advance",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-parent-advance", parents=[p0]),
        operation_final=_operation_final(
            "op-parent-advance",
            {"workspace": w43.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    w43_commit,
                    final_operation_id="op-parent-advance",
                )
            ],
            candidate_commits=[w43_commit],
        ),
        parents=(p0,),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    p0_report = manager.fsck_world(p0)
    assert p0_report.ok
    assert p0_report.pin_classification["published"] == (p0_pin,)
    assert manager.cleanup_orphan_pins(p0) == ()
    assert manager.store("store_workspace").repo.references[p0_pin].target == pygit2.Oid(hex=w42)

    manager.store("store_workspace").repo.references[p0_pin].delete()
    p1_report = manager.fsck_world(p1)

    assert not p1_report.ok
    assert p0_pin in p1_report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_publishes_long_parent_chains_without_recursive_depth_failure(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-loop-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    parent: str | None = None
    first_world: str | None = None

    for index in range(72):
        operation_id = f"op-loop-{index}"
        world_oid = manager.create_unsafe_world(
            snapshot=snapshot,
            transition=_transition(operation_id, parents=[] if parent is None else [parent]),
            operation_final=(
                _bootstrap_final(
                    manager,
                    operation_id,
                    {"workspace": workspace},
                    stores_by_binding={"workspace": "store_workspace"},
                )
                if parent is None
                else _operation_final(operation_id, {"workspace": workspace})
            ),
            parents=() if parent is None else (parent,),
        )
        assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=parent)
        first_world = first_world or world_oid
        parent = world_oid

    assert parent is not None
    assert first_world is not None
    assert manager.read_world(DEFAULT_GROUND_REF).oid == parent
    assert manager.fsck_world(parent).ok

    first_pin = world_pin_ref(manager.world_store.world_store_id, first_world, "workspace")
    manager.store("store_workspace").repo.references[first_pin].delete()
    report = manager.fsck_world(parent)

    assert not report.ok
    assert first_pin in report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_records_and_checks_world_retention_receipts(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-receipt-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-receipt"),
        operation_final=_bootstrap_final(
            manager,
            "op-receipt",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, world_oid)
    assert receipt_ref in manager.world_store.repo.references
    assert manager.fsck_world(world_oid).ok

    manager.world_store.repo.references[receipt_ref].delete()
    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert "missing_retention_receipt" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_rejects_retention_receipt_with_wrong_retained_refs(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-receipt-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-receipt-retained-refs"),
        operation_final=_bootstrap_final(
            manager,
            "op-receipt-retained-refs",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, world_oid)
    receipt = _read_receipt_payload(manager, receipt_ref)
    receipt["retained_refs"] = []
    receipt["receipt_digest"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _write_receipt_payload(manager, receipt_ref, receipt)

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert "corrupt_retention_receipt" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_rejects_retention_receipt_with_wrong_typed_retained_refs(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-receipt-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-receipt-typed-retained"),
        operation_final=_bootstrap_final(
            manager,
            "op-receipt-typed-retained",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, world_oid)
    receipt = _read_receipt_payload(manager, receipt_ref)
    receipt["retained"] = []
    receipt["receipt_digest"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _write_receipt_payload(manager, receipt_ref, receipt)

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert "corrupt_retention_receipt" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_fsck_reports_missing_selection_evidence_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _prepared_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        operation_id="op-import-workspace-evidence-ref-head",
        binding="workspace",
        payload={"label": "workspace"},
        semantic_op="import",
    )
    evidence_ref = selection_evidence_ref(
        manager.world_store,
        operation_id="op-evidence-ref",
        binding="workspace",
        store=manager.store("store_workspace"),
        head=workspace,
        evidence_kind="import",
    )
    final = attach_selection_evidence_ref(
        operation_final_with_head_selections(
            "op-evidence-ref",
            {"workspace": workspace},
            selection_kinds={"workspace": "import"},
        ),
        binding="workspace",
        evidence_ref=evidence_ref,
    )
    world_oid = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-evidence-ref"),
        operation_final=final,
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, world_oid)

    manager.world_store.repo.references[evidence_ref.ref].delete()
    structural_report = manager.fsck_world(world_oid)
    report = manager.fsck_world_deep(world_oid)

    assert structural_report.ok
    assert not report.ok
    assert "missing_evidence_ref" in {issue.code for issue in report.issue_details}
    receipt = _read_receipt_payload(manager, receipt_ref)
    assert receipt["retained_refs"] == [world_pin_ref(manager.world_store.world_store_id, world_oid, "workspace")]
    assert {"kind": "evidence-ref", "ref": evidence_ref.ref, "digest": evidence_ref.evidence_digest} in receipt[
        "retained"
    ]


def test_world_storage_manager_cleans_orphan_retention_receipt_after_cas_failure(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-cas-workspace",
        binding="workspace",
    )
    p0_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-publish"),
        operation_final=_bootstrap_final(
            manager,
            "op-publish",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    selected, selected_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-lost-cas-receipt",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace", binding="workspace", head=selected.head, role="shepherd.WorkspaceRef"
            )
        ),
        transition=_transition("op-lost-cas-receipt", parents=[p0]),
        operation_final=_operation_final(
            "op-lost-cas-receipt",
            {"workspace": selected.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    selected_commit,
                    final_operation_id="op-lost-cas-receipt",
                )
            ],
            candidate_commits=[selected_commit],
        ),
        parents=(p0,),
    )
    intervening = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-lost-cas-intervening",
    )
    assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=intervening, input_world_oid=p0)

    assert not manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=p1, input_world_oid=p0)
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, p1)
    assert receipt_ref in manager.world_store.repo.references

    deleted = manager.cleanup_orphan_pins(p1)

    assert receipt_ref in deleted
    assert receipt_ref not in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_missing_authority_pin(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-missing-authority-pin",
    )

    manager.store("store_workspace").repo.references[p0_pin].delete()

    # Trust-by-default (Part A): the detector still flags the broken prior-lineage pin, but it is off
    # the publish hot path, so the publish proceeds and writes its own new-world retention.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*missing retained refs"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    assert (
        world_pin_ref(manager.world_store.world_store_id, p1, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p1) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_missing_authority_ancestor_pin(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-authority-ancestor-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-authority-ancestor-p2",
    )

    manager.store("store_workspace").repo.references[p0_pin].delete()

    # Trust-by-default (Part A): the detector still flags the broken ancestor pin; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*missing retained refs"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)

    assert (
        world_pin_ref(manager.world_store.world_store_id, p2, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p2) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_missing_authority_ancestor_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _prepared_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        operation_id="op-import-authority-evidence-p0-head",
        binding="workspace",
        payload={"label": "workspace"},
        semantic_op="import",
    )
    evidence_ref = selection_evidence_ref(
        manager.world_store,
        operation_id="op-authority-evidence-p0",
        binding="workspace",
        store=manager.store("store_workspace"),
        head=workspace,
        evidence_kind="import",
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-authority-evidence-p0"),
        operation_final=attach_selection_evidence_ref(
            operation_final_with_head_selections(
                "op-authority-evidence-p0",
                {"workspace": workspace},
                selection_kinds={"workspace": "import"},
            ),
            binding="workspace",
            evidence_ref=evidence_ref,
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-authority-evidence-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-authority-evidence-p2",
    )

    manager.world_store.repo.references[evidence_ref.ref].delete()
    report = manager.fsck_world_deep(p1)

    assert not report.ok
    assert "missing_evidence_ref" in {issue.code for issue in report.issue_details}
    # Trust-by-default (Part A): the detector still flags the missing ancestor evidence; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*evidence ref is missing"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)
    assert (
        world_pin_ref(manager.world_store.world_store_id, p2, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p2) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_always_checks_target_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-target-ref-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-target-ref-p2",
    )
    p1_pin = world_pin_ref(manager.world_store.world_store_id, p1, "workspace")
    manager.store("store_workspace").repo.references[p1_pin].delete()

    # Trust-by-default (Part A): the detector still checks the target ref (DEFAULT_GROUND_REF) even
    # when an unrelated extra authority is supplied; but it is off the hot path, so publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*missing retained refs"):
        manager._validate_authority_retention_preflight(
            (DEFAULT_GROUND_REF, "refs/vcscore/unrelated-authority"),
            allow_same_resource_alias=False,
        )
    assert _publish_world(
        manager,
        ref=DEFAULT_GROUND_REF,
        world_oid=p2,
        expected_oid=p1,
        authority_refs=("refs/vcscore/unrelated-authority",),
    )

    assert (
        world_pin_ref(manager.world_store.world_store_id, p2, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p2) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_corrupt_authority_pin(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-corrupt-authority-pin",
    )
    wrong = manager.create_unsafe_unprepared_json_revision("store_workspace", "refs/heads/wrong", {"label": "wrong"})
    create_or_update_reference(
        manager.store("store_workspace").repo,
        p0_pin,
        pygit2.Oid(hex=wrong),
        force=True,
    )

    # Trust-by-default (Part A): the detector still flags the corrupt prior-lineage pin; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*corrupt retained refs"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    assert (
        world_pin_ref(manager.world_store.world_store_id, p1, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p1) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_missing_authority_receipt(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-missing-authority-receipt",
    )
    manager.world_store.repo.references[world_retention_receipt_ref(DEFAULT_GROUND_REF, p0)].delete()

    # Trust-by-default (Part A): the detector still flags the missing prior-lineage receipt; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*missing retention receipt"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    assert (
        world_pin_ref(manager.world_store.world_store_id, p1, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p1) in manager.world_store.repo.references


def test_world_storage_manager_publish_preflight_rejects_missing_authority_ancestor_receipt(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-authority-receipt-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-authority-receipt-p2",
    )
    manager.world_store.repo.references[world_retention_receipt_ref(DEFAULT_GROUND_REF, p0)].delete()

    # Trust-by-default (Part A): the detector still flags the missing ancestor receipt; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*missing retention receipt"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)

    assert (
        world_pin_ref(manager.world_store.world_store_id, p2, "workspace")
        in manager.store("store_workspace").repo.references
    )
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, p2) in manager.world_store.repo.references


def test_world_storage_manager_fsck_reports_missing_authority_ancestor_receipt(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-fsck-ancestor-receipt",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    manager.world_store.repo.references[world_retention_receipt_ref(DEFAULT_GROUND_REF, p0)].delete()

    report = manager.fsck_world_deep(p1)
    structural_report = manager.fsck_world(p1, mode="structural")

    assert not report.ok
    assert "missing_retention_receipt" in {issue.code for issue in report.issue_details}
    assert p0 in {issue.world_oid for issue in report.issue_details}
    assert structural_report.ok


def test_world_storage_manager_fsck_reports_and_recovers_missing_authority_ancestor_pin(tmp_path) -> None:
    # Part B: trust-by-default takes the prior-lineage retention check off the publish hot path, but
    # the integrity did not vanish -- deep fsck of the tip still detects a missing ANCESTOR pin on
    # demand (the tip's authority closure transitively includes the ancestor lineage). repin_world_retention
    # repairs it by re-pinning that closure from the immutable world commits.
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-fsck-recover-ancestor-pin-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-fsck-recover-ancestor-pin-p2",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)
    manager.store("store_workspace").repo.references[p0_pin].delete()

    report = manager.fsck_world_deep(p2)
    assert not report.ok
    assert "missing_selected_head_pins" in {issue.code for issue in report.issue_details}

    manager.repin_world_retention(p2)

    assert manager.fsck_world_deep(p2).ok
    assert p0_pin in manager.store("store_workspace").repo.references


def test_world_storage_manager_fsck_reports_corrupt_authority_pin(tmp_path) -> None:
    # Part B: deep fsck still flags a corrupt prior-lineage pin on demand (off the publish hot path).
    # Repair stays cautious for corruption (inspect before re-pin), so this asserts detection only.
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p0_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-fsck-corrupt-pin",
    )
    wrong = manager.create_unsafe_unprepared_json_revision("store_workspace", "refs/heads/wrong-partb", {"label": "x"})
    create_or_update_reference(manager.store("store_workspace").repo, p0_pin, pygit2.Oid(hex=wrong), force=True)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    report = manager.fsck_world_deep(p1)
    assert not report.ok
    assert "corrupt_selected_head_pins" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_fsck_reports_lost_fork_origin_protection(tmp_path) -> None:
    # Part B: deep fsck still detects a fork whose parent authority was rewritten out from under it
    # (lost inherited-world protection) on demand. This is an authority-level recovery, not a missing
    # pin, so repin_world_retention does not apply -- detection only.
    scope_ref = "refs/vcscore/scopes/partb-rewrite"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p0,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    unrelated = _workspace_root_world(manager, workspace_head=workspace, operation_id="op-partb-unrelated-root")
    create_or_update_reference(manager.world_store.repo, DEFAULT_GROUND_REF, pygit2.Oid(hex=unrelated), force=True)
    child = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-partb-fork-child",
    )
    assert manager.advance_world_ref(ref=scope_ref, world_oid=child, input_world_oid=p0)

    report = manager.fsck_world_deep(child, authority_refs=(scope_ref,))
    assert not report.ok
    assert "corrupt_fork_origin_receipt" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_rejects_corrupt_authority_ancestor_receipt(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-corrupt-ancestor-receipt-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p0_receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, p0)
    receipt = _read_receipt_payload(manager, p0_receipt_ref)
    receipt["retained_refs"] = []
    receipt["receipt_digest"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    _write_receipt_payload(manager, p0_receipt_ref, receipt)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-corrupt-ancestor-receipt-p2",
    )

    report = manager.fsck_world_deep(p1)

    assert not report.ok
    assert "corrupt_retention_receipt" in {issue.code for issue in report.issue_details}
    assert p0 in {issue.world_oid for issue in report.issue_details}
    # Trust-by-default (Part A): the detector still flags the corrupt ancestor receipt; publish proceeds.
    with pytest.raises(InvalidRepositoryStateError, match=r"preflight.*retention receipt.*disagrees"):
        manager._validate_authority_retention_preflight((DEFAULT_GROUND_REF,), allow_same_resource_alias=False)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)


def test_world_storage_manager_publish_computes_closure_a_constant_number_of_times(tmp_path, monkeypatch) -> None:
    """Count contract (trust-by-default, Part A): publishing world N computes the publish-retention
    closure a CONSTANT number of times, independent of how many worlds already sit on the authority
    — not O(N), as the prior-lineage preflight re-walk made it (2N-1 closure computations, Sigma=N^2).

    This is the machine-independent encoding of the ~8x publish-slope win: a reintroduced per-lineage
    re-walk would make the deep-lineage publish compute the closure far more often than the shallow
    one, and fail here. Pairs with the AST boundary guard in test_publish_preflight_boundary.py.
    """
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)

    counter = {"n": 0}
    original_closure = WorldStorageManager.compute_publish_retention_closure

    def _counting_closure(self, oid):
        counter["n"] += 1
        return original_closure(self, oid)

    monkeypatch.setattr(WorldStorageManager, "compute_publish_retention_closure", _counting_closure)

    calls_per_publish: list[int] = []
    parent, parent_workspace = p0, workspace
    for index in range(8):
        child = _workspace_advance_world(
            manager,
            parent_world_oid=parent,
            parent_workspace_head=parent_workspace,
            operation_id=f"op-closure-count-{index}",
        )
        counter["n"] = 0  # isolate the publish itself (candidate build above is not counted)
        assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=child, input_world_oid=parent)
        calls_per_publish.append(counter["n"])
        parent = child
        parent_workspace = manager.read_world(child).snapshot.head_for("workspace").head

    # Independent of lineage depth: the deepest publish computes the closure the same number of times
    # as the shallowest. Observed: exactly 1. The prior-lineage preflight made this grow as 2N-1.
    assert calls_per_publish[-1] == calls_per_publish[0], calls_per_publish
    assert max(calls_per_publish) <= 2, calls_per_publish


def test_world_storage_manager_preflight_does_not_require_receipts_for_non_input_git_parents(tmp_path) -> None:
    producer_ref = "refs/vcscore/producers/non-input"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    producer_world = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-non-input-parent-producer"),
        operation_final=_bootstrap_final(
            manager,
            "op-non-input-parent-producer",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=producer_ref, world_oid=producer_world, expected_oid=None)
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-non-input-parent-merge", parents=[p0, producer_world], input_world=p0),
        operation_final=_operation_final("op-non-input-parent-merge", {"workspace": workspace}),
        parents=(p0, producer_world),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=workspace,
        operation_id="op-after-non-input-parent",
    )

    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, producer_world) not in manager.world_store.repo.references
    assert world_retention_receipt_ref(producer_ref, producer_world) in manager.world_store.repo.references
    assert manager.fsck_world(p1).ok
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)


def test_world_storage_manager_fsck_treats_root_non_input_git_parents_as_audit_only(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    audit_parent = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-audit-parent"),
        operation_final=_bootstrap_final(
            manager,
            "op-audit-parent",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-audit-parent-merge", parents=[p0, audit_parent], input_world=p0),
        operation_final=_operation_final("op-audit-parent-merge", {"workspace": workspace}),
        parents=(p0, audit_parent),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    report = manager.fsck_world(p1)

    assert report.ok
    audit_parent_pin = world_pin_ref(manager.world_store.world_store_id, audit_parent, "workspace")
    assert audit_parent_pin not in report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_forked_scope_inherits_authority_retention(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/child-loop"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-fork-scope-p1",
    )

    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p1,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    assert world_fork_origin_receipt_ref(scope_ref) in manager.world_store.repo.references
    assert manager.fsck_world(p1, authority_refs=(scope_ref,)).ok

    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-fork-scope-p2",
    )
    assert manager.advance_world_ref(ref=scope_ref, world_oid=p2, input_world_oid=p1)
    assert manager.fsck_world(p2, authority_refs=(scope_ref,)).ok


def test_world_storage_manager_forked_scope_publishes_first_new_world_with_non_diverged_receipt(tmp_path) -> None:
    """Regression: production ``_fork_v2_scope_world`` writes a fork-origin
    receipt with ``first_world_oid == forked_from_world_oid`` (the scope has
    not diverged at fork time). The fork-origin lineage check (now the
    fsck-only detector, off the publish hot path) must not reject the scope's
    first new-world publication just because the receipt records no divergence.
    """
    scope_ref = "refs/vcscore/scopes/cr2"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)

    # Production fork shape: scope starts at the same world as the parent
    # (``world_oid == forked_from_world_oid``). The receipt therefore records
    # ``first_world_oid == forked_from_world_oid``.
    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p0,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    receipt_ref = world_fork_origin_receipt_ref(scope_ref)
    assert receipt_ref in manager.world_store.repo.references

    # First new world on the scope. Before the fix, the preflight at
    # ``_authority_lineage_segments`` raised "fork origin first_world_oid is
    # not in local authority lineage" because ``first_world_oid`` (== p0,
    # the fork base) is never in ``lineage[:fork_base_index]``.
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-cr2-first-new-world",
    )
    assert manager.advance_world_ref(ref=scope_ref, world_oid=p1, input_world_oid=p0)
    assert manager.fsck_world(p1, authority_refs=(scope_ref,)).ok


def test_world_storage_manager_zero_local_fork_accepts_child_local_advances(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/zero-local"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)

    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p0,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-zero-local-child-p1",
    )
    assert manager.advance_world_ref(ref=scope_ref, world_oid=p1, input_world_oid=p0)
    p1_workspace = manager.read_world(p1).snapshot.head_for("workspace").head
    p2 = _workspace_advance_world(
        manager,
        parent_world_oid=p1,
        parent_workspace_head=p1_workspace,
        operation_id="op-zero-local-child-p2",
    )

    assert manager.advance_world_ref(ref=scope_ref, world_oid=p2, input_world_oid=p1)
    assert manager.fsck_world(p2, authority_refs=(scope_ref,)).ok


def test_world_storage_manager_zero_local_fork_survives_parent_advance(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/zero-local-parent-advance"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p0,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    parent_p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-zero-local-parent-p1",
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=parent_p1, expected_oid=p0)
    child_p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-zero-local-child-after-parent",
    )

    assert manager.advance_world_ref(ref=scope_ref, world_oid=child_p1, input_world_oid=p0)
    assert manager.fsck_world(child_p1, authority_refs=(scope_ref,)).ok


def test_world_storage_manager_zero_local_fork_rejects_parent_rewrite_losing_base(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/zero-local-parent-rewrite"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p0,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    unrelated = _workspace_root_world(manager, workspace_head=workspace, operation_id="op-unrelated-parent-root")
    create_or_update_reference(manager.world_store.repo, DEFAULT_GROUND_REF, pygit2.Oid(hex=unrelated), force=True)
    child_p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-zero-local-child-after-rewrite",
    )

    # Trust-by-default (Part A): the detector still flags that GROUND no longer protects the inherited
    # base, but it is off the hot path, so the scope advance proceeds (fsck surfaces it on demand, Part B).
    with pytest.raises(InvalidRepositoryStateError, match="fork origin authority no longer protects inherited world"):
        manager._validate_authority_retention_preflight((scope_ref,), allow_same_resource_alias=False)
    assert manager.advance_world_ref(ref=scope_ref, world_oid=child_p1, input_world_oid=p0)


def test_world_storage_manager_fork_origin_missing_base_reports_corrupt_receipt(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/corrupt-fork-base"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    unrelated = _workspace_root_world(manager, workspace_head=workspace, operation_id="op-corrupt-fork-base-root")
    create_or_update_reference(manager.world_store.repo, scope_ref, pygit2.Oid(hex=p0), force=True)
    manager.write_world_fork_origin_receipt(
        authority_ref=scope_ref,
        first_world_oid=p0,
        forked_from_authority_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=unrelated,
    )

    with pytest.raises(InvalidRepositoryStateError, match="forked_from_world_oid is not in child input lineage"):
        manager._validate_authority_retention_preflight((scope_ref,), allow_same_resource_alias=False)
    report = manager.fsck_world(p0, authority_refs=(scope_ref,), mode="deep")

    assert not report.ok
    assert "corrupt_fork_origin_receipt" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_forked_scope_reports_missing_inherited_retention(tmp_path) -> None:
    scope_ref = "refs/vcscore/scopes/missing-inherited"
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-fork-missing-inherited",
    )
    assert manager.fork_world_ref(
        ref=scope_ref,
        world_oid=p1,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    inherited_pin = world_pin_ref(manager.world_store.world_store_id, p0, "workspace")

    manager.store("store_workspace").repo.references[inherited_pin].delete()
    report = manager.fsck_world(p1, authority_refs=(scope_ref,))

    assert not report.ok
    assert inherited_pin in report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_retains_recursive_child_world_selected_heads(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-recursive-workspace",
        binding="workspace",
    )
    s7 = _prepared_session_checkpoint(manager, "refs/checkpoints/S7", {"label": "session S7"})
    s19 = _bootstrap_revision(
        manager,
        "store_session",
        "refs/heads/parent",
        {"label": "session S19"},
        operation_id="op-bootstrap-recursive-session",
        binding="session",
        parents=(s7,),
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s19, role="shepherd.SessionState"),
        ),
        transition=_transition("op-parent-initial"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-initial",
            {"workspace": w42, "session": s19},
            stores_by_binding={"workspace": "store_workspace", "session": "store_session"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    c0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        ),
        transition=_transition("op-child-fork", parents=[p0]),
        operation_final=_checkpoint_final(manager, "op-child-fork", {"workspace": w42, "session": s7}),
        parents=(p0,),
    )
    w43, w43_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-child",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
    )
    s8, s8_commit = _prepared_candidate(
        manager,
        "store_session",
        operation_id="op-child",
        binding="session",
        payload={"label": "session S8"},
        parents=(s7,),
    )
    c1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s8.head, role="shepherd.SessionState"),
        ),
        transition=_transition("op-child", parents=[c0]),
        operation_final=_operation_final(
            "op-child",
            {"workspace": w43.head, "session": s8.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    w43_commit,
                    final_operation_id="op-child",
                ),
                _candidate_outcome(
                    manager,
                    "store_session",
                    s8_commit,
                    final_operation_id="op-child",
                ),
            ],
            candidate_commits=[w43_commit, s8_commit],
        ),
        parents=(c0,),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        ),
        transition=_transition("op-parent-merge", parents=[p0, c1], input_world=p0),
        operation_final=attach_selection_evidence_ref(
            operation_final_with_head_selections(
                "op-parent-merge",
                {"workspace": w43.head, "session": s7},
                outcomes=[
                    _candidate_outcome(
                        manager,
                        "store_workspace",
                        w43_commit,
                        final_operation_id="op-parent-merge",
                        producer_world_oid=c1,
                    )
                ],
                candidate_commits=[w43_commit],
                selection_kinds={"session": "checkpoint"},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                manager.world_store,
                operation_id="op-parent-merge",
                binding="session",
                store=manager.store("store_session"),
                head=s7,
                evidence_kind="checkpoint",
            ),
        ),
        parents=(p0, c1),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    child_session_pin = world_pin_ref(manager.world_store.world_store_id, c1, "session")
    child_world_ref = child_world_retention_ref(p1, "root.workspace.producer")
    manager.store("store_session").repo.references[candidate_ref("op-child", "session")].delete()

    assert manager.store("store_session").repo.references[child_session_pin].target == pygit2.Oid(hex=s8.head)
    assert manager.world_store.repo.references[child_world_ref].target == pygit2.Oid(hex=c1)
    assert manager.fsck_world(p1).ok

    manager.store("store_session").repo.references[child_session_pin].delete()
    report = manager.fsck_world(p1)

    assert not report.ok
    assert child_session_pin in report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_rejects_child_produced_candidate_id_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-candidate-id-workspace",
        binding="workspace",
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
        ),
        transition=_transition("op-parent-initial-candidate-id"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-initial-candidate-id",
            {"workspace": w42},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    candidate_a, commit_a = create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id="op-child-candidate-id",
        binding="workspace",
        candidate_id="a",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=manager.world_store,
    )
    child_world = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace",
                binding="workspace",
                head=candidate_a.head,
                role="shepherd.WorkspaceRef",
            ),
        ),
        transition=_transition("op-child-candidate-id", parents=[p0]),
        operation_final=_operation_final(
            "op-child-candidate-id",
            {"workspace": candidate_a.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    commit_a,
                    final_operation_id="op-child-candidate-id",
                ),
            ],
            candidate_commits=[commit_a],
        ),
        parents=(p0,),
    )
    alias_ref = candidate_ref("op-child-candidate-id", "workspace", "b")
    create_or_update_reference(manager.store("store_workspace").repo, alias_ref, pygit2.Oid(hex=candidate_a.head))
    alias_commit = CandidateCommitRecord(
        operation_id=commit_a.operation_id,
        binding=commit_a.binding,
        candidate_id="b",
        store_id=commit_a.store_id,
        resource_id=commit_a.resource_id,
        candidate_head=commit_a.candidate_head,
        candidate_ref=alias_ref,
        revision_preparation_digest=commit_a.revision_preparation_digest,
    )
    parent_world = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head(
                "store_workspace",
                binding="workspace",
                head=candidate_a.head,
                role="shepherd.WorkspaceRef",
            ),
        ),
        transition=_transition("op-parent-candidate-id", parents=[p0, child_world], input_world=p0),
        operation_final=_operation_final(
            "op-parent-candidate-id",
            {"workspace": candidate_a.head},
            outcomes=[
                {
                    **_candidate_outcome(
                        manager,
                        "store_workspace",
                        commit_a,
                        final_operation_id="op-parent-candidate-id",
                        producer_world_oid=child_world,
                    ),
                    "candidate_id": "b",
                    "candidate_commit_digest": alias_commit.candidate_commit_digest(),
                },
            ],
            candidate_commits=[alias_commit],
        ),
        parents=(p0, child_world),
    )

    report = manager.fsck_world_deep(parent_world)

    assert not report.ok
    assert any("producer world does not select candidate" in issue.message for issue in report.issue_details)


def test_world_storage_manager_recursive_cas_cleanup_removes_publish_closure(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-cas-workspace",
        binding="workspace",
    )
    s7 = _prepared_session_checkpoint(manager, "refs/checkpoints/S7", {"label": "session S7"})
    s19 = _bootstrap_revision(
        manager,
        "store_session",
        "refs/heads/parent",
        {"label": "session S19"},
        operation_id="op-bootstrap-cas-session",
        binding="session",
        parents=(s7,),
    )
    p0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s19, role="shepherd.SessionState"),
        ),
        transition=_transition("op-parent-initial-cas"),
        operation_final=_bootstrap_final(
            manager,
            "op-parent-initial-cas",
            {"workspace": w42, "session": s19},
            stores_by_binding={"workspace": "store_workspace", "session": "store_session"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    c0 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        ),
        transition=_transition("op-child-fork-cas", parents=[p0]),
        operation_final=_checkpoint_final(manager, "op-child-fork-cas", {"workspace": w42, "session": s7}),
        parents=(p0,),
    )
    w43, w43_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-child-cas",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
    )
    s8, s8_commit = _prepared_candidate(
        manager,
        "store_session",
        operation_id="op-child-cas",
        binding="session",
        payload={"label": "session S8"},
        parents=(s7,),
    )
    c1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s8.head, role="shepherd.SessionState"),
        ),
        transition=_transition("op-child-cas", parents=[c0]),
        operation_final=_operation_final(
            "op-child-cas",
            {"workspace": w43.head, "session": s8.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    w43_commit,
                    final_operation_id="op-child-cas",
                ),
                _candidate_outcome(
                    manager,
                    "store_session",
                    s8_commit,
                    final_operation_id="op-child-cas",
                ),
            ],
            candidate_commits=[w43_commit, s8_commit],
        ),
        parents=(c0,),
    )
    p1 = manager.create_unsafe_world(
        snapshot=_snapshot(
            manager.substrate_head("store_workspace", binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        ),
        transition=_transition("op-parent-merge-cas", parents=[p0, c1], input_world=p0),
        operation_final=attach_selection_evidence_ref(
            operation_final_with_head_selections(
                "op-parent-merge-cas",
                {"workspace": w43.head, "session": s7},
                outcomes=[
                    _candidate_outcome(
                        manager,
                        "store_workspace",
                        w43_commit,
                        final_operation_id="op-parent-merge-cas",
                        producer_world_oid=c1,
                    )
                ],
                candidate_commits=[w43_commit],
                selection_kinds={"session": "checkpoint"},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                manager.world_store,
                operation_id="op-parent-merge-cas",
                binding="session",
                store=manager.store("store_session"),
                head=s7,
                evidence_kind="checkpoint",
            ),
        ),
        parents=(p0, c1),
    )
    intervening = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=w42,
        operation_id="op-recursive-cas-intervening",
    )
    assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=intervening, input_world_oid=p0)

    assert not manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=p1, input_world_oid=p0)
    retained_refs = {
        world_pin_ref(manager.world_store.world_store_id, p1, "workspace"): manager.store("store_workspace").repo,
        world_pin_ref(manager.world_store.world_store_id, p1, "session"): manager.store("store_session").repo,
        world_pin_ref(manager.world_store.world_store_id, c1, "workspace"): manager.store("store_workspace").repo,
        world_pin_ref(manager.world_store.world_store_id, c1, "session"): manager.store("store_session").repo,
        world_pin_ref(manager.world_store.world_store_id, c0, "workspace"): manager.store("store_workspace").repo,
        world_pin_ref(manager.world_store.world_store_id, c0, "session"): manager.store("store_session").repo,
        child_world_retention_ref(p1, "root.workspace.producer"): manager.world_store.repo,
        world_retention_receipt_ref(DEFAULT_GROUND_REF, p1): manager.world_store.repo,
    }

    deleted = set(manager.cleanup_orphan_pins(p1))

    assert set(retained_refs) <= deleted
    for ref, repo in retained_refs.items():
        assert ref not in repo.references


def test_world_storage_manager_orphan_cleanup_preserves_in_flight_publish_retention(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-cleanup-during-publish",
    )
    original_publish = manager.world_store._publish_ref_unchecked
    deleted_during_publish: tuple[str, ...] | None = None

    def cleanup_before_cas(ref: str, world_oid: str, expected_oid: str | None) -> bool:
        nonlocal deleted_during_publish
        deleted_during_publish = manager.cleanup_orphan_pins(world_oid)
        return original_publish(ref, world_oid, expected_oid)

    monkeypatch.setattr(manager.world_store, "_publish_ref_unchecked", cleanup_before_cas)

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)

    assert deleted_during_publish == ()
    assert manager.fsck_world(p1).ok
    assert not any(
        ref.startswith(world_publication_lease_prefix() + "/") for ref in manager.world_store.repo.references
    )


def test_world_storage_manager_cleanup_removes_stale_publication_lease_after_publish(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-stale-publish-lease",
    )

    def keep_publication_lease(lease_refs: tuple[str, ...], *, world_oid: str) -> None:
        assert lease_refs
        assert world_oid == p1

    monkeypatch.setattr(manager, "_release_publication_leases", keep_publication_lease)

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    stale_leases = tuple(
        ref for ref in manager.world_store.repo.references if ref.startswith(world_publication_lease_prefix() + "/")
    )

    assert len(stale_leases) == 1
    assert manager.cleanup_stale_publication_leases() == stale_leases
    assert not any(
        ref.startswith(world_publication_lease_prefix() + "/") for ref in manager.world_store.repo.references
    )


def test_world_storage_manager_cleanup_requires_explicit_abandon_for_journalless_unpublished_lease(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-journalless-lease",
    )

    lease_refs = manager._write_publication_leases((DEFAULT_GROUND_REF,), manager.read_world(p1))
    assert len(lease_refs) == 1
    assert isinstance(_read_lease_payload(manager, lease_refs[0])["created_at_unix_ns"], int)

    assert manager.cleanup_stale_publication_leases() == ()
    assert lease_refs[0] in manager.world_store.repo.references
    assert manager.cleanup_stale_publication_leases(abandon_journalless=True) == lease_refs
    assert lease_refs[0] not in manager.world_store.repo.references


def test_world_storage_manager_cleanup_removes_stale_scope_publication_lease_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_workspace_world(manager)
    p1 = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-stale-scope-publish-lease",
    )
    scope = scope_ref("scope-a")

    def keep_publication_lease(lease_refs: tuple[str, ...], *, world_oid: str) -> None:
        assert lease_refs
        assert world_oid == p1

    monkeypatch.setattr(manager, "_release_publication_leases", keep_publication_lease)

    assert manager.fork_world_ref(
        ref=scope,
        world_oid=p1,
        forked_from_ref=DEFAULT_GROUND_REF,
        forked_from_world_oid=p0,
    )
    stale_leases = tuple(
        ref for ref in manager.world_store.repo.references if ref.startswith(world_publication_lease_prefix() + "/")
    )

    assert len(stale_leases) == 1
    assert manager.cleanup_stale_publication_leases() == stale_leases
    assert not any(
        ref.startswith(world_publication_lease_prefix() + "/") for ref in manager.world_store.repo.references
    )


def test_world_storage_manager_retains_explicit_world_ref_children(tmp_path) -> None:
    manager = _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace child"},
        operation_id="op-bootstrap-child-workspace",
        binding="workspace",
    )
    child_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-child-world"),
        operation_final=_bootstrap_final(
            manager,
            "op-child-world",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    world_ref_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/child",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id="op-parent-world-ref",
    )
    parent_snapshot = _snapshot(
        manager.substrate_head("store_child_world_ref", binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    retention_requirements = (
        RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
        RetentionPolicyRequirement(
            kind="child-world-retention",
            target=f"world:{child_world}",
            digest=child_snapshot.digest(),
        ),
    )
    final = attach_selection_evidence_ref(
        operation_final_with_head_selections(
            "op-parent-world-ref",
            {"child": world_ref_head},
            store_ids={"child": "store_child_world_ref"},
            resource_ids={"child": "world-ref:child-task"},
            selection_kinds={"child": "import"},
            retention_policy_requirements={"child": retention_requirements},
        ),
        binding="child",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id="op-parent-world-ref",
            binding="child",
            store=manager.store("store_child_world_ref"),
            head=world_ref_head,
            evidence_kind="import",
        ),
    )
    parent_world = manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition("op-parent-world-ref"),
        operation_final=final,
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=parent_world, expected_oid=None)
    retained_child_ref = child_world_retention_ref(parent_world, "root.child.world_ref")
    child_workspace_pin = world_pin_ref(manager.world_store.world_store_id, child_world, "workspace")
    parent_child_pin = world_pin_ref(manager.world_store.world_store_id, parent_world, "child")
    receipt_ref = world_retention_receipt_ref(DEFAULT_GROUND_REF, parent_world)

    assert manager.world_store.repo.references[retained_child_ref].target == pygit2.Oid(hex=child_world)
    assert manager.store("store_workspace").repo.references[child_workspace_pin].target == pygit2.Oid(hex=workspace)
    assert _read_receipt_payload(manager, receipt_ref)["retained_refs"] == sorted(
        [child_workspace_pin, parent_child_pin, retained_child_ref]
    )
    assert manager.fsck_world(parent_world).ok

    manager.store("store_workspace").repo.references[child_workspace_pin].delete()
    report = manager.fsck_world(parent_world)

    assert not report.ok
    assert child_workspace_pin in report.pin_classification["missing_for_published_world"]


def test_world_storage_manager_imports_published_world_ref_child_after_candidate_ref_gc(tmp_path) -> None:
    child_ref = "refs/vcscore/child-ground"
    manager = _world_ref_manager(tmp_path)
    child_world, child_workspace_head, child_candidate_ref = _published_candidate_child_world(
        manager,
        child_ref=child_ref,
    )
    parent_world = _world_ref_parent_world(
        manager,
        child_world=child_world,
        child_workspace_head=child_workspace_head,
    )
    manager.store("store_workspace").repo.references[child_candidate_ref].delete()

    assert manager.fsck_world(child_world, authority_refs=(child_ref,)).ok
    assert _publish_world(
        manager,
        ref=DEFAULT_GROUND_REF,
        world_oid=parent_world,
        expected_oid=None,
        authority_refs=(child_ref,),
    )
    retained_child_ref = child_world_retention_ref(parent_world, "root.child.world_ref")
    child_workspace_pin = world_pin_ref(manager.world_store.world_store_id, child_world, "workspace")

    assert manager.world_store.repo.references[retained_child_ref].target == pygit2.Oid(hex=child_world)
    assert manager.store("store_workspace").repo.references[child_workspace_pin].target == pygit2.Oid(
        hex=child_workspace_head
    )
    assert manager.fsck_world(parent_world).ok


def test_world_storage_manager_imported_world_ref_child_requires_explicit_authority_after_candidate_ref_gc(
    tmp_path,
) -> None:
    child_ref = "refs/vcscore/child-ground"
    manager = _world_ref_manager(tmp_path)
    child_world, child_workspace_head, child_candidate_ref = _published_candidate_child_world(
        manager,
        child_ref=child_ref,
    )
    parent_world = _world_ref_parent_world(
        manager,
        child_world=child_world,
        child_workspace_head=child_workspace_head,
    )
    manager.store("store_workspace").repo.references[child_candidate_ref].delete()

    with pytest.raises(InvalidRepositoryStateError, match="durable candidate ref"):
        _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=parent_world, expected_oid=None)

    parent_child_pin = world_pin_ref(manager.world_store.world_store_id, parent_world, "child")
    assert parent_child_pin not in manager.store("store_child_world_ref").repo.references
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, parent_world) not in manager.world_store.repo.references


def test_world_storage_manager_imported_world_ref_child_rejects_corrupt_authority_retention(tmp_path) -> None:
    child_ref = "refs/vcscore/child-ground"
    manager = _world_ref_manager(tmp_path)
    child_world, child_workspace_head, child_candidate_ref = _published_candidate_child_world(
        manager,
        child_ref=child_ref,
    )
    parent_world = _world_ref_parent_world(
        manager,
        child_world=child_world,
        child_workspace_head=child_workspace_head,
    )
    child_workspace_pin = world_pin_ref(manager.world_store.world_store_id, child_world, "workspace")
    manager.store("store_workspace").repo.references[child_candidate_ref].delete()
    manager.store("store_workspace").repo.references[child_workspace_pin].delete()

    # Trust-by-default (Part A): the deleted child candidate ref is part of the NEW world's own
    # closure, so the kept new-world closure validation still rejects this publish (the prior-lineage
    # preflight is no longer what catches it) — same rejection as the candidate-ref-gc sibling test.
    with pytest.raises(InvalidRepositoryStateError, match=r"durable candidate ref"):
        _publish_world(
            manager,
            ref=DEFAULT_GROUND_REF,
            world_oid=parent_world,
            expected_oid=None,
            authority_refs=(child_ref,),
        )

    parent_child_pin = world_pin_ref(manager.world_store.world_store_id, parent_world, "child")
    assert parent_child_pin not in manager.store("store_child_world_ref").repo.references
    assert world_retention_receipt_ref(DEFAULT_GROUND_REF, parent_world) not in manager.world_store.repo.references


def test_world_storage_manager_preserves_published_parent_world_ref_retention_after_authority_advances(
    tmp_path,
) -> None:
    manager = _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace child"},
        operation_id="op-bootstrap-child-workspace",
        binding="workspace",
    )
    child_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-child-world"),
        operation_final=_bootstrap_final(
            manager,
            "op-child-world",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    world_ref_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/child",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id="op-parent-world-ref",
    )
    parent_snapshot = _snapshot(
        manager.substrate_head("store_child_world_ref", binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    retention_requirements = (
        RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
        RetentionPolicyRequirement(
            kind="child-world-retention",
            target=f"world:{child_world}",
            digest=child_snapshot.digest(),
        ),
    )
    parent_world = manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition("op-parent-world-ref"),
        operation_final=attach_selection_evidence_ref(
            operation_final_with_head_selections(
                "op-parent-world-ref",
                {"child": world_ref_head},
                store_ids={"child": "store_child_world_ref"},
                resource_ids={"child": "world-ref:child-task"},
                selection_kinds={"child": "import"},
                retention_policy_requirements={"child": retention_requirements},
            ),
            binding="child",
            evidence_ref=selection_evidence_ref(
                manager.world_store,
                operation_id="op-parent-world-ref",
                binding="child",
                store=manager.store("store_child_world_ref"),
                head=world_ref_head,
                evidence_kind="import",
            ),
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=parent_world, expected_oid=None)
    retained_child_ref = child_world_retention_ref(parent_world, "root.child.world_ref")
    child_workspace_pin = world_pin_ref(manager.world_store.world_store_id, child_world, "workspace")

    next_world = manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition("op-next", parents=[parent_world]),
        operation_final=operation_final_with_head_selections(
            "op-next",
            {"child": world_ref_head},
            store_ids={"child": "store_child_world_ref"},
            resource_ids={"child": "world-ref:child-task"},
            retention_policy_requirements={"child": retention_requirements},
        ),
        parents=(parent_world,),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=next_world, expected_oid=parent_world)

    report = manager.fsck_world(parent_world)
    assert report.ok
    assert retained_child_ref in report.pin_classification["published"]
    assert child_workspace_pin in report.pin_classification["published"]
    assert manager.cleanup_orphan_pins(parent_world) == ()


def test_world_storage_manager_rejects_world_ref_child_with_invalid_world_evidence(tmp_path) -> None:
    manager = _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace child"},
        operation_id="op-bootstrap-independent-child-workspace",
        binding="workspace",
    )
    child_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-invalid-child-world"),
        operation_final=_operation_final("op-invalid-child-world", {"workspace": "1" * 40}),
    )
    world_ref_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/invalid-child",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id="op-parent-invalid-world-ref",
    )
    parent_snapshot = _snapshot(
        manager.substrate_head("store_child_world_ref", binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    final = attach_selection_evidence_ref(
        operation_final_with_head_selections(
            "op-parent-invalid-world-ref",
            {"child": world_ref_head},
            store_ids={"child": "store_child_world_ref"},
            resource_ids={"child": "world-ref:child-task"},
            selection_kinds={"child": "import"},
            retention_policy_requirements={
                "child": (
                    RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
                    RetentionPolicyRequirement(
                        kind="child-world-retention",
                        target=f"world:{child_world}",
                        digest=child_snapshot.digest(),
                    ),
                )
            },
        ),
        binding="child",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id="op-parent-invalid-world-ref",
            binding="child",
            store=manager.store("store_child_world_ref"),
            head=world_ref_head,
            evidence_kind="import",
        ),
    )
    parent_world = manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition("op-parent-invalid-world-ref"),
        operation_final=final,
    )

    with pytest.raises(InvalidRepositoryStateError, match="selected heads disagree"):
        _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=parent_world, expected_oid=None)

    report = manager.fsck_world(parent_world)

    assert not report.ok
    assert "world_validation_failed" in {issue.code for issue in report.issue_details}
    assert child_world in {issue.world_oid for issue in report.issue_details}


def test_world_storage_manager_orphan_cleanup_preserves_independently_published_child_world(tmp_path) -> None:
    child_ref = "refs/vcscore/child-ground"
    manager = _manager(
        tmp_path,
        stores=(
            *_specs(),
            SubstrateStoreSpec(
                identity=_world_ref_identity(),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace child"},
        operation_id="op-bootstrap-independent-child-workspace",
        binding="workspace",
    )
    child_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    child_world = manager.create_unsafe_world(
        snapshot=child_snapshot,
        transition=_transition("op-child-world"),
        operation_final=_bootstrap_final(
            manager,
            "op-child-world",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=child_ref, world_oid=child_world, expected_oid=None)
    child_workspace_pin = world_pin_ref(manager.world_store.world_store_id, child_world, "workspace")
    assert manager.store("store_workspace").repo.references[child_workspace_pin].target == pygit2.Oid(hex=workspace)

    world_ref_head = _prepared_world_ref_revision(
        manager,
        "refs/heads/child",
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
        operation_id="op-parent-world-ref",
    )
    parent_snapshot = _snapshot(
        manager.substrate_head("store_child_world_ref", binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    final = attach_selection_evidence_ref(
        operation_final_with_head_selections(
            "op-parent-world-ref",
            {"child": world_ref_head},
            store_ids={"child": "store_child_world_ref"},
            resource_ids={"child": "world-ref:child-task"},
            selection_kinds={"child": "import"},
            retention_policy_requirements={
                "child": (
                    RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
                    RetentionPolicyRequirement(
                        kind="child-world-retention",
                        target=f"world:{child_world}",
                        digest=child_snapshot.digest(),
                    ),
                )
            },
        ),
        binding="child",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id="op-parent-world-ref",
            binding="child",
            store=manager.store("store_child_world_ref"),
            head=world_ref_head,
            evidence_kind="import",
        ),
    )
    parent_world = manager.create_unsafe_world(
        snapshot=parent_snapshot,
        transition=_transition("op-parent-world-ref"),
        operation_final=final,
    )

    deleted = manager.cleanup_orphan_pins(parent_world, authority_refs=(DEFAULT_GROUND_REF, child_ref))

    assert child_workspace_pin not in deleted
    assert manager.store("store_workspace").repo.references[child_workspace_pin].target == pygit2.Oid(hex=workspace)
    assert manager.fsck_world(child_world, authority_refs=(child_ref,)).ok


def test_world_storage_manager_rejects_publish_when_expected_head_is_not_a_parent(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-publish-parent-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    p0 = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("p0"),
        operation_final=_bootstrap_final(
            manager,
            "p0",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    p1 = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("p1", parents=[p0]),
        operation_final=_operation_final("p1", {"workspace": workspace}),
        parents=(p0,),
    )
    p2 = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("p2", parents=[p0]),
        operation_final=_operation_final("p2", {"workspace": workspace}),
        parents=(p0,),
    )

    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    with pytest.raises(InvalidRepositoryStateError, match="input_world_oid disagrees"):
        _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p2, expected_oid=p1)

    assert manager.fsck_world(p2).pin_classification["orphaned"] == ()


def test_world_storage_manager_reports_missing_pins_for_published_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-published-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-publish"),
        operation_final=_bootstrap_final(
            manager,
            "op-publish",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    pin_ref = world_pin_ref(manager.world_store.world_store_id, world_oid, "workspace")
    manager.store("store_workspace").repo.references[pin_ref].delete()

    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert report.pin_classification["missing_for_published_world"] == (pin_ref,)
    assert "published world is missing selected-head pins" in report.issues
    assert report.issue_details[0].code == "missing_selected_head_pins"
    assert report.issue_details[0].world_oid == world_oid


def test_world_storage_manager_fsck_allows_selected_candidate_refs_to_be_gc_after_publication(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"label": "workspace W42"}
    )
    selected, selected_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-select",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=selected.head, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": selected.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    selected_commit,
                    final_operation_id="op-select",
                )
            ],
            candidate_commits=[selected_commit],
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)

    manager.store("store_workspace").repo.references[selected.ref].delete()
    assert manager.fsck_world(world_oid).ok

    pin_ref = world_pin_ref(manager.world_store.world_store_id, world_oid, "workspace")
    manager.store("store_workspace").repo.references[pin_ref].delete()
    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert report.pin_classification["missing_for_published_world"] == (pin_ref,)
    assert "selected candidate outcome lacks a durable candidate ref" in report.issues
    assert "missing_candidate_ref" in {issue.code for issue in report.issue_details}


def test_world_storage_manager_fsck_requires_selected_candidate_ref_before_publication(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"label": "workspace W42"}
    )
    selected, selected_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-select",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=selected.head, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": selected.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    selected_commit,
                    final_operation_id="op-select",
                )
            ],
            candidate_commits=[selected_commit],
        ),
    )

    manager.store("store_workspace").repo.references[selected.ref].delete()
    report = manager.fsck_world_deep(world_oid)

    assert not report.ok
    assert "selected candidate outcome lacks a durable candidate ref" in report.issues
    assert report.issue_details[0].code == "missing_candidate_ref"


def test_world_storage_manager_fsck_requires_selected_candidate_ref_for_unpublished_orphan_pins(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-lost-cas-workspace",
        binding="workspace",
    )
    p0_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-publish"),
        operation_final=_bootstrap_final(
            manager,
            "op-publish",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    selected, selected_commit = _prepared_candidate(
        manager,
        "store_workspace",
        operation_id="op-lost-cas",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
    )
    p1_snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=selected.head, role="shepherd.WorkspaceRef")
    )
    p1 = manager.create_unsafe_world(
        snapshot=p1_snapshot,
        transition=_transition("op-lost-cas", parents=[p0]),
        operation_final=_operation_final(
            "op-lost-cas",
            {"workspace": selected.head},
            outcomes=[
                _candidate_outcome(
                    manager,
                    "store_workspace",
                    selected_commit,
                    final_operation_id="op-lost-cas",
                )
            ],
            candidate_commits=[selected_commit],
        ),
        parents=(p0,),
    )
    intervening = _workspace_advance_world(
        manager,
        parent_world_oid=p0,
        parent_workspace_head=workspace,
        operation_id="op-lost-cas-selected-ref-intervening",
    )
    assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=intervening, input_world_oid=p0)

    assert not manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=p1, input_world_oid=p0)
    manager.store("store_workspace").repo.references[selected.ref].delete()
    report = manager.fsck_world_deep(p1)

    assert not report.ok
    assert report.pin_classification["orphaned"] == (
        world_pin_ref(manager.world_store.world_store_id, p1, "workspace"),
    )
    assert "selected candidate outcome lacks a durable candidate ref" in report.issues


def test_world_storage_manager_rejects_installation_identity_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    config = json.loads((manager.root / "world-stores.json").read_text(encoding="utf-8"))

    assert config["stores"]["store_workspace"]["locator"] == "substrates/workspace.git"
    with pytest.raises(InvalidRepositoryStateError, match="identity mismatch"):
        _manager(tmp_path, stores=_specs(workspace_identity=_workspace_identity(resource_id="fs:other")))


def test_world_storage_manager_rejects_implicit_locator_rewrites(tmp_path) -> None:
    manager = _manager(tmp_path)
    config_path = manager.root / "world-stores.json"
    original_config = config_path.read_bytes()

    with pytest.raises(InvalidRepositoryStateError, match="locator mismatch"):
        _manager(tmp_path, stores=_specs(workspace_locator="substrates/workspace-moved.git"))

    assert config_path.read_bytes() == original_config
    assert not (manager.root / "substrates" / "workspace-moved.git").exists()


def test_world_storage_manager_rejects_missing_configured_stores_on_reopen(tmp_path) -> None:
    manager = _manager(tmp_path)
    shutil.rmtree(manager.root / "substrates" / "workspace.git")

    with pytest.raises(InvalidRepositoryStateError, match="configured substrate store is missing"):
        _manager(tmp_path)


def test_world_storage_manager_rebinds_locator_to_existing_matching_store(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = _bootstrap_revision(
        manager,
        "store_workspace",
        "refs/heads/main",
        {"label": "workspace"},
        operation_id="op-bootstrap-rebind-workspace",
        binding="workspace",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-publish"),
        operation_final=_bootstrap_final(
            manager,
            "op-publish",
            {"workspace": workspace},
            stores_by_binding={"workspace": "store_workspace"},
        ),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)

    shutil.copytree(manager.root / "substrates" / "workspace.git", manager.root / "substrates" / "workspace-copy.git")
    WorldStorageManager.rebind_store_locator(
        manager.root,
        world_store_id="store_world_test",
        store_id="store_workspace",
        locator="substrates/workspace-copy.git",
    )
    reopened = _manager(tmp_path, stores=_specs(workspace_locator="substrates/workspace-copy.git"))

    assert reopened.locator_hints()["store_workspace"] == "substrates/workspace-copy.git"
    assert reopened.fsck_world(world_oid).ok


def test_world_storage_manager_rebind_rejects_missing_or_mismatched_targets(tmp_path) -> None:
    manager = _manager(tmp_path)
    config_path = manager.root / "world-stores.json"
    original_config = config_path.read_bytes()

    with pytest.raises(InvalidRepositoryStateError, match="configured substrate store is missing"):
        WorldStorageManager.rebind_store_locator(
            manager.root,
            world_store_id="store_world_test",
            store_id="store_workspace",
            locator="substrates/missing.git",
        )
    assert config_path.read_bytes() == original_config

    SubstrateStore.open_or_init(
        manager.root / "substrates" / "workspace-other.git",
        _workspace_identity(resource_id="fs:other"),
    )
    with pytest.raises(InvalidRepositoryStateError, match="identity mismatch"):
        WorldStorageManager.rebind_store_locator(
            manager.root,
            world_store_id="store_world_test",
            store_id="store_workspace",
            locator="substrates/workspace-other.git",
        )
    assert config_path.read_bytes() == original_config


def test_world_storage_manager_rejects_unsafe_store_locators() -> None:
    with pytest.raises(ValueError, match="relative path"):
        SubstrateStoreSpec(identity=_workspace_identity(), locator="../workspace.git")
