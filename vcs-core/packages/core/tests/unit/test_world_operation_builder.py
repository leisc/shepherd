# under-test: vcs_core._world_operation_builder
"""Unit tests for production operation-final construction."""

from __future__ import annotations

import pytest
from vcs_core import WORLD_TRANSITION_SCHEMA, InvalidRepositoryStateError, WorldSnapshot
from vcs_core._transition_kernel_records import CandidateCommitRecord, RetentionPolicyRequirement
from vcs_core._world_operation_builder import (
    CandidateSelection,
    FinalizedWorldOperation,
    OperationFinalBuilder,
    PreparedCandidateTupleRecord,
    PreparedWorldOperation,
    SelectionRequirementPlan,
)
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import DEFAULT_GROUND_REF, SubstrateStoreSpec, WorldStorageManager


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_workspace",
                    kind="filesystem",
                    resource_id="fs:repo-main",
                ),
                locator="substrates/workspace.git",
            ),
        ),
    )


def _transition(operation_id: str, *, parents: tuple[str, ...] = ()) -> dict[str, object]:
    extra = {"input_world": parents[0]} if parents else {}
    return {
        "schema": WORLD_TRANSITION_SCHEMA,
        "operation_id": operation_id,
        "parent_worlds": list(parents),
        **extra,
    }


def _candidate_bundle(
    manager: WorldStorageManager,
    *,
    operation_id: str,
    binding: str = "workspace",
    candidate_id: str = "primary",
    payload: dict[str, object],
    parents: tuple[str, ...],
):
    return manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id=operation_id,
        binding=binding,
        candidate_id=candidate_id,
        payload=payload,
        parents=parents,
    )


def _candidate_plan(
    manager: WorldStorageManager,
    *,
    operation_id: str,
    bundle,
    **kwargs,
):
    return manager.plan_candidate_selection(
        operation_id=operation_id,
        selection=CandidateSelection.from_bundle(bundle),
        role="r",
        **kwargs,
    )


def _unchanged_plan(*, operation_id: str, selected_head: str, binding: str = "workspace") -> SelectionRequirementPlan:
    return SelectionRequirementPlan(
        operation_id=operation_id,
        binding=binding,
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head=selected_head,
        selection_kind="unchanged",
        retention_policy_requirements=(RetentionPolicyRequirement(kind="selected-head-pin", target=selected_head),),
    )


def test_operation_final_builder_constructs_candidate_backed_final(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    finalized = (
        OperationFinalBuilder("op-build")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build", bundle=bundle))
        .build(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )

    assert finalized.selected == {"workspace": bundle.candidate.head}
    assert finalized.operation_final.payload["candidate_commits"] == [bundle.candidate_commit.to_json()]
    assert finalized.operation_final.payload["candidate_outcomes"][0] == {
        "binding": "workspace",
        "candidate": bundle.candidate.head,
        "outcome": "selected",
        "store_id": bundle.candidate.store_id,
        "resource_id": bundle.candidate.resource_id,
        "transition_digest": bundle.transition.transition_digest(),
        "revision_plan_digest": bundle.plan.revision_plan_digest(),
        "content_digest": bundle.plan.content_digest,
        "revision_preparation_digest": bundle.preparation.revision_preparation_digest(),
        "candidate_commit_digest": bundle.candidate_commit.candidate_commit_digest(),
        "evidence_digests": list(bundle.preparation.evidence_digests),
        "evidence_refs": [ref.to_json() for ref in bundle.preparation.evidence_refs],
    }
    assert finalized.operation_final_digest == finalized.operation_final.digest()
    assert finalized.snapshot_digest == snapshot.digest()


def test_operation_final_builder_constructs_strict_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-prepared",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    prepared = (
        OperationFinalBuilder("op-build-prepared")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-prepared", bundle=bundle))
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-prepared", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )

    assert PreparedWorldOperation.from_json(prepared.to_json()) == prepared
    assert prepared.finalize().operation_final_digest == prepared.to_json()["operation_final_digest"]

    mutated = prepared.to_json()
    mutated["snapshot_digest"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="snapshot_digest disagrees"):
        PreparedWorldOperation.from_json(mutated)

    mutated = prepared.to_json()
    mutated["prepared_operation_digest"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="prepared_operation_digest disagrees"):
        PreparedWorldOperation.from_json(mutated)

    mutated = prepared.to_json()
    mutated["candidate_outcomes"] = [
        outcome.to_json(final_operation_id="op-build-prepared") for outcome in prepared.candidate_outcomes
    ]
    with pytest.raises(ValueError, match="unsupported candidate outcome schema"):
        PreparedWorldOperation.from_json(mutated)

    mutated = prepared.to_json()
    mutated["candidate_outcomes"] = [prepared.candidate_outcomes[0].to_record_json(final_operation_id="other-op")]
    with pytest.raises(ValueError, match="candidate outcome operation_id disagrees"):
        PreparedWorldOperation.from_json(mutated)


def test_operation_records_reject_explicit_empty_selected_for_nonempty_snapshot(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-explicit-empty-selected",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )
    prepared = (
        OperationFinalBuilder("op-explicit-empty-selected")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-explicit-empty-selected", bundle=bundle))
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-explicit-empty-selected", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )
    finalized = prepared.finalize()

    with pytest.raises(ValueError, match="prepared operation selected heads disagree with snapshot"):
        PreparedWorldOperation(
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
            selection_evidence=prepared.selection_evidence,
            selected={},
            parents=prepared.parents,
        )

    with pytest.raises(ValueError, match="finalized operation selected heads disagree with operation-final"):
        FinalizedWorldOperation(
            operation_id=finalized.operation_id,
            operation_kind=finalized.operation_kind,
            target_ref=finalized.target_ref,
            input_world_oid=finalized.input_world_oid,
            snapshot=finalized.snapshot,
            transition=finalized.transition,
            operation_final=finalized.operation_final,
            candidate_refs=finalized.candidate_refs,
            candidate_commits=finalized.candidate_commits,
            candidate_outcomes=finalized.candidate_outcomes,
            selected={},
            parents=finalized.parents,
        )


def test_operation_final_builder_embeds_prepared_candidate_tuples(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-build-tuple",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    prepared = (
        OperationFinalBuilder("op-build-tuple")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-tuple", bundle=bundle))
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-tuple", parents=("f" * 40,)),
            parents=("f" * 40,),
        )
    )

    prepared_json = prepared.to_json()
    round_tripped = PreparedWorldOperation.from_json(prepared_json)

    assert prepared.candidate_refs == (bundle.candidate,)
    assert prepared.candidate_commits == (bundle.candidate_commit,)
    assert prepared_json["candidate_tuples"][0]["preparation"]["revision_preparation_digest"] == (
        bundle.preparation.revision_preparation_digest()
    )
    assert round_tripped == prepared


def test_operation_final_builder_constructs_root_bootstrap_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id="op-root-workspace",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 1},
        semantic_op="bootstrap",
    )
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r")
    selection_plan = manager.plan_existing_head_selection(
        operation_id="op-root-bootstrap",
        head=head,
        selection_kind="bootstrap",
    )
    snapshot = WorldSnapshot((head,))

    prepared = (
        OperationFinalBuilder("op-root-bootstrap")
        .select_existing(plan=selection_plan)
        .build_prepared(
            operation_kind="bootstrap",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=snapshot,
            transition=_transition("op-root-bootstrap"),
            parents=(),
        )
    )

    assert PreparedWorldOperation.from_json(prepared.to_json()) == prepared
    assert prepared.input_world_oid is None
    assert prepared.parents == ()
    assert prepared.selected == {"workspace": workspace}
    assert prepared.head_selections[0]["selection_kind"] == "bootstrap"


def test_operation_final_builder_rejects_unchanged_selection_for_root_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id="op-root-unchanged-workspace",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 1},
        semantic_op="bootstrap",
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r"),)
    )

    with pytest.raises(ValueError, match="root prepared operation requires explicit"):
        (
            OperationFinalBuilder("op-root-unchanged")
            .select_unchanged(
                plan=SelectionRequirementPlan(
                    operation_id="op-root-unchanged",
                    binding="workspace",
                    store_id="store_workspace",
                    resource_id="fs:repo-main",
                    selected_head=workspace,
                    selection_kind="unchanged",
                    retention_policy_requirements=(
                        RetentionPolicyRequirement(kind="selected-head-pin", target=workspace),
                    ),
                )
            )
            .build_prepared(
                operation_kind="bootstrap",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid=None,
                snapshot=snapshot,
                transition=_transition("op-root-unchanged"),
                parents=(),
            )
        )


def test_operation_final_builder_preserves_non_primary_candidate_id(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-secondary",
        candidate_id="retry-2",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    finalized = (
        OperationFinalBuilder("op-build-secondary")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-secondary", bundle=bundle))
        .build(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-secondary", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )

    outcome = finalized.operation_final.payload["candidate_outcomes"][0]
    assert outcome["binding"] == "workspace"
    assert outcome["candidate"] == bundle.candidate.head
    assert outcome["candidate_id"] == "retry-2"
    assert outcome["outcome"] == "selected"
    assert outcome["candidate_commit_digest"] == bundle.candidate_commit.candidate_commit_digest()


def test_operation_final_builder_uses_canonical_final_evidence_order(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle_a = _candidate_bundle(
        manager,
        operation_id="op-build-canonical-order",
        binding="workspace-a",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    bundle_b = _candidate_bundle(
        manager,
        operation_id="op-build-canonical-order",
        binding="workspace-b",
        payload={"schema": "example/workspace", "n": 3},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head("store_workspace", binding="workspace-a", head=bundle_a.candidate.head, role="r"),
            manager.substrate_head("store_workspace", binding="workspace-b", head=bundle_b.candidate.head, role="r"),
        )
    )

    finalized = (
        OperationFinalBuilder("op-build-canonical-order")
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-canonical-order", bundle=bundle_a))
        .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-canonical-order", bundle=bundle_b))
        .build(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-canonical-order", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle_a.candidate, bundle_b.candidate),
        )
    )

    assert list(finalized.candidate_outcome_payloads) == finalized.operation_final.payload["candidate_outcomes"]
    assert [commit.to_json() for commit in finalized.candidate_commits] == finalized.operation_final.payload[
        "candidate_commits"
    ]


def test_prepared_operation_digest_uses_canonical_record_order(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        {"schema": "example/workspace", "n": 1},
    )
    bundle_a = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-prepared-canonical-order",
        binding="workspace-a",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    bundle_b = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-prepared-canonical-order",
        binding="workspace-b",
        payload={"schema": "example/workspace", "n": 3},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head("store_workspace", binding="workspace-a", head=bundle_a.candidate.head, role="r"),
            manager.substrate_head("store_workspace", binding="workspace-b", head=bundle_b.candidate.head, role="r"),
        )
    )
    prepared = (
        OperationFinalBuilder("op-prepared-canonical-order")
        .select_candidate_plan(
            plan=_candidate_plan(manager, operation_id="op-prepared-canonical-order", bundle=bundle_a)
        )
        .select_candidate_plan(
            plan=_candidate_plan(manager, operation_id="op-prepared-canonical-order", bundle=bundle_b)
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-prepared-canonical-order", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle_a.candidate, bundle_b.candidate),
        )
    )
    reordered = PreparedWorldOperation(
        operation_id=prepared.operation_id,
        operation_kind=prepared.operation_kind,
        target_ref=prepared.target_ref,
        input_world_oid=prepared.input_world_oid,
        snapshot=prepared.snapshot,
        transition=prepared.transition,
        candidate_tuples=tuple(reversed(prepared.candidate_tuples)),
        candidate_refs=tuple(reversed(prepared.candidate_refs)),
        candidate_commits=tuple(reversed(prepared.candidate_commits)),
        candidate_outcomes=tuple(reversed(prepared.candidate_outcomes)),
        head_selections=tuple(reversed(prepared.head_selections)),
        selection_evidence=tuple(reversed(prepared.selection_evidence)),
        selected=prepared.selected,
        parents=prepared.parents,
    )

    assert reordered.prepared_operation_digest() == prepared.prepared_operation_digest()
    assert (
        PreparedWorldOperation.from_json(reordered.to_json()).prepared_operation_digest()
        == prepared.prepared_operation_digest()
    )


def test_candidate_selection_rejects_mismatched_commit_record(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-mismatch",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    mismatched_commit = CandidateCommitRecord(
        operation_id=bundle.candidate_commit.operation_id,
        binding="other",
        store_id=bundle.candidate_commit.store_id,
        resource_id=bundle.candidate_commit.resource_id,
        candidate_head=bundle.candidate_commit.candidate_head,
        candidate_ref=bundle.candidate_commit.candidate_ref,
        revision_preparation_digest=bundle.candidate_commit.revision_preparation_digest,
        candidate_id=bundle.candidate_commit.candidate_id,
    )

    with pytest.raises(ValueError, match="binding disagrees"):
        CandidateSelection(bundle.candidate, mismatched_commit, PreparedCandidateTupleRecord.from_bundle(bundle))


def test_operation_final_builder_rejects_candidate_ref_without_outcome(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-missing-outcome",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    with pytest.raises(ValueError, match="has no candidate outcome"):
        (
            OperationFinalBuilder("op-build-missing-outcome")
            .select_unchanged(
                plan=_unchanged_plan(operation_id="op-build-missing-outcome", selected_head=bundle.candidate.head)
            )
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=snapshot,
                transition=_transition("op-build-missing-outcome", parents=("f" * 40,)),
                parents=("f" * 40,),
                candidate_refs=(bundle.candidate,),
            )
        )


def test_operation_final_builder_includes_archived_candidate_commit(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-archive",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),))

    finalized = (
        OperationFinalBuilder("op-build-archive")
        .select_unchanged(plan=_unchanged_plan(operation_id="op-build-archive", selected_head=parent))
        .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
        .build(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-archive", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )

    assert finalized.operation_final.payload["candidate_commits"] == [bundle.candidate_commit.to_json()]
    outcome = finalized.operation_final.payload["candidate_outcomes"][0]
    assert outcome["binding"] == "workspace"
    assert outcome["candidate"] == bundle.candidate.head
    assert outcome["outcome"] == "archived"
    assert outcome["candidate_commit_digest"] == bundle.candidate_commit.candidate_commit_digest()


def test_operation_final_builder_archived_candidate_uses_canonical_evidence_refs(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-archive-evidence",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),))

    finalized = (
        OperationFinalBuilder("op-build-archive-evidence")
        .select_unchanged(plan=_unchanged_plan(operation_id="op-build-archive-evidence", selected_head=parent))
        .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
        .build(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-build-archive-evidence", parents=("f" * 40,)),
            parents=("f" * 40,),
            candidate_refs=(bundle.candidate,),
        )
    )

    outcome = finalized.operation_final.payload["candidate_outcomes"][0]
    assert outcome["evidence_refs"] == [ref.to_json() for ref in bundle.preparation.evidence_refs]


def test_operation_final_builder_archived_candidate_validates_as_world_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id="op-bootstrap-archive-parent",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 1},
        semantic_op="bootstrap",
    )
    parent_head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r")
    parent_snapshot = WorldSnapshot((parent_head,))
    parent_prepared = (
        OperationFinalBuilder("op-bootstrap-archive-parent-world")
        .select_existing(
            plan=manager.plan_existing_head_selection(
                operation_id="op-bootstrap-archive-parent-world",
                head=parent_head,
                selection_kind="bootstrap",
            )
        )
        .build_prepared(
            operation_kind="bootstrap",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=parent_snapshot,
            transition=_transition("op-bootstrap-archive-parent-world"),
            parents=(),
        )
    )
    parent_world = manager.create_world_from_prepared(parent_prepared)
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-archive-validated",
        payload={"schema": "example/workspace", "n": 2},
        parents=(workspace,),
    )
    prepared = (
        OperationFinalBuilder("op-build-archive-validated")
        .select_unchanged(
            plan=manager.plan_unchanged_selection(
                operation_id="op-build-archive-validated",
                head=parent_head,
                input_world_oid=parent_world,
            )
        )
        .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=parent_world,
            snapshot=parent_snapshot,
            transition=_transition("op-build-archive-validated", parents=(parent_world,)),
            parents=(parent_world,),
            candidate_refs=(bundle.candidate,),
        )
    )

    world_oid = manager.create_world_from_prepared(prepared)

    manager.world_store.validate_world_commit(world_oid, manager.stores)


def test_operation_final_builder_rejects_unknown_selection_intent(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),))

    with pytest.raises(ValueError, match="unknown binding 'missing'"):
        (
            OperationFinalBuilder("op-build-unknown")
            .select_unchanged(
                plan=SelectionRequirementPlan(
                    operation_id="op-build-unknown",
                    binding="missing",
                    store_id="store_workspace",
                    resource_id="fs:repo-main",
                    selected_head=parent,
                    selection_kind="unchanged",
                    retention_policy_requirements=(
                        RetentionPolicyRequirement(kind="selected-head-pin", target=parent),
                    ),
                )
            )
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=snapshot,
                transition=_transition("op-build-unknown", parents=("f" * 40,)),
                parents=("f" * 40,),
            )
        )


def test_operation_final_builder_requires_explicit_selection_plan_for_every_binding(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),))

    with pytest.raises(ValueError, match="requires explicit selection plan for binding 'workspace'"):
        OperationFinalBuilder("op-missing-selection").build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid="f" * 40,
            snapshot=snapshot,
            transition=_transition("op-missing-selection", parents=("f" * 40,)),
            parents=("f" * 40,),
        )


def test_operation_final_builder_rejects_selection_plan_operation_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    with pytest.raises(ValueError, match="unchanged selection plan operation_id disagrees"):
        OperationFinalBuilder("op-builder").select_unchanged(
            plan=_unchanged_plan(operation_id="op-plan", selected_head=parent)
        )

    workspace = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/bootstrap",
        operation_id="op-create-bootstrap",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 1},
        semantic_op="bootstrap",
    )
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r")
    existing_plan = manager.plan_existing_head_selection(
        operation_id="op-existing-plan",
        head=head,
        selection_kind="bootstrap",
    )
    with pytest.raises(ValueError, match="existing-head selection plan operation_id disagrees"):
        OperationFinalBuilder("op-builder").select_existing(plan=existing_plan)

    bundle = _candidate_bundle(
        manager,
        operation_id="op-candidate-plan",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    candidate_plan = _candidate_plan(manager, operation_id="op-candidate-plan", bundle=bundle)
    with pytest.raises(ValueError, match="candidate selection plan operation_id disagrees"):
        OperationFinalBuilder("op-builder").select_candidate_plan(plan=candidate_plan)


def test_operation_final_builder_rejects_candidate_selection_head_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-head-mismatch",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),))

    with pytest.raises(ValueError, match="disagrees with snapshot head"):
        (
            OperationFinalBuilder("op-build-head-mismatch")
            .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-head-mismatch", bundle=bundle))
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=snapshot,
                transition=_transition("op-build-head-mismatch", parents=("f" * 40,)),
                parents=("f" * 40,),
            )
        )


def test_operation_final_builder_rejects_child_produced_candidate_without_producer_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-child",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="child-produced selection requires producer_world_oid"):
        manager.plan_candidate_selection(
            operation_id="op-parent",
            selection=CandidateSelection.from_bundle(bundle),
            role="r",
        )


def test_operation_final_builder_rejects_candidate_producer_operation_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-child-a",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )

    plan = manager.plan_candidate_selection(
        operation_id="op-parent",
        selection=CandidateSelection.from_bundle(bundle),
        producer_operation_id="op-child-b",
        producer_world_oid="e" * 40,
        role="r",
    )
    with pytest.raises(ValueError, match="candidate outcome lacks matching candidate commit record"):
        (
            OperationFinalBuilder("op-parent")
            .select_candidate_plan(plan=plan)
            .build_prepared(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=snapshot,
                transition=_transition("op-parent", parents=("f" * 40,)),
                parents=("f" * 40,),
                candidate_refs=(bundle.candidate,),
            )
        )


def test_operation_final_builder_rejects_invalid_archived_candidate_outcomes(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 1}
    )
    bundle = _candidate_bundle(
        manager,
        operation_id="op-build-bad-archive",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
    )

    selected_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle.candidate.head, role="r"),)
    )
    with pytest.raises(ValueError, match="must not name selected head"):
        (
            OperationFinalBuilder("op-build-bad-archive")
            .select_candidate_plan(plan=_candidate_plan(manager, operation_id="op-build-bad-archive", bundle=bundle))
            .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=selected_snapshot,
                transition=_transition("op-build-bad-archive", parents=("f" * 40,)),
                parents=("f" * 40,),
            )
        )

    unchanged_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=parent, role="r"),)
    )
    with pytest.raises(ValueError, match="duplicate candidate outcome"):
        (
            OperationFinalBuilder("op-build-duplicate-archive")
            .select_unchanged(plan=_unchanged_plan(operation_id="op-build-duplicate-archive", selected_head=parent))
            .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
            .archive_candidate(selection=CandidateSelection.from_bundle(bundle))
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid="f" * 40,
                snapshot=unchanged_snapshot,
                transition=_transition("op-build-duplicate-archive", parents=("f" * 40,)),
                parents=("f" * 40,),
            )
        )
