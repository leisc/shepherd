# under-test: vcs_core._world_recovery
"""Unit tests for conservative private v2 world recovery helpers."""

from __future__ import annotations

import pygit2
import pytest
from vcs_core import WORLD_TRANSITION_SCHEMA, InvalidRepositoryStateError, WorldSnapshot
from vcs_core._transition_kernel_records import CandidateCommitRecord, CandidateOutcomeRecord
from vcs_core._world_operation_builder import PreparedCandidateTupleRecord, PreparedWorldOperation
from vcs_core._world_publication_plan import PublicationPlan
from vcs_core._world_recovery import (
    _commit_finalized_world,
    archive_failed_operation,
    cleanup_orphan_pins,
    cleanup_stale_publication_leases,
    complete_committed_operation,
    complete_journaled_operation,
    reconcile_open_operation_journal_index,
)
from vcs_core._world_refs import world_publication_lease_prefix
from vcs_core._world_types import OperationFinalRecord
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import (
    DEFAULT_GROUND_REF,
    CandidateSelection,
    OperationFinalBuilder,
    SubstrateStoreSpec,
    WorldStorageManager,
    operation_journal_ref,
)

from .world_vectors_v2_helpers import (
    attach_selection_evidence_ref,
    candidate_outcome_for_commit,
    create_prepared_candidate,
    operation_final_with_head_selections,
    selection_evidence_ref,
)


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity("store_workspace", "filesystem", "fs:repo-main"),
                locator="substrates/workspace.git",
            ),
        ),
    )


def _publish_world(
    manager: WorldStorageManager,
    *,
    ref: str,
    world_oid: str,
    expected_oid: str | None,
) -> bool:
    if expected_oid is None:
        return manager.publish_root_world(ref=ref, world_oid=world_oid)
    return manager.advance_world_ref(ref=ref, world_oid=world_oid, input_world_oid=expected_oid)


def _transition(operation_id: str, *, parents: list[str] | None = None) -> dict[str, object]:
    resolved_parents = parents or []
    extra = {"input_world": resolved_parents[0]} if resolved_parents else {}
    return {"schema": WORLD_TRANSITION_SCHEMA, "operation_id": operation_id, "parent_worlds": resolved_parents, **extra}


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


def _prepared_workspace_candidate(manager: WorldStorageManager, operation_id: str, parent: str):
    return create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id=operation_id,
        binding="workspace",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
        world_store=manager.world_store,
    )


def _candidate_outcome(
    manager: WorldStorageManager,
    candidate_commit: CandidateCommitRecord,
    *,
    final_operation_id: str,
) -> dict[str, object]:
    return candidate_outcome_for_commit(
        manager.store("store_workspace"),
        candidate_commit,
        final_operation_id=final_operation_id,
        world_store=manager.world_store,
    )


def _published_base(manager: WorldStorageManager) -> tuple[str, str]:
    workspace = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id="op-initial-workspace",
        binding="workspace",
        payload={"schema": "example/workspace"},
        semantic_op="bootstrap",
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r"),)
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", workspace),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=None)
    return workspace, world_oid


def _bootstrap_operation_final(manager: WorldStorageManager, operation_id: str, workspace: str) -> dict[str, object]:
    return attach_selection_evidence_ref(
        operation_final_with_head_selections(
            operation_id,
            {"workspace": workspace},
            selection_kinds={"workspace": "bootstrap"},
        ),
        binding="workspace",
        evidence_ref=selection_evidence_ref(
            manager.world_store,
            operation_id=operation_id,
            binding="workspace",
            store=manager.store("store_workspace"),
            head=workspace,
            evidence_kind="bootstrap",
        ),
    )


def _record_finalized_world(
    manager: WorldStorageManager,
    operation_id: str,
    world_oid: str,
    *,
    input_world_oid: str,
    candidate_refs=(),
    operation_kind: str = "merge",
    target_ref: str = DEFAULT_GROUND_REF,
) -> None:
    manager.record_operation_finalized(operation_id)


def _record_prepared_world(
    manager: WorldStorageManager,
    operation_id: str,
    world_oid: str,
    *,
    input_world_oid: str,
    candidate_refs=(),
    operation_kind: str = "merge",
    target_ref: str = DEFAULT_GROUND_REF,
) -> None:
    world = manager.read_world(world_oid)
    final = OperationFinalRecord(dict(world.operation_final))
    candidate_tuples = [_candidate_tuple(manager, candidate) for candidate in candidate_refs]
    prepared = PreparedWorldOperation(
        operation_id=operation_id,
        operation_kind=operation_kind,
        target_ref=target_ref,
        input_world_oid=input_world_oid,
        snapshot=world.snapshot,
        transition=dict(world.transition),
        candidate_tuples=tuple(candidate_tuples),
        candidate_refs=tuple(candidate_refs),
        candidate_commits=tuple(
            CandidateCommitRecord.from_json(dict(item)) for item in final.payload["candidate_commits"]
        ),
        candidate_outcomes=tuple(
            CandidateOutcomeRecord.from_operation_final_json(dict(item)) for item in final.payload["candidate_outcomes"]
        ),
        head_selections=tuple(dict(item) for item in final.payload["head_selections"]),
        selection_evidence=tuple(dict(item) for item in final.payload["selection_evidence"]),
        selected=dict(final.payload["selected"]),
        parents=world.parent_oids,
    )
    manager.record_operation_prepared(operation_id, prepared=prepared)


def _candidate_tuple(manager: WorldStorageManager, candidate) -> PreparedCandidateTupleRecord:
    store = manager.store(candidate.store_id)
    provenance = store.validate_prepared_candidate(
        candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    return PreparedCandidateTupleRecord(
        candidate=candidate,
        transition=provenance.transition,
        plan=provenance.plan,
        preparation=provenance.preparation,
        candidate_commit=store.candidate_commit_record(
            candidate,
            evidence_resolver=manager.world_store.resolve_evidence_ref,
        ),
    )


def _record_publishing(
    manager: WorldStorageManager,
    operation_id: str,
    *,
    world_oid: str,
    input_world_oid: str,
) -> None:
    publication_plan = manager.build_advance_publication_plan(
        ref=DEFAULT_GROUND_REF,
        world_oid=world_oid,
        expected_oid=input_world_oid,
        input_world_oid=input_world_oid,
    )
    manager.record_operation_publishing(operation_id, world_oid=world_oid, publication_plan=publication_plan)


def test_recovery_cleanup_orphan_pins_is_idempotent(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    intervening_head, intervening_commit = _prepared_workspace_candidate(manager, "op-intervening", workspace)
    intervening_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=intervening_head.head, role="r"),)
    )
    intervening_world = manager.create_unsafe_world(
        snapshot=intervening_snapshot,
        transition=_transition("op-intervening", parents=[p0]),
        operation_final=_operation_final(
            "op-intervening",
            {"workspace": intervening_head.head},
            outcomes=[_candidate_outcome(manager, intervening_commit, final_operation_id="op-intervening")],
            candidate_commits=[intervening_commit],
        ),
        parents=(p0,),
    )
    assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=intervening_world, input_world_oid=p0)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-conflict", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-conflict", parents=[p0]),
        operation_final=_operation_final(
            "op-conflict",
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-conflict")],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    assert not manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=world_oid, input_world_oid=p0)

    first = cleanup_orphan_pins(manager, world_oid)
    second = cleanup_orphan_pins(manager, world_oid)

    assert [action.code for action in first.actions] == [
        "orphan_pin_deleted",
        "orphan_retention_receipt_deleted",
    ]
    assert second.actions == ()
    assert manager.fsck_world(world_oid).pin_classification["orphaned"] == ()


def test_recovery_archives_failed_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    _, p0 = _published_base(manager)
    manager.open_operation_journal(
        operation_id="op-failed",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.fail_operation_journal("op-failed", error="boom")

    report = archive_failed_operation(manager, "op-failed")

    assert [action.code for action in report.actions] == ["operation_archived"]
    assert manager.read_operation_journal("op-failed", family="archived").tip.payload["status"] == "archived"


def test_recovery_prunes_stale_open_ref_for_already_archived_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    _, p0 = _published_base(manager)
    manager.open_operation_journal(
        operation_id="op-stale-archived",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.fail_operation_journal("op-stale-archived", error="boom")
    manager.archive_operation_journal("op-stale-archived")
    # atomic archive no longer leaves a stale open ref; simulate an OUT-OF-MODEL one (manual edit /
    # pre-index data) at the archived tip for recovery to prune.
    archived_tip = manager.read_operation_journal("op-stale-archived", family="archived").tip.oid
    manager.world_store.repo.references.create(
        operation_journal_ref("open", "op-stale-archived"), pygit2.Oid(hex=archived_tip)
    )
    assert operation_journal_ref("open", "op-stale-archived") in manager.world_store.repo.references

    report = archive_failed_operation(manager, "op-stale-archived")

    assert [action.code for action in report.actions] == [
        "stale_open_journal_deleted",
        "operation_already_archived",
    ]
    assert operation_journal_ref("open", "op-stale-archived") not in manager.world_store.repo.references
    # the recovery cleanup co-write left the open-journal index consistent with the authority
    assert manager.verify_open_operation_journal_index().status == "fresh"


def test_recovery_reconcile_open_operation_journal_index_heals_out_of_model_drift(tmp_path) -> None:
    manager = _manager(tmp_path)
    _, p0 = _published_base(manager)
    manager.open_operation_journal(
        operation_id="op-live",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    # An out-of-model writer leaves an open ref the co-write never indexed -> stale under-report.
    live_tip = manager.read_operation_journal("op-live", family="open").tip.oid
    manager.world_store.repo.references.create(operation_journal_ref("open", "op-drift"), pygit2.Oid(hex=live_tip))
    assert manager.verify_open_operation_journal_index().status == "stale"

    report = reconcile_open_operation_journal_index(manager)

    assert report.ok
    assert [action.code for action in report.actions] == ["open_operation_journal_index_reconciled"]
    assert manager.verify_open_operation_journal_index().status == "fresh"
    assert operation_journal_ref("open", "op-drift") in manager._open_journal_index().read_open_refs()


def test_recovery_cleans_stale_publication_lease_after_successful_publish(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-stale-lease", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-stale-lease", parents=[p0]),
        operation_final=_operation_final(
            "op-stale-lease",
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-stale-lease")],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )

    with monkeypatch.context() as patched:
        patched.setattr(manager._pubret, "_release_publication_leases", lambda _refs, *, world_oid: None)
        assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=world_oid, input_world_oid=p0)

    assert any(ref.startswith(world_publication_lease_prefix() + "/") for ref in manager.world_store.repo.references)

    report = cleanup_stale_publication_leases(manager)

    assert [action.code for action in report.actions] == ["stale_publication_lease_deleted"]
    assert not any(
        ref.startswith(world_publication_lease_prefix() + "/") for ref in manager.world_store.repo.references
    )


def test_recovery_completes_finalized_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-finalized-recover", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    prepared = (
        OperationFinalBuilder("op-finalized-recover")
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id="op-finalized-recover",
                selection=CandidateSelection(candidate, candidate_commit, _candidate_tuple(manager, candidate)),
                role="r",
            )
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
            snapshot=snapshot,
            transition=_transition("op-finalized-recover", parents=[p0]),
            parents=(p0,),
            candidate_tuples=(_candidate_tuple(manager, candidate),),
            candidate_refs=(candidate,),
        )
    )
    manager.open_operation_journal(
        operation_id="op-finalized-recover",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.record_operation_prepared("op-finalized-recover", prepared=prepared)
    manager.record_operation_finalized("op-finalized-recover")

    report = complete_journaled_operation(manager, "op-finalized-recover")

    assert [action.code for action in report.actions] == ["operation_completed"]
    assert manager.read_world(DEFAULT_GROUND_REF).snapshot.head_for("workspace").head == candidate.head
    closed = manager.read_operation_journal("op-finalized-recover", family="closed")
    assert [entry.payload["status"] for entry in closed.entries] == [
        "opened",
        "prepared",
        "finalized",
        "world_committed",
        "publishing",
        "published",
        "closed",
    ]


def test_recovery_rejects_finalized_replay_without_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(InvalidRepositoryStateError, match="requires prepared_world_operation"):
        _commit_finalized_world(
            manager,
            {
                "status": "finalized",
                "snapshot": {},
                "transition": {},
                "operation_final": {},
                "parents": [],
            },
        )


def test_recovery_completes_world_committed_operation_when_ref_still_matches(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-recover", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-recover",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-recover")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-recover", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-recover",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-recover", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-recover", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-recover", world_oid=world_oid)

    report = complete_committed_operation(manager, "op-recover")

    assert [action.code for action in report.actions] == ["operation_completed"]
    assert manager.read_world(DEFAULT_GROUND_REF).oid == world_oid
    assert manager.read_operation_journal("op-recover", family="closed").tip.payload["status"] == "closed"

    second = complete_committed_operation(manager, "op-recover")

    assert [action.code for action in second.actions] == ["operation_already_closed"]


def test_recovery_prunes_stale_open_ref_for_already_closed_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-stale-closed", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-stale-closed",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-stale-closed")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-stale-closed", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-stale-closed",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-stale-closed", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-stale-closed", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-stale-closed", world_oid=world_oid)
    _record_publishing(manager, "op-stale-closed", world_oid=world_oid, input_world_oid=p0)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=p0)
    manager.record_operation_published("op-stale-closed", world_oid=world_oid)
    manager.close_operation_journal("op-stale-closed", world_oid=world_oid)
    # atomic close no longer leaves a stale open ref; simulate an OUT-OF-MODEL one (manual edit /
    # pre-index data) at the closed tip for recovery to prune.
    closed_tip = manager.read_operation_journal("op-stale-closed", family="closed").tip.oid
    manager.world_store.repo.references.create(
        operation_journal_ref("open", "op-stale-closed"), pygit2.Oid(hex=closed_tip)
    )
    assert operation_journal_ref("open", "op-stale-closed") in manager.world_store.repo.references

    report = complete_committed_operation(manager, "op-stale-closed")

    assert [action.code for action in report.actions] == [
        "stale_open_journal_deleted",
        "operation_already_closed",
    ]
    assert operation_journal_ref("open", "op-stale-closed") not in manager.world_store.repo.references
    # the recovery cleanup co-write left the open-journal index consistent with the authority
    assert manager.verify_open_operation_journal_index().status == "fresh"


def test_recovery_blocks_world_committed_when_publish_succeeded_before_publication_intent(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-post-cas", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-post-cas",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-post-cas")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-post-cas", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-post-cas",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-post-cas", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-post-cas", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-post-cas", world_oid=world_oid)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=p0)

    report = complete_committed_operation(manager, "op-post-cas")

    assert [action.code for action in report.actions] == ["operation_complete_blocked"]
    assert "before publication intent was journaled" in report.actions[0].message
    assert manager.read_operation_journal("op-post-cas").tip.payload["status"] == "world_committed"


def test_recovery_replays_journaled_publication_plan(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-plan-replay", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-plan-replay",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-plan-replay")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-plan-replay", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-plan-replay",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-plan-replay", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-plan-replay", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-plan-replay", world_oid=world_oid)
    publication_plan = manager.build_publication_plan(
        ref=DEFAULT_GROUND_REF,
        world_oid=world_oid,
        expected_oid=p0,
        input_world_oid=p0,
        allow_same_resource_alias=True,
    )
    manager.record_operation_publishing("op-plan-replay", world_oid=world_oid, publication_plan=publication_plan)
    # V2.2c: prepare_publication moved to the controller; patch it there (the WSM shim would be
    # bypassed by the moved caller). The `intercepted` flag makes a vacuous patch loud (rule 9).
    original_prepare = manager._pubret.prepare_publication
    intercepted: list[bool] = []

    def assert_journaled_plan(plan):
        intercepted.append(True)
        assert plan.allow_same_resource_alias
        return original_prepare(plan)

    monkeypatch.setattr(manager._pubret, "prepare_publication", assert_journaled_plan)

    report = complete_committed_operation(manager, "op-plan-replay")

    assert intercepted, "prepare_publication fault injection did not intercept (vacuous patch)"
    assert [action.code for action in report.actions] == ["operation_completed"]
    assert manager.read_operation_journal("op-plan-replay", family="closed").tip.payload["status"] == "closed"


def test_publication_plan_validation_binds_plan_to_journaled_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-plan-bound", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-plan-bound", parents=[p0]),
        operation_final=_operation_final(
            "op-plan-bound",
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-plan-bound")],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-plan-bound",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-plan-bound", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-plan-bound", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-plan-bound", world_oid=world_oid)

    wrong_world_plan = PublicationPlan(
        authority_ref=DEFAULT_GROUND_REF,
        authority_refs=(DEFAULT_GROUND_REF,),
        world_store_id=manager.world_store.world_store_id,
        world_oid=p0,
        expected_oid=p0,
        input_world_oid=p0,
    )
    with pytest.raises(InvalidRepositoryStateError, match="world_oid disagrees"):
        manager.record_operation_publishing("op-plan-bound", world_oid=world_oid, publication_plan=wrong_world_plan)

    wrong_ref_plan = PublicationPlan(
        authority_ref="refs/vcscore/other",
        authority_refs=("refs/vcscore/other",),
        world_store_id=manager.world_store.world_store_id,
        world_oid=world_oid,
        expected_oid=p0,
        input_world_oid=p0,
    )
    with pytest.raises(InvalidRepositoryStateError, match="authority_ref disagrees"):
        manager.record_operation_publishing("op-plan-bound", world_oid=world_oid, publication_plan=wrong_ref_plan)


def test_publication_plan_rejects_noncanonical_or_wrong_manager_scope(tmp_path) -> None:
    manager = _manager(tmp_path)
    _workspace, p0 = _published_base(manager)

    with pytest.raises(InvalidRepositoryStateError, match=r"exactly once|deduplicated"):
        PublicationPlan(
            authority_ref=DEFAULT_GROUND_REF,
            authority_refs=(DEFAULT_GROUND_REF, DEFAULT_GROUND_REF),
            world_store_id=manager.world_store.world_store_id,
            world_oid=p0,
            expected_oid=None,
            input_world_oid=None,
        )

    plan = PublicationPlan(
        authority_ref=DEFAULT_GROUND_REF,
        authority_refs=(DEFAULT_GROUND_REF,),
        world_store_id="other_world_store",
        world_oid=p0,
        expected_oid=None,
        input_world_oid=None,
    )
    with pytest.raises(InvalidRepositoryStateError, match="world_store_id disagrees"):
        manager.prepare_publication(plan)


def test_recovery_closes_published_open_journal_when_ref_matches_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-published", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-published",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-published")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-published", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-published",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-published", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-published", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-published", world_oid=world_oid)
    _record_publishing(manager, "op-published", world_oid=world_oid, input_world_oid=p0)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=p0)
    manager.record_operation_published("op-published", world_oid=world_oid)

    report = complete_committed_operation(manager, "op-published")

    assert [action.code for action in report.actions] == ["operation_completed"]
    assert manager.read_operation_journal("op-published", family="closed").tip.payload["status"] == "closed"


def test_recovery_closes_publishing_operation_when_authority_advanced_to_descendant(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-publishing-ancestor", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-publishing-ancestor", parents=[p0]),
        operation_final=_operation_final(
            "op-publishing-ancestor",
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-publishing-ancestor")],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-publishing-ancestor",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(
        manager,
        "op-publishing-ancestor",
        world_oid,
        input_world_oid=p0,
        candidate_refs=(candidate,),
    )
    _record_finalized_world(
        manager, "op-publishing-ancestor", world_oid, input_world_oid=p0, candidate_refs=(candidate,)
    )
    manager.record_operation_world_committed("op-publishing-ancestor", world_oid=world_oid)
    _record_publishing(manager, "op-publishing-ancestor", world_oid=world_oid, input_world_oid=p0)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=world_oid, expected_oid=p0)

    descendant_head, descendant_commit = _prepared_workspace_candidate(manager, "op-descendant", candidate.head)
    descendant_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=descendant_head.head, role="r"),)
    )
    descendant_world = manager.create_unsafe_world(
        snapshot=descendant_snapshot,
        transition=_transition("op-descendant", parents=[world_oid]),
        operation_final=_operation_final(
            "op-descendant",
            {"workspace": descendant_head.head},
            outcomes=[_candidate_outcome(manager, descendant_commit, final_operation_id="op-descendant")],
            candidate_commits=[descendant_commit],
        ),
        parents=(world_oid,),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=descendant_world, expected_oid=world_oid)

    report = complete_committed_operation(manager, "op-publishing-ancestor")

    assert [action.code for action in report.actions] == ["operation_completed"]
    assert manager.read_world(DEFAULT_GROUND_REF).oid == descendant_world
    assert manager.read_operation_journal("op-publishing-ancestor", family="closed").tip.payload["status"] == "closed"


def test_recovery_blocks_when_authority_moved_to_unrelated_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace, p0 = _published_base(manager)
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-blocked", workspace)
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=candidate.head, role="r"),)
    )
    operation_final = _operation_final(
        "op-blocked",
        {"workspace": candidate.head},
        outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-blocked")],
        candidate_commits=[candidate_commit],
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-blocked", parents=[p0]),
        operation_final=operation_final,
        parents=(p0,),
    )
    other_head, other_commit = _prepared_workspace_candidate(
        manager,
        "op-other",
        workspace,
    )
    other_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=other_head.head, role="r"),)
    )
    other_world = manager.create_unsafe_world(
        snapshot=other_snapshot,
        transition=_transition("op-other", parents=[p0]),
        operation_final=_operation_final(
            "op-other",
            {"workspace": other_head.head},
            outcomes=[_candidate_outcome(manager, other_commit, final_operation_id="op-other")],
            candidate_commits=[other_commit],
        ),
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id="op-blocked",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-blocked", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-blocked", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-blocked", world_oid=world_oid)
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=other_world, expected_oid=p0)

    report = complete_committed_operation(manager, "op-blocked")

    assert [action.code for action in report.actions] == ["operation_complete_blocked"]
    assert manager.read_operation_journal("op-blocked").tip.payload["status"] == "world_committed"
