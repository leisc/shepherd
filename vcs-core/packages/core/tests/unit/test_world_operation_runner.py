# under-test: vcs_core._world_operation_runner
"""Unit tests for the private v2 world operation runner."""

from __future__ import annotations

import pytest
from vcs_core import WORLD_TRANSITION_SCHEMA, InvalidRepositoryStateError, WorldSnapshot
from vcs_core._world_operation_protocol import validate_operation_status_fields
from vcs_core._world_operation_runner import WorldOperationRunner
from vcs_core._world_recovery import complete_committed_operation
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import (
    DEFAULT_GROUND_REF,
    CandidateSelection,
    OperationFinalBuilder,
    SubstrateStoreSpec,
    WorldStorageManager,
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
                identity=SubstrateStoreIdentity(
                    store_id="store_workspace",
                    kind="filesystem",
                    resource_id="fs:repo-main",
                ),
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
    return {
        "schema": WORLD_TRANSITION_SCHEMA,
        "operation_id": operation_id,
        "parent_worlds": resolved_parents,
        **extra,
    }


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
    candidate_commit,
    *,
    final_operation_id: str,
) -> dict[str, object]:
    return candidate_outcome_for_commit(
        manager.store("store_workspace"),
        candidate_commit,
        final_operation_id=final_operation_id,
        world_store=manager.world_store,
    )


def _bootstrap_workspace(manager: WorldStorageManager, operation_id: str = "op-initial-workspace") -> str:
    return manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id=operation_id,
        binding="workspace",
        payload={"schema": "example/workspace", "n": 42},
        semantic_op="bootstrap",
    )


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


def _prepared_candidate_operation(
    *,
    manager: WorldStorageManager,
    operation_id: str,
    input_world_oid: str,
    snapshot: WorldSnapshot,
    candidate,
    selection: CandidateSelection,
):
    return (
        OperationFinalBuilder(operation_id)
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id=operation_id,
                selection=selection,
                role="r",
            )
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=input_world_oid,
            snapshot=snapshot,
            transition=_transition(operation_id, parents=[input_world_oid]),
            parents=(input_world_oid,),
            candidate_refs=(candidate,),
        )
    )


def test_world_operation_runner_publishes_and_closes_journal(tmp_path) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)
    w42 = _bootstrap_workspace(manager)
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    w43_bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-runner",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w43_bundle.candidate.head, role="r"),)
    )

    prepared = _prepared_candidate_operation(
        manager=manager,
        operation_id="op-runner",
        input_world_oid=p0,
        snapshot=snapshot,
        candidate=w43_bundle.candidate,
        selection=CandidateSelection.from_bundle(w43_bundle),
    )

    result = runner.publish_prepared_world(prepared)

    assert result.status == "closed"
    assert result.published
    assert result.world_oid is not None
    assert manager.read_world(DEFAULT_GROUND_REF).oid == result.world_oid
    closed = manager.read_operation_journal("op-runner", family="closed")
    assert [entry.payload["status"] for entry in closed.entries] == [
        "opened",
        "prepared",
        "finalized",
        "world_committed",
        "publishing",
        "published",
        "closed",
    ]
    assert closed.tip.payload["status"] == "closed"
    assert closed.tip.payload["publication_plan"]["authority_ref"] == DEFAULT_GROUND_REF
    assert (
        closed.tip.payload["publication_plan_digest"]
        == closed.tip.payload["publication_plan"]["publication_plan_digest"]
    )
    prepared_payload = closed.entries[1].payload
    prepared_record = prepared_payload["prepared_world_operation"]
    assert prepared_record["schema"] == "vcscore/prepared-world-operation/v1"
    assert prepared_payload["prepared_world_operation_digest"] == prepared_record["prepared_operation_digest"]
    assert prepared_record["operation_final_digest"] == closed.entries[2].payload["operation_final_digest"]
    assert [summary.operation_id for summary in manager.list_operation_journals(family="closed")] == ["op-runner"]

    tampered = dict(closed.tip.payload)
    prepared_record = dict(tampered["prepared_world_operation"])
    prepared_record["selected"] = {"workspace": "0" * 40}
    tampered["prepared_world_operation"] = prepared_record
    with pytest.raises(InvalidRepositoryStateError, match="prepared operation"):
        validate_operation_status_fields(tampered)


def test_world_operation_runner_rejects_tuple_free_candidate_publication(tmp_path) -> None:
    manager = _manager(tmp_path)
    parent = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        {"schema": "example/workspace", "n": 1},
    )
    candidate, candidate_commit = create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id="op-tuple-free-candidate",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 2},
        parents=(parent,),
        world_store=manager.world_store,
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head(
                "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
            ),
        )
    )
    with pytest.raises(ValueError, match="prepared candidate tuple"):
        (
            OperationFinalBuilder("op-tuple-free-candidate")
            .select_candidate_plan(
                plan=manager.plan_candidate_selection(
                    operation_id="op-tuple-free-candidate",
                    selection=CandidateSelection(candidate, candidate_commit, None),  # type: ignore[arg-type]
                    role="shepherd.WorkspaceRef",
                )
            )
            .build_prepared(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid=None,
                snapshot=snapshot,
                transition=_transition("op-tuple-free-candidate"),
                candidate_refs=(candidate,),
            )
        )


def test_world_operation_runner_publishes_root_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)
    workspace = _bootstrap_workspace(manager, operation_id="op-root-runner-workspace")
    head = manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="r")
    snapshot = WorldSnapshot((head,))
    selection_plan = manager.plan_existing_head_selection(
        operation_id="op-root-runner",
        head=head,
        selection_kind="bootstrap",
    )
    prepared = (
        OperationFinalBuilder("op-root-runner")
        .select_existing(plan=selection_plan)
        .build_prepared(
            operation_kind="bootstrap",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=snapshot,
            transition=_transition("op-root-runner"),
            parents=(),
        )
    )

    result = runner.publish_prepared_world(prepared)

    assert result.status == "closed"
    assert result.published
    assert result.world_oid is not None
    assert manager.read_world(DEFAULT_GROUND_REF).oid == result.world_oid
    closed = manager.read_operation_journal("op-root-runner", family="closed")
    assert closed.entries[0].payload["input_world_oid"] is None
    assert closed.tip.payload["publication_plan"]["expected_oid"] is None
    assert closed.tip.payload["publication_plan"]["input_world_oid"] is None
    assert closed.entries[1].payload["prepared_world_operation"]["input_world_oid"] is None


def test_world_operation_runner_exposes_prepared_publication_api(tmp_path) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)

    assert not hasattr(runner, "publish_finalized_world")
    assert hasattr(runner, "publish_prepared_world")


def test_operation_finalization_is_derived_from_prepared_operation(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_workspace(manager)
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    bundle_a = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-prepared-authority",
        binding="workspace",
        candidate_id="a",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    bundle_b = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-prepared-authority",
        binding="workspace",
        candidate_id="b",
        payload={"schema": "example/workspace", "n": 44},
        parents=(w42,),
    )
    snapshot_a = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle_a.candidate.head, role="r"),)
    )
    snapshot_b = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=bundle_b.candidate.head, role="r"),)
    )
    prepared_a = (
        OperationFinalBuilder("op-prepared-authority")
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id="op-prepared-authority",
                selection=CandidateSelection.from_bundle(bundle_a),
                role="r",
            )
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
            snapshot=snapshot_a,
            transition=_transition("op-prepared-authority", parents=[p0]),
            parents=(p0,),
            candidate_refs=(bundle_a.candidate,),
        )
    )
    prepared_b = (
        OperationFinalBuilder("op-prepared-authority")
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id="op-prepared-authority",
                selection=CandidateSelection.from_bundle(bundle_b),
                role="r",
            )
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
            snapshot=snapshot_b,
            transition=_transition("op-prepared-authority", parents=[p0]),
            parents=(p0,),
            candidate_refs=(bundle_b.candidate,),
        )
    )

    manager.open_operation_journal(
        operation_id="op-prepared-authority",
        operation_kind="merge",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.record_operation_prepared("op-prepared-authority", prepared=prepared_a)

    with pytest.raises(InvalidRepositoryStateError, match="invalid operation journal transition"):
        manager.record_operation_prepared("op-prepared-authority", prepared=prepared_b)

    finalized_entry = manager.record_operation_finalized("op-prepared-authority")
    assert finalized_entry.payload["selected"] == prepared_a.selected


def test_world_operation_runner_rejects_finalized_input_world_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = _bootstrap_workspace(manager)
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    w43_bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-runner-input-mismatch",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w43_bundle.candidate.head, role="r"),)
    )

    with pytest.raises(ValueError, match="input_world_oid disagrees"):
        (
            OperationFinalBuilder("op-runner-input-mismatch")
            .select_candidate_plan(
                plan=manager.plan_candidate_selection(
                    operation_id="op-runner-input-mismatch",
                    selection=CandidateSelection.from_bundle(w43_bundle),
                    role="r",
                )
            )
            .build(
                operation_kind="merge",
                target_ref=DEFAULT_GROUND_REF,
                input_world_oid=p0,
                snapshot=snapshot,
                transition={**_transition("op-runner-input-mismatch", parents=[p0]), "input_world": "f" * 40},
                parents=(p0,),
                candidate_refs=(w43_bundle.candidate,),
            )
        )


def test_world_operation_runner_records_failed_journal_for_cas_conflict(tmp_path) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)
    w42 = _bootstrap_workspace(manager)
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    w44, w44_commit = create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id="op-intervening",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 44},
        parents=(w42,),
        world_store=manager.world_store,
    )
    p1_snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w44.head, role="r"),)
    )
    p1 = manager.create_unsafe_world(
        snapshot=p1_snapshot,
        transition=_transition("op-intervening", parents=[p0]),
        operation_final=_operation_final(
            "op-intervening",
            {"workspace": w44.head},
            outcomes=[_candidate_outcome(manager, w44_commit, final_operation_id="op-intervening")],
            candidate_commits=[w44_commit],
        ),
        parents=(p0,),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p1, expected_oid=p0)
    w43_bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-conflict",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w43_bundle.candidate.head, role="r"),)
    )

    prepared = _prepared_candidate_operation(
        manager=manager,
        operation_id="op-conflict",
        input_world_oid=p0,
        snapshot=snapshot,
        candidate=w43_bundle.candidate,
        selection=CandidateSelection.from_bundle(w43_bundle),
    )
    result = runner.publish_prepared_world(prepared)

    assert result.status == "failed"
    assert not result.published
    assert result.world_oid is not None
    assert manager.read_world(DEFAULT_GROUND_REF).oid == p1
    assert manager.read_operation_journal("op-conflict").tip.payload["status"] == "failed"
    assert manager.fsck_world(result.world_oid).pin_classification["orphaned"]


def test_world_operation_runner_keeps_published_world_recoverable_after_bookkeeping_failure(
    tmp_path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)
    w42 = _bootstrap_workspace(manager)
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    assert _publish_world(manager, ref=DEFAULT_GROUND_REF, world_oid=p0, expected_oid=None)
    w43_bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-post-publish-fail",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w43_bundle.candidate.head, role="r"),)
    )

    def fail_record_published(operation_id: str, *, world_oid: str) -> None:
        raise RuntimeError(f"simulated post-publish failure for {operation_id} at {world_oid}")

    with monkeypatch.context() as patched:
        patched.setattr(manager, "record_operation_published", fail_record_published)
        prepared = _prepared_candidate_operation(
            manager=manager,
            operation_id="op-post-publish-fail",
            input_world_oid=p0,
            snapshot=snapshot,
            candidate=w43_bundle.candidate,
            selection=CandidateSelection.from_bundle(w43_bundle),
        )
        result = runner.publish_prepared_world(prepared)

    assert result.status == "recovery_required"
    assert result.published
    assert result.world_oid is not None
    assert result.journal_family == "open"
    assert manager.read_world(DEFAULT_GROUND_REF).oid == result.world_oid
    assert manager.read_operation_journal("op-post-publish-fail").tip.payload["status"] == "publishing"

    recovery = complete_committed_operation(manager, "op-post-publish-fail")

    assert [action.code for action in recovery.actions] == ["operation_completed"]
    assert manager.read_operation_journal("op-post-publish-fail", family="closed").tip.payload["status"] == "closed"
