# under-test: vcs_core._world_operation_journal
"""Unit tests for private v2 world operation journals."""

from __future__ import annotations

import pygit2
import pytest
from vcs_core import (
    WORLD_TRANSITION_SCHEMA,
    InvalidRepositoryStateError,
    WorldSnapshot,
    canonical_bytes,
    canonical_digest,
)
from vcs_core._ref_txn import UpdateRefStdinResult
from vcs_core._transition_kernel_records import CandidateCommitRecord, CandidateOutcomeRecord
from vcs_core._world_operation_builder import PreparedCandidateTupleRecord, PreparedWorldOperation
from vcs_core._world_operation_journal import OPERATION_JOURNAL_PATH, OPERATION_JOURNAL_SCHEMA
from vcs_core._world_refs import world_open_operation_journal_index_ref
from vcs_core._world_types import OperationFinalRecord, SubstrateHead
from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import DEFAULT_GROUND_REF, SubstrateStoreSpec, WorldStorageManager, operation_journal_ref

from .world_vectors_v2_helpers import (
    attach_selection_evidence_ref,
    candidate_outcome_for_commit,
    create_prepared_candidate,
    operation_final_with_head_selections,
    selection_evidence_ref,
)


def _workspace_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="fs:repo-main")


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(SubstrateStoreSpec(identity=_workspace_identity(), locator="substrates/workspace.git"),),
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


def _prepared_workspace_candidate(manager: WorldStorageManager, operation_id: str, parent: str):
    return create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id=operation_id,
        binding="workspace",
        payload={"label": "workspace W43"},
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


def _transition(operation_id: str, *, parents: list[str] | None = None) -> dict[str, object]:
    resolved_parents = parents or []
    extra = {"input_world": resolved_parents[0]} if resolved_parents else {}
    return {
        "schema": WORLD_TRANSITION_SCHEMA,
        "operation_id": operation_id,
        "parent_worlds": resolved_parents,
        **extra,
    }


def _snapshot(*heads: SubstrateHead) -> WorldSnapshot:
    return WorldSnapshot(tuple(heads))


def _published_base_world(manager: WorldStorageManager) -> tuple[str, str]:
    workspace = _prepared_existing_revision(
        manager,
        "refs/heads/main",
        operation_id="op-base-workspace",
        payload={"label": "workspace W42"},
        semantic_op="bootstrap",
    )
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-base"),
        operation_final=_bootstrap_operation_final(manager, "op-base", workspace),
    )
    assert manager.publish_root_world(ref=DEFAULT_GROUND_REF, world_oid=world_oid)
    return world_oid, workspace


def _prepared_existing_revision(
    manager: WorldStorageManager,
    ref: str,
    *,
    operation_id: str,
    payload: dict[str, object],
    semantic_op: str,
    parents: tuple[str, ...] = (),
) -> str:
    return manager.create_prepared_json_revision(
        "store_workspace",
        ref,
        operation_id=operation_id,
        binding="workspace",
        payload=payload,
        parents=parents,
        semantic_op=semantic_op,
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


def _world_for_operation(
    manager: WorldStorageManager,
    operation_id: str,
    workspace: str,
    *,
    parent_world: str | None = None,
) -> str:
    snapshot = _snapshot(
        manager.substrate_head("store_workspace", binding="workspace", head=workspace, role="shepherd.WorkspaceRef")
    )
    parents = () if parent_world is None else (parent_world,)
    operation_final = _operation_final(operation_id, {"workspace": workspace})
    if parent_world is None:
        operation_final = _bootstrap_operation_final(manager, operation_id, workspace)
    if parent_world is not None:
        parent_head = manager.read_world(parent_world).snapshot.head_for("workspace").head
        if parent_head != workspace:
            operation_final = attach_selection_evidence_ref(
                operation_final_with_head_selections(
                    operation_id,
                    {"workspace": workspace},
                    selection_kinds={"workspace": "import"},
                ),
                binding="workspace",
                evidence_ref=selection_evidence_ref(
                    manager.world_store,
                    operation_id=operation_id,
                    binding="workspace",
                    store=manager.store("store_workspace"),
                    head=workspace,
                    evidence_kind="import",
                ),
            )
    return manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition(operation_id, parents=list(parents)),
        operation_final=operation_final,
        parents=parents,
    )


def _journal_replay_payload(manager: WorldStorageManager, world_oid: str) -> dict[str, object]:
    world = manager.read_world(world_oid)
    final = OperationFinalRecord(dict(world.operation_final))
    return {
        "snapshot": world.snapshot.to_json(),
        "snapshot_digest": world.snapshot.digest(),
        "transition": dict(world.transition),
        "parents": list(world.parent_oids),
        "operation_final": final.payload,
        "operation_final_digest": final.digest(),
        "candidate_commits": list(final.payload["candidate_commits"]),
    }


def _publication_plan_payload(
    manager: WorldStorageManager,
    world_oid: str,
    *,
    input_world_oid: str | None,
) -> dict[str, object]:
    if input_world_oid is None:
        plan = manager.build_root_publication_plan(ref=DEFAULT_GROUND_REF, world_oid=world_oid)
    else:
        plan = manager.build_advance_publication_plan(
            ref=DEFAULT_GROUND_REF,
            world_oid=world_oid,
            expected_oid=input_world_oid,
            input_world_oid=input_world_oid,
        )
    return {"publication_plan": plan.to_json(), "publication_plan_digest": plan.digest()}


def _record_publishing(
    manager: WorldStorageManager,
    operation_id: str,
    *,
    world_oid: str,
    input_world_oid: str,
) -> None:
    plan = manager.build_advance_publication_plan(
        ref=DEFAULT_GROUND_REF,
        world_oid=world_oid,
        expected_oid=input_world_oid,
        input_world_oid=input_world_oid,
    )
    manager.record_operation_publishing(operation_id, world_oid=world_oid, publication_plan=plan)


def _record_finalized_world(
    manager: WorldStorageManager,
    operation_id: str,
    world_oid: str,
    *,
    input_world_oid: str,
    candidate_refs=(),
    operation_kind: str = "shepherd.task",
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
    operation_kind: str = "shepherd.task",
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


def test_world_operation_journal_round_trips_closed_chain_after_reopen(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-loop",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    candidate, candidate_commit = _prepared_workspace_candidate(manager, "op-loop", workspace)
    selected = {"workspace": candidate.head}
    outcomes = (_candidate_outcome(manager, candidate_commit, final_operation_id="op-loop"),)
    p1_snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
        )
    )
    p1 = manager.create_unsafe_world(
        snapshot=p1_snapshot,
        transition=_transition("op-loop", parents=[p0]),
        operation_final=_operation_final(
            "op-loop", selected, outcomes=[dict(outcomes[0])], candidate_commits=[candidate_commit]
        ),
        parents=(p0,),
    )
    _record_prepared_world(manager, "op-loop", p1, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, "op-loop", p1, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed("op-loop", world_oid=p1)
    _record_publishing(manager, "op-loop", world_oid=p1, input_world_oid=p0)
    assert manager.advance_world_ref(ref=DEFAULT_GROUND_REF, world_oid=p1, input_world_oid=p0)
    manager.record_operation_published("op-loop", world_oid=p1)
    manager.close_operation_journal("op-loop", world_oid=p1)

    reopened = _manager(tmp_path)
    history = reopened.read_operation_journal("op-loop", family="closed")

    assert [entry.payload["status"] for entry in history.entries] == [
        "opened",
        "prepared",
        "finalized",
        "world_committed",
        "publishing",
        "published",
        "closed",
    ]
    assert history.tip.payload["world_oid"] == p1
    assert operation_journal_ref("open", "op-loop") not in reopened.world_store.repo.references
    assert reopened.fsck_operation_journal("op-loop", family="closed").ok


def test_world_operation_journal_world_committed_derives_final_evidence_from_world(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    operation_id = "op-derive-final"
    candidate, candidate_commit = _prepared_workspace_candidate(manager, operation_id, workspace)
    snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
        )
    )
    outcome = candidate_outcome_for_commit(
        manager.store("store_workspace"),
        candidate_commit,
        final_operation_id=operation_id,
        world_store=manager.world_store,
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition(operation_id, parents=[p0]),
        operation_final=_operation_final(
            operation_id,
            {"workspace": candidate.head},
            outcomes=[outcome],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id=operation_id,
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, operation_id, world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, operation_id, world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.record_operation_world_committed(operation_id, world_oid=world_oid)

    tip = manager.read_operation_journal(operation_id).tip.payload
    assert tip["selected"] == {"workspace": candidate.head}
    assert tip["candidate_outcomes"] == [outcome]
    assert tip["operation_final_digest"] == manager.read_world(world_oid).transition["operation_final"]["digest"]


def _fail_next_co_write(monkeypatch) -> None:
    """Force the next atomic co-write transaction to be rejected, so nothing moves."""
    monkeypatch.setattr(
        "vcs_core._incremental._co_write.run_update_ref_stdin",
        lambda _repo, _moves: UpdateRefStdinResult(ok=False, detail="simulated co-write rejection"),
    )


def test_world_operation_journal_close_transaction_failure_is_all_or_none(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-close-fail", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-close-fail",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-close-fail", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-close-fail", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-close-fail", world_oid=world_oid)
    _record_publishing(manager, "op-close-fail", world_oid=world_oid, input_world_oid=p0)
    manager.record_operation_published("op-close-fail", world_oid=world_oid)
    open_ref = operation_journal_ref("open", "op-close-fail")
    assert manager._open_journal_index().read_open_refs() == {open_ref}

    _fail_next_co_write(monkeypatch)
    with pytest.raises(InvalidRepositoryStateError):
        manager.close_operation_journal("op-close-fail", world_oid=world_oid)

    # all-or-none: nothing moved -> open stays nonterminal, no closed ref, index entry intact
    assert manager.read_operation_journal("op-close-fail").tip.payload["status"] == "published"
    with pytest.raises(InvalidRepositoryStateError, match="ref is missing"):
        manager.read_operation_journal("op-close-fail", family="closed")
    assert manager._open_journal_index().read_open_refs() == {open_ref}


def test_world_operation_journal_close_atomically_publishes_terminal_and_deletes_open(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-close-ok", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-close-ok",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-close-ok", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-close-ok", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-close-ok", world_oid=world_oid)
    _record_publishing(manager, "op-close-ok", world_oid=world_oid, input_world_oid=p0)
    manager.record_operation_published("op-close-ok", world_oid=world_oid)

    manager.close_operation_journal("op-close-ok", world_oid=world_oid)

    # atomic success: terminal published, open ref deleted together (no split-brain), index tombstoned
    assert manager.read_operation_journal("op-close-ok", family="closed").tip.payload["status"] == "closed"
    assert operation_journal_ref("open", "op-close-ok") not in manager.world_store.repo.references
    assert manager._open_journal_index().read_open_refs() == frozenset()


def test_world_operation_journal_open_index_tracks_opens(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-a", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
    )
    manager.open_operation_journal(
        operation_id="op-b", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
    )
    index = manager._open_journal_index()
    expected = {operation_journal_ref("open", "op-a"), operation_journal_ref("open", "op-b")}
    assert index.read_open_refs() == expected
    assert index.read_open_refs() == manager._scan_open_operation_journal_refs()  # index == authority scan
    assert index.verify_against_authority().ok


def test_world_operation_journal_corrupt_open_index_blocks_open(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-first", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
    )
    # corrupt the open-journal index record (a commit with no index blob)
    repo = manager.world_store.repo
    index_ref = world_open_operation_journal_index_ref(manager.world_store.world_store_id)
    sig = pygit2.Signature("t", "t@e.invalid")
    corrupt = repo.create_commit(None, sig, sig, "corrupt", repo.TreeBuilder().write(), [])
    repo.references.create(index_ref, corrupt, force=True)

    # a new open fails closed (the co-write's prepare reads the corrupt index and raises); no partial write
    with pytest.raises(InvalidRepositoryStateError):
        manager.open_operation_journal(
            operation_id="op-second", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
        )
    assert operation_journal_ref("open", "op-second") not in repo.references


def test_world_operation_journal_cleanup_accepts_legacy_open_ref_at_terminal_tip(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-legacy-open-terminal", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-legacy-open-terminal",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-legacy-open-terminal", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-legacy-open-terminal", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-legacy-open-terminal", world_oid=world_oid)
    _record_publishing(manager, "op-legacy-open-terminal", world_oid=world_oid, input_world_oid=p0)
    manager.record_operation_published("op-legacy-open-terminal", world_oid=world_oid)
    manager.close_operation_journal("op-legacy-open-terminal", world_oid=world_oid)
    closed_tip = manager.read_operation_journal("op-legacy-open-terminal", family="closed").tip.oid
    open_ref = operation_journal_ref("open", "op-legacy-open-terminal")
    manager.world_store.repo.references.create(open_ref, pygit2.Oid(hex=closed_tip))
    # The manual open ref is out-of-model: the close already tombstoned the index, so the authority
    # now has an open ref the index lacks -> stale drift (the corrupting direction the backstop guards).
    assert manager.verify_open_operation_journal_index().status == "stale"

    assert manager.cleanup_stale_terminal_operation_open_ref(
        "op-legacy-open-terminal",
        terminal_family="closed",
    )

    # The cleanup co-write atomically deletes the open ref; the index tombstone is an idempotent
    # no-op (it never indexed this out-of-model ref), so index and authority agree again.
    assert open_ref not in manager.world_store.repo.references
    assert manager._open_journal_index().read_open_refs() == frozenset()
    assert manager.verify_open_operation_journal_index().status == "fresh"


def test_world_operation_journal_cleanup_transaction_failure_keeps_open_ref(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-cleanup-fail", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-cleanup-fail",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-cleanup-fail", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-cleanup-fail", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-cleanup-fail", world_oid=world_oid)
    _record_publishing(manager, "op-cleanup-fail", world_oid=world_oid, input_world_oid=p0)
    manager.record_operation_published("op-cleanup-fail", world_oid=world_oid)
    manager.close_operation_journal("op-cleanup-fail", world_oid=world_oid)
    closed_tip = manager.read_operation_journal("op-cleanup-fail", family="closed").tip.oid
    open_ref = operation_journal_ref("open", "op-cleanup-fail")
    manager.world_store.repo.references.create(open_ref, pygit2.Oid(hex=closed_tip))

    _fail_next_co_write(monkeypatch)
    with pytest.raises(InvalidRepositoryStateError):
        manager.cleanup_stale_terminal_operation_open_ref("op-cleanup-fail", terminal_family="closed")

    # all-or-none: the co-write was rejected, so the stale open ref is NOT deleted
    assert open_ref in manager.world_store.repo.references


def test_world_operation_journal_rebuild_reconciles_out_of_model_open_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-indexed", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
    )
    assert manager.verify_open_operation_journal_index().status == "fresh"

    # An out-of-model writer creates an open ref bypassing the co-write -> the index misses it. This
    # is the residual STALE-under-report hazard the writer model admits (a manual/private-ref edit).
    indexed_tip = manager.read_operation_journal("op-indexed", family="open").tip.oid
    out_of_model = operation_journal_ref("open", "op-out-of-model")
    manager.world_store.repo.references.create(out_of_model, pygit2.Oid(hex=indexed_tip))
    assert manager.verify_open_operation_journal_index().status == "stale"

    # The recovery rebuild reconciles the drift from the authoritative open-ref scan.
    manager.rebuild_open_operation_journal_index()
    assert manager.verify_open_operation_journal_index().status == "fresh"
    assert out_of_model in manager._open_journal_index().read_open_refs()


def test_world_operation_journal_open_retries_on_index_ref_contention(tmp_path, monkeypatch) -> None:
    """The journal-level proof of the co-write retry: a concurrent process advances the shared index
    ref mid-open, our batch loses the index CAS, and the retry re-reads -> re-prepares -> re-batches
    and converges -- the open journal created exactly once, no orphan, index and authority consistent.
    (The primitive-level classified retry is covered in test_incremental_contract; this is the real
    manager path through open_operation_journal.)"""
    import vcs_core._incremental._co_write as co_write_mod

    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    open_ref = operation_journal_ref("open", "op-contended")
    competitor_ref = operation_journal_ref("open", "op-competitor")
    real_run = co_write_mod.run_update_ref_stdin
    state = {"n": 0}

    def injected(repo, moves):
        state["n"] += 1
        if state["n"] == 1:
            # A concurrent process opens a DIFFERENT operation, advancing the shared index ref off
            # the base our batch prepared against (uses cas_update_ref / references.create, not the
            # patched --stdin, so no recursion). Our batch then loses the index CAS.
            repo.references.create(competitor_ref, pygit2.Oid(hex=p0))
            manager._open_journal_index().rebuild_from_durable_history()
            return UpdateRefStdinResult(ok=False, detail="lost the index CAS to a concurrent open")
        return real_run(repo, moves)

    monkeypatch.setattr(co_write_mod, "run_update_ref_stdin", injected)
    entry = manager.open_operation_journal(
        operation_id="op-contended", operation_kind="shepherd.task", target_ref=DEFAULT_GROUND_REF, input_world_oid=p0
    )

    assert state["n"] >= 2  # attempt 1 lost the index CAS and retried (non-vacuous)
    assert entry.payload["status"] == "opened"
    assert open_ref in manager.world_store.repo.references  # the open journal was created...
    assert manager.read_operation_journal("op-contended").tip.oid == entry.oid  # ...exactly once, no orphan
    # re-folded onto the competitor's base: index holds BOTH opens and equals the authority
    assert manager._open_journal_index().read_open_refs() == {open_ref, competitor_ref}
    assert manager.verify_open_operation_journal_index().status == "fresh"


def test_world_operation_journal_open_surfaces_authority_conflict_without_retry(tmp_path, monkeypatch) -> None:
    """The negative case: an AUTHORITY-ref precondition failure (a same-operation race created the
    open ref) is a real conflict -- surfaced immediately, never retried as index contention."""
    import vcs_core._incremental._co_write as co_write_mod

    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    open_ref = operation_journal_ref("open", "op-conflict")
    real_run = co_write_mod.run_update_ref_stdin
    state = {"n": 0}

    def injected(repo, moves):
        state["n"] += 1
        if state["n"] == 1:
            # A concurrent process created the SAME open ref: the create-only authority precondition
            # now fails. This is a conflict, not index contention -> must surface, not retry.
            repo.references.create(open_ref, pygit2.Oid(hex=p0))
            return UpdateRefStdinResult(ok=False, detail="lost the authority CAS")
        return real_run(repo, moves)

    monkeypatch.setattr(co_write_mod, "run_update_ref_stdin", injected)
    with pytest.raises(InvalidRepositoryStateError, match="authority ref precondition"):
        manager.open_operation_journal(
            operation_id="op-conflict",
            operation_kind="shepherd.task",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
        )

    assert state["n"] == 1  # surfaced immediately, NOT retried


def test_world_operation_journal_archive_transaction_failure_is_all_or_none(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-archive-fail",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.fail_operation_journal("op-archive-fail", error="boom")
    open_ref = operation_journal_ref("open", "op-archive-fail")

    _fail_next_co_write(monkeypatch)
    with pytest.raises(InvalidRepositoryStateError):
        manager.archive_operation_journal("op-archive-fail")

    # all-or-none: nothing moved -> open stays failed, no archived ref, index entry intact
    assert manager.read_operation_journal("op-archive-fail").tip.payload["status"] == "failed"
    with pytest.raises(InvalidRepositoryStateError, match="ref is missing"):
        manager.read_operation_journal("op-archive-fail", family="archived")
    assert manager._open_journal_index().read_open_refs() == {open_ref}


def test_world_operation_journal_rejects_operation_id_reuse_across_families(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-reused-open",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )

    with pytest.raises(InvalidRepositoryStateError, match="already exists"):
        manager.open_operation_journal(
            operation_id="op-reused-open",
            operation_kind="shepherd.task",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
        )

    manager.open_operation_journal(
        operation_id="op-reused-archived",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.fail_operation_journal("op-reused-archived", error="boom")
    manager.archive_operation_journal("op-reused-archived")

    with pytest.raises(InvalidRepositoryStateError, match="already exists"):
        manager.open_operation_journal(
            operation_id="op-reused-archived",
            operation_kind="shepherd.task",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
        )


def test_world_operation_journal_rejects_invalid_lifecycle_transition(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-invalid", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-invalid",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )

    with pytest.raises(InvalidRepositoryStateError, match="invalid operation journal transition"):
        manager.close_operation_journal("op-invalid", world_oid=world_oid)


def test_world_operation_journal_requires_finalized_before_world_committed(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-unfinalized", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-unfinalized",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-unfinalized", world_oid, input_world_oid=p0)

    with pytest.raises(InvalidRepositoryStateError, match="prepared -> world_committed"):
        manager.record_operation_world_committed("op-unfinalized", world_oid=world_oid)


def test_world_operation_journal_requires_publishing_before_published(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-unpublished-intent", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-unpublished-intent",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-unpublished-intent", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-unpublished-intent", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-unpublished-intent", world_oid=world_oid)

    with pytest.raises(InvalidRepositoryStateError, match="world_committed -> published"):
        manager.record_operation_published("op-unpublished-intent", world_oid=world_oid)


def test_world_operation_journal_rejects_world_oid_change_after_committed(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-world-oid-change", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-world-oid-change",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-world-oid-change", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-world-oid-change", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-world-oid-change", world_oid=world_oid)
    _record_publishing(manager, "op-world-oid-change", world_oid=world_oid, input_world_oid=p0)
    other_workspace = _prepared_existing_revision(
        manager,
        "refs/heads/other",
        operation_id="op-world-oid-change-import",
        payload={"label": "other"},
        semantic_op="import",
    )
    other_world_oid = _world_for_operation(manager, "op-world-oid-change", other_workspace, parent_world=p0)

    with pytest.raises(InvalidRepositoryStateError, match="world_oid cannot change"):
        manager.record_operation_published("op-world-oid-change", world_oid=other_world_oid)

    tip = manager.read_operation_journal("op-world-oid-change").tip.payload
    assert tip["status"] == "publishing"
    assert tip["world_oid"] == world_oid


def test_world_operation_journal_rejects_world_oid_change_after_published(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-final-digest-change", workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id="op-final-digest-change",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, "op-final-digest-change", world_oid, input_world_oid=p0)
    _record_finalized_world(manager, "op-final-digest-change", world_oid, input_world_oid=p0)
    manager.record_operation_world_committed("op-final-digest-change", world_oid=world_oid)
    _record_publishing(manager, "op-final-digest-change", world_oid=world_oid, input_world_oid=p0)
    manager.record_operation_published("op-final-digest-change", world_oid=world_oid)
    other_workspace = _prepared_existing_revision(
        manager,
        "refs/heads/other",
        operation_id="op-final-digest-change-import",
        payload={"label": "other"},
        semantic_op="import",
    )
    other_world_oid = _world_for_operation(manager, "op-final-digest-change", other_workspace, parent_world=p0)

    with pytest.raises(InvalidRepositoryStateError, match="world_oid cannot change"):
        manager.close_operation_journal("op-final-digest-change", world_oid=other_world_oid)

    tip = manager.read_operation_journal("op-final-digest-change").tip.payload
    assert tip["status"] == "published"
    assert tip["operation_final_digest"] == manager.read_world(world_oid).transition["operation_final"]["digest"]


def test_world_operation_journal_fsck_reports_missing_candidate_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-missing-candidate",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    candidate, candidate_commit = create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id="op-missing-candidate",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(workspace,),
        world_store=manager.world_store,
    )
    snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
        )
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition("op-missing-candidate", parents=[p0]),
        operation_final=_operation_final(
            "op-missing-candidate",
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id="op-missing-candidate")],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    _record_prepared_world(manager, "op-missing-candidate", world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    manager.store("store_workspace").repo.references[candidate.ref].delete()

    report = manager.fsck_operation_journal("op-missing-candidate")

    assert not report.ok
    assert report.issues == (f"operation journal candidate ref is missing: {candidate.ref}",)


def test_world_operation_journal_archived_ref_survives_after_open_ref_delete(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    manager.open_operation_journal(
        operation_id="op-archive",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    manager.fail_operation_journal("op-archive", error="child failed")
    manager.archive_operation_journal("op-archive")

    history = manager.read_operation_journal("op-archive", family="archived")

    assert [entry.payload["status"] for entry in history.entries] == ["opened", "failed", "archived"]
    assert history.tip.payload["error"] == "child failed"
    assert operation_journal_ref("open", "op-archive") not in manager.world_store.repo.references


def test_world_operation_journal_fsck_rejects_noncanonical_journal_record(tmp_path) -> None:
    manager = _manager(tmp_path)
    _write_manual_journal_commit(
        manager,
        operation_id="op-corrupt",
        payload_bytes=b'{"schema":"vcscore/operation-journal/v1"}',
        ref=operation_journal_ref("open", "op-corrupt"),
    )

    report = manager.fsck_operation_journal("op-corrupt")

    assert not report.ok
    assert "canonical record is missing the vcs-core v2 domain prefix" in report.issues[0]


@pytest.mark.parametrize(
    ("status", "payload_extra", "expected"),
    [
        ("world_committed", {"world_oid": "1" * 40}, "operation_final_digest"),
        (
            "publishing",
            {"world_oid": "1" * 40, "operation_final_digest": "sha256:" + "0" * 64},
            "publication_plan",
        ),
        ("published", {"operation_final_digest": "sha256:" + "0" * 64}, "world_oid"),
        ("published", {"world_oid": "1" * 40, "operation_final_digest": "sha256:" + "0" * 64}, "publication_plan"),
        ("closed", {"world_oid": "1" * 40}, "operation_final_digest"),
        ("closed", {"world_oid": "1" * 40, "operation_final_digest": "sha256:" + "0" * 64}, "publication_plan"),
        ("failed", {}, "error"),
        ("archived", {}, "error"),
    ],
)
def test_world_operation_journal_fsck_requires_status_specific_fields(
    tmp_path,
    status: str,
    payload_extra: dict[str, object],
    expected: str,
) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": f"op-missing-{expected}-{status}",
        "operation_kind": "shepherd.task",
        "status": status,
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": p0,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
        **payload_extra,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=str(payload["operation_id"]),
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", str(payload["operation_id"])),
    )

    report = manager.fsck_operation_journal(str(payload["operation_id"]))

    assert not report.ok
    assert expected in report.issues[0]


def test_world_operation_journal_fsck_rejects_self_digested_malformed_publication_plan(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    operation_id = "op-malformed-publication-plan"
    raw_plan = {
        "schema": "vcscore/world-publication-plan/v1",
        "authority_ref": DEFAULT_GROUND_REF,
        "world_store_id": manager.world_store.world_store_id,
        "world_oid": p0,
        "expected_oid": None,
        "input_world_oid": None,
    }
    plan_digest = canonical_digest(raw_plan)
    plan = {**raw_plan, "publication_plan_digest": plan_digest}
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": operation_id,
        "operation_kind": "shepherd.task",
        "status": "publishing",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": None,
        "world_oid": p0,
        "operation_final_digest": "sha256:" + "0" * 64,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "publication_plan": plan,
        "publication_plan_digest": plan_digest,
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
    )

    report = manager.fsck_operation_journal(operation_id)

    assert not report.ok
    assert "missing publication plan fields" in report.issues[0]


def test_world_operation_journal_fsck_rejects_tampered_finalized_replay_payload(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    operation_id = "op-bad-finalized-replay"
    world_oid = _world_for_operation(manager, operation_id, workspace, parent_world=p0)
    world = manager.read_world(world_oid)
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": operation_id,
        "operation_kind": "shepherd.task",
        "status": "finalized",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": p0,
        "candidate_refs": [],
        "candidate_outcomes": list(world.operation_final["candidate_outcomes"]),
        "selected": dict(world.operation_final["selected"]),
        **_journal_replay_payload(manager, world_oid),
        "snapshot_digest": "sha256:" + "0" * 64,
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
    )

    report = manager.fsck_operation_journal(operation_id)

    assert not report.ok
    assert "snapshot_digest disagrees with snapshot" in report.issues[0]


def test_world_operation_journal_fsck_rejects_selected_drift_from_world_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    _p0, workspace = _published_base_world(manager)
    operation_id = "op-selected-drift"
    world_oid = _world_for_operation(manager, operation_id, workspace)
    digest = str(manager.read_world(world_oid).transition["operation_final"]["digest"])
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": operation_id,
        "operation_kind": "shepherd.task",
        "status": "closed",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": world_oid,
        "world_oid": world_oid,
        "operation_final_digest": digest,
        "candidate_refs": [],
        "candidate_outcomes": [],
        **_journal_replay_payload(manager, world_oid),
        **_publication_plan_payload(manager, world_oid, input_world_oid=None),
        "selected": {"workspace": "0" * 40},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("closed", operation_id),
    )

    report = manager.fsck_operation_journal(operation_id, family="closed")

    assert not report.ok
    assert "selected heads disagree with operation_final" in report.issues[0]


def test_world_operation_journal_fsck_rejects_candidate_outcome_drift_from_world_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    operation_id = "op-outcome-drift"
    candidate, candidate_commit = _prepared_workspace_candidate(manager, operation_id, workspace)
    snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
        )
    )
    outcome = {"binding": "workspace", "candidate": candidate.head, "outcome": "selected"}
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition(operation_id, parents=[p0]),
        operation_final=_operation_final(
            operation_id,
            {"workspace": candidate.head},
            outcomes=[outcome],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    digest = str(manager.read_world(world_oid).transition["operation_final"]["digest"])
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": operation_id,
        "operation_kind": "shepherd.task",
        "status": "closed",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": p0,
        "world_oid": world_oid,
        "operation_final_digest": digest,
        "candidate_refs": [],
        "selected": {"workspace": candidate.head},
        **_journal_replay_payload(manager, world_oid),
        **_publication_plan_payload(manager, world_oid, input_world_oid=p0),
        "candidate_outcomes": [],
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("closed", operation_id),
    )

    report = manager.fsck_operation_journal(operation_id, family="closed")

    assert not report.ok
    assert "candidate_outcomes disagree with operation_final" in report.issues[0]


def test_world_operation_journal_fsck_rejects_operation_id_drift_from_world_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    _p0, workspace = _published_base_world(manager)
    world_oid = _world_for_operation(manager, "op-world", workspace)
    digest = str(manager.read_world(world_oid).transition["operation_final"]["digest"])
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-journal",
        "operation_kind": "shepherd.task",
        "status": "closed",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": world_oid,
        "world_oid": world_oid,
        "operation_final_digest": digest,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {"workspace": workspace},
        **_journal_replay_payload(manager, world_oid),
        **_publication_plan_payload(manager, world_oid, input_world_oid=None),
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id="op-journal",
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("closed", "op-journal"),
    )

    report = manager.fsck_operation_journal("op-journal", family="closed")

    assert not report.ok
    assert "operation id disagrees with operation-final" in report.issues[0]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("world_oid", "2" * 40),
        ("operation_final_digest", "sha256:" + "1" * 64),
        ("selected", {"workspace": "0" * 40}),
        ("candidate_outcomes", [{"binding": "workspace", "candidate": "0" * 40, "outcome": "selected"}]),
    ],
)
def test_world_operation_journal_fsck_rejects_monotonic_field_changes(
    tmp_path,
    field: str,
    replacement: object,
) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    operation_id = f"op-mutated-{field}"
    world_oid = _world_for_operation(manager, operation_id, workspace, parent_world=p0)
    manager.open_operation_journal(
        operation_id=operation_id,
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, operation_id, world_oid, input_world_oid=p0)
    _record_finalized_world(manager, operation_id, world_oid, input_world_oid=p0)
    manager.record_operation_world_committed(operation_id, world_oid=world_oid)
    prior = manager.read_operation_journal(operation_id).tip
    payload = {
        **prior.payload,
        **_publication_plan_payload(manager, world_oid, input_world_oid=p0),
        field: replacement,
        "status": "publishing",
        "seq": int(prior.payload["seq"]) + 1,
        "previous_journal_oid": prior.oid,
        "updated_at_unix_ns": int(prior.payload["updated_at_unix_ns"]) + 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
        parents=(prior.oid,),
    )

    report = manager.fsck_operation_journal(operation_id)

    assert not report.ok
    assert f"{field} cannot change" in report.issues[0] or field in report.issues[0]


def test_world_operation_journal_fsck_rejects_candidate_ref_drift_after_finalization(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, workspace = _published_base_world(manager)
    operation_id = "op-mutated-candidate-refs"
    candidate, candidate_commit = _prepared_workspace_candidate(manager, operation_id, workspace)
    snapshot = _snapshot(
        manager.substrate_head(
            "store_workspace", binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef"
        )
    )
    world_oid = manager.create_unsafe_world(
        snapshot=snapshot,
        transition=_transition(operation_id, parents=[p0]),
        operation_final=_operation_final(
            operation_id,
            {"workspace": candidate.head},
            outcomes=[_candidate_outcome(manager, candidate_commit, final_operation_id=operation_id)],
            candidate_commits=[candidate_commit],
        ),
        parents=(p0,),
    )
    manager.open_operation_journal(
        operation_id=operation_id,
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=p0,
    )
    _record_prepared_world(manager, operation_id, world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    _record_finalized_world(manager, operation_id, world_oid, input_world_oid=p0, candidate_refs=(candidate,))
    prior = manager.read_operation_journal(operation_id).tip
    payload = {
        **prior.payload,
        "candidate_refs": [],
        "world_oid": world_oid,
        "status": "world_committed",
        "seq": int(prior.payload["seq"]) + 1,
        "previous_journal_oid": prior.oid,
        "updated_at_unix_ns": int(prior.payload["updated_at_unix_ns"]) + 1,
    }
    _write_manual_journal_commit(
        manager,
        operation_id=operation_id,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
        parents=(prior.oid,),
    )

    report = manager.fsck_operation_journal(operation_id)

    assert not report.ok
    assert "candidate_refs cannot change after finalization" in report.issues[0]


def test_world_operation_journal_fsck_rejects_broken_parent_chain(tmp_path) -> None:
    manager = _manager(tmp_path)
    p0, _workspace = _published_base_world(manager)
    root_payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-broken",
        "operation_kind": "shepherd.task",
        "status": "opened",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": p0,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    root_oid = _write_manual_journal_commit(
        manager,
        operation_id="op-broken",
        payload_bytes=canonical_bytes(root_payload),
        ref=None,
    )
    child_payload = {
        **root_payload,
        "status": "prepared",
        "seq": 1,
        "previous_journal_oid": root_oid,
        "updated_at_unix_ns": 2,
    }
    _write_manual_journal_commit(
        manager,
        operation_id="op-broken",
        payload_bytes=canonical_bytes(child_payload),
        ref=operation_journal_ref("open", "op-broken"),
        parents=(),
    )

    report = manager.fsck_operation_journal("op-broken")

    assert not report.ok
    assert "previous_journal_oid disagrees with Git parent" in report.issues[0]


def _write_manual_journal_commit(
    manager: WorldStorageManager,
    *,
    operation_id: str,
    payload_bytes: bytes,
    ref: str | None,
    parents: tuple[str, ...] = (),
) -> str:
    repo = manager.world_store.repo
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        OPERATION_JOURNAL_PATH.split("/")[-1],
        repo.create_blob(payload_bytes),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root_builder = repo.TreeBuilder()
    insert_tree_entry(repo, root_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("vcs-core operation journal", "vcs-core@example.invalid")
    oid = create_commit_with_recovery(
        repo,
        None,
        signature,
        signature,
        f"manual journal {operation_id}",
        root_builder.write(),
        [pygit2.Oid(hex=parent) for parent in parents],
    )
    if ref is not None:
        repo.references.create(ref, oid, force=True)
    return str(oid)
