# under-test: vcs_core._world_queries
"""Unit tests for private v2 world query summaries."""

from __future__ import annotations

from vcs_core import WORLD_TRANSITION_SCHEMA, WorldSnapshot
from vcs_core._world_operation_runner import WorldOperationRunner
from vcs_core._world_queries import summarize_world
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


def test_world_summary_reports_selected_heads_pins_and_linked_journals(tmp_path) -> None:
    manager = _manager(tmp_path)
    runner = WorldOperationRunner(manager)
    w42 = manager.create_prepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        operation_id="op-initial-workspace",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 42},
        semantic_op="bootstrap",
    )
    p0_snapshot = WorldSnapshot((manager.substrate_head("store_workspace", binding="workspace", head=w42, role="r"),))
    p0 = manager.create_unsafe_world(
        snapshot=p0_snapshot,
        transition=_transition("op-initial"),
        operation_final=_bootstrap_operation_final(manager, "op-initial", w42),
    )
    assert manager.publish_root_world(ref=DEFAULT_GROUND_REF, world_oid=p0)
    w43_bundle = manager.create_prepared_json_candidate_bundle(
        "store_workspace",
        operation_id="op-summary",
        binding="workspace",
        payload={"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    snapshot = WorldSnapshot(
        (manager.substrate_head("store_workspace", binding="workspace", head=w43_bundle.candidate.head, role="r"),)
    )
    prepared = (
        OperationFinalBuilder("op-summary")
        .select_candidate_plan(
            plan=manager.plan_candidate_selection(
                operation_id="op-summary",
                selection=CandidateSelection.from_bundle(w43_bundle),
                role="r",
            )
        )
        .build_prepared(
            operation_kind="merge",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=p0,
            snapshot=snapshot,
            transition=_transition("op-summary", parents=[p0]),
            parents=(p0,),
            candidate_refs=(w43_bundle.candidate,),
        )
    )
    result = runner.publish_prepared_world(prepared)

    summary = summarize_world(manager, DEFAULT_GROUND_REF)

    assert summary.ok
    assert summary.world_oid == result.world_oid
    assert summary.operation_id == "op-summary"
    assert summary.selected_heads == {"workspace": w43_bundle.candidate.head}
    assert summary.pin_classification["published"]
    assert summary.journal_statuses == ("closed:closed",)
