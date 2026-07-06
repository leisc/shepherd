# under-test: vcs_core._query_trace
from __future__ import annotations

from vcs_core._query_inventory import (
    InventoryIssue,
    InventoryItem,
    InventorySnapshot,
    issue_id,
    present_invalid,
    present_valid,
)
from vcs_core._query_trace import project_inventory_trace


def test_project_inventory_trace_cites_source_items() -> None:
    item = InventoryItem(
        id="operation_journal:b64u_b3Blbg:b64u_b3A",
        domain="operation_journal",
        kind="v2_world_operation_journal",
        locator="refs/vcscore/ops/b64u_b3Blbg/b64u_b3A",
        source_kind="git_ref",
        source_store="coordinator",
        health=present_valid(authority_role="authoritative"),
        fields={"family": "open", "operation_id": "op", "status": "opened", "seq": 0},
    )
    snapshot = InventorySnapshot.create(items=(item,))

    events = project_inventory_trace(snapshot)

    assert len(events) == 1
    assert events[0].kind == "journal.entry_recorded"
    assert events[0].source_item_id == item.id
    assert events[0].subject_id == "op"
    assert events[0].to_json()["fields"] == {"family": "open", "operation_id": "op", "status": "opened", "seq": 0}


def test_project_inventory_trace_preserves_recovery_issue_codes() -> None:
    item_id = "recovery:orphaned_scope:refs/vcscore/scopes/task"
    issue = InventoryIssue(
        id=issue_id(item_id, "recovery_orphaned_scope_ref"),
        code="recovery_orphaned_scope_ref",
        message="orphaned",
        subject_id=item_id,
    )
    item = InventoryItem(
        id=item_id,
        domain="recovery",
        kind="orphaned_scope_ref",
        locator="refs/vcscore/scopes/task",
        source_kind="runtime_state",
        source_store="coordinator",
        health=present_invalid(
            primary_issue="dangling_dependency",
            issue_codes=("recovery_orphaned_scope_ref",),
            authority_role="projection",
            status="recovery_required",
        ),
        fields={"scope_ref": "refs/vcscore/scopes/task"},
        issues=(issue,),
    )
    snapshot = InventorySnapshot.create(items=(item,))

    events = project_inventory_trace(snapshot)

    assert events[0].kind == "recovery.blocked"
    assert events[0].source_item_id == item_id
    assert events[0].issue_codes == ("recovery_orphaned_scope_ref",)
