from __future__ import annotations

from typing import TYPE_CHECKING

from vcs_core._app_blockers import AppBlocker
from vcs_core._app_readiness_projection import (
    app_blocker_from_inventory_item,
    app_blockers_from_inventory_items,
    app_blockers_from_readiness_result,
)
from vcs_core._query_inventory import (
    RECOVERY_MATERIALIZATION_RUN,
    RECOVERY_ORPHANED_OPERATION_REF,
    RECOVERY_ORPHANED_SCOPE_REF,
    RECOVERY_SCOPE_REGISTRY_MISMATCH,
    RECOVERY_SIBLING_GROUP_BLOCKER,
    WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,
    Health,
    InventoryIssue,
    InventoryItem,
    issue_id,
    missing,
    present_invalid,
)
from vcs_core._query_readiness import ReadinessRequest
from vcs_core._workspace_authority import WorkspaceAuthorityPending, write_pending_workspace_authority

if TYPE_CHECKING:
    from vcs_core.vcscore import VcsCore


def test_app_blocker_projection_adds_workspace_authority_provenance() -> None:
    item = _item(
        item_id="workspace_authority_pending:file:broken.json",
        domain="workspace_authority",
        kind="workspace_authority_pending",
        locator="/repo/.vcscore/workspace-authority/broken.json",
        health=present_invalid(
            primary_issue="corrupt",
            issue_codes=(WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,),
            authority_role="authoritative",
            status="present_corrupt",
        ),
        fields={},
        issue_code=WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,
    )

    blocker = app_blocker_from_inventory_item(item)

    assert blocker == AppBlocker(
        kind="workspace_authority",
        subject="broken.json (present_corrupt)",
        detail="Workspace authority operation 'broken.json (present_corrupt)' requires recovery before mutation.",
        hint="Run `vcs-core recover-workspace-authority` before mutating or materializing.",
        source_item_id=item.id,
        source_issue_id=item.issues[0].id,
        source_issue_code=WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,
    )


def test_app_blocker_projection_maps_recovery_facts_without_owner_context() -> None:
    orphaned_scope = _recovery_item(
        item_id="recovery:orphaned_scope:refs/vcscore/scopes/task",
        kind="orphaned_scope_ref",
        locator="refs/vcscore/scopes/task",
        fields={"scope_ref": "refs/vcscore/scopes/task", "scope_name": "task"},
        issue_code=RECOVERY_ORPHANED_SCOPE_REF,
    )
    orphaned_operation = _recovery_item(
        item_id="recovery:orphaned_operation:refs/vcscore/ops/op-1",
        kind="orphaned_operation_ref",
        locator="refs/vcscore/ops/op-1",
        fields={"operation_id": "op-1"},
        issue_code=RECOVERY_ORPHANED_OPERATION_REF,
    )
    sibling_group = _recovery_item(
        item_id="recovery:sibling_group:group-1",
        kind="sibling_group_blocker",
        locator="refs/vcscore/sibling-groups/group-1",
        fields={"label": "group-1 (open)", "group_id": "group-1"},
        issue_code=RECOVERY_SIBLING_GROUP_BLOCKER,
    )
    registry_mismatch = _recovery_item(
        item_id="recovery:scope_registry_mismatch:task",
        kind="scope_registry_mismatch",
        locator="refs/vcscore/scopes/task",
        fields={"scope_name": "task", "detail": "task registry entry points at a stale ref"},
        issue_code=RECOVERY_SCOPE_REGISTRY_MISMATCH,
    )
    materialization_run = _recovery_item(
        item_id="recovery:materialization_run:run-1",
        kind="materialization_run",
        locator="/repo/.vcscore/materialization-run.json",
        fields={"run_id": "run-1"},
        issue_code=RECOVERY_MATERIALIZATION_RUN,
    )

    blockers = app_blockers_from_inventory_items(
        (orphaned_scope, orphaned_operation, sibling_group, registry_mismatch, materialization_run)
    )

    assert [blocker.kind for blocker in blockers] == [
        "orphaned_scope",
        "orphaned_operation",
        "sibling_group",
        "scope_registry_mismatch",
        "materialization_recovery",
    ]
    assert {blocker.source_item_id for blocker in blockers} == {
        orphaned_scope.id,
        orphaned_operation.id,
        sibling_group.id,
        registry_mismatch.id,
        materialization_run.id,
    }
    assert blockers[0].subject == "task"
    assert blockers[1].subject == "op-1"
    assert blockers[4].source_issue_code == RECOVERY_MATERIALIZATION_RUN


def test_app_blocker_projection_skips_non_app_visible_inventory() -> None:
    item = _item(
        item_id="workspace_authority_pending:file:missing.json",
        domain="workspace_authority",
        kind="workspace_authority_pending",
        locator="/repo/.vcscore/workspace-authority/missing.json",
        health=missing(
            issue_codes=("workspace_authority_missing_file",), lifecycle="recoverable", authority_role="authoritative"
        ),
        fields={},
        issue_code="workspace_authority_missing_file",
    )

    assert app_blocker_from_inventory_item(item) is None


def test_app_blocker_projection_uses_readiness_policy_issue_for_workspace_authority(mg: VcsCore) -> None:
    _write_workspace_authority_pending(mg, "wv-projection")

    result = mg.query_readiness(
        ReadinessRequest.create(command="vcscore.push-status", requested_freshness="locked", allow_best_effort=False)
    )
    blockers = app_blockers_from_readiness_result(result)

    blocker = next(blocker for blocker in blockers if blocker.kind == "workspace_authority")
    item_id = "workspace_authority_pending:file:b64u_d3YtcHJvamVjdGlvbg.json"
    assert blocker.subject == "wv-projection"
    assert blocker.source_item_id == item_id
    assert blocker.source_issue_code == "readiness_workspace_authority_pending"
    assert blocker.source_issue_id == f"issue:{item_id}:readiness_workspace_authority_pending"


def _recovery_item(
    *,
    item_id: str,
    kind: str,
    locator: str,
    fields: dict[str, object],
    issue_code: str,
) -> InventoryItem:
    return _item(
        item_id=item_id,
        domain="recovery",
        kind=kind,
        locator=locator,
        health=present_invalid(
            primary_issue="dangling_dependency",
            issue_codes=(issue_code,),
            lifecycle="recoverable",
            authority_role="projection",
            status="recovery_required",
        ),
        fields=fields,
        issue_code=issue_code,
    )


def _item(
    *,
    item_id: str,
    domain: str,
    kind: str,
    locator: str,
    health: Health,
    fields: dict[str, object],
    issue_code: str,
) -> InventoryItem:
    issue = InventoryIssue(
        id=issue_id(item_id, issue_code),
        code=issue_code,
        message=f"{issue_code}: {item_id}",
        subject_id=item_id,
        locator=locator,
        recovery_hint="Recover before mutating.",
    )
    return InventoryItem(
        id=item_id,
        domain=domain,
        kind=kind,
        locator=locator,
        source_kind="fixture",
        source_store="coordinator",
        health=health,
        role=("recovery",),
        fields=fields,
        source_identity={"fixture": item_id},
        issues=(issue,),
    )


def _write_workspace_authority_pending(mg: VcsCore, operation_id: str) -> None:
    write_pending_workspace_authority(
        mg._repo_path,
        WorkspaceAuthorityPending(
            operation_id=operation_id,
            source_operation_id=f"{operation_id}-source",
            driver_command="test",
            scope_name=mg.ground.name,
            scope_ref=mg.ground.ref,
            scope_instance_id=mg.ground.instance_id,
            scope_world_id=mg.ground.world_id,
            expected_input_world_oid=None,
            scalar_source_commit=None,
        ),
    )
