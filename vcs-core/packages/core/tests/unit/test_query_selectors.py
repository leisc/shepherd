# under-test: vcs_core._query_selectors
from __future__ import annotations

import pytest
from vcs_core._query_inventory import (
    InventoryIssue,
    InventoryItem,
    InventorySnapshot,
    issue_id,
    present_invalid,
    present_valid,
)
from vcs_core._query_selectors import InventorySelectorError, parse_selector, select_inventory_items


def _item(
    item_id: str,
    *,
    domain: str,
    kind: str,
    status: str,
    validity: str,
    issue_code: str | None = None,
    role: tuple[str, ...] = (),
    fields: dict[str, object] | None = None,
) -> InventoryItem:
    issues = (
        (
            InventoryIssue(
                id=issue_id(item_id, issue_code),
                code=issue_code,
                message=issue_code,
                subject_id=item_id,
            ),
        )
        if issue_code is not None
        else ()
    )
    health = (
        present_valid(authority_role="authoritative")
        if validity == "valid"
        else present_invalid(
            primary_issue="corrupt",
            issue_codes=tuple(issue.code for issue in issues),
            authority_role="authoritative",
            status=status,
        )
    )
    return InventoryItem(
        id=item_id,
        domain=domain,
        kind=kind,
        locator=item_id,
        source_kind="test",
        source_store="coordinator",
        health=health,
        role=role,
        fields=dict(fields or {}),
        issues=issues,
    )


def test_selector_filters_invalid_items_by_domain_issue_and_field() -> None:
    valid = _item(
        "workspace-ok",
        domain="workspace_authority",
        kind="workspace_authority_pending",
        status="present_valid",
        validity="valid",
        role=("authority", "recovery"),
        fields={"operation_id": "op-ok"},
    )
    invalid = _item(
        "workspace-bad",
        domain="workspace_authority",
        kind="workspace_authority_pending",
        status="present_corrupt",
        validity="invalid",
        issue_code="workspace_authority_payload_corrupt",
        role=("authority", "recovery"),
        fields={"operation_id": "op-bad"},
    )
    journal = _item(
        "journal-bad",
        domain="operation_journal",
        kind="v2_world_operation_journal",
        status="present_corrupt",
        validity="invalid",
        issue_code="operation_journal_payload_corrupt",
        fields={"operation_id": "op-journal"},
    )
    snapshot = InventorySnapshot.create(items=(valid, invalid, journal))

    selected = select_inventory_items(
        snapshot,
        "domain=workspace_authority health.validity=invalid issue=workspace_authority_payload_corrupt",
    )

    assert selected == (invalid,)
    assert invalid.issues[0].code == "workspace_authority_payload_corrupt"
    assert select_inventory_items(snapshot, "field.operation_id=op-bad") == (invalid,)


def test_selector_supports_union_and_negation_without_reordering() -> None:
    first = _item("first", domain="recovery", kind="orphaned_scope_ref", status="recovery_required", validity="invalid")
    second = _item("second", domain="recovery", kind="dirty_push", status="recovery_required", validity="invalid")
    third = _item(
        "third", domain="operation_journal", kind="v2_world_operation_journal", status="present_valid", validity="valid"
    )
    snapshot = InventorySnapshot.create(items=(first, second, third))

    selected = select_inventory_items(snapshot, "domain=recovery -kind=dirty_push | kind=v2_world_operation_journal")

    assert selected == (first, third)


def test_selector_rejects_unknown_tokens() -> None:
    with pytest.raises(InventorySelectorError, match="unknown inventory selector token"):
        parse_selector("unknown=value")
