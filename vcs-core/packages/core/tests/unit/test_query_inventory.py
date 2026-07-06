# under-test: vcs_core._query_inventory
from __future__ import annotations

from vcs_core._query_inventory import (
    InventoryIssue,
    InventoryItem,
    InventorySnapshot,
    health_to_json,
    issue_id,
    missing,
    present_invalid,
)


def test_inventory_snapshot_serializes_shared_dtos() -> None:
    item_id = "operation_journal:open:b64u_b3A"
    issue = InventoryIssue(
        id=issue_id(item_id, "operation_journal_payload_corrupt"),
        code="operation_journal_payload_corrupt",
        message="operation journal payload is not readable",
        subject_id=item_id,
        locator="refs/vcscore/ops/b64u_b3Blbg/b64u_b3A",
    )
    item = InventoryItem(
        id=item_id,
        domain="operation_journal",
        kind="v2_world_operation_journal",
        locator="refs/vcscore/ops/b64u_b3Blbg/b64u_b3A",
        source_kind="git_ref",
        source_store="coordinator",
        health=present_invalid(
            primary_issue="corrupt",
            issue_codes=("operation_journal_payload_corrupt",),
            authority_role="authoritative",
        ),
        role=("journal", "authority"),
        fields={"family": "open"},
        source_identity={"ref_target_oid": "1" * 40},
        issues=(issue,),
    )

    snapshot = InventorySnapshot.create(items=(item,), issues=(issue,))
    payload = snapshot.to_json()
    item_payload = item.to_json()

    assert payload["consistency"] == "best_effort"
    assert payload["items"] == [item_payload]
    assert payload["edges"] == []
    assert payload["issues"][0]["code"] == "operation_journal_payload_corrupt"
    assert item_payload["health"]["status"] == "present_corrupt"
    assert item_payload["issues"][0]["code"] == "operation_journal_payload_corrupt"


def test_health_absent_is_targeted_observation() -> None:
    health = missing(issue_codes=("operation_journal_missing_ref",), authority_role="authoritative")

    assert health_to_json(health)["presence"] == "absent"
    assert health_to_json(health)["primary_issue"] == "missing"
    assert health_to_json(health)["status"] == "missing"
