# under-test: vcs_core._execution_wire
from __future__ import annotations

from vcs_core._execution_wire import (
    serialize_operation_history,
    serialize_operation_summary,
    serialize_recovery_snapshot,
)
from vcs_core.types import CommitInfo, OperationHistory, OperationSummary, RecoverySnapshot


def test_serialize_operation_summary_uses_explicit_field_inventory() -> None:
    summary = OperationSummary(
        operation_id="op-123",
        label="Example Operation",
        kind="test.example",
        status="ok",
        visibility="archived",
        world_id="world-123",
        world_name="example",
        world_ref="refs/vcscore/scopes/example",
        carrier_ref="refs/vcscore/archive/example-123",
        archived_via="discarded_world_ref",
        parent_operation_id="parent-1",
        effect_count=3,
        started_at=10.0,
        closed_at=20.0,
        anchor_oid="deadbeef",
        final_phase="completed",
    )

    assert serialize_operation_summary(summary) == {
        "operation_id": "op-123",
        "label": "Example Operation",
        "kind": "test.example",
        "status": "ok",
        "visibility": "archived",
        "world_id": "world-123",
        "world_name": "example",
        "world_ref": "refs/vcscore/scopes/example",
        "carrier_ref": "refs/vcscore/archive/example-123",
        "anchor_oid": "deadbeef",
        "effect_count": 3,
        "parent_operation_id": "parent-1",
        "final_phase": "completed",
        "archived_via": "discarded_world_ref",
    }


def test_serialize_operation_history_uses_explicit_summary_and_commit_fields() -> None:
    history = OperationHistory(
        summary=OperationSummary(
            operation_id="op-123",
            label=None,
            kind="test.example",
            status="ok",
            visibility="visible",
            world_id="world-123",
            world_name="example",
            world_ref="refs/vcscore/scopes/example",
            carrier_ref="refs/vcscore/scopes/example",
            effect_count=1,
        ),
        commits=(
            CommitInfo(
                oid="deadbeef",
                message="effect:Marker scope:example",
                timestamp=123.0,
                metadata={"type": "Marker", "label": "hello"},
                parent_oids=["cafebabe"],
            ),
        ),
    )

    assert serialize_operation_history(history) == {
        "summary": {
            "operation_id": "op-123",
            "label": None,
            "kind": "test.example",
            "status": "ok",
            "visibility": "visible",
            "world_id": "world-123",
            "world_name": "example",
            "world_ref": "refs/vcscore/scopes/example",
            "carrier_ref": "refs/vcscore/scopes/example",
            "anchor_oid": None,
            "effect_count": 1,
            "parent_operation_id": None,
            "final_phase": None,
            "archived_via": None,
        },
        "commits": [
            {
                "oid": "deadbeef",
                "message": "effect:Marker scope:example",
                "timestamp": 123.0,
                "metadata": {"type": "Marker", "label": "hello"},
                "parent_oids": ["cafebabe"],
            }
        ],
    }


def test_serialize_recovery_snapshot_uses_explicit_field_inventory() -> None:
    summary = OperationSummary(
        operation_id="op-archived",
        label="Archived Op",
        kind="test.example",
        status="error",
        visibility="archived",
        world_id="world-123",
        world_name="example",
        world_ref="refs/vcscore/scopes/example",
        carrier_ref="refs/vcscore/archive/ops/op-archived",
        archived_via="operation_ref",
        effect_count=2,
        final_phase="aborted",
    )
    snapshot = RecoverySnapshot(
        orphaned_scope_refs=("refs/vcscore/scopes/example",),
        open_operations=(),
        archived_recovery_operations=(summary,),
        orphaned_operations=(),
        workspace_authority_pending=("wv_scan_op_123",),
    )

    assert serialize_recovery_snapshot(snapshot) == {
        "orphaned_scope_refs": ["refs/vcscore/scopes/example"],
        "open_operations": [],
        "archived_recovery_operations": [
            {
                "operation_id": "op-archived",
                "label": "Archived Op",
                "kind": "test.example",
                "status": "error",
                "visibility": "archived",
                "world_id": "world-123",
                "world_name": "example",
                "world_ref": "refs/vcscore/scopes/example",
                "carrier_ref": "refs/vcscore/archive/ops/op-archived",
                "anchor_oid": None,
                "effect_count": 2,
                "parent_operation_id": None,
                "final_phase": "aborted",
                "archived_via": "operation_ref",
            }
        ],
        "orphaned_operations": [],
        "workspace_authority_pending": ["wv_scan_op_123"],
    }
