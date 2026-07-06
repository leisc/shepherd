# under-test: vcs_core._workspace_authority
from __future__ import annotations

import json
from pathlib import Path

import pytest
from vcs_core._query_inventory import (
    WORKSPACE_AUTHORITY_IDENTITY_MISMATCH,
    WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,
)
from vcs_core._workspace_authority import (
    WorkspaceAuthorityPending,
    clear_pending_workspace_authority,
    clear_pending_workspace_authority_for_scope,
    pending_workspace_authority_records,
    pending_workspace_authority_records_for_scope,
    read_pending_workspace_authority,
    workspace_authority_operation_labels,
    write_pending_workspace_authority,
)
from vcs_core._workspace_authority_inventory import probe_workspace_authority_pending


def test_workspace_authority_pending_round_trips_canonical_record(tmp_path: Path) -> None:
    pending = WorkspaceAuthorityPending(
        operation_id="wv_scan_op_123",
        source_operation_id="op_123",
        driver_command="scan",
        scope_name="task",
        scope_ref="refs/vcscore/scopes/task",
        scope_instance_id="scope-1",
        scope_world_id="world-1",
        expected_input_world_oid=None,
        scalar_source_commit="abc123",
    ).with_update(phase="scalar_committed")

    write_pending_workspace_authority(tmp_path, pending)

    assert read_pending_workspace_authority(tmp_path, "wv_scan_op_123") == pending
    assert pending_workspace_authority_records(tmp_path) == (pending,)
    assert pending_workspace_authority_records_for_scope(tmp_path, "refs/vcscore/scopes/task") == (pending,)
    assert workspace_authority_operation_labels(tmp_path) == ("wv_scan_op_123",)
    assert pending.workspace_output_binding == "workspace"

    clear_pending_workspace_authority(tmp_path, "wv_scan_op_123")
    assert pending_workspace_authority_records(tmp_path) == ()


def test_workspace_authority_pending_decodes_legacy_record_without_output_binding() -> None:
    pending = WorkspaceAuthorityPending.from_dict(
        {
            "schema": "vcscore/workspace-authority-pending/v1",
            "operation_id": "wv_scan_op_123",
            "source_operation_id": "op_123",
            "driver_command": "scan",
            "scope_name": "task",
            "scope_ref": "refs/vcscore/scopes/task",
            "scope_instance_id": "scope-1",
        }
    )

    assert pending.workspace_output_binding == "workspace"


def test_workspace_authority_pending_writes_reversible_locator(tmp_path: Path) -> None:
    pending = WorkspaceAuthorityPending(
        operation_id="wv/scan op 123",
        source_operation_id="op_123",
        driver_command="scan",
        scope_name="task",
        scope_ref="refs/vcscore/scopes/task",
        scope_instance_id="scope-1",
        scope_world_id="world-1",
        expected_input_world_oid=None,
        scalar_source_commit="abc123",
    ).with_update(phase="scalar_committed")

    write_pending_workspace_authority(tmp_path, pending)

    items = probe_workspace_authority_pending(tmp_path)
    assert len(items) == 1
    assert Path(str(items[0].locator)).name.startswith("b64u_")
    assert items[0].fields["locator_encoding"] == "b64u"
    assert items[0].fields["locator_operation_id"] == "wv/scan op 123"


def test_clear_pending_workspace_authority_for_scope_only_removes_matching_records(tmp_path: Path) -> None:
    task = WorkspaceAuthorityPending(
        operation_id="wv_scan_task",
        source_operation_id="op_task",
        driver_command="scan",
        scope_name="task",
        scope_ref="refs/vcscore/scopes/task",
        scope_instance_id="scope-1",
        scope_world_id="world-1",
        expected_input_world_oid=None,
        scalar_source_commit="abc123",
    ).with_update(phase="scalar_committed")
    ground = WorkspaceAuthorityPending(
        operation_id="wv_scan_ground",
        source_operation_id="op_ground",
        driver_command="scan",
        scope_name="ground",
        scope_ref="refs/vcscore/ground",
        scope_instance_id="ground",
        scope_world_id="world-ground",
        expected_input_world_oid=None,
        scalar_source_commit="def456",
    ).with_update(phase="scalar_committed")
    write_pending_workspace_authority(tmp_path, task)
    write_pending_workspace_authority(tmp_path, ground)

    assert clear_pending_workspace_authority_for_scope(tmp_path, "refs/vcscore/scopes/task") == ("wv_scan_task",)

    assert pending_workspace_authority_records(tmp_path) == (ground,)


def test_workspace_authority_pending_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="unsupported schema"):
        WorkspaceAuthorityPending.from_dict(
            {
                "schema": "wrong",
                "operation_id": "wv",
                "source_operation_id": "op",
                "driver_command": "scan",
                "scope_name": "ground",
                "scope_ref": "refs/vcscore/ground",
                "scope_instance_id": "ground",
            }
        )


def test_workspace_authority_inventory_reports_malformed_pending_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace-authority" / "pending"
    root.mkdir(parents=True)
    (root / "broken.json").write_text("not json")

    items = probe_workspace_authority_pending(tmp_path)

    assert len(items) == 1
    assert items[0].health.status == "present_corrupt"
    assert items[0].health.issue_codes == (WORKSPACE_AUTHORITY_PAYLOAD_CORRUPT,)
    assert workspace_authority_operation_labels(tmp_path) == ("broken.json (present_corrupt)",)


def test_workspace_authority_inventory_rejects_plain_locator(tmp_path: Path) -> None:
    pending = WorkspaceAuthorityPending(
        operation_id="wv_scan_plain",
        source_operation_id="op_legacy",
        driver_command="scan",
        scope_name="task",
        scope_ref="refs/vcscore/scopes/task",
        scope_instance_id="scope-1",
        scope_world_id="world-1",
        expected_input_world_oid=None,
        scalar_source_commit="abc123",
    ).with_update(phase="scalar_committed")
    root = tmp_path / "workspace-authority" / "pending"
    root.mkdir(parents=True)
    (root / "wv_scan_plain.json").write_text(json.dumps(pending.to_dict(), sort_keys=True, separators=(",", ":")))

    items = probe_workspace_authority_pending(tmp_path)

    assert pending_workspace_authority_records(tmp_path) == ()
    assert items[0].health.status == "identity_mismatch"
    assert items[0].health.issue_codes == (WORKSPACE_AUTHORITY_IDENTITY_MISMATCH,)
