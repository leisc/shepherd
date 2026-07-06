# under-test: vcs_core._runtime_types
"""Tests for internal runtime-handle types and boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from vcs_core._runtime_types import ExecutionContext, OperationRefInfo
from vcs_core.types import ScopeInfo


def test_execution_context_captures_scope_identity() -> None:
    scope = ScopeInfo(
        name="task",
        ref="refs/vcscore/scopes/task",
        instance_id="scope-123",
        creation_oid="abc123",
        world_id="world-123",
    )

    context = ExecutionContext.from_scope(scope, session_id="session-1", parent_operation_id="parent-op")

    assert context.scope_ref == scope.ref
    assert context.scope_name == "task"
    assert context.scope_instance_id == "scope-123"
    assert context.world_id == "world-123"
    assert context.session_id == "session-1"
    assert context.parent_operation_id == "parent-op"
    assert context.matches_scope(scope)


def test_execution_context_rejects_scope_without_world_id() -> None:
    scope = ScopeInfo(
        name="task",
        ref="refs/vcscore/scopes/task",
        instance_id="scope-123",
        creation_oid="abc123",
    )

    with pytest.raises(RuntimeError, match="missing durable world_id"):
        ExecutionContext.from_scope(scope)


def test_operation_ref_info_exposes_durable_identity_helpers() -> None:
    operation = OperationRefInfo(
        handle_id="legacy-op",
        kind="marker.runtime",
        ref="refs/vcscore/ops/op_123",
        scope_ref="refs/vcscore/scopes/task",
        scope_instance_id="scope-123",
        parent_op_ref=None,
        base_oid="abc123",
        operation_id="op_123",
        operation_label="marker-step",
    )

    assert operation.durable_id == "op_123"
    assert operation.display_label == "marker-step"
    assert operation.handle_id == "legacy-op"


def test_operation_ref_info_is_internal_to_runtime_modules() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "vcs_core"
    allowed_importers = {
        "_vcscore_lifecycle.py",
        "_vcscore_runtime.py",
        "_operation_start_authority.py",
        "_store_operation_queries.py",
        "vcscore.py",
        "recording.py",
        "store.py",
    }
    importers: set[str] = set()

    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "vcs_core._runtime_types":
                continue
            if any(alias.name == "OperationRefInfo" for alias in node.names):
                importers.add(path.relative_to(src_root).as_posix())

    assert importers == allowed_importers
