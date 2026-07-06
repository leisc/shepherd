# under-test: vcs_core._workspace_capture_manifest
"""Tests for scalar capture reduction to workspace-state manifest bridging."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from vcs_core import Store, canonical_digest
from vcs_core._workspace_capture_manifest import (
    WORKSPACE_CAPTURE_REDUCER_VERSION,
    workspace_capture_reduction_from_effects,
    workspace_capture_state_from_store,
    workspace_state_payload_from_store,
)
from vcs_core._world_substrate_adapters import WORKSPACE_REVISION_SCHEMA, WORKSPACE_STATE_MANIFEST_SCHEMA
from vcs_core.types import EffectRecord, ScopeInfo


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class _FakeCommit:
    id = "c" * 40


class _FakeStore:
    def __init__(self, files: dict[str, tuple[bytes, int]]) -> None:
        self._files = files

    def list_workspace_files(self, ref: str) -> list[tuple[str, str, int]]:
        assert ref == "refs/vcscore/scopes/task"
        return [(path, "b" * 40, mode) for path, (_content, mode) in self._files.items()]

    def read_workspace_file(self, ref: str, path: str) -> bytes | None:
        assert ref == "refs/vcscore/scopes/task"
        record = self._files.get(path)
        return None if record is None else record[0]

    def resolve_to_commit(self, commitish: str) -> _FakeCommit:
        assert commitish == "refs/vcscore/scopes/task"
        return _FakeCommit()


def _scope() -> ScopeInfo:
    return ScopeInfo(
        name="task",
        ref="refs/vcscore/scopes/task",
        instance_id="scope-instance",
        creation_oid="a" * 40,
    )


def test_workspace_capture_reduction_from_effects_builds_digest_only_payload() -> None:
    reduction = workspace_capture_reduction_from_effects(
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="FilePatch",
                metadata={"path": "b.sh"},
                workspace_changes=(("b.sh", b"#!/bin/sh\n", 0o100755),),
            ),
            EffectRecord(
                effect_type="FileCreate",
                metadata={"path": "a.txt"},
                workspace_changes=(("a.txt", b"hello"),),
            ),
        ),
        covered_paths=("b.sh", "a.txt"),
        event_count=3,
    )

    assert reduction.payload["schema"] == WORKSPACE_REVISION_SCHEMA
    manifest = reduction.payload["state_manifest"]
    assert manifest == {
        "schema": WORKSPACE_STATE_MANIFEST_SCHEMA,
        "byte_authority": "digest-only",
        "entries": [
            {"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"hello")},
            {"path": "b.sh", "state": "present", "mode": 0o100755, "content_digest": _digest(b"#!/bin/sh\n")},
        ],
    }
    assert reduction.reduced_state_proof == {
        "command_operation_id": "cmd-1",
        "byte_authority": "digest-only",
        "manifest_digest": canonical_digest(manifest),
        "covered_paths": ["b.sh", "a.txt"],
        "event_count": 3,
        "reduced_effect_count": 2,
        "reducer": WORKSPACE_CAPTURE_REDUCER_VERSION,
    }


def test_workspace_capture_reduction_from_effects_records_deletes_and_last_path_state() -> None:
    reduction = workspace_capture_reduction_from_effects(
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="FilePatch",
                metadata={"path": "same.txt"},
                workspace_changes=(("same.txt", b"before"),),
            ),
            EffectRecord(
                effect_type="FileDelete",
                metadata={"path": "same.txt"},
                workspace_changes=(("same.txt", None),),
            ),
        ),
        covered_paths=("same.txt",),
        event_count=2,
    )

    assert reduction.payload["state_manifest"]["entries"] == [{"path": "same.txt", "state": "deleted"}]


def test_workspace_capture_reduction_from_effects_preserves_empty_noop_manifest() -> None:
    reduction = workspace_capture_reduction_from_effects(
        command_operation_id="cmd-1",
        effects=(),
        covered_paths=("transient.txt",),
        event_count=2,
    )

    assert reduction.payload["state_manifest"]["entries"] == []
    assert reduction.reduced_state_proof["covered_paths"] == ["transient.txt"]
    assert reduction.reduced_state_proof["reduced_effect_count"] == 0


def test_workspace_capture_reduction_from_effects_preserves_failed_origin() -> None:
    failed_origin = {"operation_id": "cmd-1", "exit_code": 7, "signal": None}

    reduction = workspace_capture_reduction_from_effects(
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="FileCreate",
                metadata={"path": "out.txt", "failed_command_origin": failed_origin},
                workspace_changes=(("out.txt", b"failed-final"),),
            ),
        ),
        covered_paths=("out.txt",),
        event_count=1,
        failed_command_origin=failed_origin,
    )

    assert reduction.reduced_state_proof["failed_command_origin"] == failed_origin
    assert reduction.reduced_state_proof["failed_command_origin"] is not failed_origin


def test_workspace_capture_reduction_from_effects_rejects_invalid_manifest_entry() -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        workspace_capture_reduction_from_effects(
            command_operation_id="cmd-1",
            effects=(
                EffectRecord(
                    effect_type="FilePatch",
                    metadata={"path": "../outside.txt"},
                    workspace_changes=(("../outside.txt", b"bad"),),
                ),
            ),
            covered_paths=("../outside.txt",),
            event_count=1,
        )


def test_workspace_capture_state_from_store_builds_full_state_manifest() -> None:
    reduction = workspace_capture_state_from_store(
        store=_FakeStore(
            {
                "kept.txt": (b"kept", 0o100644),
                "new.txt": (b"new", 0o100644),
                "run.sh": (b"#!/bin/sh\n", 0o100755),
            }
        ),
        scope=_scope(),
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="FileCreate",
                metadata={"path": "new.txt"},
                workspace_changes=(("new.txt", b"new"),),
            ),
        ),
        covered_paths=("new.txt",),
        event_count=1,
    )

    manifest = reduction.payload["state_manifest"]
    assert manifest["entries"] == [
        {"path": "kept.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"kept")},
        {"path": "new.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"new")},
        {"path": "run.sh", "state": "present", "mode": 0o100755, "content_digest": _digest(b"#!/bin/sh\n")},
    ]
    assert reduction.reduced_state_proof["manifest_digest"] == canonical_digest(manifest)
    assert reduction.reduced_state_proof["state_source"] == "scalar-scope-tree"
    assert reduction.reduced_state_proof["scope_name"] == "task"
    assert reduction.reduced_state_proof["state_source_commit"] == "c" * 40
    assert reduction.reduced_state_proof["deleted_paths"] == []


def test_workspace_state_payload_from_store_builds_full_state_manifest() -> None:
    result = workspace_state_payload_from_store(
        store=_FakeStore(
            {
                "a.txt": (b"a", 0o100644),
                "run.sh": (b"#!/bin/sh\n", 0o100755),
            }
        ),
        scope=_scope(),
    )

    assert result.workspace_tree_oid is None
    assert result.payload["schema"] == WORKSPACE_REVISION_SCHEMA
    assert result.payload["state_manifest"] == {
        "schema": WORKSPACE_STATE_MANIFEST_SCHEMA,
        "byte_authority": "digest-only",
        "entries": [
            {"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"a")},
            {"path": "run.sh", "state": "present", "mode": 0o100755, "content_digest": _digest(b"#!/bin/sh\n")},
        ],
    }


def test_workspace_capture_state_from_store_records_absent_covered_paths() -> None:
    reduction = workspace_capture_state_from_store(
        store=_FakeStore({}),
        scope=_scope(),
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="FileDelete",
                metadata={"path": "old.txt"},
                workspace_changes=(("old.txt", None),),
            ),
        ),
        covered_paths=("transient.txt", "old.txt"),
        event_count=2,
    )

    assert reduction.payload["state_manifest"]["entries"] == []
    assert reduction.reduced_state_proof["deleted_paths"] == ["old.txt", "transient.txt"]


def test_workspace_capture_state_tree_backed_applies_runtime_effects(tmp_path: Path) -> None:
    store = Store(str(tmp_path / ".vcscore"))
    store.create_root_commit()
    scope = store.fork(Store.GROUND_REF, "task")
    store._emit_effect(
        scope,
        "Seed",
        {},
        workspace_changes=[("kept.txt", b"kept"), ("old.txt", b"old")],
        substrate="test",
    )

    reduction = workspace_capture_state_from_store(
        store=store,
        scope=scope,
        command_operation_id="cmd-1",
        effects=(
            EffectRecord(
                effect_type="RuntimeChanges",
                metadata={},
                workspace_changes=(("runtime.txt", b"runtime"), ("old.txt", None)),
            ),
        ),
        covered_paths=("runtime.txt", "old.txt"),
        event_count=1,
        tree_backed=True,
    )

    assert reduction.workspace_tree_oid is not None
    manifest = reduction.payload["state_manifest"]
    assert manifest["byte_authority"] == "tree-backed"
    assert manifest["entries"] == [
        {"path": "kept.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"kept")},
        {
            "path": "runtime.txt",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _digest(b"runtime"),
        },
    ]
    assert reduction.reduced_state_proof["deleted_paths"] == ["old.txt"]
    assert reduction.reduced_state_proof["state_derivation"] == "scalar-scope-tree+runtime-effects/v1"
