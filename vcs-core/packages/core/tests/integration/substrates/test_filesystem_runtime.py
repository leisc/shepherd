# under-test: vcs_core._substrate_runtime
"""Filesystem substrate overlay and capture integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from vcs_core._capture_reducer import CaptureJournalEvent
from vcs_core._claims import ResourceClaim
from vcs_core._fs_capture import FsCaptureEvent
from vcs_core._substrate_runtime import BuiltInRuntimeBinding, build_builtin_substrate_context
from vcs_core.substrates import DeclarativeFilesystemSubstrate, FilesystemSubstrate
from vcs_core.types import ScopeInfo

from ...support.overlays import MockOverlayBackend as _MockOverlayBackend
from ...support.scopes import scope_runtime as _scope_runtime
from ...support.scopes import set_scope as _set_scope

if TYPE_CHECKING:
    from vcs_core.store import Store


def _capture_event(
    command_id: str,
    op: str,
    path: str,
    *,
    global_seq: int,
    pid: int = 123,
    proc_seq: int | None = None,
) -> CaptureJournalEvent:
    return CaptureJournalEvent(
        command_operation_id=command_id,
        binding_name="filesystem",
        op=op,  # type: ignore[arg-type]
        path=path,
        scope="task",
        scope_instance_id="iid-1",
        pid=pid,
        proc_seq=global_seq if proc_seq is None else proc_seq,
        global_seq=global_seq,
        event_seq=global_seq,
    )


def test_filesystem_branch_isolated_requires_backend(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")

    with pytest.raises(RuntimeError, match="no overlay backend"):
        fs.branch("task-overlay", parent_scope=parent, hints={"isolated": True})


def test_filesystem_execute_overlay_write_mutates_backend_and_suppresses_effects(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-overlay-write"))

    task = store.fork(store.GROUND_REF, "task-overlay-write")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(task.name, parent_scope=parent, hints={"isolated": True})
    outcome = fs.execute("write", task, path="overlay.py", content=b"content")

    assert outcome.effects == ()
    assert backend.diff_layer(task.name) == [("overlay.py", b"content", 0o100644)]


def test_filesystem_execute_overlay_write_preserves_explicit_filemode(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-overlay-mode"))

    task = store.fork(store.GROUND_REF, "task-overlay-mode")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(task.name, parent_scope=parent, hints={"isolated": True})

    executable = fs.execute("write", task, path="overlay.py", content=b"content", mode=0o100755)
    regular = fs.execute("write", task, path="overlay.py", content=b"content", mode=0o100644)

    assert executable.effects == ()
    assert regular.effects == ()
    assert backend.diff_layer(task.name) == [("overlay.py", b"content", 0o100644)]


def test_filesystem_execute_overlay_delete_mutates_backend_and_suppresses_effects(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-overlay-delete"))

    task = store.fork(store.GROUND_REF, "task-overlay-delete")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(task.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file(task.name, "overlay.py", b"content")

    outcome = fs.execute("delete", task, path="overlay.py")

    assert outcome.effects == ()
    assert backend.diff_layer(task.name) == [("overlay.py", None, 0)]


def test_filesystem_prepare_merge_generates_effects_from_overlay_diff(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)

    seed = store.fork(store.GROUND_REF, "task-overlay-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes(
        [
            ("existing.py", b"before"),
            ("delete.py", b"remove-me"),
        ]
    )
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-overlay-prepare")
    parent = ScopeInfo(
        name="ground",
        ref=store.GROUND_REF,
        instance_id="ground",
        creation_oid="",
    )

    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file(child.name, "existing.py", b"after")
    backend.write_file(child.name, "created.py", b"new", mode=0o100755)
    backend.delete_file(child.name, "delete.py")

    effects = fs.prepare_merge(child, parent)

    effect_types = {effect.metadata["path"]: effect.effect_type for effect in effects}
    assert effect_types == {
        "existing.py": "FilePatch",
        "created.py": "FileCreate",
        "delete.py": "FileDelete",
    }
    for effect in effects:
        assert effect.metadata["capture_mode"] == "reconciled"
        assert effect.metadata["capture_mechanism"] == "overlay-diff"
    assert {effect.metadata["path"]: effect.metadata["reconcile_reason"] for effect in effects} == {
        "existing.py": "missing_direct_create_or_patch",
        "created.py": "missing_direct_create_or_patch",
        "delete.py": "missing_direct_delete",
    }
    assert {effect.metadata["path"]: effect.workspace_changes for effect in effects}["created.py"] == (
        ("created.py", b"new", 0o100755),
    )


def test_filesystem_overlay_changes_preserve_filemode(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)

    child = store.fork(store.GROUND_REF, "task-overlay-status")
    parent = ScopeInfo(
        name="ground",
        ref=store.GROUND_REF,
        instance_id="ground",
        creation_oid="",
    )

    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file(child.name, "bin/run.sh", b"#!/bin/sh\n", mode=0o100755)
    backend.delete_file(child.name, "deleted.txt")

    assert fs.overlay_changes(child.name) == [
        ("bin/run.sh", b"#!/bin/sh\n", 0o100755),
        ("deleted.txt", None),
    ]


def test_filesystem_prepare_merge_skips_paths_already_reflected_in_scope_tip(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)

    seed = store.fork(store.GROUND_REF, "task-overlay-direct-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("existing.py", b"before")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-overlay-direct")
    parent = ScopeInfo(
        name="ground",
        ref=store.GROUND_REF,
        instance_id="ground",
        creation_oid="",
    )
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})

    store._emit_effect(
        child,
        "FilePatch",
        {"path": "existing.py"},
        workspace_changes=[("existing.py", b"after")],
        substrate="filesystem",
    )

    backend.write_file(child.name, "existing.py", b"after")

    assert fs.prepare_merge(child, parent) == []


def test_filesystem_prepare_merge_skips_delete_when_scope_tip_already_absent(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)

    child = store.fork(store.GROUND_REF, "task-overlay-delete-suppressed")
    parent = ScopeInfo(
        name="ground",
        ref=store.GROUND_REF,
        instance_id="ground",
        creation_oid="",
    )
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.delete_file(child.name, "missing.py")

    assert fs.prepare_merge(child, parent) == []


def test_filesystem_effect_for_captured_write_close_creates_new_file(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-create"))

    child = store.fork(store.GROUND_REF, "task-capture-create")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-create", "captured.txt", b"hello")

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="write_close",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.txt",
            pid=123,
            proc_seq=1,
        ),
        seq=10,
    )

    assert effect is not None
    assert effect.effect_type == "FileCreate"
    assert effect.metadata == {
        "path": "captured.txt",
        "capture_mode": "direct",
        "capture_mechanism": "preload",
        "pid": 123,
        "proc_seq": 1,
        "seq": 10,
    }
    assert effect.workspace_changes == (("captured.txt", b"hello"),)


def test_filesystem_effect_for_captured_write_close_patches_existing_file(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-patch"))

    seed = store.fork(store.GROUND_REF, "task-capture-patch-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("captured.txt", b"before")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-patch")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-patch", "captured.txt", b"after")

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="write_close",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.txt",
            pid=123,
            proc_seq=2,
        ),
        seq=11,
    )

    assert effect is not None
    assert effect.effect_type == "FilePatch"
    assert effect.workspace_changes == (("captured.txt", b"after"),)


def test_filesystem_effect_for_captured_write_close_skips_identical_bytes(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-same"))

    seed = store.fork(store.GROUND_REF, "task-capture-same-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("captured.txt", b"same")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-same")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-same", "captured.txt", b"same")

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="write_close",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.txt",
            pid=123,
            proc_seq=3,
        ),
        seq=12,
    )

    assert effect is None


def test_filesystem_effect_for_captured_metadata_change_to_executable(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-executable"))

    seed = store.fork(store.GROUND_REF, "task-capture-executable-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("captured.sh", b"same")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-executable")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-executable", "captured.sh", b"same", mode=0o100755)

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="metadata_change",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.sh",
            pid=123,
            proc_seq=6,
        ),
        seq=15,
    )

    assert effect is not None
    assert effect.effect_type == "FilePatch"
    assert effect.workspace_changes == (("captured.sh", b"same", 0o100755),)


def test_filesystem_effect_for_captured_metadata_change_to_regular(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-regular"))

    seed = store.fork(store.GROUND_REF, "task-capture-regular-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("captured.sh", b"same", 0o100755)])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-regular")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-regular", "captured.sh", b"same", mode=0o100644)

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="metadata_change",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.sh",
            pid=123,
            proc_seq=7,
        ),
        seq=16,
    )

    assert effect is not None
    assert effect.effect_type == "FilePatch"
    assert effect.workspace_changes == (("captured.sh", b"same"),)


def test_filesystem_effect_for_captured_unlink_deletes_existing_file(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-delete"))

    seed = store.fork(store.GROUND_REF, "task-capture-delete-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("captured.txt", b"before")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-delete")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="unlink",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="captured.txt",
            pid=123,
            proc_seq=4,
        ),
        seq=13,
    )

    assert effect is not None
    assert effect.effect_type == "FileDelete"
    assert effect.workspace_changes == (("captured.txt", None),)


def test_filesystem_effect_for_captured_unlink_skips_already_absent_file(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-missing"))

    child = store.fork(store.GROUND_REF, "task-capture-missing")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="unlink",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="missing.txt",
            pid=123,
            proc_seq=5,
        ),
        seq=14,
    )

    assert effect is None


def test_filesystem_capture_reduction_collapses_create_then_delete(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-collapse"))

    child = store.fork(store.GROUND_REF, "task-capture-collapse")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.delete_file("task-capture-collapse", "transient.txt")

    effects = fs.effects_for_capture_reduction(
        child,
        (
            _capture_event("cmd-1", "write_close", "transient.txt", global_seq=1),
            _capture_event("cmd-1", "unlink", "transient.txt", global_seq=2),
        ),
    )

    assert effects == ()


def test_filesystem_capture_reduction_records_final_state_and_failed_origin(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-reduce"))

    child = store.fork(store.GROUND_REF, "task-capture-reduce")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-reduce", "out.txt", b"final")

    effects = fs.effects_for_capture_reduction(
        child,
        (_capture_event("cmd-2", "write_close", "out.txt", global_seq=4),),
        failed_command_origin={"operation_id": "cmd-2", "exit_code": 7, "signal": None},
    )

    assert len(effects) == 1
    effect = effects[0]
    assert effect.effect_type == "FileCreate"
    assert effect.workspace_changes == (("out.txt", b"final"),)
    assert effect.metadata["command_operation_id"] == "cmd-2"
    assert effect.metadata["capture_record"] == "reduction"
    assert effect.metadata["failed_command_origin"] == {
        "operation_id": "cmd-2",
        "exit_code": 7,
        "signal": None,
    }


def test_filesystem_capture_reduction_deduplicates_interleaved_same_path_events(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-interleaved"))

    child = store.fork(store.GROUND_REF, "task-capture-interleaved")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.write_file("task-capture-interleaved", "out.txt", b"final")

    effects = fs.effects_for_capture_reduction(
        child,
        (
            _capture_event("cmd-4", "write_close", "out.txt", global_seq=4, pid=202, proc_seq=2),
            _capture_event("cmd-4", "metadata_change", "out.txt", global_seq=2, pid=101, proc_seq=8),
            _capture_event("cmd-4", "write_close", "out.txt", global_seq=3, pid=202, proc_seq=1),
        ),
    )

    assert len(effects) == 1
    assert effects[0].effect_type == "FileCreate"
    assert effects[0].workspace_changes == (("out.txt", b"final"),)
    assert effects[0].metadata["command_operation_id"] == "cmd-4"


def test_filesystem_capture_reduction_deletes_pre_existing_file(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-reduce-delete"))

    seed = store.fork(store.GROUND_REF, "task-capture-reduce-delete-seed")
    seed_ctx = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    _set_scope(seed_ctx, seed)
    seed_ctx.record_changes([("old.txt", b"before")])
    store.merge(seed, store.GROUND_REF)

    child = store.fork(store.GROUND_REF, "task-capture-reduce-delete")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})
    backend.delete_file("task-capture-reduce-delete", "old.txt")

    effects = fs.effects_for_capture_reduction(
        child,
        (_capture_event("cmd-3", "unlink", "old.txt", global_seq=5),),
    )

    assert len(effects) == 1
    assert effects[0].effect_type == "FileDelete"
    assert effects[0].workspace_changes == (("old.txt", None),)


@pytest.mark.parametrize(
    "path",
    ["", ".", "../outside.txt", "subdir/../../outside.txt", "/tmp/outside.txt", ".vcscore/config.toml"],
)
def test_filesystem_effect_for_captured_event_ignores_invalid_workspace_paths(store: Store, path: str) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    fs.bind_runtime(_scope_runtime(fs._pipeline, isolated=True, overlay_base="task-capture-invalid"))

    child = store.fork(store.GROUND_REF, "task-capture-invalid")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})

    effect = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="write_close",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path=path,
            pid=123,
            proc_seq=6,
        ),
        seq=15,
    )

    assert effect is None


def test_filesystem_claim_policy_suppresses_authoritative_paths_only(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    workspace = Path(store.repo_path).parent.resolve()
    claimed_path = workspace / "shadow.db"

    fs.bind_runtime(
        BuiltInRuntimeBinding(
            pipeline=fs._pipeline,
            is_scope_or_ancestor_isolated=lambda _scope: False,
            overlay_base_scope_name=lambda _scope: "ground",
            working_directory_for_scope=lambda _scope: workspace,
            lookup_claim=lambda path: (
                ResourceClaim(
                    substrate="sqlite",
                    target_id="sqlite:main",
                    path=str(claimed_path),
                    policy="authoritative_suppress_fs",
                )
                if Path(path).resolve() == claimed_path
                else None
            ),
        )
    )

    assert fs._rel(claimed_path) is None

    fs.bind_runtime(
        BuiltInRuntimeBinding(
            pipeline=fs._pipeline,
            is_scope_or_ancestor_isolated=lambda _scope: False,
            overlay_base_scope_name=lambda _scope: "ground",
            working_directory_for_scope=lambda _scope: workspace,
            lookup_claim=lambda path: (
                ResourceClaim(
                    substrate="sqlite",
                    target_id="sqlite:main",
                    path=str(claimed_path),
                    policy="exclusive",
                )
                if Path(path).resolve() == claimed_path
                else None
            ),
        )
    )

    assert fs._rel(claimed_path) == "shadow.db"


def test_filesystem_capture_and_reconcile_ignore_authoritative_claims(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    workspace = Path(store.repo_path).parent.resolve()
    claimed_path = workspace / "shadow.db"
    fs.bind_runtime(
        BuiltInRuntimeBinding(
            pipeline=fs._pipeline,
            is_scope_or_ancestor_isolated=lambda _scope: True,
            overlay_base_scope_name=lambda _scope: "task-claimed-path",
            working_directory_for_scope=lambda _scope: workspace,
            lookup_claim=lambda path: (
                ResourceClaim(
                    substrate="sqlite",
                    target_id="sqlite:main",
                    path=str(claimed_path),
                    policy="authoritative_suppress_fs",
                )
                if Path(path).resolve() == claimed_path
                else None
            ),
        )
    )

    child = store.fork(store.GROUND_REF, "task-claimed-path")
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")
    fs.branch(child.name, parent_scope=parent, hints={"isolated": True})

    captured = fs.effect_for_captured_event(
        child,
        FsCaptureEvent(
            op="write_close",
            scope=child.name,
            scope_instance_id=child.instance_id,
            path="shadow.db",
            pid=123,
            proc_seq=6,
        ),
        seq=15,
    )
    reconciled = fs._reconciled_effect_for_change(child, "shadow.db", b"payload")

    assert captured is None
    assert reconciled is None


def test_filesystem_commit_and_discard_delegate_to_overlay_backend(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)
    parent = ScopeInfo(name="ground", ref=store.GROUND_REF, instance_id="ground", creation_oid="")

    fs.branch("task-commit", parent_scope=parent, hints={"isolated": True})
    fs.commit_merge("task-commit", parent_scope=parent)
    fs.branch("task-discard", parent_scope=parent, hints={"isolated": True})
    fs.discard("task-discard")

    assert backend.committed == [("task-commit", "ground")]
    assert backend.discarded == ["task-discard"]
    assert fs._overlay_scopes == set()


def test_filesystem_activate_and_deactivate_delegate_to_backend_but_push_uses_store_diff(store: Store) -> None:
    backend = _MockOverlayBackend()
    fs = FilesystemSubstrate(build_builtin_substrate_context(store), backend=backend)

    fs.activate()
    fs.push()
    fs.deactivate()

    assert "ground" in backend.layers
    assert backend.pushed == []
    assert backend.deactivated is True


def test_bind_runtime_updates_internal_runtime_binding(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    queries = _scope_runtime(fs._pipeline, isolated=True, overlay_base="task-bind")

    fs.bind_runtime(queries)

    assert fs._runtime is queries


def test_bind_runtime_replaces_pipeline(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    replacement = _scope_runtime(fs._pipeline, isolated=True, overlay_base="task-rebind")

    fs.bind_runtime(replacement)

    assert fs._runtime is replacement
