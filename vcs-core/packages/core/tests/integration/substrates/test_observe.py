# under-test: vcs_core._substrate_runtime
"""Marker and declarative filesystem substrate integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vcs_core._substrate_runtime import build_builtin_substrate_context
from vcs_core.substrates import DeclarativeFilesystemSubstrate, FilesystemSubstrate, MarkerSubstrate

from ...support.scopes import set_scope as _set_scope

if TYPE_CHECKING:
    from vcs_core.store import Store


def test_marker_substrate_writes_commit(store: Store) -> None:
    marker = MarkerSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-m")
    _set_scope(marker, task)

    oid = marker.mark("checkpoint", {"phase": "start"})
    assert oid

    log = store.log(ref=task.ref)
    assert any(e.metadata.get("type") == "Marker" for e in log)


def test_marker_substrate_no_workspace_changes(store: Store) -> None:
    marker = MarkerSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-m2")
    _set_scope(marker, task)

    marker.mark("test")

    diff = store.diff()
    assert len(diff.files) == 0


def test_marker_substrate_requires_scope(store: Store) -> None:
    marker = MarkerSubstrate(build_builtin_substrate_context(store))
    with pytest.raises(RuntimeError, match="No execution context"):
        marker.mark("test")


def test_marker_substrate_execute_returns_effect(store: Store) -> None:
    marker = MarkerSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-m-exec")

    outcome = marker.execute("mark", task, label="checkpoint", metadata={"phase": "start"})
    effects = outcome.effects

    assert len(effects) == 1
    assert effects[0].effect_type == "Marker"
    assert effects[0].metadata == {"label": "checkpoint", "metadata": {"phase": "start"}}


def test_marker_substrate_accepts_explicit_scope_without_ambient_context(store: Store) -> None:
    marker = MarkerSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-m-explicit")

    oid = marker.mark("checkpoint", {"phase": "explicit"}, scope=task)

    assert oid
    effects = store.filter_effects(effect_type="Marker", ref=task.ref)
    assert any(effect.metadata.get("label") == "checkpoint" for effect in effects)


def test_declarative_filesystem_record_changes(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f")
    _set_scope(fs, task)

    oids = fs.record_changes(
        [
            ("src/main.py", b"hello"),
            ("src/lib.py", b"world"),
        ]
    )
    assert len(oids) == 2

    store.merge(task, store.GROUND_REF)

    diff = store.diff()
    paths = {f.path for f in diff.files}
    assert "src/main.py" in paths
    assert "src/lib.py" in paths


def test_declarative_filesystem_record_read(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-r")
    _set_scope(fs, task)

    oid = fs.record_read("src/auth.py")
    assert oid

    effects = store.filter_effects(effect_type="FileRead", ref=task.ref)
    assert len(effects) == 1
    assert effects[0].metadata["path"] == "src/auth.py"


def test_declarative_filesystem_accepts_explicit_scope_without_ambient_context(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f-explicit")

    oids = fs.record_changes([("src/explicit.py", b"content")], scope=task)
    read_oid = fs.record_read("src/explicit.py", scope=task)

    assert len(oids) == 1
    assert read_oid
    effects = store.filter_effects(ref=task.ref)
    assert any(effect.metadata.get("path") == "src/explicit.py" for effect in effects)


def test_declarative_filesystem_delete(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))

    task = store.fork(store.GROUND_REF, "task-create")
    _set_scope(fs, task)
    fs.record_changes([("a.py", b"content")])
    store.merge(task, store.GROUND_REF)

    task2 = store.fork(store.GROUND_REF, "task-delete")
    _set_scope(fs, task2)
    fs.record_changes([("a.py", None)])
    store.merge(task2, store.GROUND_REF)

    effects = store.filter_effects(effect_type="FileDelete")
    assert len(effects) >= 1


def test_declarative_filesystem_distinguishes_create_vs_patch(store: Store) -> None:
    fs = DeclarativeFilesystemSubstrate(build_builtin_substrate_context(store))

    task = store.fork(store.GROUND_REF, "task-c")
    _set_scope(fs, task)
    fs.record_changes([("src/auth.py", b"original")])
    store.merge(task, store.GROUND_REF)

    creates = store.filter_effects(effect_type="FileCreate")
    assert any(e.metadata.get("path") == "src/auth.py" for e in creates)

    task2 = store.fork(store.GROUND_REF, "task-p")
    _set_scope(fs, task2)
    fs.record_changes([("src/auth.py", b"modified")])
    store.merge(task2, store.GROUND_REF)

    patches = store.filter_effects(effect_type="FilePatch")
    assert any(e.metadata.get("path") == "src/auth.py" for e in patches)


def test_filesystem_execute_write_returns_create_for_new_file(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f-exec-create")

    outcome = fs.execute("write", task, path="src/new.py", content=b"hello")
    effects = outcome.effects

    assert len(effects) == 1
    assert effects[0].effect_type == "FileCreate"
    assert effects[0].metadata["path"] == "src/new.py"
    assert effects[0].workspace_changes == (("src/new.py", b"hello"),)


def test_filesystem_execute_write_returns_patch_for_existing_file(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    seed = store.fork(store.GROUND_REF, "task-f-exec-seed")
    _set_scope(fs, seed)
    fs.record_changes([("src/existing.py", b"original")])
    store.merge(seed, store.GROUND_REF)

    task = store.fork(store.GROUND_REF, "task-f-exec-patch")
    outcome = fs.execute("write", task, path="src/existing.py", content=b"updated")
    effects = outcome.effects

    assert len(effects) == 1
    assert effects[0].effect_type == "FilePatch"
    assert effects[0].metadata["path"] == "src/existing.py"


def test_filesystem_execute_delete_returns_delete(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    seed = store.fork(store.GROUND_REF, "task-f-exec-del-seed")
    _set_scope(fs, seed)
    fs.record_changes([("src/delete.py", b"content")])
    store.merge(seed, store.GROUND_REF)

    task = store.fork(store.GROUND_REF, "task-f-exec-delete")
    outcome = fs.execute("delete", task, path="src/delete.py")
    effects = outcome.effects

    assert len(effects) == 1
    assert effects[0].effect_type == "FileDelete"
    assert effects[0].metadata["path"] == "src/delete.py"
    assert effects[0].workspace_changes == (("src/delete.py", None),)


def test_filesystem_execute_write_rejects_none_content(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f-exec-none")

    with pytest.raises(ValueError, match="Use command='delete'"):
        fs.execute("write", task, path="src/delete.py", content=None)


def test_filesystem_execute_read_returns_file_read(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f-exec-read")

    outcome = fs.execute("read", task, path="src/readme.py")
    effects = outcome.effects

    assert len(effects) == 1
    assert effects[0].effect_type == "FileRead"
    assert effects[0].metadata["path"] == "src/readme.py"
    assert effects[0].workspace_changes == ()


def test_filesystem_execute_unknown_command_raises(store: Store) -> None:
    fs = FilesystemSubstrate(build_builtin_substrate_context(store))
    task = store.fork(store.GROUND_REF, "task-f-exec-unknown")

    with pytest.raises(ValueError, match="Unknown filesystem command"):
        fs.execute("rename", task, path="src/old.py", new_path="src/new.py")
