"""Store checkout and workspace-extraction integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from vcs_core.store import Store


def test_list_workspace_files(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-list")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "src/main.py"},
        workspace_changes=[("src/main.py", b"print('hello')")],
        substrate="filesystem",
    )
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "README.md"},
        workspace_changes=[("README.md", b"# readme")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    files = store.list_workspace_files(Store.GROUND_REF)
    paths = {path for path, _oid, _mode in files}
    assert "src/main.py" in paths
    assert "README.md" in paths


def test_checkout_workspace_tree(store: Store, tmp_path) -> None:
    task = store.fork(Store.GROUND_REF, "task-checkout")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "data/config.toml"},
        workspace_changes=[("data/config.toml", b"[settings]\nkey = 1\n")],
        substrate="filesystem",
    )
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "app.py"},
        workspace_changes=[("app.py", b"import sys\n")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    dest = str(tmp_path / "checkout-dest")
    count = store.checkout_workspace_tree(Store.GROUND_REF, dest)

    assert count == 2
    assert (Path(dest) / "app.py").read_bytes() == b"import sys\n"
    assert (Path(dest) / "data" / "config.toml").read_bytes() == b"[settings]\nkey = 1\n"


def test_checkout_by_oid(store: Store, tmp_path) -> None:
    task = store.fork(Store.GROUND_REF, "task-oid")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "oid-test.txt"},
        workspace_changes=[("oid-test.txt", b"oid content")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    tip_oid = store.log(max_count=1)[0].oid

    dest = str(tmp_path / "oid-checkout")
    count = store.checkout_workspace_tree(tip_oid, dest)
    assert count >= 1
    assert (Path(dest) / "oid-test.txt").read_bytes() == b"oid content"


def test_checkout_by_short_oid(store: Store, tmp_path) -> None:
    task = store.fork(Store.GROUND_REF, "task-short")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "short.txt"},
        workspace_changes=[("short.txt", b"short prefix")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    tip_oid = store.log(max_count=1)[0].oid
    short_oid = tip_oid[:12]

    dest = str(tmp_path / "short-checkout")
    count = store.checkout_workspace_tree(short_oid, dest)
    assert count >= 1
    assert (Path(dest) / "short.txt").read_bytes() == b"short prefix"


def test_checkout_cleans_dest_on_reuse(store: Store, tmp_path) -> None:
    t1 = store.fork(Store.GROUND_REF, "task-v1")
    store._emit_effect(
        t1,
        "FileCreate",
        {"path": "keep.txt"},
        workspace_changes=[("keep.txt", b"v1")],
        substrate="filesystem",
    )
    store._emit_effect(
        t1,
        "FileCreate",
        {"path": "stale.txt"},
        workspace_changes=[("stale.txt", b"will be gone")],
        substrate="filesystem",
    )
    store.merge(t1, Store.GROUND_REF)
    v1_oid = store.log(max_count=1)[0].oid

    t2 = store.fork(Store.GROUND_REF, "task-v2")
    store._emit_effect(
        t2,
        "FilePatch",
        {"path": "keep.txt"},
        workspace_changes=[("keep.txt", b"v2")],
        substrate="filesystem",
    )
    store._emit_effect(
        t2,
        "FileDelete",
        {"path": "stale.txt"},
        workspace_changes=[("stale.txt", None)],
        substrate="filesystem",
    )
    store.merge(t2, Store.GROUND_REF)
    v2_oid = store.log(max_count=1)[0].oid

    dest = str(tmp_path / "reuse-dest")

    store.checkout_workspace_tree(v1_oid, dest)
    assert (Path(dest) / "stale.txt").exists()

    store.checkout_workspace_tree(v2_oid, dest)
    assert (Path(dest) / "keep.txt").read_bytes() == b"v2"
    assert not (Path(dest) / "stale.txt").exists()


def test_resolve_to_commit(store: Store) -> None:
    commit = store.resolve_to_commit(Store.GROUND_REF)
    assert commit is not None

    tip_oid = store.log(max_count=1)[0].oid
    commit2 = store.resolve_to_commit(tip_oid)
    assert commit2 is not None
    assert str(commit2.id) == tip_oid

    commit3 = store.resolve_to_commit(tip_oid[:10])
    assert commit3 is not None
    assert str(commit3.id) == tip_oid

    assert store.resolve_to_commit("nonexistent") is None
    assert store.resolve_to_commit("deadbeef" * 5) is None


def test_checkout_raises_on_bad_ref(store: Store, tmp_path) -> None:
    from vcs_core import RefResolutionError

    dest = str(tmp_path / "bad-ref-dest")
    with pytest.raises(RefResolutionError, match="Cannot resolve ref"):
        store.checkout_workspace_tree("nonexistent-ref", dest)

    assert not Path(dest).exists()


def test_list_workspace_files_raises_on_bad_ref(store: Store) -> None:
    from vcs_core import RefResolutionError

    with pytest.raises(RefResolutionError):
        store.list_workspace_files("nonexistent-ref")


def test_checkout_refuses_dangerous_dest(store: Store) -> None:
    with pytest.raises(ValueError, match="protected path"):
        store.checkout_workspace_tree(Store.GROUND_REF, str(Path.home()))

    with pytest.raises(ValueError, match="protected path"):
        store.checkout_workspace_tree(Store.GROUND_REF, "/")

    with pytest.raises(ValueError, match="protected path"):
        store.checkout_workspace_tree(Store.GROUND_REF, store.repo_path)


def test_checkout_refuses_unmarked_existing_dest(store: Store, tmp_path) -> None:
    task = store.fork(Store.GROUND_REF, "task-marker")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "f.txt"},
        workspace_changes=[("f.txt", b"data")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    existing = tmp_path / "user-dir"
    existing.mkdir()
    (existing / "important.txt").write_text("do not delete")

    with pytest.raises(ValueError, match="not created by checkout_workspace_tree"):
        store.checkout_workspace_tree(Store.GROUND_REF, str(existing))

    assert (existing / "important.txt").read_text() == "do not delete"


def test_checkout_marker_allows_re_extraction(store: Store, tmp_path) -> None:
    t1 = store.fork(Store.GROUND_REF, "task-re-v1")
    store._emit_effect(
        t1,
        "FileCreate",
        {"path": "a.txt"},
        workspace_changes=[("a.txt", b"v1")],
        substrate="filesystem",
    )
    store.merge(t1, Store.GROUND_REF)

    dest = str(tmp_path / "safe-dest")
    store.checkout_workspace_tree(Store.GROUND_REF, dest)

    assert (Path(dest) / ".vcscore-checkout").exists()

    t2 = store.fork(Store.GROUND_REF, "task-re-v2")
    store._emit_effect(
        t2,
        "FileCreate",
        {"path": "b.txt"},
        workspace_changes=[("b.txt", b"v2")],
        substrate="filesystem",
    )
    store.merge(t2, Store.GROUND_REF)

    count = store.checkout_workspace_tree(Store.GROUND_REF, dest)
    assert count >= 2


def test_checkout_preserves_executable_bit(store: Store, tmp_path) -> None:
    import os

    task = store.fork(Store.GROUND_REF, "task-exec")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "run.sh"},
        workspace_changes=[("run.sh", b"#!/bin/sh\necho hello", 0o100755)],
        substrate="filesystem",
    )
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "lib.py"},
        workspace_changes=[("lib.py", b"pass")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    dest = str(tmp_path / "exec-checkout")
    store.checkout_workspace_tree(Store.GROUND_REF, dest)

    run_sh = Path(dest) / "run.sh"
    lib_py = Path(dest) / "lib.py"
    assert run_sh.read_bytes() == b"#!/bin/sh\necho hello"
    assert os.access(run_sh, os.X_OK), "run.sh should be executable"
    assert not os.access(lib_py, os.X_OK), "lib.py should not be executable"


def test_list_workspace_files_includes_mode(store: Store) -> None:
    import pygit2 as _pygit2

    task = store.fork(Store.GROUND_REF, "task-mode-list")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "script.sh"},
        workspace_changes=[("script.sh", b"#!/bin/sh", 0o100755)],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    files = store.list_workspace_files(Store.GROUND_REF)
    by_name = {path: mode for path, _oid, mode in files}
    assert by_name["script.sh"] == _pygit2.GIT_FILEMODE_BLOB_EXECUTABLE


def test_checkout_refuses_vcscore_subdirectory(store: Store) -> None:
    repo_path = Path(store.repo_path).resolve()
    with pytest.raises(ValueError, match="protected path"):
        store.checkout_workspace_tree(Store.GROUND_REF, str(repo_path / "refs" / "heads"))

    with pytest.raises(ValueError, match="protected path"):
        store.checkout_workspace_tree(Store.GROUND_REF, str(repo_path / "objects"))


def test_checkout_allows_non_protected_subdirectory(store: Store, tmp_path) -> None:
    task = store.fork(Store.GROUND_REF, "task-ok-dest")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "ok.txt"},
        workspace_changes=[("ok.txt", b"fine")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    dest = str(tmp_path / "deep" / "nested" / "dir")
    count = store.checkout_workspace_tree(Store.GROUND_REF, dest)
    assert count >= 1
    assert (Path(dest) / "ok.txt").read_bytes() == b"fine"


def test_workspace_file_mode_returns_stored_mode(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-mode")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "run.sh"},
        workspace_changes=[("run.sh", b"#!/bin/sh", 0o100755)],
        substrate="filesystem",
    )
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "lib.py"},
        workspace_changes=[("lib.py", b"pass")],
        substrate="filesystem",
    )
    store.merge(task, Store.GROUND_REF)

    assert store.workspace_file_mode(Store.GROUND_REF, "run.sh") == 0o100755
    assert store.workspace_file_mode(Store.GROUND_REF, "lib.py") == 0o100644


def test_workspace_file_mode_returns_none_for_missing(store: Store) -> None:
    assert store.workspace_file_mode(Store.GROUND_REF, "nonexistent.txt") is None
