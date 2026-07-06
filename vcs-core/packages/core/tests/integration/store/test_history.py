"""Store history and merge-behavior integration tests."""

from __future__ import annotations

import pytest
from vcs_core import MergePreconditionError, StaleScopeError
from vcs_core.store import Store


def test_complete_task_lifecycle(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-fix-auth")
    store._emit_effect(task, "TaskStarted", {"task": "fix-auth"}, substrate="agent")
    store._emit_effect(
        task,
        "FilePatch",
        {"path": "src/auth.py"},
        workspace_changes=[("src/auth.py", b"fixed code")],
        substrate="filesystem",
    )
    store._emit_effect(task, "TaskCompleted", {"task": "fix-auth"}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    status = store.status()
    assert status.commits_ahead > 0
    assert status.local_changes == 1

    log = store.log()
    assert len(log) > 1
    assert any(e.metadata.get("type") == "TaskStarted" for e in log)

    diff = store.diff()
    assert len(diff.files) == 1
    assert diff.files[0].path == "src/auth.py"
    assert diff.files[0].status == "added"

    store.advance_materialized()
    status = store.status()
    assert status.commits_ahead == 0
    assert status.local_changes == 0


def test_nested_scopes(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-fix")
    store._emit_effect(task, "TaskStarted", {"task": "fix"}, substrate="agent")

    step = store.fork(task.ref, "step-analyze")
    store._emit_effect(step, "StepStarted", {"step": "analyze"}, substrate="agent")

    tool = store.fork(step.ref, "tool-search")
    store._emit_effect(tool, "ToolCallStarted", {"tool": "search"}, substrate="agent")
    store._emit_effect(tool, "FileRead", {"path": "src/auth.py"}, substrate="filesystem")
    store._emit_effect(tool, "ToolCallCompleted", {"tool": "search"}, substrate="agent")
    store.merge(tool, step.ref)

    tool2 = store.fork(step.ref, "tool-edit")
    store._emit_effect(tool2, "ToolCallStarted", {"tool": "edit"}, substrate="agent")
    store._emit_effect(
        tool2,
        "FilePatch",
        {"path": "src/auth.py"},
        workspace_changes=[("src/auth.py", b"patched")],
        substrate="filesystem",
    )
    store._emit_effect(tool2, "ToolCallCompleted", {"tool": "edit"}, substrate="agent")
    store.merge(tool2, step.ref)

    store._emit_effect(step, "StepCompleted", {"step": "analyze"}, substrate="agent")
    store.merge(step, task.ref)

    store._emit_effect(task, "TaskCompleted", {"task": "fix"}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    log = store.log(max_count=200)
    assert len(log) > 10

    tool_starts = store.filter_effects(effect_type="ToolCallStarted")
    assert len(tool_starts) == 2

    substrate_effects = store.filter_effects(substrate="filesystem")
    assert len(substrate_effects) >= 2


def test_rollback_retry(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-fix")
    store._emit_effect(task, "TaskStarted", {"task": "fix"}, substrate="agent")

    tool = store.fork(task.ref, "tool-edit-0")
    store._emit_effect(
        tool,
        "FilePatch",
        {"path": "src/auth.py"},
        workspace_changes=[("src/auth.py", b"wrong fix")],
        substrate="filesystem",
    )
    archive_ref = store.discard(tool)
    assert archive_ref.startswith("refs/vcscore/archive/tool-edit-0-")

    tool2 = store.fork(task.ref, "tool-edit-1")
    store._emit_effect(
        tool2,
        "FilePatch",
        {"path": "src/auth.py"},
        workspace_changes=[("src/auth.py", b"correct fix")],
        substrate="filesystem",
    )
    store.merge(tool2, task.ref)

    store._emit_effect(task, "TaskCompleted", {"task": "fix"}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    diff = store.diff()
    assert len(diff.files) == 1
    assert diff.files[0].path == "src/auth.py"


def test_scope_name_validation(store: Store) -> None:
    with pytest.raises(ValueError, match="contains '/'"):
        store.fork(Store.GROUND_REF, "task/bad-name")


def test_filter_effects_by_type(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-t")
    store._emit_effect(task, "TaskStarted", {}, substrate="agent")
    store._emit_effect(task, "FileRead", {"path": "a.py"}, substrate="filesystem")
    store._emit_effect(task, "TaskCompleted", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    results = store.filter_effects(effect_type="FileRead")
    assert len(results) == 1
    assert results[0].metadata["type"] == "FileRead"


def test_filter_effects_by_substrate(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-s")
    store._emit_effect(task, "TaskStarted", {}, substrate="agent")
    store._emit_effect(task, "Marker", {"label": "test"}, substrate="marker")
    store.merge(task, Store.GROUND_REF)

    results = store.filter_effects(substrate="marker")
    assert len(results) == 1


def test_filter_effects_by_scope(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-sc")
    store._emit_effect(task, "TaskStarted", {}, substrate="agent")
    store._emit_effect(task, "FileRead", {"path": "a.py"}, substrate="filesystem")
    store.merge(task, Store.GROUND_REF)

    results = store.filter_effects(scope="task-sc")
    assert len(results) == 2
    assert all(r.metadata.get("scope") == "task-sc" for r in results)

    results = store.filter_effects(scope="nonexistent")
    assert len(results) == 0


def test_read_workspace_file_returns_bytes_for_committed_file(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-read-file")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "src/nested/file.py"},
        workspace_changes=[("src/nested/file.py", b"payload")],
        substrate="filesystem",
    )

    assert store.read_workspace_file(task.ref, "src/nested/file.py") == b"payload"


def test_read_workspace_file_returns_none_for_missing_path(store: Store) -> None:
    assert store.read_workspace_file(Store.GROUND_REF, "missing.py") is None


def test_read_workspace_file_returns_none_for_missing_ref(store: Store) -> None:
    assert store.read_workspace_file("refs/vcscore/scopes/missing", "file.py") is None


def test_read_workspace_file_returns_none_for_directory_path(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-read-dir")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "src/nested/file.py"},
        workspace_changes=[("src/nested/file.py", b"payload")],
        substrate="filesystem",
    )

    assert store.read_workspace_file(task.ref, "src") is None
    assert store.read_workspace_file(task.ref, "src/nested") is None
    assert not store.file_exists_in_workspace(task.ref, "src")


def test_advance_materialized(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-a")
    store._emit_effect(task, "Test", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    assert store.status().commits_ahead > 0
    store.advance_materialized()
    assert store.status().commits_ahead == 0


def test_reset_ground_to_materialized(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-r")
    store._emit_effect(task, "Test", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    ahead = store.status().commits_ahead
    assert ahead > 0
    discarded = store.reset_ground_to_materialized()
    assert discarded == ahead
    assert store.status().commits_ahead == 0


def test_stale_scope_error(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-stale")
    store.discard(task)
    with pytest.raises(StaleScopeError):
        store.merge(task, Store.GROUND_REF)


def test_merge_precondition_error(store: Store) -> None:
    task1 = store.fork(Store.GROUND_REF, "task-1")
    store._emit_effect(task1, "Test", {}, substrate="agent")
    store.merge(task1, Store.GROUND_REF)

    task2 = store.fork(Store.GROUND_REF, "task-2")

    from vcs_core.types import ScopeInfo

    stale_scope = ScopeInfo(
        name=task2.name,
        ref=task2.ref,
        instance_id=task2.instance_id,
        creation_oid=task1.creation_oid,
    )
    with pytest.raises(MergePreconditionError):
        store.merge(stale_scope, Store.GROUND_REF)


def test_assert_mergeable_accepts_fresh_scope(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-mergeable")

    store.assert_mergeable(task, Store.GROUND_REF)


def test_assert_mergeable_raises_before_fast_forward(store: Store) -> None:
    task1 = store.fork(Store.GROUND_REF, "task-preflight-1")
    task2 = store.fork(Store.GROUND_REF, "task-preflight-2")
    store._emit_effect(task1, "Test", {}, substrate="agent")
    store.merge(task1, Store.GROUND_REF)

    with pytest.raises(MergePreconditionError, match="sequential live-child policy"):
        store.assert_mergeable(task2, Store.GROUND_REF)

    assert task2.ref in store._repo.references


def test_prune_archives(store: Store) -> None:
    for i in range(5):
        task = store.fork(Store.GROUND_REF, f"task-{i}")
        store.discard(task)

    pruned = store.prune_archives(keep_recent=2)
    assert pruned == 3


def test_prune_archives_keeps_newest_archive_commit_not_lexical_ref(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def set_commit_time(timestamp: int) -> None:
        monkeypatch.setattr("vcs_core.store.time.time", lambda: timestamp)
        monkeypatch.setattr("vcs_core.git_store.time.time", lambda: timestamp)

    set_commit_time(1_000)
    old = store.fork(Store.GROUND_REF, "z-old")
    store._emit_effect(old, "OldArchive", {}, substrate="agent")
    old_archive = store.discard(old)

    set_commit_time(2_000)
    new = store.fork(Store.GROUND_REF, "a-new")
    store._emit_effect(new, "NewArchive", {}, substrate="agent")
    new_archive = store.discard(new)

    assert sorted((old_archive, new_archive)) == [new_archive, old_archive]

    pruned = store.prune_archives(keep_recent=1)

    assert pruned == 1
    assert old_archive not in store._repo.references
    assert new_archive in store._repo.references


def test_dual_tree_structure(store: Store) -> None:
    import json

    import pygit2

    task = store.fork(Store.GROUND_REF, "task-dt")
    store._emit_effect(task, "Test", {"key": "value"}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    repo = pygit2.Repository(store._repo_path)
    tip = repo.references[Store.GROUND_REF].peel(pygit2.Commit)
    for commit in repo.walk(tip.id, pygit2.GIT_SORT_TOPOLOGICAL):
        tree_names = [e.name for e in commit.tree]
        assert "workspace" in tree_names
        assert "meta" in tree_names

        meta_tree = repo.get(commit.tree["meta"].id)
        meta_entries = [e.name for e in meta_tree]
        assert "effect.json" in meta_entries

        blob = repo.get(meta_tree["effect.json"].id)
        effect = json.loads(blob.data.decode())
        assert "type" in effect
        assert "substrate" in effect
        assert ".vcscore" not in tree_names


def test_rebase_not_implemented(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-rebase")
    store._emit_effect(task, "Test", {}, substrate="agent")

    with pytest.raises(NotImplementedError, match="three-way merge"):
        store.rebase(task, Store.GROUND_REF)


def test_walk_pending_empty(store: Store) -> None:
    assert store.walk_pending() == []


def test_walk_pending_returns_causal_order(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-wp")
    store._emit_effect(task, "First", {}, substrate="agent")
    store._emit_effect(task, "Second", {}, substrate="agent")
    store._emit_effect(task, "Third", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    pending = store.walk_pending()
    assert len(pending) >= 3
    types = [c.metadata.get("type") for c in pending]
    assert types.index("First") < types.index("Second") < types.index("Third")


def test_walk_pending_cleared_after_advance(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-wpa")
    store._emit_effect(task, "Test", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    assert len(store.walk_pending()) > 0
    store.advance_materialized()
    assert store.walk_pending() == []


def test_store_fork_allows_primitive_siblings_but_merge_rejects_stale_parent(store: Store) -> None:
    task1 = store.fork(Store.GROUND_REF, "task-sib-1")
    store._emit_effect(task1, "Work1", {}, substrate="agent")

    task2 = store.fork(Store.GROUND_REF, "task-sib-2")
    store._emit_effect(task2, "Work2", {}, substrate="agent")

    store.merge(task1, Store.GROUND_REF)

    with pytest.raises(MergePreconditionError, match="sequential live-child policy"):
        store.merge(task2, Store.GROUND_REF)


def test_is_ancestor(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task-anc")
    store._emit_effect(task, "Test", {}, substrate="agent")
    store.merge(task, Store.GROUND_REF)

    log = store.log(max_count=100)
    root_oid = log[-1].oid
    tip_oid = log[0].oid

    assert store._is_ancestor(root_oid, tip_oid)
    assert not store._is_ancestor(tip_oid, root_oid)


def test_list_archive_refs(store: Store) -> None:
    assert store.list_archive_refs() == []

    task = store.fork(Store.GROUND_REF, "to-archive")
    store._emit_effect(
        task,
        "FileCreate",
        {"path": "x.txt"},
        workspace_changes=[("x.txt", b"data")],
        substrate="filesystem",
    )
    store.discard(task)

    archives = store.list_archive_refs()
    assert len(archives) == 1
    assert archives[0].startswith("refs/vcscore/archive/to-archive-")
