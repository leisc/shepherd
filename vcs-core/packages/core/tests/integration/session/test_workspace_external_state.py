# under-test: vcs_core._workspace_external_state
"""Physical workspace admission checks for snapshot-backed sessions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner
from vcs_core._workspace_external_state import assert_workspace_admissible, external_workspace_blockers
from vcs_core.cli import main
from vcs_core.store import Store


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True, text=True)


def _commit_git_baseline(workspace: Path) -> None:
    _run_git(workspace, "init", "-q")
    _run_git(workspace, "config", "user.name", "Meta Git Test")
    _run_git(workspace, "config", "user.email", "vcs-core-test@example.invalid")
    (workspace / "README.md").write_text("hello\n")
    _run_git(workspace, "add", "README.md")
    _run_git(workspace, "commit", "-qm", "initial")


def test_git_head_must_be_adopted_before_session_start(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))

    blockers = external_workspace_blockers(store, tmp_path)

    assert [(blocker.path, blocker.reason) for blocker in blockers] == [("README.md", "git-head-not-adopted")]


def test_git_subdirectory_admission_uses_workspace_relative_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    selected = repo / "subws"
    selected.mkdir(parents=True)
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.name", "Meta Git Test")
    _run_git(repo, "config", "user.email", "vcs-core-test@example.invalid")
    (repo / "root.txt").write_text("root\n")
    (selected / "inside.txt").write_text("inside\n")
    _run_git(repo, "add", "root.txt", "subws/inside.txt")
    _run_git(repo, "commit", "-qm", "initial")
    (repo / "outer.txt").write_text("outer\n")
    (selected / "notes.txt").write_text("notes\n")
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(selected)])
    assert init_result.exit_code == 0, init_result.output
    store = Store.open_existing(str(selected / ".vcscore"))

    blockers = external_workspace_blockers(store, selected)

    assert [(blocker.path, blocker.reason) for blocker in blockers] == [
        ("inside.txt", "git-head-not-adopted"),
        ("notes.txt", "git-worktree-dirty"),
    ]


def test_git_head_adoption_allows_ignored_files_but_blocks_untracked_dirt(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / ".gitignore").write_text("debug.log\n")
    _run_git(tmp_path, "add", ".gitignore")
    _run_git(tmp_path, "commit", "-qm", "ignore debug")
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))

    assert_workspace_admissible(store, tmp_path)
    (tmp_path / "debug.log").write_text("ignored\n")
    assert_workspace_admissible(store, tmp_path)
    (tmp_path / "notes.txt").write_text("untracked\n")

    blockers = external_workspace_blockers(store, tmp_path)

    assert [(blocker.path, blocker.reason) for blocker in blockers] == [("notes.txt", "git-worktree-dirty")]


def test_git_admission_handles_nul_quoted_status_paths(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))

    (tmp_path / "notes\nfile.txt").write_text("untracked\n")

    blockers = external_workspace_blockers(store, tmp_path)

    assert [(blocker.path, blocker.reason) for blocker in blockers] == [("notes\nfile.txt", "git-worktree-dirty")]


def test_worktree_adoption_blocks_later_physical_drift(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("adopted\n")
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert_workspace_admissible(store, tmp_path)

    (tmp_path / "notes.txt").write_text("dirty\n")

    blockers = external_workspace_blockers(store, tmp_path)

    assert [(blocker.path, blocker.reason) for blocker in blockers] == [("notes.txt", "worktree-not-adopted")]
