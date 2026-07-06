# under-test: vcs_core._session_paths
"""Regression checks for deterministic short session runtime paths."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from vcs_core._session import SessionDaemon
from vcs_core._session_paths import (
    session_hook_socket_path,
    session_runtime_root,
    session_socket_path,
)


def test_session_runtime_paths_are_deterministic_and_short(tmp_path: Path) -> None:
    workspace = tmp_path / ("nested-" * 8) / ("deeper-" * 8) / "project"
    repo_path = workspace / ".vcscore"

    first_root = session_runtime_root(str(repo_path))
    second_root = session_runtime_root(str(repo_path))
    socket_path = session_socket_path(str(repo_path))
    hook_socket = session_hook_socket_path(str(repo_path))
    old_hook_socket = repo_path / "session-hook.sock"

    assert first_root == second_root
    assert first_root.name
    assert socket_path.startswith(str(first_root))
    assert hook_socket.startswith(str(first_root))
    assert len(hook_socket) <= 103
    assert len(str(old_hook_socket)) > len(hook_socket)
    assert str(first_root).startswith("/tmp/")
    assert first_root.parent.name.startswith("vcs-core-session-")


def test_session_runtime_paths_stay_short_with_long_workspace_basename(tmp_path: Path) -> None:
    workspace = tmp_path / ("workspace-" + ("a" * 80))
    repo_path = workspace / ".vcscore"

    runtime_root = session_runtime_root(str(repo_path))
    hook_socket = session_hook_socket_path(str(repo_path))

    assert runtime_root.name.startswith("repo-")
    assert len(runtime_root.name) == len("repo-") + 16
    assert len(hook_socket) <= 103


def test_session_runtime_root_is_prepared_owner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime-root"
    monkeypatch.setattr("vcs_core._session.session_runtime_root", lambda repo_path: runtime_root)

    daemon = SessionDaemon(str(tmp_path / "workspace"))
    daemon._prepare_runtime_root()

    assert stat.S_IMODE(runtime_root.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(runtime_root.parent.stat().st_mode) & 0o077 == 0
