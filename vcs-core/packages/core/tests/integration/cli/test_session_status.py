"""Session status CLI rendering tests."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from vcs_core.cli import main
from vcs_core.testing import SessionInfo

from ...support.cli import init_repo as _init


def _session_info(workspace: Path, *, started_at: float | None = None) -> SessionInfo:
    return SessionInfo(
        pid=os.getpid(),
        socket_path="/tmp/fake.sock",
        mount_path="/stale/path",
        workspace=str(workspace),
        started_at=time.time() if started_at is None else started_at,
    )


def test_session_status_no_session(tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    result = runner.invoke(main, ["session", "status"])

    assert result.exit_code == 0
    assert "No session running" in result.output


def test_session_status_prefers_live_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    info = _session_info(tmp_path, started_at=time.time() - 120)

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        del params
        assert method == "status_summary"
        return {
            "ok": True,
            "result": {
                "pid": info.pid,
                "current_scope": "experiment",
                "current_world_id": "world_experiment",
                "mount_path": str(tmp_path),
                "workspace": str(tmp_path),
                "started_at": info.started_at,
                "overlay_change_count": 1,
                "local_changes": 0,
                "commits_ahead": 0,
                "live_scopes": ["experiment"],
                "retained_scopes": ["sealed-a", "sealed-b"],
                "blockers": [],
                "pending_operations": 0,
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["session", "status"])

    assert result.exit_code == 0, result.output
    assert f"Managed workspace: {tmp_path.resolve()}" in result.output
    assert "Environment: host state outside workspace is untracked" in result.output
    assert f"Mount path: {tmp_path}" in result.output
    assert "Workspace:" not in result.output
    assert "Scope:      experiment" in result.output
    assert "World ID:   world_experiment" in result.output
    assert "Retained scopes: 2" in result.output
    assert "/stale/path" not in result.output


def test_top_level_status_delegates_to_live_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    info = _session_info(tmp_path, started_at=time.time() - 30)

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: info)

    def fake_send_session_request(info_arg: SessionInfo, method: str, params: dict | None = None) -> dict:
        assert info_arg == info
        del params
        assert method == "status_summary"
        return {
            "ok": True,
            "result": {
                "pid": info.pid,
                "current_scope": "experiment",
                "current_world_id": "world_experiment",
                "mount_path": str(tmp_path / "overlay"),
                "workspace": str(tmp_path),
                "started_at": info.started_at,
                "overlay_change_count": 1,
                "local_changes": 0,
                "commits_ahead": 0,
                "live_scopes": ["experiment"],
                "retained_scopes": ["sealed-a"],
                "blockers": [],
                "pending_operations": 0,
            },
        }

    monkeypatch.setattr("vcs_core._cli_ipc.send_session_request", fake_send_session_request)

    result = runner.invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "Session active" in result.output
    assert f"Managed workspace: {tmp_path.resolve()}" in result.output
    assert "Environment: host state outside workspace is untracked" in result.output
    assert "Scope:      experiment" in result.output
    assert "Overlay changes: 1" in result.output
    assert "Local changes:   0" in result.output
    assert "Commits ahead:   0" in result.output
    assert "Retained scopes: 1" in result.output
