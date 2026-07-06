"""Session lifecycle CLI behavior tests."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from vcs_core.cli import main
from vcs_core.testing import SessionInfo

from ...support.cli import init_repo as _init


def _session_info(workspace: Path) -> SessionInfo:
    return SessionInfo(
        pid=os.getpid(),
        socket_path="/tmp/fake.sock",
        mount_path="/stale/path",
        workspace=str(workspace),
        started_at=time.time(),
    )


def test_session_start_reports_mount_path_from_live_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    info = _session_info(tmp_path)

    monkeypatch.setattr("vcs_core._session.daemonize", lambda workspace, foreground=False: 43210)
    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: {
            "ok": True,
            "result": {
                "pid": info.pid,
                "current_scope": "ground",
                "mount_path": str(tmp_path / "overlay"),
                "workspace": str(tmp_path),
                "started_at": info.started_at,
            },
        },
    )

    result = runner.invoke(main, ["session", "start"])

    assert result.exit_code == 0, result.output
    assert "Session started (PID 43210)" in result.output
    assert f"Working directory: {tmp_path / 'overlay'}" in result.output
    assert "vcs-core session shell --scope task --create" in result.output


def test_session_start_requires_existing_repo(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["session", "start"])

        assert result.exit_code != 0
        assert "not a vcs-core repository" in result.output
        assert not Path(".vcscore").exists()


def test_session_start_foreground_skips_followup_ipc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr("vcs_core._session.daemonize", lambda workspace, foreground=False: 12345)
    monkeypatch.setattr(
        "vcs_core._cli_ipc.try_session_ipc",
        lambda method, params=None: (_ for _ in ()).throw(
            AssertionError("foreground start should not query IPC state")
        ),
    )

    result = runner.invoke(main, ["session", "start", "--foreground"])

    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_session_start_reports_daemonize_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr(
        "vcs_core._session.daemonize",
        lambda workspace, foreground=False: (_ for _ in ()).throw(RuntimeError("overlay unavailable")),
    )

    result = runner.invoke(main, ["session", "start"])

    assert result.exit_code != 0
    assert "overlay unavailable" in result.output


def test_session_start_reports_unreachable_daemon_after_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    info = _session_info(tmp_path)

    monkeypatch.setattr("vcs_core._session.daemonize", lambda workspace, foreground=False: 12345)
    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: (_ for _ in ()).throw(ConnectionError("boom")),
    )

    result = runner.invoke(main, ["session", "start"])

    assert result.exit_code != 0
    assert "session daemon is recorded as running but unreachable" in result.output


def test_session_stop_reports_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    calls: list[str] = []

    def fake_stop_session(workspace: str) -> None:
        calls.append(workspace)

    monkeypatch.setattr("vcs_core._session.stop_session", fake_stop_session)

    result = runner.invoke(main, ["session", "stop"])

    assert result.exit_code == 0, result.output
    assert calls == ["."]
    assert "Session stopped." in result.output


def test_session_stop_reports_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr(
        "vcs_core._session.stop_session",
        lambda workspace: (_ for _ in ()).throw(RuntimeError("No session")),
    )

    result = runner.invoke(main, ["session", "stop"])

    assert result.exit_code != 0
    assert "No session" in result.output
