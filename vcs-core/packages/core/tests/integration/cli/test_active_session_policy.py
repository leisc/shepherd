"""Top-level CLI policy tests while a persistent session is active."""

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
        mount_path=str(workspace / "overlay"),
        workspace=str(workspace),
        started_at=time.time(),
    )


def test_run_rejects_when_session_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    script = tmp_path / "noop.py"
    script.write_text("print('hello')\n")

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: _session_info(tmp_path))

    result = runner.invoke(main, ["run", str(script)])

    assert result.exit_code != 0
    assert "not supported while a persistent session is active" in result.output
    assert "vcs-core session exec" in result.output


def test_run_missing_script_still_rejects_when_session_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: _session_info(tmp_path))

    result = runner.invoke(main, ["run", str(tmp_path / "missing.py")])

    assert result.exit_code != 0
    assert "not supported while a persistent session is active" in result.output
    assert "vcs-core session exec" in result.output
    assert "script does not exist" not in result.output


def test_diff_rejects_when_session_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: _session_info(tmp_path))

    result = runner.invoke(main, ["diff"])

    assert result.exit_code != 0
    assert "not supported while a persistent session is active" in result.output
    assert "vcs-core session status" in result.output


def test_push_rejects_when_session_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: _session_info(tmp_path))

    result = runner.invoke(main, ["push"])

    assert result.exit_code != 0
    assert "not supported while a persistent session is active" in result.output
    assert "vcs-core session stop" in result.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("archive-orphaned-scopes", "vcs-core session stop"),
        ("archive-orphaned-operations", "vcs-core session stop"),
    ],
)
def test_archive_cleanup_rejects_when_session_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    expected: str,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    monkeypatch.setattr("vcs_core._cli_ipc.live_session_info", lambda: _session_info(tmp_path))

    result = runner.invoke(main, [command])

    assert result.exit_code != 0
    assert "not supported while a persistent session is active" in result.output
    assert expected in result.output
