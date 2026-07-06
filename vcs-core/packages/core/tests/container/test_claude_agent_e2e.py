# under-test: vcs_core._session
"""N+2 live thread: the Claude Agent SDK edits a Git workspace under vcs-core
capture, reversibly, in Podman.

This is the full thread the spine was built for: instead of the deterministic
`/bin/bash` stand-in, a real Claude Agent SDK process (run via `agent_edit.py`)
runs under `vcs-core session exec --capture`, with cwd = the overlay mount, so
its file edits are captured by vcs-core's overlay and are reversible — `merge`
persists them to ground, `discard` reverts them.

Gated: runs only when a `claude` CLI and ANTHROPIC_API_KEY are present (i.e. the
combined `vcs-core-agent` image with the key passed in). It SKIPS in the normal
`make test_container` run (vcs-core-test image has no Claude runtime). Live and
nondeterministic in content — the assertions check the *deterministic* outcome
(the named file is captured into ground / reverted), not the agent's prose.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from vcs_core._ipc import is_session_alive, read_session_info, send_request
from vcs_core.cli import main

_AGENT_SCRIPT = (
    Path(__file__).parents[4]
    / "design"
    / "spikes"
    / "260515-world-vectors"
    / "260526-claude-agent-thread"
    / "agent_edit.py"
)

pytestmark = [
    pytest.mark.container,
    pytest.mark.skipif(
        sys.platform != "linux" or os.geteuid() != 0,
        reason="Live Claude agent thread requires Linux with root.",
    ),
    pytest.mark.skipif(
        shutil.which("claude") is None or not os.environ.get("ANTHROPIC_API_KEY"),
        reason="Live Claude smoke: needs the `claude` CLI on PATH and ANTHROPIC_API_KEY.",
    ),
]


@pytest.fixture
def agent_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "project"
    ws.mkdir()
    (ws / "original.txt").write_text("original content")
    (ws / "vcscore.toml").write_text('[bindings.filesystem]\ntype = "filesystem"\nbackend = "kernel"\n')
    result = CliRunner().invoke(main, ["init", "--adopt", "worktree", "--all", str(ws)])
    assert result.exit_code == 0, result.output
    return ws


@contextlib.contextmanager
def _session_daemon(ws: Path):
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(ws))
    thread = threading.Thread(target=daemon._run, daemon=True)
    thread.start()
    repo_path = str(ws / ".vcscore")
    deadline = time.time() + 5
    while time.time() < deadline and not is_session_alive(repo_path):
        time.sleep(0.1)
    info = read_session_info(repo_path)
    assert info is not None, "session daemon did not start"
    try:
        yield info
    finally:
        with contextlib.suppress(Exception):
            send_request(info.socket_path, "stop")
        thread.join(timeout=5)


def _run_agent_under_capture(scope: str) -> None:
    """session exec the Claude agent under capture; assert the agent succeeded."""
    result = CliRunner().invoke(
        main,
        [
            "session",
            "exec",
            "--scope",
            scope,
            "--create",
            "--capture",
            "--",
            "/usr/bin/python3",
            str(_AGENT_SCRIPT),
        ],
    )
    assert result.exit_code == 0, f"agent exec failed: {result.output}"


def test_claude_agent_edit_is_captured_and_merged(agent_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Claude agent's file edit is captured and merges into ground."""
    from vcs_core.store import Store

    repo_path = str(agent_workspace / ".vcscore")
    with _session_daemon(agent_workspace) as info:
        monkeypatch.chdir(agent_workspace)
        _run_agent_under_capture("claude-task")
        merged = send_request(info.socket_path, "merge", {"name": "claude-task"})
        assert merged["ok"] is True, merged

    note = Store(repo_path).read_workspace_file(Store.GROUND_REF, "AGENT_NOTE.md")
    assert note is not None, "agent's AGENT_NOTE.md did not reach ground after merge"
    assert b"claude agent" in note, f"unexpected AGENT_NOTE.md content: {note!r}"


def test_claude_agent_edit_discard_reverts(agent_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Claude agent's file edit is reverted when the scope is discarded."""
    from vcs_core.store import Store

    repo_path = str(agent_workspace / ".vcscore")
    with _session_daemon(agent_workspace) as info:
        monkeypatch.chdir(agent_workspace)
        _run_agent_under_capture("claude-throwaway")
        discarded = send_request(info.socket_path, "discard", {"name": "claude-throwaway"})
        assert discarded["ok"] is True, discarded

    assert Store(repo_path).read_workspace_file(Store.GROUND_REF, "AGENT_NOTE.md") is None
