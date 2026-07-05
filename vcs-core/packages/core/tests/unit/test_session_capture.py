"""Unit tests for the public session-capture facade."""

from __future__ import annotations

from pathlib import Path

import pytest
from vcs_core._ipc import SessionInfo
from vcs_core.session_capture import CapturedExecOutcome, CaptureSession, SessionCaptureError, start_capture_session


def _session_info(workspace: Path, *, socket_path: str = "/tmp/mg.sock") -> SessionInfo:
    return SessionInfo(
        pid=1,
        socket_path=socket_path,
        mount_path=str(workspace / ".vcscore" / "session"),
        workspace=str(workspace),
        started_at=0,
        daemon_instance_id="daemon-1",
    )


def test_capture_session_exec_passes_request_and_returns_outcome(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._cli_session_runtime as session_runtime

    recorded: dict[str, object] = {}

    def fake_run_managed_exec(**kwargs: object) -> int:
        recorded.update(kwargs)
        kwargs["on_started"]("op-123")  # type: ignore[index, operator]
        kwargs["stdout"].write(b"out")  # type: ignore[union-attr]
        kwargs["stderr"].write(b"err")  # type: ignore[union-attr]
        return 0

    monkeypatch.setattr(session_runtime, "run_managed_exec", fake_run_managed_exec)

    before = Path.cwd()
    outcome = CaptureSession(_session_info(tmp_path)).exec_capture(
        scope="task-1",
        argv=["/bin/bash", "-lc", "true"],
        env={"AGENT_INSTRUCTION": "do it"},
        capture=True,
    )

    assert isinstance(outcome, CapturedExecOutcome)
    assert outcome.ok is True
    assert outcome.returncode == 0
    assert outcome.scope == "task-1"
    assert outcome.stdout == b"out"
    assert outcome.stderr == b"err"
    assert outcome.operation_id == "op-123"
    assert Path.cwd() == before

    assert recorded["argv"] == ("/bin/bash", "-lc", "true")
    assert recorded["scope_name"] == "task-1"
    assert recorded["create"] is True
    assert recorded["capture_requested"] is True
    assert recorded["session_info"] == _session_info(tmp_path)
    assert recorded["env"]["AGENT_INSTRUCTION"] == "do it"  # type: ignore[index]


def test_capture_session_is_opaque_but_reports_workspace(tmp_path: Path) -> None:
    session = CaptureSession(_session_info(tmp_path))

    assert session.workspace == tmp_path
    assert not hasattr(session, "socket_path")


def test_capture_session_merge_and_discard_dispatch(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._ipc as ipc

    calls: list[tuple[str, str, object]] = []

    def fake_send_request(socket_path: str, method: str, params: object = None) -> dict[str, object]:
        calls.append((socket_path, method, params))
        return {"ok": True, "result": {"method": method}}

    monkeypatch.setattr(ipc, "send_request", fake_send_request)

    session = CaptureSession(_session_info(tmp_path, socket_path="/tmp/s.sock"))
    assert session.merge("task-1") == {"method": "merge"}
    assert session.discard("task-2") == {"method": "discard"}

    assert calls == [
        ("/tmp/s.sock", "merge", {"name": "task-1"}),
        ("/tmp/s.sock", "discard", {"name": "task-2"}),
    ]


def test_capture_session_merge_and_discard_raise_on_daemon_error(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._ipc as ipc

    monkeypatch.setattr(ipc, "send_request", lambda *args, **kwargs: {"ok": False, "error": "blocked"})

    session = CaptureSession(_session_info(tmp_path))
    with pytest.raises(SessionCaptureError, match="blocked"):
        session.merge("task-1")


def test_start_capture_session_starts_and_stops_daemon(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._ipc as ipc
    import vcs_core._session as session_mod

    daemon_workspaces: list[str] = []
    requests: list[tuple[str, str, object]] = []
    info = _session_info(tmp_path, socket_path="/tmp/live.sock")

    class FakeDaemon:
        def __init__(self, workspace: str) -> None:
            daemon_workspaces.append(workspace)

        def _run(self) -> None:
            return None

    def fake_send_request(socket_path: str, method: str, params: object = None) -> dict[str, bool]:
        requests.append((socket_path, method, params))
        return {"ok": True, "result": {}}

    monkeypatch.setattr(
        session_mod,
        "_prepare_session_start",
        lambda workspace: (str(workspace), str(Path(workspace) / ".vcscore")),
    )
    monkeypatch.setattr(session_mod, "SessionDaemon", FakeDaemon)
    monkeypatch.setattr(ipc, "is_session_alive", lambda repo_path: True)
    monkeypatch.setattr(ipc, "read_session_info", lambda repo_path: info)
    monkeypatch.setattr(ipc, "send_request", fake_send_request)

    with start_capture_session(tmp_path) as session:
        assert session.workspace == tmp_path

    assert daemon_workspaces == [str(tmp_path)]
    assert requests == [
        ("/tmp/live.sock", "get_state", {"hook_capabilities": []}),
        ("/tmp/live.sock", "stop", None),
    ]


def test_start_capture_session_rejects_existing_live_session(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._session as session_mod

    def fail_prepare(workspace: object) -> tuple[str, str]:
        del workspace
        raise RuntimeError("Session already running (PID 123).")

    monkeypatch.setattr(session_mod, "_prepare_session_start", fail_prepare)

    with pytest.raises(SessionCaptureError, match="Session already running"), start_capture_session(tmp_path):
        pass


def test_start_capture_session_surfaces_daemon_start_error(monkeypatch, tmp_path: Path) -> None:
    import vcs_core._ipc as ipc
    import vcs_core._session as session_mod

    class FakeDaemon:
        def __init__(self, workspace: str) -> None:
            del workspace

        def _run(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        session_mod,
        "_prepare_session_start",
        lambda workspace: (str(workspace), str(Path(workspace) / ".vcscore")),
    )
    monkeypatch.setattr(session_mod, "SessionDaemon", FakeDaemon)
    monkeypatch.setattr(ipc, "read_session_info", lambda repo_path: None)

    with (
        pytest.raises(SessionCaptureError, match="boom") as excinfo,
        start_capture_session(
            tmp_path,
            startup_timeout=1.0,
        ),
    ):
        pass

    assert isinstance(excinfo.value.__cause__, RuntimeError)
