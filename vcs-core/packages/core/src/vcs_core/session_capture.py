"""Public session-capture facade over vcs-core's private daemon transport.

This module is the supported Python surface for callers that need the mature
session/overlay capture lane. It exposes semantic operations: start a capture
session, run a command in a captured scope, then merge or discard that scope.
The Unix-socket IPC protocol, daemon frame format, and session metadata files
remain private implementation details.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from vcs_core._ipc import SessionInfo

SESSION_CAPTURE_API_VERSION = "v0.1"


class SessionCaptureError(VcsCoreError, RuntimeError):
    """Raised when the daemon-backed capture facade cannot complete a request."""


@dataclass(frozen=True)
class CapturedExecOutcome:
    """Result of running one command through a captured session scope."""

    returncode: int
    scope: str
    stdout: bytes
    stderr: bytes
    operation_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CaptureSession:
    """Opaque handle for a live daemon-backed capture session."""

    def __init__(self, info: SessionInfo) -> None:
        self.__info = info

    @property
    def workspace(self) -> Path:
        """Workspace root for this session."""
        return Path(self.__info.workspace)

    def exec_capture(
        self,
        *,
        scope: str,
        argv: Sequence[str],
        env: Mapping[str, str] | None = None,
        capture: bool = True,
        create: bool = True,
        parent: str | None = None,
        cwd_subpath: str | None = None,
    ) -> CapturedExecOutcome:
        """Run ``argv`` in ``scope`` through the daemon-managed capture lane."""
        from vcs_core import _cli_session_runtime as session_runtime

        child_env = dict(os.environ)
        if env:
            child_env.update(env)

        operation_id: str | None = None

        def _record_operation_id(value: str) -> None:
            nonlocal operation_id
            operation_id = value

        stdout, stderr = BytesIO(), BytesIO()
        try:
            returncode = session_runtime.run_managed_exec(
                argv=tuple(argv),
                scope_name=scope,
                create=create,
                parent=parent,
                cwd_subpath=cwd_subpath,
                capture_requested=capture,
                capture_debug=None,
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                exit_code=1,
                on_started=_record_operation_id,
                session_info=self.__info,
            )
        except session_runtime.SessionCliError as exc:
            raise SessionCaptureError(str(exc)) from exc

        return CapturedExecOutcome(
            returncode=returncode,
            scope=scope,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            operation_id=operation_id,
        )

    def merge(self, scope: str) -> Mapping[str, Any]:
        """Merge ``scope`` into ground, persisting captured edits."""
        return self.__scope_request("merge", scope)

    def discard(self, scope: str) -> Mapping[str, Any]:
        """Discard ``scope``, reverting captured edits."""
        return self.__scope_request("discard", scope)

    def __scope_request(self, method: str, scope: str) -> Mapping[str, Any]:
        from vcs_core import _ipc

        try:
            response = _ipc.send_request(self.__info.socket_path, method, {"name": scope})
        except (ConnectionError, OSError) as exc:
            raise SessionCaptureError(str(exc)) from exc
        if not response["ok"]:
            raise SessionCaptureError(response["error"])
        result = response["result"]
        if not isinstance(result, dict):
            raise SessionCaptureError(f"vcs-core session {method} returned a non-object result")
        return result


@contextmanager
def start_capture_session(
    workspace: str | Path,
    *,
    startup_timeout: float = 5.0,
    shutdown_timeout: float = 5.0,
) -> Iterator[CaptureSession]:
    """Start a daemon-backed capture session for ``workspace`` and stop it on exit."""
    from vcs_core import _ipc
    from vcs_core._session import SessionDaemon, _prepare_session_start

    try:
        ws, repo_path = _prepare_session_start(workspace)
    except RuntimeError as exc:
        raise SessionCaptureError(str(exc)) from exc

    daemon = SessionDaemon(ws)
    startup_errors: list[BaseException] = []

    def _run_daemon() -> None:
        try:
            daemon._run()
        except BaseException as exc:  # noqa: BLE001
            startup_errors.append(exc)

    thread = threading.Thread(target=_run_daemon, daemon=True)
    thread.start()

    info: SessionInfo | None = None
    try:
        info = _wait_for_ready_session(
            repo_path,
            workspace=ws,
            startup_timeout=startup_timeout,
            startup_errors=startup_errors,
        )
    except Exception:
        maybe_info = _ipc.read_session_info(repo_path)
        if maybe_info is not None:
            _stop_session(maybe_info)
        thread.join(timeout=shutdown_timeout)
        raise

    try:
        yield CaptureSession(info)
    finally:
        _stop_session(info)
        thread.join(timeout=shutdown_timeout)


def _wait_for_ready_session(
    repo_path: str,
    *,
    workspace: str,
    startup_timeout: float,
    startup_errors: list[BaseException],
) -> SessionInfo:
    from vcs_core import _ipc

    deadline = time.monotonic() + startup_timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if startup_errors:
            startup_exc = startup_errors[0]
            raise SessionCaptureError(
                f"vcs-core capture session daemon failed to start for {workspace!r}: {startup_exc}"
            ) from startup_exc

        info = _ipc.read_session_info(repo_path)
        if info is not None and _ipc.is_session_alive(repo_path):
            try:
                response = _ipc.send_request(info.socket_path, "get_state", {"hook_capabilities": []})
            except (ConnectionError, OSError) as exc:
                last_error = exc
            else:
                if response["ok"]:
                    return info
                last_error = SessionCaptureError(response["error"])
        time.sleep(0.05)

    if startup_errors:
        startup_exc = startup_errors[0]
        raise SessionCaptureError(
            f"vcs-core capture session daemon failed to start for {workspace!r}: {startup_exc}"
        ) from startup_exc
    raise SessionCaptureError(f"vcs-core capture session daemon did not start for {workspace!r}") from last_error


def _stop_session(info: SessionInfo) -> None:
    from vcs_core import _ipc

    with suppress(Exception):
        _ipc.send_request(info.socket_path, "stop")


__all__ = [
    "SESSION_CAPTURE_API_VERSION",
    "CaptureSession",
    "CapturedExecOutcome",
    "SessionCaptureError",
    "start_capture_session",
]
