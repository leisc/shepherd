"""Support helpers for daemon-backed session CLI commands."""

from __future__ import annotations

import os
import socket
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from vcs_core import _cli_ipc
from vcs_core._admission.identifiers import ParseError, parse_optional_scope_name
from vcs_core._cli_errors import prefixed_error_message
from vcs_core._errors import VcsCoreError
from vcs_core._managed_exec_protocol import (
    ManagedExecExitFrame,
    ManagedExecMessageFrame,
    ManagedExecStartedFrame,
    ManagedExecStreamFrame,
    decode_managed_exec_response_frame,
    encode_managed_exec_request,
    sanitized_managed_exec_env,
)

AUTO_CAPTURE_DEBUG_SENTINEL = "__AUTO__"

if TYPE_CHECKING:
    from collections.abc import Callable

    from vcs_core._ipc import JsonObject, SessionInfo, SessionResponse


class SessionCliError(VcsCoreError, RuntimeError):
    """CLI-facing session error with an explicit process exit code."""

    def __init__(self, message: str, *, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _ManagedExecOutputClosedError(VcsCoreError):
    """The invoking CLI can no longer forward managed exec output."""


@dataclass(frozen=True)
class PreparedSessionContext:
    mount_path: str
    active_scope: str
    workspace: str
    session_socket_path: str
    daemon_instance_id: str | None
    env: dict[str, str]


@dataclass(frozen=True)
class LiveSessionStatus:
    pid: int
    started_at: float
    workspace: str
    mount_path: str
    current_scope: str
    current_world_id: str | None
    overlay_change_count: int
    local_changes: int = 0
    commits_ahead: int = 0
    live_scope_count: int = 0
    retained_scope_count: int = 0
    blocker_count: int = 0
    pending_operations: int | None = None


@dataclass(frozen=True)
class ExecEnvelope:
    operation_id: str
    env: dict[str, str]


@dataclass(frozen=True)
class ShellCaptureLease:
    lease_id: str


@dataclass(frozen=True)
class _HookEnvView:
    env: dict[str, str]
    prepend_path: tuple[str, ...]
    prepend_env: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _SessionStateView:
    pid: int
    current_scope: str
    current_world_id: str | None
    mount_path: str
    workspace: str
    started_at: float
    daemon_instance_id: str | None
    hook_static: _HookEnvView
    hook_scope: _HookEnvView


@dataclass(frozen=True)
class _OverlayStatusView:
    change_count: int


def _apply_hook_runtime_env(base_env: dict[str, str], state: _SessionStateView) -> dict[str, str]:
    env = dict(base_env)
    env.update(state.hook_static.env)
    env.update(state.hook_scope.env)

    prepend_parts: list[str] = []
    prepend_parts.extend(state.hook_static.prepend_path)
    prepend_parts.extend(state.hook_scope.prepend_path)
    if prepend_parts:
        existing_path = env.get("PATH", "")
        env["PATH"] = (
            os.pathsep.join([*prepend_parts, existing_path]) if existing_path else os.pathsep.join(prepend_parts)
        )

    for prepend_state in (state.hook_static.prepend_env, state.hook_scope.prepend_env):
        for key, values in prepend_state.items():
            prepend_values = list(values)
            if not prepend_values:
                continue
            existing_value = env.get(key, "")
            env[key] = (
                os.pathsep.join([*prepend_values, existing_value])
                if existing_value
                else os.pathsep.join(prepend_values)
            )

    return env


def _decode_session_state(
    result: JsonObject,
    *,
    default_pid: int,
    default_mount_path: str,
    default_workspace: str,
    default_started_at: float,
) -> _SessionStateView:
    return _SessionStateView(
        pid=_int_value(result.get("pid"), default=default_pid),
        current_scope=_string_value(result.get("current_scope"), default="ground"),
        current_world_id=_optional_string_value(result.get("current_world_id")),
        mount_path=_string_value(result.get("mount_path"), default=default_mount_path),
        workspace=_string_value(result.get("workspace"), default=default_workspace),
        started_at=_float_value(result.get("started_at"), default=default_started_at),
        daemon_instance_id=_optional_string_value(result.get("daemon_instance_id")),
        hook_static=_decode_hook_env(result, prefix="hook_static"),
        hook_scope=_decode_hook_env(result, prefix="hook_scope"),
    )


def _decode_overlay_status(result: JsonObject) -> _OverlayStatusView:
    raw_changes = result.get("changes", [])
    change_count = len(raw_changes) if isinstance(raw_changes, list) else 0
    return _OverlayStatusView(change_count=change_count)


def _decode_hook_env(result: JsonObject, *, prefix: str) -> _HookEnvView:
    return _HookEnvView(
        env=_string_mapping(result.get(f"{prefix}_env")),
        prepend_path=_string_sequence(result.get(f"{prefix}_prepend_path")),
        prepend_env=_string_sequence_mapping(result.get(f"{prefix}_prepend_env")),
    )


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    rendered: dict[str, str] = {}
    for key, item in value.items():
        rendered[str(key)] = str(item)
    return rendered


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _string_sequence_mapping(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    rendered: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        rendered[str(key)] = _string_sequence(item)
    return rendered


def _string_value(value: object, *, default: str) -> str:
    return value if isinstance(value, str) else default


def _optional_string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _optional_int_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_value(value: object, *, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def prepare_session_context(
    *,
    scope_name: str | None,
    create: bool,
    parent: str | None,
    capture: bool,
    usage_error_exit_code: int,
    env_error_exit_code: int,
) -> PreparedSessionContext:
    try:
        scope_name = parse_optional_scope_name(scope_name, allow_ground=not create)
        parent = parse_optional_scope_name(parent)
    except ParseError as exc:
        raise SessionCliError(str(exc), exit_code=usage_error_exit_code) from exc

    if create and scope_name is None:
        raise SessionCliError(
            "`vcs-core session exec --create` requires `--scope <name>`."
            if usage_error_exit_code == 2
            else "`vcs-core session shell --create` requires `--scope <name>`.",
            exit_code=usage_error_exit_code,
        )
    if parent is not None and not create:
        raise SessionCliError(
            "`--parent` is only valid together with `--create`.",
            exit_code=usage_error_exit_code,
        )

    info = _cli_ipc.live_session_info()
    if info is None:
        raise SessionCliError(
            "no session running. Start one with `vcs-core session start`.",
            exit_code=env_error_exit_code,
        )

    try:
        if create:
            fork_response = _cli_ipc.send_session_request(
                info,
                "fork",
                {"name": scope_name, "parent": parent or "ground", "isolated": True},
            )
            if not _cli_ipc.response_ok(fork_response):
                raise SessionCliError(
                    _cli_ipc.response_error(fork_response),
                    exit_code=env_error_exit_code,
                )

        if scope_name is not None:
            switch_response = _cli_ipc.send_session_request(info, "switch", {"name": scope_name})
            if not _cli_ipc.response_ok(switch_response):
                raise SessionCliError(
                    _cli_ipc.response_error(switch_response),
                    exit_code=env_error_exit_code,
                )

        requested_capabilities = ["fs_capture"] if capture else []
        state_response = _cli_ipc.send_session_request(
            info,
            "get_state",
            {"hook_capabilities": requested_capabilities},
        )
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=env_error_exit_code) from exc

    if not _cli_ipc.response_ok(state_response):
        raise SessionCliError(
            _cli_ipc.response_error(state_response),
            exit_code=env_error_exit_code,
        )

    state = _decode_session_state(
        _cli_ipc.response_result(state_response),
        default_pid=info.pid,
        default_mount_path=info.mount_path,
        default_workspace=info.workspace,
        default_started_at=info.started_at,
    )

    if not Path(state.mount_path).is_dir():
        raise SessionCliError(
            f"overlay mount path does not exist: {state.mount_path}",
            exit_code=env_error_exit_code,
        )
    if state.current_scope == "ground":
        raise SessionCliError(
            "session shell/exec on ground is disabled for snapshot-backed sessions; "
            "use `--scope <name> --create` to work in an isolated scope.",
            exit_code=usage_error_exit_code,
        )

    return PreparedSessionContext(
        mount_path=state.mount_path,
        active_scope=state.current_scope,
        workspace=state.workspace,
        session_socket_path=info.socket_path,
        daemon_instance_id=state.daemon_instance_id or info.daemon_instance_id,
        env=_apply_hook_runtime_env({**os.environ, "VCS_CORE_SESSION": "1"}, state),
    )


def load_live_session_status(*, exit_code: int = 1) -> LiveSessionStatus | None:
    info = _cli_ipc.live_session_info()
    if info is None:
        return None

    try:
        summary_response = _cli_ipc.send_session_request(info, "status_summary")
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=exit_code) from exc

    if not _cli_ipc.response_ok(summary_response):
        raise SessionCliError(_cli_ipc.response_error(summary_response), exit_code=exit_code)

    result = _cli_ipc.response_result(summary_response)
    live_scopes = result.get("live_scopes", [])
    retained_scopes = result.get("retained_scopes", [])
    blockers = result.get("blockers", [])

    return LiveSessionStatus(
        pid=_int_value(result.get("pid"), default=info.pid),
        started_at=_float_value(result.get("started_at"), default=info.started_at),
        workspace=_string_value(result.get("workspace"), default=info.workspace),
        mount_path=_string_value(result.get("mount_path"), default=info.mount_path),
        current_scope=_string_value(result.get("current_scope"), default="ground"),
        current_world_id=_optional_string_value(result.get("current_world_id")),
        overlay_change_count=_int_value(result.get("overlay_change_count"), default=0),
        local_changes=_int_value(result.get("local_changes"), default=0),
        commits_ahead=_int_value(result.get("commits_ahead"), default=0),
        live_scope_count=len(live_scopes) if isinstance(live_scopes, list) else 0,
        retained_scope_count=len(retained_scopes) if isinstance(retained_scopes, list) else 0,
        blocker_count=len(blockers) if isinstance(blockers, list) else 0,
        pending_operations=_optional_int_value(result.get("pending_operations")),
    )


def resolve_exec_cwd(mount_path: str, subpath: str | None) -> str:
    mount_root = Path(mount_path).resolve()
    if subpath is None:
        return str(mount_root)
    candidate = (mount_root / subpath).resolve()
    if os.path.commonpath([str(mount_root), str(candidate)]) != str(mount_root):
        raise SessionCliError(
            f"--cwd '{subpath}' escapes overlay mount.",
            exit_code=2,
        )
    if not candidate.is_dir():
        raise SessionCliError(
            f"--cwd '{subpath}' does not exist under overlay mount.",
            exit_code=2,
        )
    return str(candidate)


def begin_exec_envelope(
    *,
    argv: tuple[str, ...],
    cwd: str,
    scope_name: str,
    capture_requested: bool,
    exit_code: int,
) -> ExecEnvelope:
    info = _cli_ipc.live_session_info()
    if info is None:
        raise SessionCliError(
            "no session running. Start one with `vcs-core session start`.",
            exit_code=exit_code,
        )
    params: dict[str, object] = {
        "argv": list(argv),
        "cwd": cwd,
        "scope": scope_name,
        "capture_requested": capture_requested,
        "started_at": time.time(),
        "client_pid": os.getpid(),
    }
    try:
        response = _cli_ipc.send_session_request(info, "exec_envelope_begin", params)
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=exit_code) from exc
    if not _cli_ipc.response_ok(response):
        raise SessionCliError(_cli_ipc.response_error(response), exit_code=exit_code)

    result = _cli_ipc.response_result(response)
    operation_id = _string_value(result.get("operation_id"), default="")
    if not operation_id:
        raise SessionCliError("session daemon returned an empty exec operation id.", exit_code=exit_code)
    child_env = _string_mapping(result.get("env"))
    if "VCS_CORE_COMMAND_OPERATION_ID" not in child_env:
        child_env["VCS_CORE_COMMAND_OPERATION_ID"] = operation_id
    return ExecEnvelope(operation_id=operation_id, env=child_env)


def begin_shell_command_envelope(
    *,
    command_text: str,
    cwd: str,
    scope_name: str,
    shell_pid: int,
    shell_lease_id: str,
    socket_path: str | None = None,
    daemon_instance_id: str | None = None,
    exit_code: int,
) -> ExecEnvelope:
    """Begin a capture envelope for one interactive Bash prompt command."""
    label = command_text.strip() or "<empty shell command>"
    started_at = time.time()
    params: dict[str, object] = {
        "argv": ["bash", "-lc", label],
        "cwd": cwd,
        "scope": scope_name,
        "capture_requested": True,
        "capture_policy": "shell_command",
        "transport": "shell",
        "submitted_text": label,
        "shell_pid": shell_pid,
        "shell_lease_id": shell_lease_id,
        "started_at": started_at,
        "client_pid": shell_pid,
    }
    if daemon_instance_id is not None:
        params["daemon_instance_id"] = daemon_instance_id
    try:
        response = _send_envelope_request(
            socket_path=socket_path,
            method="exec_envelope_begin",
            params=params,
            missing_session_message="no session running. Start one with `vcs-core session start`.",
            exit_code=exit_code,
        )
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        _record_shell_command_not_admitted(
            command_text=label,
            cwd=cwd,
            scope_name=scope_name,
            shell_pid=shell_pid,
            shell_lease_id=shell_lease_id,
            socket_path=socket_path,
            daemon_instance_id=daemon_instance_id,
            started_at=started_at,
            admission_error=str(exc),
        )
        raise SessionCliError(
            str(exc),
            exit_code=exit_code,
        ) from exc
    if not _cli_ipc.response_ok(response):
        error = _cli_ipc.response_error(response)
        _record_shell_command_not_admitted(
            command_text=label,
            cwd=cwd,
            scope_name=scope_name,
            shell_pid=shell_pid,
            shell_lease_id=shell_lease_id,
            socket_path=socket_path,
            daemon_instance_id=daemon_instance_id,
            started_at=started_at,
            admission_error=error,
        )
        raise SessionCliError(error, exit_code=exit_code)

    result = _cli_ipc.response_result(response)
    operation_id = _string_value(result.get("operation_id"), default="")
    if not operation_id:
        raise SessionCliError("session daemon returned an empty shell command operation id.", exit_code=exit_code)
    child_env = _string_mapping(result.get("env"))
    if "VCS_CORE_COMMAND_OPERATION_ID" not in child_env:
        child_env["VCS_CORE_COMMAND_OPERATION_ID"] = operation_id
    return ExecEnvelope(operation_id=operation_id, env=child_env)


def _record_shell_command_not_admitted(
    *,
    command_text: str,
    cwd: str,
    scope_name: str,
    shell_pid: int,
    shell_lease_id: str,
    socket_path: str | None,
    daemon_instance_id: str | None,
    started_at: float,
    admission_error: str,
) -> None:
    params: dict[str, object] = {
        "cwd": cwd,
        "scope": scope_name,
        "capture_requested": True,
        "capture_policy": "shell_command",
        "transport": "shell",
        "submitted_text": command_text,
        "shell_pid": shell_pid,
        "shell_lease_id": shell_lease_id,
        "started_at": started_at,
        "ended_at": time.time(),
        "client_pid": shell_pid,
        "admission_error": admission_error,
    }
    if daemon_instance_id is not None:
        params["daemon_instance_id"] = daemon_instance_id
    try:
        response = _send_envelope_request(
            socket_path=socket_path,
            method="shell_command_not_admitted",
            params=params,
            missing_session_message="session disappeared before shell capture diagnostic could be recorded.",
            exit_code=3,
        )
    except Exception:  # noqa: BLE001
        return
    if not _cli_ipc.response_ok(response):
        return


def new_shell_capture_lease_id() -> str:
    return f"shl_{uuid.uuid4().hex}"


def begin_shell_capture_lease(
    *,
    lease_id: str,
    scope_name: str,
    shell_pid: int,
    socket_path: str,
    daemon_instance_id: str | None,
    exit_code: int,
) -> ShellCaptureLease:
    params: dict[str, object] = {
        "lease_id": lease_id,
        "scope": scope_name,
        "capture_requested": True,
        "shell_pid": shell_pid,
        "started_at": time.time(),
        "client_pid": os.getpid(),
    }
    if daemon_instance_id is not None:
        params["daemon_instance_id"] = daemon_instance_id
    try:
        response = _send_envelope_request(
            socket_path=socket_path,
            method="shell_capture_lease_begin",
            params=params,
            missing_session_message="no session running. Start one with `vcs-core session start`.",
            exit_code=exit_code,
        )
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=exit_code) from exc
    if not _cli_ipc.response_ok(response):
        raise SessionCliError(_cli_ipc.response_error(response), exit_code=exit_code)

    result = _cli_ipc.response_result(response)
    returned_lease_id = _string_value(result.get("lease_id"), default="")
    if returned_lease_id != lease_id:
        raise SessionCliError("session daemon returned an unexpected shell capture lease id.", exit_code=exit_code)
    return ShellCaptureLease(lease_id=returned_lease_id)


def finish_shell_capture_lease(
    *,
    lease_id: str,
    return_code: int,
    socket_path: str,
    daemon_instance_id: str | None,
) -> None:
    params: dict[str, object] = {
        "operation_id": lease_id,
        "ended_at": time.time(),
    }
    if return_code >= 0:
        params["outcome"] = "success" if return_code == 0 else "failed_exit"
        params["exit_code"] = return_code
    else:
        params["outcome"] = "signaled"
        params["signal"] = -return_code
    if daemon_instance_id is not None:
        params["daemon_instance_id"] = daemon_instance_id
    try:
        response = _send_envelope_request(
            socket_path=socket_path,
            method="shell_capture_lease_outcome",
            params=params,
            missing_session_message="session disappeared before shell capture lease could be released.",
            exit_code=3,
        )
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=3) from exc
    if not _cli_ipc.response_ok(response):
        raise SessionCliError(_cli_ipc.response_error(response), exit_code=3)


def finish_exec_envelope(
    *,
    operation_id: str,
    outcome: str,
    exit_code_value: int | None = None,
    signal_value: int | None = None,
    launch_error: str | None = None,
    abandoned_reason: str | None = None,
    socket_path: str | None = None,
    daemon_instance_id: str | None = None,
) -> None:
    params: dict[str, object] = {
        "operation_id": operation_id,
        "outcome": outcome,
        "ended_at": time.time(),
    }
    if exit_code_value is not None:
        params["exit_code"] = exit_code_value
    if signal_value is not None:
        params["signal"] = signal_value
    if launch_error is not None:
        params["launch_error"] = launch_error
    if abandoned_reason is not None:
        params["abandoned_reason"] = abandoned_reason
    if daemon_instance_id is not None:
        params["daemon_instance_id"] = daemon_instance_id
    try:
        response = _send_envelope_request(
            socket_path=socket_path,
            method="exec_envelope_outcome",
            params=params,
            missing_session_message="session disappeared before exec outcome could be recorded.",
            exit_code=3,
        )
    except (_cli_ipc.SessionIpcError, ValueError) as exc:
        raise SessionCliError(str(exc), exit_code=3) from exc
    if not _cli_ipc.response_ok(response):
        raise SessionCliError(_cli_ipc.response_error(response), exit_code=3)


def _send_envelope_request(
    *,
    socket_path: str | None,
    method: str,
    params: dict[str, object],
    missing_session_message: str,
    exit_code: int,
) -> SessionResponse:
    if socket_path is not None:
        return _cli_ipc.send_session_request_to_socket(socket_path, method, params)

    info = _cli_ipc.live_session_info()
    if info is None:
        raise SessionCliError(missing_session_message, exit_code=exit_code)
    return _cli_ipc.send_session_request(info, method, params)


def _debug_log_inside_tracked_workspace(value: str, workspace: str) -> bool:
    """True when an explicit ``--capture-debug`` path lands in the tracked workspace.

    Capture debug logs are control-plane metadata: auto-mode routes them under
    ``.vcscore/var/logs/`` (which ``push`` excludes). A path inside the tracked
    workspace (but outside ``.vcscore/``) would be refused by ``push`` as
    un-adopted, so the caller rejects it up front. The control plane itself is
    allowed (it is excluded from materialization by construction).
    """
    workspace_root = Path(workspace).resolve()
    debug_path = Path(os.path.abspath(value))
    if not debug_path.is_relative_to(workspace_root):
        return False
    return not debug_path.is_relative_to(workspace_root / ".vcscore")


def run_managed_exec(
    *,
    argv: tuple[str, ...],
    scope_name: str | None,
    create: bool,
    parent: str | None,
    cwd_subpath: str | None,
    capture_requested: bool,
    capture_debug: str | None,
    env: dict[str, str],
    stdout: BinaryIO,
    stderr: BinaryIO,
    exit_code: int,
    on_started: Callable[[str], None] | None = None,
    session_info: SessionInfo | None = None,
) -> int:
    """Run session exec through the daemon-owned streaming protocol."""
    info = session_info or _cli_ipc.live_session_info()
    if info is None:
        raise SessionCliError(
            "no session running. Start one with `vcs-core session start`.",
            exit_code=exit_code,
        )

    child_env = sanitized_managed_exec_env(dict(env))
    capture_debug_log = None
    if capture_debug is not None:
        if (
            capture_requested
            and capture_debug not in {AUTO_CAPTURE_DEBUG_SENTINEL, "--"}
            and _debug_log_inside_tracked_workspace(capture_debug, info.workspace)
        ):
            raise SessionCliError(
                f"--capture-debug path {capture_debug!r} is inside the tracked "
                "workspace; capture debug logs are control-plane metadata. Use "
                "--capture-debug=-- to write under .vcscore/var/logs/, or pass a "
                "path outside the workspace.",
                exit_code=exit_code,
            )
        debug_log_path, announced = resolve_debug_log_path(
            capture_debug,
            scope_name or "session",
            info.workspace,
        )
        capture_debug_log = debug_log_path
        if not capture_requested:
            stderr.write(b"Warning: --capture-debug has no effect without --capture.\n")
            stderr.flush()
        if announced:
            stderr.write(f"Capture debug log: {debug_log_path}\n".encode())
            stderr.flush()

    request_params = {
        "argv": list(argv),
        "scope": scope_name,
        "create": create,
        "parent": parent,
        "cwd_subpath": cwd_subpath,
        "capture_requested": capture_requested,
        "capture_debug_log": capture_debug_log,
        "env": child_env,
        "started_at": time.time(),
        "client_pid": os.getpid(),
    }
    operation_holder: dict[str, str | None] = {"operation_id": None}

    def _record_started(value: str) -> None:
        operation_holder["operation_id"] = value
        if on_started is not None:
            on_started(value)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            sock.connect(info.socket_path)
            sock.sendall(encode_managed_exec_request(request_params))
            sock.shutdown(socket.SHUT_WR)
            return _read_managed_exec_stream(
                sock,
                on_started=_record_started,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        except _ManagedExecOutputClosedError:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            return 1
        except KeyboardInterrupt:
            operation_id = operation_holder["operation_id"]
            if operation_id:
                _send_managed_exec_signal(info, operation_id=operation_id, signal_value=2)
                return _read_managed_exec_stream_after_interrupt(
                    sock, stdout=stdout, stderr=stderr, exit_code=exit_code
                )
            raise
        except OSError as exc:
            msg = f"Could not reach session daemon at {info.socket_path}: {exc}"
            raise SessionCliError(msg, exit_code=exit_code) from exc
    finally:
        sock.close()


def _read_managed_exec_stream(
    sock: socket.socket,
    *,
    on_started: Callable[[str], None] | None,
    stdout: BinaryIO,
    stderr: BinaryIO,
    exit_code: int,
) -> int:
    buffer = b""
    saw_exit = False
    stdout_open = True
    stderr_open = True
    final_exit_code = 126
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            raw_line, buffer = buffer.split(b"\n", 1)
            if not raw_line.strip():
                continue
            frame = _decode_managed_exec_frame(raw_line, exit_code=exit_code)
            if isinstance(frame, ManagedExecStartedFrame):
                if callable(on_started):
                    on_started(frame.operation_id)
                continue
            if isinstance(frame, ManagedExecStreamFrame) and frame.type == "stdout":
                if stdout_open:
                    try:
                        stdout.write(frame.data)
                        stdout.flush()
                    except BrokenPipeError:
                        stdout_open = False
                        raise _ManagedExecOutputClosedError from None
                continue
            if isinstance(frame, ManagedExecStreamFrame) and frame.type == "stderr":
                if stderr_open:
                    try:
                        stderr.write(frame.data)
                        stderr.flush()
                    except BrokenPipeError:
                        stderr_open = False
                        raise _ManagedExecOutputClosedError from None
                continue
            if isinstance(frame, ManagedExecMessageFrame) and frame.type == "error":
                message = frame.message
                if stderr_open:
                    try:
                        stderr.write(f"{prefixed_error_message(message)}\n".encode())
                        stderr.flush()
                    except BrokenPipeError:
                        stderr_open = False
                continue
            if isinstance(frame, ManagedExecMessageFrame) and frame.type == "recording_error":
                message = frame.message
                if stderr_open:
                    try:
                        stderr.write(f"Warning: failed to record session exec outcome: {message}\n".encode())
                        stderr.flush()
                    except BrokenPipeError:
                        stderr_open = False
                continue
            if isinstance(frame, ManagedExecExitFrame):
                final_exit_code = frame.exit_code
                saw_exit = True
                continue
            raise SessionCliError(f"session daemon sent unsupported exec frame: {frame!r}", exit_code=exit_code)
    if not saw_exit:
        raise SessionCliError("session daemon closed managed exec stream without an exit frame.", exit_code=exit_code)
    return final_exit_code


def _read_managed_exec_stream_after_interrupt(
    sock: socket.socket,
    *,
    stdout: BinaryIO,
    stderr: BinaryIO,
    exit_code: int,
) -> int:
    return _read_managed_exec_stream(
        sock,
        on_started=None,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
    )


def _send_managed_exec_signal(
    info: SessionInfo,
    *,
    operation_id: str,
    signal_value: int,
) -> None:
    response = _cli_ipc.send_session_request(
        info,
        "exec_managed_signal",
        {"operation_id": operation_id, "signal": signal_value},
    )
    if not _cli_ipc.response_ok(response):
        raise SessionCliError(_cli_ipc.response_error(response), exit_code=3)


def _decode_managed_exec_frame(
    raw_line: bytes,
    *,
    exit_code: int,
) -> ManagedExecStartedFrame | ManagedExecStreamFrame | ManagedExecMessageFrame | ManagedExecExitFrame:
    try:
        return decode_managed_exec_response_frame(raw_line)
    except (TypeError, ValueError) as exc:
        raise SessionCliError(f"session daemon sent invalid exec frame: {exc}", exit_code=exit_code) from exc


def resolve_debug_log_path(value: str, scope_name: str, workspace: str) -> tuple[str, bool]:
    workspace_root = Path(workspace).resolve()
    if value in {AUTO_CAPTURE_DEBUG_SENTINEL, "--"}:
        debug_path = workspace_root / ".vcscore" / "var" / "logs" / f"shim-{scope_name}-{int(time.time())}.log"
        announced = True
    else:
        debug_path = Path(os.path.abspath(value))
        announced = False
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    return str(debug_path), announced
