"""Managed execution transaction service for the session daemon."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

from vcs_core._admission.identifiers import ParseError, parse_optional_scope_name
from vcs_core._app import AppCommandBlocked, AppError, VcsCoreApp, app_error_message
from vcs_core._app_blockers import AppBlocker
from vcs_core._capture_reducer import ordered_capture_events
from vcs_core._errors import VcsCoreError
from vcs_core._managed_exec_process import (
    StreamItem,
    launch_process,
    process_group_id,
    pump_stream,
    shell_exit_code,
    terminate_process_group,
)
from vcs_core._managed_exec_protocol import (
    ManagedExecFrame,
    ManagedExecRequest,
    error_frame,
    exit_frame,
    managed_exec_request_from_params,
    recording_error_frame,
    started_frame,
    stream_frame,
)
from vcs_core._operation_start_authority import (
    begin_executable_operation,
    begin_not_admitted_shell_command_operation,
)
from vcs_core._operation_tx import OperationArchiveResult, archive_operation_with_fallback
from vcs_core._query_readiness import ReadinessOperationAuthority
from vcs_core._session_exec_envelope import (
    command_label,
    completion_command_metadata,
    new_capture_epoch_id,
    new_unique_command_operation_id,
    new_unique_shell_lease_operation_id,
    validate_exec_outcome,
)

if TYPE_CHECKING:
    from vcs_core._ipc import JsonObject
    from vcs_core.store import Store


logger = logging.getLogger(__name__)

_STREAM_QUEUE_MAX = 16
_PROCESS_POLL_SECONDS = 0.01
_POST_TERMINATE_DRAIN_SECONDS = 5.0


class ManagedExecutionHost(Protocol):
    _lock: Any
    _mg: Any
    _current_scope_name: str
    _managed_execs: dict[str, ManagedExecState]
    _daemon_instance_id: str
    _hook_frontier: Any
    _hook_accepted_seq: int
    _hook_processed_seq: int

    def _dispatch(self, method: str, params: JsonObject) -> JsonObject: ...


class ManagedExecUsageError(VcsCoreError, ValueError):
    """CLI usage error resolved by the daemon after request admission."""


@dataclass
class ManagedExecState:
    operation_id: str
    process: Any
    pgid: int


@dataclass(frozen=True)
class _PreparedExec:
    argv: list[str]
    cwd: str
    scope_name: str
    capture_requested: bool
    started_at: float
    client_pid: int
    env: dict[str, str]


@dataclass(frozen=True)
class _ParsedCommandOutcome:
    operation_id: str
    outcome: str
    ended_at: float
    exit_code: int | None
    signal_value: int | None
    launch_error: str | None
    abandoned_reason: str | None
    transport_status: str | None
    daemon_instance_id: str | None


@dataclass(frozen=True)
class _OpenCommandEnvelope:
    operation: Any
    start_metadata: dict[str, object]
    start_command: dict[str, object] | None
    capture_requested: bool


@dataclass(frozen=True)
class _CaptureCompletion:
    complete: bool
    incomplete_reason: str | None = None
    drain: Any | None = None


@dataclass(frozen=True)
class _ScopeWriterConflict:
    operation_id: str
    label: str


class ManagedExecutionService:
    """Own daemon-side command envelope and capture completion policy."""

    def __init__(self, daemon: ManagedExecutionHost) -> None:
        self._daemon = daemon

    def assert_scope_lifecycle_unblocked(self, scope_name: str, *, action: str) -> None:
        """Reject scope lifecycle mutation while command writers are open."""
        self.assert_scope_writer_unblocked(scope_name, action=action)

    def assert_scope_writer_unblocked(
        self,
        scope_name: str,
        *,
        action: str,
        allowed_shell_lease_id: str | None = None,
    ) -> None:
        """Reject same-scope mutations while a session writer is open."""
        with self._daemon._lock:
            with VcsCoreApp.active_view(self._daemon._mg, current_scope=self._daemon._current_scope_name) as app:
                scope = app.resolve_scope(scope_name)
            conflict = _find_open_scope_writer(
                self._daemon._mg.store,
                scope_ref=scope.ref,
                allowed_shell_lease_id=allowed_shell_lease_id,
            )
            if conflict is None:
                return
            raise ValueError(
                f"Cannot {action} scope {scope.name!r}: "
                f"{conflict.label} {conflict.operation_id!r} is still open for that scope."
            )

    def run_params(self, params: JsonObject) -> Any:
        try:
            request = managed_exec_request_from_params(params)
        except (TypeError, ValueError) as exc:
            yield error_frame(str(exc))
            yield exit_frame(3)
            return
        yield from self.run(request)

    def run(self, request: ManagedExecRequest) -> Any:
        operation_id = ""
        process = None
        pgid: int | None = None
        return_code: int | None = None
        terminal_recorded = False
        registered = False

        try:
            try:
                prepared = self._prepare_exec(request)
                begin = self.begin_envelope(
                    {
                        "argv": prepared.argv,
                        "cwd": prepared.cwd,
                        "scope": prepared.scope_name,
                        "capture_requested": prepared.capture_requested,
                        "managed": True,
                        "started_at": prepared.started_at,
                        "client_pid": prepared.client_pid,
                    }
                )
                operation_id = str(begin["operation_id"])
                child_env = dict(prepared.env)
                child_env.update(_string_mapping(begin.get("env")))
                process = launch_process(argv=prepared.argv, cwd=prepared.cwd, env=child_env)
            except FileNotFoundError:
                command = prepared.argv[0] if "prepared" in locals() and prepared.argv else "session exec"
                yield from self._record_launch_error_frames(
                    operation_id,
                    exit_code=127,
                    message=f"command not found: {command}",
                )
                terminal_recorded = True
                return
            except PermissionError:
                command = prepared.argv[0] if "prepared" in locals() and prepared.argv else "session exec"
                yield from self._record_launch_error_frames(
                    operation_id,
                    exit_code=126,
                    message=f"target not executable: {command}",
                )
                terminal_recorded = True
                return
            except OSError as exc:
                detail = exc.strerror or str(exc)
                command = prepared.argv[0] if "prepared" in locals() and prepared.argv else "session exec"
                yield from self._record_launch_error_frames(
                    operation_id,
                    exit_code=126,
                    message=f"failed to launch {command}: {detail}",
                )
                terminal_recorded = True
                return
            except ManagedExecUsageError as exc:
                yield error_frame(str(exc))
                yield exit_frame(2)
                terminal_recorded = True
                return
            except AppError as exc:
                yield error_frame(app_error_message(exc))
                yield exit_frame(3)
                terminal_recorded = True
                return
            except Exception as exc:  # noqa: BLE001
                yield error_frame(str(exc) or exc.__class__.__name__)
                yield exit_frame(3)
                terminal_recorded = True
                return

            pgid = process_group_id(process)
            with self._daemon._lock:
                self._daemon._managed_execs[operation_id] = ManagedExecState(
                    operation_id=operation_id,
                    process=process,
                    pgid=pgid,
                )
                registered = True

            yield started_frame(operation_id=operation_id, pid=process.pid, pgid=pgid)
            return_code, stream_connected = yield from self._stream_process(process, pgid=pgid)
            try:
                result = self._finish_process(
                    operation_id,
                    return_code=return_code,
                    transport_status=None if stream_connected else "client_disconnected",
                )
                terminal_recorded = True
                message = result.get("recording_error")
                if isinstance(message, str) and message and stream_connected:
                    yield recording_error_frame(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to record managed session exec outcome %s", operation_id, exc_info=True)
                terminal_recorded = True
                if stream_connected:
                    yield recording_error_frame(str(exc) or exc.__class__.__name__)
            if stream_connected:
                yield exit_frame(shell_exit_code(return_code))
        except GeneratorExit:
            if process is not None and pgid is not None and operation_id:
                if return_code is None:
                    terminate_process_group(pgid)
                    with suppress(Exception):
                        process.wait(timeout=1.0)
                    self._finish_abandoned(operation_id, reason="client disconnected")
                elif not terminal_recorded:
                    with suppress(Exception):
                        self._finish_process(
                            operation_id,
                            return_code=return_code,
                            transport_status="client_disconnected",
                        )
            raise
        finally:
            if registered:
                with self._daemon._lock:
                    self._daemon._managed_execs.pop(operation_id, None)

    def _prepare_exec(self, request: ManagedExecRequest) -> _PreparedExec:
        try:
            scope_name = parse_optional_scope_name(request.scope_name, allow_ground=not request.create)
            parent = parse_optional_scope_name(request.parent)
        except ParseError as exc:
            raise ManagedExecUsageError(str(exc)) from exc
        if request.create and scope_name is None:
            raise ManagedExecUsageError("`vcs-core session exec --create` requires `--scope <name>`.")
        if parent is not None and not request.create:
            raise ManagedExecUsageError("`--parent` is only valid together with `--create`.")

        with self._daemon._lock:
            if request.create:
                fork = self._daemon._dispatch(
                    "fork", {"name": scope_name, "parent": parent or "ground", "isolated": True}
                )
                scope_name = str(fork["name"])
            if scope_name is not None:
                self._daemon._dispatch("switch", {"name": scope_name})
            state = self._daemon._dispatch(
                "get_state",
                {"hook_capabilities": ["fs_capture"] if request.capture_requested else []},
            )

        active_scope = _require_str(state, "current_scope")
        if active_scope == "ground":
            raise ManagedExecUsageError(
                "session shell/exec on ground is disabled for snapshot-backed sessions; "
                "use `--scope <name> --create` to work in an isolated scope."
            )
        mount_path = _require_str(state, "mount_path")
        if not Path(mount_path).is_dir():
            raise ManagedExecUsageError(f"overlay mount path does not exist: {mount_path}")
        cwd = _resolve_exec_cwd(mount_path, request.cwd_subpath)
        return _PreparedExec(
            argv=request.argv,
            cwd=cwd,
            scope_name=active_scope,
            capture_requested=request.capture_requested,
            started_at=request.started_at,
            client_pid=request.client_pid,
            env=_apply_hook_runtime_env(request.env, state),
        )

    def _stream_process(self, process: Any, *, pgid: int) -> Any:
        stream_queue: queue.Queue[StreamItem] = queue.Queue(maxsize=_STREAM_QUEUE_MAX)
        stop_streams = threading.Event()
        stdout_thread = threading.Thread(
            target=pump_stream,
            args=("stdout", process.stdout, stream_queue, stop_streams),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump_stream,
            args=("stderr", process.stderr, stream_queue, stop_streams),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        streams_done: set[str] = set()
        return_code: int | None = None
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                item = _get_stream_item(stream_queue)
                if item is None:
                    continue
                frame = _stream_item_frame(item, streams_done)
                if frame is not None:
                    yield frame

            terminate_process_group(pgid)
            if len(streams_done) < 2:
                stream_connected = yield from self._drain_remaining_streams(stream_queue, streams_done)
                if not stream_connected:
                    return (return_code, False)
        finally:
            stop_streams.set()
            stdout_thread.join(timeout=0.1)
            stderr_thread.join(timeout=0.1)

        if return_code is None:
            return_code = process.wait()
        return (return_code, True)

    def _drain_remaining_streams(
        self,
        stream_queue: queue.Queue[StreamItem],
        streams_done: set[str],
    ) -> Any:
        deadline = time.monotonic() + _POST_TERMINATE_DRAIN_SECONDS
        while len(streams_done) < 2:
            timeout = max(0.0, min(_PROCESS_POLL_SECONDS, deadline - time.monotonic()))
            if timeout == 0.0:
                return True
            try:
                item = stream_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            frame = _stream_item_frame(item, streams_done)
            if frame is not None:
                yield frame
        return True

    def _record_launch_error_frames(self, operation_id: str, *, exit_code: int, message: str) -> Any:
        if operation_id:
            try:
                result = self.record_outcome(
                    {
                        "operation_id": operation_id,
                        "outcome": "launch_error",
                        "ended_at": time.time(),
                        "exit_code": exit_code,
                        "launch_error": message,
                    }
                )
                recording_error = result.get("recording_error")
                if isinstance(recording_error, str) and recording_error:
                    yield recording_error_frame(recording_error)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to record managed session exec launch error %s", operation_id, exc_info=True)
                yield recording_error_frame(str(exc) or exc.__class__.__name__)
        yield error_frame(message)
        yield exit_frame(exit_code)

    def _finish_process(
        self,
        operation_id: str,
        *,
        return_code: int,
        transport_status: str | None = None,
    ) -> JsonObject:
        if return_code < 0:
            params: dict[str, object] = {
                "operation_id": operation_id,
                "outcome": "signaled",
                "ended_at": time.time(),
                "signal": -return_code,
            }
            if transport_status is not None:
                params["transport_status"] = transport_status
            return dict(self.record_outcome(params))
        params = {
            "operation_id": operation_id,
            "outcome": "success" if return_code == 0 else "failed_exit",
            "ended_at": time.time(),
            "exit_code": return_code,
        }
        if transport_status is not None:
            params["transport_status"] = transport_status
        return dict(self.record_outcome(params))

    def _finish_abandoned(self, operation_id: str, *, reason: str) -> None:
        try:
            self.record_outcome(
                {
                    "operation_id": operation_id,
                    "outcome": "abandoned",
                    "ended_at": time.time(),
                    "abandoned_reason": reason,
                }
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to record abandoned managed session exec %s", operation_id, exc_info=True)

    def begin_shell_capture_lease(self, params: JsonObject) -> JsonObject:
        with self._daemon._lock:
            self._validate_daemon_instance_id(_optional_str(params, "daemon_instance_id", default=None))
            scope_name = (
                _parse_optional_scope_name_for_method(
                    "session shell", _optional_str(params, "scope", default=self._daemon._current_scope_name)
                )
                or self._daemon._current_scope_name
            )
            capture_requested = _optional_bool(params, "capture_requested", default=True)
            if not capture_requested:
                raise ValueError("shell capture lease requires capture_requested=true.")
            client_pid = _optional_int(params, "client_pid", default=0)
            shell_pid = _optional_int(params, "shell_pid", default=None)
            started_at = _optional_float(params, "started_at", default=time.time())
            requested_lease_id = _optional_str(params, "lease_id", default=None)
            lease_id = requested_lease_id or new_unique_shell_lease_operation_id(self._daemon._mg.store)
            if self._daemon._mg.store.operation_id_exists(lease_id):
                raise ValueError(f"Shell capture lease id {lease_id!r} already exists.")

            with VcsCoreApp.active_view(self._daemon._mg, current_scope=self._daemon._current_scope_name) as app:
                scope = app.resolve_scope(scope_name)
                app.retain_restored_scope(scope_name)
            conflict = _find_open_scope_writer(
                self._daemon._mg.store,
                scope_ref=scope.ref,
            )
            if conflict is not None:
                raise ValueError(
                    f"Cannot start captured session shell on scope {scope.name!r}: "
                    f"{conflict.label} {conflict.operation_id!r} is still open for that scope."
                )

            shell: dict[str, object] = {
                "scope": scope.name,
                "capture_requested": True,
                "client_pid": client_pid,
                "started_at": started_at,
                "daemon_instance_id": self._daemon._daemon_instance_id,
            }
            if shell_pid is not None:
                shell["shell_pid"] = shell_pid
            operation = begin_executable_operation(
                self._daemon._mg,
                scope,
                attempted=f"open session shell capture lease {lease_id}",
                handle_id=lease_id,
                kind="vcs_core.session_shell",
                world_id=scope.world_id or self._daemon._mg._scope_world_id(scope),
                scope_instance_id=scope.instance_id,
                operation_id=lease_id,
                operation_label=f"session shell --capture: {scope.name}",
                session_id=self._daemon._mg._session_id,
                metadata={"shell": shell},
            )
            # M3: tag the daemon-owned VcsCore with this daemon's instance id
            # (the same id written into the lease metadata above), so
            # query_readiness excludes this lease from orphaned-operation
            # blockers. Co-located with the lease daemon-id write so the two
            # cannot drift; the lease-discovery in query_readiness is the real
            # gate (the tag is harmless when no lease is open).
            self._daemon._mg._active_daemon_instance_id = self._daemon._daemon_instance_id
        return {
            "lease_id": operation.durable_id,
            "operation_ref": operation.ref,
        }

    def finish_shell_capture_lease(self, params: JsonObject) -> JsonObject:
        parsed = _parse_command_outcome(params)
        with self._daemon._lock:
            operation = _find_open_shell_lease_operation(self._daemon._mg.store, parsed.operation_id)
            start_metadata = self._daemon._mg.store._read_operation_start_metadata(operation.ref)
            self._validate_shell_capture_lease_outcome(parsed.daemon_instance_id, start_metadata=start_metadata)
            shell_metadata = _shell_lease_completion_metadata(start_metadata, parsed=parsed)
            archive_ref = self._daemon._mg.store.complete_operation_to_archive(
                operation,
                metadata={"shell": shell_metadata},
                status="ok" if parsed.outcome == "success" else "error",
            )
        return {"lease_id": parsed.operation_id, "archive_ref": archive_ref}

    def begin_envelope(self, params: JsonObject) -> JsonObject:
        with self._daemon._lock:
            argv = _require_string_list(params, "argv")
            cwd = _require_str(params, "cwd")
            capture_requested = _optional_bool(params, "capture_requested", default=False)
            capture_policy = _optional_str(params, "capture_policy", default=None)
            if capture_policy not in {None, "shell_command"}:
                raise ValueError(f"Unsupported capture policy: {capture_policy!r}")
            transport = _optional_str(params, "transport", default="exec") or "exec"
            if transport not in {"exec", "shell"}:
                raise ValueError(f"Unsupported session command transport: {transport!r}")
            submitted_text = _optional_str(params, "submitted_text", default=None)
            daemon_instance_id = _optional_str(params, "daemon_instance_id", default=None)
            if transport == "shell":
                self._validate_daemon_instance_id(daemon_instance_id)
            shell_lease_id = _optional_str(params, "shell_lease_id", default=None)
            shell_pid = _optional_int(params, "shell_pid", default=None)
            if transport == "shell" and shell_pid is None:
                shell_pid = _optional_int(params, "client_pid", default=0)
            managed = _optional_bool(params, "managed", default=False)
            started_at = _optional_float(params, "started_at", default=time.time())
            client_pid = _optional_int(params, "client_pid", default=0)
            scope_name = (
                _parse_optional_scope_name_for_method(
                    "session exec", _optional_str(params, "scope", default=self._daemon._current_scope_name)
                )
                or self._daemon._current_scope_name
            )
            with VcsCoreApp.active_view(self._daemon._mg, current_scope=self._daemon._current_scope_name) as app:
                scope = app.resolve_scope(scope_name)
                app.retain_restored_scope(scope_name)
            if transport == "shell":
                if not shell_lease_id:
                    raise ValueError("shell command capture requires an active shell capture lease.")
                self._validate_shell_capture_lease_begin(
                    shell_lease_id,
                    scope_ref=scope.ref,
                    daemon_instance_id=daemon_instance_id,
                    shell_pid=shell_pid,
                )
            conflict = _find_open_scope_writer(
                self._daemon._mg.store,
                scope_ref=scope.ref,
                allowed_shell_lease_id=shell_lease_id if transport == "shell" else None,
            )
            if conflict is not None:
                capture_word = "captured " if capture_requested else ""
                msg = (
                    f"Cannot start {capture_word}session exec on scope {scope.name!r}: "
                    f"{conflict.label} {conflict.operation_id!r} is still open for that scope."
                )
                raise ValueError(msg)

            operation_id = new_unique_command_operation_id(self._daemon._mg.store)
            capture_epoch = new_capture_epoch_id() if capture_requested else None
            metadata: dict[str, object] = {
                "command": {
                    "argv": argv,
                    "cwd": cwd,
                    "scope": scope.name,
                    "capture_requested": capture_requested,
                    "managed": managed,
                    "client_pid": client_pid,
                    "started_at": started_at,
                    "transport": transport,
                }
            }
            if transport == "shell":
                command = metadata["command"]
                assert isinstance(command, dict)
                if submitted_text is not None:
                    command["submitted_text"] = submitted_text
                if shell_pid is not None:
                    command["shell_pid"] = shell_pid
                if shell_lease_id is not None:
                    command["shell_lease_id"] = shell_lease_id
                if daemon_instance_id is not None:
                    command["daemon_instance_id"] = daemon_instance_id
            if capture_epoch is not None:
                command = metadata["command"]
                assert isinstance(command, dict)
                command["capture_epoch"] = capture_epoch
                if capture_policy is not None:
                    command["capture_policy"] = capture_policy
            operation = begin_executable_operation(
                self._daemon._mg,
                scope,
                attempted=f"open session exec envelope {operation_id}",
                handle_id=operation_id,
                kind="vcs_core.session_exec",
                world_id=scope.world_id or self._daemon._mg._scope_world_id(scope),
                scope_instance_id=scope.instance_id,
                operation_id=operation_id,
                operation_label=command_label(argv, transport=transport, submitted_text=submitted_text),
                session_id=self._daemon._mg._session_id,
                metadata=metadata,
                authorized_operations=(
                    (
                        ReadinessOperationAuthority(
                            operation_id=shell_lease_id,
                            kind="vcs_core.session_shell",
                            scope_ref=scope.ref,
                            scope_instance_id=scope.instance_id,
                            session_id=self._daemon._mg._session_id,
                        ),
                    )
                    if transport == "shell" and shell_lease_id is not None
                    else ()
                ),
            )
            capture_authority = getattr(self._daemon, "_capture_authority", None)
            if capture_requested and capture_authority is not None:
                if capture_policy == "shell_command":
                    capture_authority.begin(
                        operation.durable_id,
                        capture_policy="shell_command",
                        shell_pid=shell_pid if shell_pid is not None and shell_pid > 0 else None,
                    )
                else:
                    capture_authority.begin(operation.durable_id, require_lifecycle=managed)

        env = {"VCS_CORE_COMMAND_OPERATION_ID": operation.durable_id}
        if capture_epoch is not None:
            env["VCS_CORE_CAPTURE_EPOCH"] = capture_epoch
            env["VCS_CORE_CAPTURE_ACTIVE"] = "1"
        return {
            "operation_id": operation.durable_id,
            "operation_ref": operation.ref,
            "env": env,
        }

    def record_shell_command_not_admitted(self, params: JsonObject) -> JsonObject:
        with self._daemon._lock:
            daemon_instance_id = _optional_str(params, "daemon_instance_id", default=None)
            self._validate_daemon_instance_id(daemon_instance_id)
            cwd = _require_str(params, "cwd")
            submitted_text = _optional_str(params, "submitted_text", default=None)
            label = (submitted_text or "").strip() or "<empty shell command>"
            shell_pid = _optional_int(params, "shell_pid", default=None)
            if shell_pid is None:
                shell_pid = _optional_int(params, "client_pid", default=0)
            shell_lease_id = _optional_str(params, "shell_lease_id", default=None)
            started_at = _optional_float(params, "started_at", default=time.time())
            ended_at = _optional_float(params, "ended_at", default=time.time())
            admission_error = _optional_str(params, "admission_error", default=None)
            scope_name = (
                _parse_optional_scope_name_for_method(
                    "session shell", _optional_str(params, "scope", default=self._daemon._current_scope_name)
                )
                or self._daemon._current_scope_name
            )

            with VcsCoreApp.active_view(self._daemon._mg, current_scope=self._daemon._current_scope_name) as app:
                scope = app.resolve_scope(scope_name)
                app.retain_restored_scope(scope_name)

            operation_id = new_unique_command_operation_id(self._daemon._mg.store)
            command: dict[str, object] = {
                "argv": ["bash", "-lc", label],
                "cwd": cwd,
                "scope": scope.name,
                "capture_requested": True,
                "capture_policy": "shell_command",
                "managed": False,
                "client_pid": shell_pid,
                "started_at": started_at,
                "transport": "shell",
                "submitted_text": label,
                "shell_pid": shell_pid,
                "daemon_instance_id": daemon_instance_id or self._daemon._daemon_instance_id,
            }
            if shell_lease_id is not None:
                command["shell_lease_id"] = shell_lease_id
            operation = begin_not_admitted_shell_command_operation(
                self._daemon._mg,
                scope,
                handle_id=operation_id,
                world_id=scope.world_id or self._daemon._mg._scope_world_id(scope),
                scope_instance_id=scope.instance_id,
                operation_id=operation_id,
                operation_label=command_label(["bash", "-lc", label], transport="shell", submitted_text=label),
                session_id=self._daemon._mg._session_id,
                metadata={"command": command},
            )
            command_metadata = completion_command_metadata(
                {"command": command},
                outcome="abandoned",
                ended_at=ended_at,
                exit_code=None,
                signal=None,
                launch_error=None,
                abandoned_reason="shell_command_not_admitted",
            )
            command_metadata["capture_status"] = "incomplete"
            command_metadata["capture_stream_status"] = "not_admitted"
            command_metadata["capture_incomplete_reason"] = "shell_command_not_admitted"
            if admission_error:
                command_metadata["admission_error"] = admission_error
            archive_ref = self._daemon._mg.store.abort_operation(
                operation,
                metadata={"command": command_metadata},
                status="error",
            )
        return {"operation_id": operation_id, "archive_ref": archive_ref}

    def record_outcome(self, params: JsonObject) -> JsonObject:
        parsed = _parse_command_outcome(params)
        with self._daemon._lock:
            envelope = self._open_command_envelope(parsed.operation_id)
            if _is_shell_command(envelope.start_command):
                self._validate_daemon_instance_id(parsed.daemon_instance_id, start_command=envelope.start_command)
            hook_accepted_snapshot = self._daemon._hook_frontier.accepted_seq
            self._daemon._hook_accepted_seq = hook_accepted_snapshot
            self._daemon._hook_processed_seq = self._daemon._hook_frontier.processed_seq

        capture = self._complete_capture(
            envelope.capture_requested,
            parsed.operation_id,
            hook_accepted_snapshot=hook_accepted_snapshot,
        )

        with self._daemon._lock:
            envelope = self._open_command_envelope(parsed.operation_id)
            capture = self._validate_capture_journal_shape(envelope, capture)
            command_metadata = self._completion_metadata(
                envelope,
                parsed=parsed,
                capture=capture,
            )
            reduction_error = self._reduce_capture_if_complete(envelope, capture, command_metadata)
            finalize_error = None
            try:
                archive = self._archive_command_outcome(envelope, parsed, command_metadata)
            finally:
                if envelope.capture_requested:
                    finalize_error = self._finalize_capture_authority(envelope)
            recording_error = _combine_recording_errors(
                reduction_error,
                archive.recording_error,
                finalize_error,
            )

        result: JsonObject = {"operation_id": parsed.operation_id, "archive_ref": archive.archive_ref}
        if recording_error is not None:
            result["recording_error"] = recording_error
        return result

    def _open_command_envelope(self, operation_id: str) -> _OpenCommandEnvelope:
        operation = _find_open_operation(self._daemon._mg.store, operation_id)
        start_metadata = self._daemon._mg.store._read_operation_start_metadata(operation.ref)
        start_command = start_metadata.get("command")
        typed_start_command = start_command if isinstance(start_command, dict) else None
        capture_requested = typed_start_command is not None and typed_start_command.get("capture_requested") is True
        return _OpenCommandEnvelope(
            operation=operation,
            start_metadata=start_metadata,
            start_command=typed_start_command,
            capture_requested=capture_requested,
        )

    def _complete_capture(
        self,
        capture_requested: bool,
        operation_id: str,
        *,
        hook_accepted_snapshot: int,
    ) -> _CaptureCompletion:
        if not capture_requested:
            return _CaptureCompletion(complete=True)
        wait_for_hook_drain = getattr(self._daemon, "_wait_for_hook_drain", None)
        if callable(wait_for_hook_drain) and not wait_for_hook_drain(min_accepted_seq=hook_accepted_snapshot):
            return _CaptureCompletion(complete=False, incomplete_reason="hook_drain_timeout")
        wait_for_capture_drain = getattr(self._daemon, "_wait_for_capture_drain", None)
        if callable(wait_for_capture_drain):
            drain = wait_for_capture_drain(operation_id)
            return _CaptureCompletion(
                complete=bool(drain.complete),
                incomplete_reason=drain.reason,
                drain=drain,
            )
        return _CaptureCompletion(complete=True)

    def _validate_capture_journal_shape(
        self,
        envelope: _OpenCommandEnvelope,
        capture: _CaptureCompletion,
    ) -> _CaptureCompletion:
        if not envelope.capture_requested or not capture.complete or not _is_shell_command(envelope.start_command):
            return capture
        history = self._daemon._mg.store.read_operation_history(envelope.operation.ref)
        events = ordered_capture_events(history.commits)
        open_writers: dict[tuple[int, str], int] = {}
        for event in events:
            key = (event.pid, event.path)
            if event.op == "write_open":
                if open_writers.get(key, 0) > 0:
                    return _CaptureCompletion(
                        complete=False,
                        incomplete_reason="fd_context_crossed_command",
                        drain=capture.drain,
                    )
                open_writers[key] = 1
                continue
            if event.op == "write_observed":
                if key not in open_writers:
                    return _CaptureCompletion(
                        complete=False,
                        incomplete_reason="fd_context_crossed_command",
                        drain=capture.drain,
                    )
                continue
            if event.op == "write_close":
                if key not in open_writers:
                    return _CaptureCompletion(
                        complete=False,
                        incomplete_reason="fd_context_crossed_command",
                        drain=capture.drain,
                    )
                del open_writers[key]
        if open_writers:
            return _CaptureCompletion(
                complete=False,
                incomplete_reason="dirty_fd_left_open",
                drain=capture.drain,
            )
        return capture

    def _completion_metadata(
        self,
        envelope: _OpenCommandEnvelope,
        *,
        parsed: _ParsedCommandOutcome,
        capture: _CaptureCompletion,
    ) -> dict[str, object]:
        command_metadata = completion_command_metadata(
            envelope.start_metadata,
            outcome=parsed.outcome,
            ended_at=parsed.ended_at,
            exit_code=parsed.exit_code,
            signal=parsed.signal_value,
            launch_error=parsed.launch_error,
            abandoned_reason=parsed.abandoned_reason,
        )
        if envelope.start_command is not None:
            capture_epoch = envelope.start_command.get("capture_epoch")
            if isinstance(capture_epoch, str) and capture_epoch:
                command_metadata["capture_epoch"] = capture_epoch
        if parsed.transport_status is not None:
            command_metadata["transport_status"] = parsed.transport_status
        if envelope.capture_requested:
            if capture.complete:
                command_metadata["capture_status"] = "complete"
                command_metadata["capture_stream_status"] = "drained"
            else:
                command_metadata["capture_status"] = "incomplete"
                command_metadata["capture_stream_status"] = "incomplete"
                command_metadata["capture_incomplete_reason"] = capture.incomplete_reason or "capture_incomplete"
            if capture.drain is not None:
                command_metadata["capture_registered_processes"] = capture.drain.registered_count
                command_metadata["capture_finished_processes"] = capture.drain.finished_count
                command_metadata["capture_event_count"] = capture.drain.accepted_count
        return command_metadata

    def _reduce_capture_if_complete(
        self,
        envelope: _OpenCommandEnvelope,
        capture: _CaptureCompletion,
        command_metadata: dict[str, object],
    ) -> str | None:
        if not envelope.capture_requested or not capture.complete:
            return None
        if not hasattr(self._daemon._mg, "_reduce_capture_for_command_operation"):
            return None
        try:
            self._daemon._mg._reduce_capture_for_command_operation(
                envelope.operation.durable_id,
                command_metadata=command_metadata,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to reduce capture for session exec %s",
                envelope.operation.durable_id,
                exc_info=True,
            )
            command_metadata["capture_status"] = "incomplete"
            command_metadata["capture_stream_status"] = "incomplete"
            command_metadata["capture_incomplete_reason"] = "capture_reduction_failed"
            capture_authority = getattr(self._daemon, "_capture_authority", None)
            if capture_authority is not None:
                capture_authority.mark_failed(
                    envelope.operation.durable_id,
                    global_seq=0,
                    reason="capture_reduction_failed",
                )
            return f"capture reduction failed: {str(exc) or exc.__class__.__name__}"

    def _archive_command_outcome(
        self,
        envelope: _OpenCommandEnvelope,
        parsed: _ParsedCommandOutcome,
        command_metadata: dict[str, object],
    ) -> OperationArchiveResult:
        metadata: dict[str, object] = {"command": command_metadata}
        if parsed.outcome == "abandoned":
            archive_ref = self._daemon._mg.store.abort_operation(
                envelope.operation,
                metadata=metadata,
                status="error",
            )
            return OperationArchiveResult(archive_ref=archive_ref)
        result = archive_operation_with_fallback(
            self._daemon._mg.store,
            envelope.operation,
            metadata=metadata,
            status="ok" if parsed.outcome == "success" else "error",
            fallback_error_prefix="session exec outcome archive failed",
        )
        if result.recording_error is not None:
            logger.warning("Failed to archive session exec outcome %s", parsed.operation_id)
        return result

    def _finalize_capture_authority(self, envelope: _OpenCommandEnvelope) -> str | None:
        capture_authority = getattr(self._daemon, "_capture_authority", None)
        if capture_authority is None:
            return None
        try:
            capture_authority.finalize(envelope.operation.durable_id)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to finalize capture authority for session exec %s",
                envelope.operation.durable_id,
                exc_info=True,
            )
            return f"capture authority finalize failed: {str(exc) or exc.__class__.__name__}"

    def _validate_daemon_instance_id(
        self,
        provided: str | None,
        *,
        start_command: dict[str, object] | None = None,
    ) -> None:
        expected = getattr(self._daemon, "_daemon_instance_id", None)
        if isinstance(expected, str) and expected and provided != expected:
            raise ValueError("stale shell capture helper for daemon instance")

        if start_command is None:
            return
        started = start_command.get("daemon_instance_id")
        if isinstance(started, str) and started and provided != started:
            raise ValueError("stale shell capture helper for daemon instance")

    def _validate_shell_capture_lease_begin(
        self,
        lease_id: str,
        *,
        scope_ref: str,
        daemon_instance_id: str | None,
        shell_pid: int | None,
    ) -> None:
        operation = _find_open_shell_lease_operation(self._daemon._mg.store, lease_id)
        if operation.scope_ref != scope_ref:
            raise ValueError("shell command capture lease belongs to a different scope.")
        start_metadata = self._daemon._mg.store._read_operation_start_metadata(operation.ref)
        shell = start_metadata.get("shell")
        if not isinstance(shell, dict) or shell.get("capture_requested") is not True:
            raise ValueError("shell command capture lease is not capture-enabled.")
        started_daemon_id = shell.get("daemon_instance_id")
        if isinstance(started_daemon_id, str) and started_daemon_id and daemon_instance_id != started_daemon_id:
            raise ValueError("stale shell capture helper for daemon instance")
        leased_shell_pid = shell.get("shell_pid")
        if (
            isinstance(leased_shell_pid, int)
            and not isinstance(leased_shell_pid, bool)
            and leased_shell_pid > 0
            and shell_pid != leased_shell_pid
        ):
            raise ValueError("stale shell capture helper for shell pid")

    def _validate_shell_capture_lease_outcome(
        self,
        daemon_instance_id: str | None,
        *,
        start_metadata: dict[str, object],
    ) -> None:
        self._validate_daemon_instance_id(daemon_instance_id)
        shell = start_metadata.get("shell")
        if not isinstance(shell, dict):
            raise TypeError("shell capture lease is missing start metadata.")
        started_daemon_id = shell.get("daemon_instance_id")
        if isinstance(started_daemon_id, str) and started_daemon_id and daemon_instance_id != started_daemon_id:
            raise ValueError("stale shell capture helper for daemon instance")


def _require_str(params: JsonObject, key: str) -> str:
    value = params.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected string parameter '{key}'.")


def _get_stream_item(stream_queue: queue.Queue[StreamItem]) -> StreamItem | None:
    try:
        return stream_queue.get(timeout=_PROCESS_POLL_SECONDS)
    except queue.Empty:
        return None


def _stream_item_frame(item: StreamItem, streams_done: set[str]) -> ManagedExecFrame | None:
    if item.data is None:
        streams_done.add(item.name)
        return None
    if item.name == "stdout":
        return stream_frame("stdout", item.data)
    if item.name == "stderr":
        return stream_frame("stderr", item.data)
    return None


def _apply_hook_runtime_env(base_env: dict[str, str], state: JsonObject) -> dict[str, str]:
    env = dict(base_env)
    env.update(_string_mapping(state.get("hook_static_env")))
    env.update(_string_mapping(state.get("hook_scope_env")))

    prepend_parts = [
        *_string_sequence(state.get("hook_static_prepend_path")),
        *_string_sequence(state.get("hook_scope_prepend_path")),
    ]
    if prepend_parts:
        existing_path = env.get("PATH", "")
        env["PATH"] = (
            os.pathsep.join([*prepend_parts, existing_path]) if existing_path else os.pathsep.join(prepend_parts)
        )

    for prepend_state in (
        _string_sequence_mapping(state.get("hook_static_prepend_env")),
        _string_sequence_mapping(state.get("hook_scope_prepend_env")),
    ):
        for key, values in prepend_state.items():
            if not values:
                continue
            existing_value = env.get(key, "")
            env[key] = os.pathsep.join([*values, existing_value]) if existing_value else os.pathsep.join(values)
    return env


def _resolve_exec_cwd(mount_path: str, subpath: str | None) -> str:
    mount_root = Path(mount_path).resolve()
    if subpath is None:
        return str(mount_root)
    candidate = (mount_root / subpath).resolve()
    if os.path.commonpath([str(mount_root), str(candidate)]) != str(mount_root):
        raise ManagedExecUsageError(f"--cwd '{subpath}' escapes overlay mount.")
    if not candidate.is_dir():
        raise ManagedExecUsageError(f"--cwd '{subpath}' does not exist under overlay mount.")
    return str(candidate)


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _string_sequence_mapping(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        result[str(key)] = _string_sequence(item)
    return result


def _require_string_list(params: JsonObject, key: str) -> list[str]:
    value = params.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"Expected string-list parameter '{key}'.")


def _parse_command_outcome(params: JsonObject) -> _ParsedCommandOutcome:
    return _ParsedCommandOutcome(
        operation_id=_require_str(params, "operation_id"),
        outcome=_require_str(params, "outcome"),
        ended_at=_optional_float(params, "ended_at", default=time.time()),
        exit_code=_optional_int(params, "exit_code", default=None),
        signal_value=_optional_int(params, "signal", default=None),
        launch_error=_optional_str(params, "launch_error"),
        abandoned_reason=_optional_str(params, "abandoned_reason"),
        transport_status=_optional_str(params, "transport_status"),
        daemon_instance_id=_optional_str(params, "daemon_instance_id", default=None),
    )


def _combine_recording_errors(*errors: str | None) -> str | None:
    present = [error for error in errors if error]
    if not present:
        return None
    return "; ".join(present)


def _is_shell_command(command: dict[str, object] | None) -> bool:
    return command is not None and command.get("transport") == "shell"


def _shell_lease_completion_metadata(
    start_metadata: dict[str, object],
    *,
    parsed: _ParsedCommandOutcome,
) -> dict[str, object]:
    validate_exec_outcome(
        parsed.outcome,
        exit_code=parsed.exit_code,
        signal=parsed.signal_value,
        launch_error=parsed.launch_error,
        abandoned_reason=parsed.abandoned_reason,
    )
    start_shell = start_metadata.get("shell")
    shell: dict[str, object] = {
        "status": parsed.outcome,
        "ended_at": parsed.ended_at,
    }
    if isinstance(start_shell, dict):
        for field in (
            "scope",
            "capture_requested",
            "client_pid",
            "shell_pid",
            "daemon_instance_id",
            "started_at",
        ):
            value = start_shell.get(field)
            if value is not None:
                shell[field] = value
        started_at = start_shell.get("started_at")
        if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
            shell["duration_seconds"] = max(0.0, parsed.ended_at - float(started_at))
    if parsed.exit_code is not None:
        shell["exit_code"] = parsed.exit_code
    if parsed.signal_value is not None:
        shell["signal"] = parsed.signal_value
    if parsed.abandoned_reason:
        shell["abandoned_reason"] = parsed.abandoned_reason
    shell["capture_status"] = "complete" if parsed.outcome == "success" else "incomplete"
    if parsed.outcome != "success":
        shell["capture_stream_status"] = "incomplete"
        shell["capture_incomplete_reason"] = "shell_exited_nonzero"
    return shell


def _parse_optional_scope_name_for_method(
    method: str,
    raw: str | None,
    *,
    allow_ground: bool = True,
) -> str | None:
    try:
        return parse_optional_scope_name(raw, allow_ground=allow_ground)
    except ParseError as exc:
        raise AppCommandBlocked(
            command=method,
            blockers=(AppBlocker(kind="invalid_input", subject="" if raw is None else raw, detail=str(exc)),),
        ) from exc


def _optional_str(params: JsonObject, key: str, default: str | None = None) -> str | None:
    value = params.get(key, default)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"Expected string parameter '{key}'.")


def _optional_bool(params: JsonObject, key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected boolean parameter '{key}'.")


@overload
def _optional_int(params: JsonObject, key: str, default: int) -> int: ...


@overload
def _optional_int(params: JsonObject, key: str, default: None) -> int | None: ...


def _optional_int(params: JsonObject, key: str, default: int | None) -> int | None:
    value = params.get(key, default)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Expected integer parameter '{key}'.")


def _optional_float(params: JsonObject, key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Expected number parameter '{key}'.")


def _find_open_operation(store: Store, operation_id: str) -> Any:
    for operation in store.list_open_operations():
        if operation.durable_id == operation_id and operation.kind == "vcs_core.session_exec":
            return operation
    raise ValueError(f"No open session exec envelope matches operation id {operation_id!r}.")


def _find_open_shell_lease_operation(store: Store, operation_id: str) -> Any:
    for operation in store.list_open_operations():
        if operation.durable_id == operation_id and operation.kind == "vcs_core.session_shell":
            return operation
    raise ValueError(f"No open shell capture lease matches operation id {operation_id!r}.")


def _find_open_scope_writer(
    store: Store,
    *,
    scope_ref: str,
    allowed_shell_lease_id: str | None = None,
) -> _ScopeWriterConflict | None:
    for operation in store.list_open_operations():
        if operation.scope_ref != scope_ref:
            continue
        if operation.kind == "vcs_core.session_shell":
            if operation.durable_id == allowed_shell_lease_id:
                continue
            return _ScopeWriterConflict(operation_id=operation.durable_id, label="session shell")
        if operation.kind == "vcs_core.session_exec":
            return _ScopeWriterConflict(operation_id=operation.durable_id, label="session exec")
    return None
