"""Shared CLI helpers for daemon-backed session IPC."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from vcs_core._ipc import JsonObject, SessionErrorResponse, SessionInfo, SessionOkResponse, SessionResponse


class SessionIpcError(VcsCoreError, RuntimeError):
    """Raised when a daemon-backed session exists but cannot be reached."""


def live_session_info(repo_path: str | None = None) -> SessionInfo | None:
    from vcs_core._ipc import is_session_alive, read_session_info

    resolved_repo_path = repo_path or os.path.join(os.path.abspath("."), ".vcscore")  # noqa: PTH118
    info = read_session_info(resolved_repo_path)
    if info is None or not is_session_alive(resolved_repo_path):
        return None
    return info


def send_session_request(info: SessionInfo, method: str, params: JsonObject | None = None) -> SessionResponse:
    return send_session_request_to_socket(info.socket_path, method, params)


def send_session_request_to_socket(socket_path: str, method: str, params: JsonObject | None = None) -> SessionResponse:
    from vcs_core._ipc import send_request

    try:
        return send_request(socket_path, method, params or {})
    except TypeError as exc:
        msg = f"session request payload is not JSON-serializable: {exc}"
        raise SessionIpcError(msg) from exc
    except (ConnectionError, OSError) as exc:
        msg = (
            "session daemon is recorded as running but unreachable. "
            "Stop it with `vcs-core session stop` and restart with `vcs-core session start`."
        )
        raise SessionIpcError(msg) from exc


def try_session_ipc(method: str, params: JsonObject | None = None) -> SessionResponse | None:
    """If a session daemon is running, send an IPC request and return the response."""
    info = live_session_info()
    if info is None:
        return None
    return send_session_request(info, method, params)


def response_ok(response: SessionResponse | None) -> bool:
    return response is not None and response["ok"]


def response_error(response: SessionResponse | None) -> str:
    if response is None:
        return "unknown error"
    if response["ok"]:
        return "unknown error"
    return response["error"]


def response_result(response: SessionResponse | None) -> JsonObject:
    if response is None:
        return {}
    if response["ok"]:
        result = response["result"]
        if isinstance(result, dict):
            return result
        raise ValueError("session response payload must be an object.")
    return {}


def ok_response(result: JsonObject) -> SessionOkResponse:
    return {"ok": True, "result": result}


def error_response(error: str) -> SessionErrorResponse:
    return {"ok": False, "error": error}
