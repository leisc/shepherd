# under-test: vcs_core._ipc
"""Tests for IPC protocol types (no platform gating -- pure Python)."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import pytest
from vcs_core._ipc import (
    SESSION_INFO_FILE,
    SessionInfo,
    is_session_alive,
    read_session_info,
    remove_session_info,
    send_request,
    write_session_info,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- SessionInfo persistence ---


def test_write_read_roundtrip(tmp_path: Path) -> None:
    repo = str(tmp_path)
    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/test.sock",
        mount_path="/tmp/mount",
        workspace="/tmp/workspace",
        started_at=1000.0,
        daemon_instance_id="daemon-1",
    )
    write_session_info(repo, info)
    loaded = read_session_info(repo)
    assert loaded is not None
    assert loaded.pid == 12345
    assert loaded.socket_path == "/tmp/test.sock"
    assert loaded.mount_path == "/tmp/mount"
    assert loaded.workspace == "/tmp/workspace"
    assert loaded.started_at == 1000.0
    assert loaded.daemon_instance_id == "daemon-1"


def test_read_session_info_accepts_legacy_file_without_daemon_instance_id(tmp_path: Path) -> None:
    (tmp_path / SESSION_INFO_FILE).write_text(
        json.dumps(
            {
                "pid": 12345,
                "socket_path": "/tmp/test.sock",
                "mount_path": "/tmp/mount",
                "workspace": "/tmp/workspace",
                "started_at": 1000.0,
            }
        )
    )

    loaded = read_session_info(str(tmp_path))

    assert loaded is not None
    assert loaded.daemon_instance_id is None


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_session_info(str(tmp_path)) is None


def test_read_malformed_returns_none(tmp_path: Path) -> None:
    (tmp_path / SESSION_INFO_FILE).write_text("not json")
    assert read_session_info(str(tmp_path)) is None


def test_remove_session_info(tmp_path: Path) -> None:
    info = SessionInfo(pid=1, socket_path="", mount_path="", workspace="", started_at=0)
    write_session_info(str(tmp_path), info)
    assert read_session_info(str(tmp_path)) is not None
    remove_session_info(str(tmp_path))
    assert read_session_info(str(tmp_path)) is None


def test_remove_missing_is_noop(tmp_path: Path) -> None:
    remove_session_info(str(tmp_path))  # should not raise


# --- PID liveness ---


def test_is_session_alive_no_file(tmp_path: Path) -> None:
    assert is_session_alive(str(tmp_path)) is False


def test_is_session_alive_dead_pid(tmp_path: Path) -> None:
    info = SessionInfo(
        pid=999999999,  # almost certainly not a real PID
        socket_path="/tmp/nonexistent.sock",
        mount_path="/tmp/mount",
        workspace="/tmp/workspace",
        started_at=time.time(),
    )
    write_session_info(str(tmp_path), info)
    assert is_session_alive(str(tmp_path)) is False


def test_is_session_alive_current_pid(tmp_path: Path) -> None:
    info = SessionInfo(
        pid=os.getpid(),
        socket_path="/tmp/test.sock",
        mount_path="/tmp/mount",
        workspace="/tmp/workspace",
        started_at=time.time(),
    )
    write_session_info(str(tmp_path), info)
    assert is_session_alive(str(tmp_path)) is True


# --- IPC send/receive ---


class _FakeSocket:
    def __init__(
        self,
        *,
        recv_chunks: list[bytes] | None = None,
        connect_error: OSError | None = None,
        send_error: OSError | None = None,
        recv_error: OSError | None = None,
    ) -> None:
        self.recv_chunks = list(recv_chunks or [])
        self.connect_error = connect_error
        self.send_error = send_error
        self.recv_error = recv_error
        self.connected_to: str | None = None
        self.sent: bytes = b""
        self.shutdown_arg: int | None = None
        self.closed = False

    def connect(self, address: str) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected_to = address

    def sendall(self, data: bytes) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent += data

    def shutdown(self, how: int) -> None:
        self.shutdown_arg = how

    def recv(self, bufsize: int) -> bytes:
        del bufsize
        if self.recv_error is not None:
            raise self.recv_error
        if not self.recv_chunks:
            return b""
        return self.recv_chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def test_send_receive_roundtrip() -> None:
    sock_path = "/tmp/vcs-core-test.sock"
    expected = {"ok": True, "result": {"key": "value"}}
    fake = _FakeSocket(recv_chunks=[json.dumps(expected).encode() + b"\n"])

    result = send_request(sock_path, "test_method", {"param": 1}, socket_factory=lambda: fake)
    assert result == expected
    assert fake.connected_to == sock_path
    assert fake.sent == b'{"method": "test_method", "params": {"param": 1}}\n'
    assert fake.closed is True


def test_send_request_reads_chunked_response() -> None:
    sock_path = "/tmp/vcs-core-test.sock"
    fake = _FakeSocket(recv_chunks=[b'{"ok": true, ', b'"result": {"key": ', b'"value"}}\n'])

    result = send_request(sock_path, "test_method", socket_factory=lambda: fake)

    assert result == {"ok": True, "result": {"key": "value"}}
    assert fake.closed is True


def test_unknown_method_error_response() -> None:
    sock_path = "/tmp/vcs-core-test.sock"
    error_resp = {"ok": False, "error": "Unknown method: 'bad'"}
    fake = _FakeSocket(recv_chunks=[json.dumps(error_resp).encode() + b"\n"])

    result = send_request(sock_path, "bad", socket_factory=lambda: fake)
    assert result["ok"] is False
    assert "Unknown method" in result["error"]


def test_send_request_raises_connection_error_on_empty_response() -> None:
    fake = _FakeSocket(recv_chunks=[])

    with pytest.raises(ConnectionError, match="Empty response"):
        send_request("/tmp/vcs-core-test.sock", "test", socket_factory=lambda: fake)

    assert fake.closed is True


def test_send_request_raises_json_decode_error_on_malformed_response() -> None:
    fake = _FakeSocket(recv_chunks=[b"not-json\n"])

    with pytest.raises(json.JSONDecodeError):
        send_request("/tmp/vcs-core-test.sock", "test", socket_factory=lambda: fake)

    assert fake.closed is True


def test_send_request_wraps_connect_failure_as_connection_error() -> None:
    fake = _FakeSocket(connect_error=FileNotFoundError("missing socket"))

    with pytest.raises(ConnectionError, match="Could not reach session daemon"):
        send_request("/tmp/nonexistent-vcscore-test.sock", "test", socket_factory=lambda: fake)

    assert fake.closed is True


def test_send_request_wraps_send_failure_as_connection_error() -> None:
    fake = _FakeSocket(send_error=BrokenPipeError("broken pipe"))

    with pytest.raises(ConnectionError, match="Could not reach session daemon"):
        send_request("/tmp/vcs-core-test.sock", "test", socket_factory=lambda: fake)

    assert fake.closed is True


def test_send_request_wraps_recv_failure_as_connection_error() -> None:
    fake = _FakeSocket(recv_error=ConnectionResetError("reset"))

    with pytest.raises(ConnectionError, match="Could not reach session daemon"):
        send_request("/tmp/vcs-core-test.sock", "test", socket_factory=lambda: fake)

    assert fake.closed is True


def test_connection_error_on_missing_socket() -> None:
    with pytest.raises(ConnectionError, match="Could not reach session daemon"):
        send_request(
            "/tmp/nonexistent-vcscore-test.sock",
            "test",
            socket_factory=lambda: _FakeSocket(connect_error=FileNotFoundError("missing socket")),
        )
