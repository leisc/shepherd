# under-test: vcs_core._managed_exec_protocol
from __future__ import annotations

import pytest
from vcs_core._managed_exec_protocol import (
    ManagedExecExitFrame,
    ManagedExecMessageFrame,
    ManagedExecStartedFrame,
    ManagedExecStreamFrame,
    decode_managed_exec_request,
    decode_managed_exec_response_frame,
    encode_managed_exec_frame,
    encode_managed_exec_request,
    error_frame,
    exit_frame,
    managed_exec_request_from_params,
    recording_error_frame,
    sanitized_managed_exec_env,
    started_frame,
    stream_frame,
)


def test_managed_exec_response_frames_round_trip() -> None:
    frames = [
        started_frame(operation_id="cmd-test", pid=123, pgid=456),
        stream_frame("stdout", b"hello\n"),
        stream_frame("stderr", b"warning\n"),
        error_frame("failed"),
        recording_error_frame("recording failed"),
        exit_frame(7),
    ]

    decoded = [decode_managed_exec_response_frame(encode_managed_exec_frame(frame)) for frame in frames]

    assert decoded == frames
    assert isinstance(decoded[0], ManagedExecStartedFrame)
    assert isinstance(decoded[1], ManagedExecStreamFrame)
    assert isinstance(decoded[3], ManagedExecMessageFrame)
    assert isinstance(decoded[5], ManagedExecExitFrame)


def test_managed_exec_response_frame_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown exec frame"):
        decode_managed_exec_response_frame(b'{"type":"mystery"}\n')


def test_managed_exec_response_frame_rejects_invalid_stream_payload() -> None:
    with pytest.raises(ValueError, match="base64-encoded"):
        decode_managed_exec_response_frame(b'{"type":"stdout","data_b64":"%%%"}\n')


def test_managed_exec_response_frame_rejects_bool_exit_code() -> None:
    with pytest.raises(ValueError, match="integer frame field 'exit_code'"):
        decode_managed_exec_response_frame(b'{"type":"exit","exit_code":true}\n')


def test_managed_exec_request_round_trip() -> None:
    request = decode_managed_exec_request(
        encode_managed_exec_request(
            {
                "argv": ["python", "-c", "print(1)"],
                "scope": "task",
                "create": False,
                "parent": None,
                "cwd_subpath": None,
                "capture_requested": True,
                "capture_debug_log": "/tmp/capture.log",
                "env": {"PATH": "/bin", "USER_VAR": "value"},
                "started_at": 10.5,
                "client_pid": 123,
            }
        )
    )

    assert request.argv == ["python", "-c", "print(1)"]
    assert request.scope_name == "task"
    assert request.capture_requested is True
    assert request.env["PATH"] == "/bin"
    assert request.env["VCS_CORE_SESSION"] == "1"
    assert request.env["VCS_CORE_FS_CAPTURE_DEBUG_LOG"] == "/tmp/capture.log"


def test_managed_exec_request_rejects_non_string_env() -> None:
    with pytest.raises(TypeError, match="string keys and values"):
        managed_exec_request_from_params({"argv": ["true"], "env": {"OK": 1}})


def test_managed_exec_request_rejects_daemon_owned_env() -> None:
    for key in (
        "VCS_CORE_CAPTURE_ACTIVE",
        "VCS_CORE_FS_CAPTURE_SUPPRESS",
        "VCS_CORE_HOOK_SUPPRESS",
        "VCS_CORE_SCOPE",
    ):
        with pytest.raises(ValueError, match="daemon-owned key"):
            managed_exec_request_from_params({"argv": ["true"], "env": {key: "stale"}})


def test_sanitized_managed_exec_env_drops_daemon_owned_keys() -> None:
    assert sanitized_managed_exec_env(
        {
            "VCS_CORE_CAPTURE_ACTIVE": "1",
            "VCS_CORE_FS_CAPTURE_SUPPRESS": "1",
            "VCS_CORE_HOOK_SUPPRESS": "1",
            "VCS_CORE_SCOPE": "stale",
            "USER_VAR": "value",
        }
    ) == {"USER_VAR": "value"}
