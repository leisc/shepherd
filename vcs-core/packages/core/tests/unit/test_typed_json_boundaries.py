from __future__ import annotations

import pytest
from vcs_core._cli_session_runtime import _float_value, _int_value
from vcs_core._lifecycle_run import LifecycleRun
from vcs_core._session_dispatch import _optional_int
from vcs_core._typed_json import decode_typed_json
from vcs_core.types import encode_typed_json


def _valid_lifecycle_payload() -> dict[str, object]:
    scope = {
        "name": "task",
        "ref": "refs/vcscore/scopes/task",
        "instance_id": "scope-instance",
        "creation_oid": "abc123",
        "world_id": "world-task",
        "isolated": True,
    }
    parent = {
        "name": "ground",
        "ref": "refs/vcscore/ground",
        "instance_id": "ground-instance",
        "creation_oid": "def456",
        "world_id": "world-ground",
        "isolated": False,
    }
    return {
        "session_id": "session",
        "operation": "merge",
        "phase": "prepare",
        "scope": scope,
        "parent": parent,
        "prepared_effect_counts": [("filesystem", 1)],
        "timestamp": 123.0,
    }


def test_session_dispatch_integer_params_reject_bool() -> None:
    with pytest.raises(ValueError, match="Expected integer parameter 'max_count'"):
        _optional_int({"max_count": True}, "max_count", default=20)


def test_cli_session_numeric_decoders_reject_bool() -> None:
    assert _int_value(True, default=123) == 123
    assert _float_value(False, default=1.5) == 1.5


def test_lifecycle_run_rejects_bool_effect_count() -> None:
    payload = _valid_lifecycle_payload()
    payload["prepared_effect_counts"] = [("filesystem", True)]

    with pytest.raises(TypeError, match="integer counts"):
        LifecycleRun.from_dict(payload)


def test_lifecycle_run_rejects_non_bool_isolated_flag() -> None:
    payload = _valid_lifecycle_payload()
    scope = dict(payload["scope"])  # type: ignore[arg-type]
    scope["isolated"] = "false"
    payload["scope"] = scope

    with pytest.raises(TypeError, match="isolated flag"):
        LifecycleRun.from_dict(payload)


def test_lifecycle_run_rejects_bool_timestamp() -> None:
    payload = _valid_lifecycle_payload()
    payload["timestamp"] = True

    with pytest.raises(TypeError, match="timestamp"):
        LifecycleRun.from_dict(payload)


def test_typed_json_round_trips_bytes_payloads() -> None:
    encoded = encode_typed_json({"content": b"hello", "items": (b"a", "b")})

    assert encoded == {
        "content": {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="},
        "items": [{"__type__": "bytes", "encoding": "base64", "data": "YQ=="}, "b"],
    }
    assert decode_typed_json(encoded) == {"content": b"hello", "items": [b"a", "b"]}


def test_typed_json_rejects_non_string_object_keys() -> None:
    with pytest.raises(TypeError, match="object keys must be strings"):
        encode_typed_json({1: "bad"})


def test_typed_json_rejects_malformed_bytes_payloads() -> None:
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_typed_json({"__type__": "bytes", "encoding": "base64", "data": "not base64!"})
