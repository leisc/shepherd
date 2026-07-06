# under-test: vcs_core._cli_delegation
from __future__ import annotations

import pytest
from vcs_core import _cli_delegation


def test_with_session_result_returns_fallback_when_no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cli_delegation._cli_ipc, "try_session_ipc", lambda method, params: None)

    result = _cli_delegation.with_session_result(
        "operations",
        {"mode": "visible"},
        on_result=lambda payload: ("session", payload),
        on_fallback=lambda: ("fallback", None),
    )

    assert result == ("fallback", None)


def test_with_session_result_passes_normalized_result_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _cli_delegation._cli_ipc,
        "try_session_ipc",
        lambda method, params: {"ok": True, "result": {"requested_mode": "visible"}},
    )

    result = _cli_delegation.with_session_result(
        "operations",
        {"mode": "visible"},
        on_result=lambda payload: payload["requested_mode"],
        on_fallback=lambda: "fallback",
    )

    assert result == "visible"


def test_with_session_result_exits_cleanly_when_session_daemon_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(method: str, params: dict[str, object] | None) -> None:
        del method, params
        raise _cli_delegation._cli_ipc.SessionIpcError("daemon unavailable")

    monkeypatch.setattr(_cli_delegation._cli_ipc, "try_session_ipc", _raise)

    with pytest.raises(SystemExit) as exc_info:
        _cli_delegation.with_session_result(
            "operations",
            {"mode": "visible"},
            on_result=lambda payload: payload,
            on_fallback=lambda: None,
        )

    assert exc_info.value.code == 1
    assert "Error: daemon unavailable" in capsys.readouterr().out


def test_with_session_result_exits_on_non_ok_session_response(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        _cli_delegation._cli_ipc,
        "try_session_ipc",
        lambda method, params: {"ok": False, "error": "bad state"},
    )

    with pytest.raises(SystemExit) as exc_info:
        _cli_delegation.with_session_result(
            "operations",
            {"mode": "visible"},
            on_result=lambda payload: payload,
            on_fallback=lambda: None,
        )

    assert exc_info.value.code == 1
    assert "Error: bad state" in capsys.readouterr().out


def test_with_session_result_exits_on_non_mapping_result_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        _cli_delegation._cli_ipc,
        "try_session_ipc",
        lambda method, params: {"ok": True, "result": ["not", "a", "mapping"]},
    )

    with pytest.raises(SystemExit) as exc_info:
        _cli_delegation.with_session_result(
            "operations",
            {"mode": "visible"},
            on_result=lambda payload: payload,
            on_fallback=lambda: None,
        )

    assert exc_info.value.code == 1
    assert "Error: session response payload must be an object." in capsys.readouterr().out
