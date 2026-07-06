"""Tests for environment capability probes."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..support.capabilities import probe_local_bind_capability


class _FakeSocket:
    def __init__(self) -> None:
        self.bound_path: str | None = None
        self.closed = False

    def bind(self, path: str) -> None:
        self.bound_path = path

    def close(self) -> None:
        self.closed = True


def test_probe_local_bind_capability_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_socket = _FakeSocket()
    monkeypatch.setattr("tests.support.capabilities.socket.socket", lambda *args, **kwargs: fake_socket)

    result = probe_local_bind_capability(tmp_path)

    assert result.available is True
    assert result.reason is None
    assert fake_socket.bound_path is not None
    assert "/.vcscore/" in fake_socket.bound_path
    assert fake_socket.bound_path.endswith("/s.sock")
    assert fake_socket.closed is True


def test_probe_local_bind_capability_reports_permission_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DeniedSocket:
        def bind(self, path: str) -> None:
            del path
            raise PermissionError(1, "Operation not permitted")

        def close(self) -> None:
            return None

    monkeypatch.setattr("tests.support.capabilities.socket.socket", lambda *args, **kwargs: _DeniedSocket())

    result = probe_local_bind_capability(tmp_path)

    assert result.available is False
    assert result.reason is not None
    assert "workspace-local listener bind" in result.reason
    assert "not permitted" in result.reason


def test_probe_local_bind_capability_reports_generic_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenSocket:
        def bind(self, path: str) -> None:
            del path
            raise OSError("socket path too long")

        def close(self) -> None:
            return None

    monkeypatch.setattr("tests.support.capabilities.socket.socket", lambda *args, **kwargs: _BrokenSocket())

    result = probe_local_bind_capability(tmp_path)

    assert result.available is False
    assert result.reason is not None
    assert "workspace-local listener bind probe failed" in result.reason
