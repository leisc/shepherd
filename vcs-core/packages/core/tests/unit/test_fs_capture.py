# under-test: vcs_core._fs_capture
"""Tests for filesystem preload shim build helpers."""

from __future__ import annotations

import subprocess

import pytest
from vcs_core._fs_capture import (
    ensure_fs_capture_shim,
    normalize_fs_capture_op,
    normalize_fs_capture_path,
    shim_source_path,
)


def test_ensure_fs_capture_shim_rejects_non_linux(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("sys.platform", "darwin")

    with pytest.raises(RuntimeError, match="only supported on Linux"):
        ensure_fs_capture_shim(tmp_path)


def test_ensure_fs_capture_shim_compiles_when_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    source = tmp_path / "fs_capture_shim.c"
    source.write_text("/* fake source */")
    calls: list[list[str]] = []

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("vcs_core._fs_capture.shim_source_path", lambda: source)

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        calls.append(cmd)
        output = tmp_path / ".vcscore" / "native" / "fs_capture_shim.so"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("shim")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = ensure_fs_capture_shim(tmp_path / ".vcscore")

    assert result.endswith("fs_capture_shim.so")
    assert calls


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("write_open", "write_open"),
        ("write_observed", "write_observed"),
        ("write_close", "write_close"),
        ("metadata_change", "metadata_change"),
        ("unlink", "unlink"),
        ("rename", None),
        (None, None),
        ([], None),
        ({}, None),
    ],
)
def test_normalize_fs_capture_op_treats_invalid_values_as_no_effect(value: object, expected: str | None) -> None:
    assert normalize_fs_capture_op(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("note.txt", "note.txt"),
        ("subdir/./note.txt", "subdir/note.txt"),
        ("", None),
        (".", None),
        ("../outside.txt", None),
        ("subdir/../../outside.txt", None),
        ("/tmp/outside.txt", None),
        (".vcscore/config.toml", None),
        ("bad\0path", None),
        (None, None),
        ([], None),
    ],
)
def test_normalize_fs_capture_path_treats_invalid_values_as_no_effect(value: object, expected: str | None) -> None:
    assert normalize_fs_capture_path(value) == expected


def test_fs_capture_shim_intercepts_chmod_family() -> None:
    source = shim_source_path().read_text()

    assert "int chmod(const char *pathname, mode_t mode)" in source
    assert "int fchmod(int fd, mode_t mode)" in source
    assert "int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags)" in source
    assert 'emit_path_event("metadata_change", rel_path)' in source


def test_fs_capture_shim_classifies_paths_canonically_without_blocking_syscalls() -> None:
    source = shim_source_path().read_text()

    assert "realpath(workspace, resolved_workspace)" in source
    assert "resolve_existing_rel_path" in source
    assert "resolve_unlink_rel_path" in source
    assert "realpath(candidate, resolved)" in source
    assert "realpath(parent, resolved_parent)" in source
    assert "fd >= 0 && resolve_fd_rel_path(fd, rel_path, sizeof(rel_path))" in source
    assert "should_emit = resolve_unlink_rel_path(AT_FDCWD, pathname, rel_path, sizeof(rel_path));" in source
    assert "result = real_unlink_fn(pathname);" in source
    assert "result = real_chmod_fn(pathname, mode);" in source


def test_fs_capture_shim_forwards_fcntl_varargs_by_command_shape() -> None:
    source = shim_source_path().read_text()

    assert "typedef enum" in source
    assert "FCNTL_ARG_INT" in source
    assert "FCNTL_ARG_PTR" in source
    assert "case F_DUPFD:" in source
    assert "case F_SETFD:" in source
    assert "case F_GETLK:" in source
    assert "case F_SETLK:" in source
    assert "F_ADD_SEALS" in source
    assert "F_SETOWN_EX" in source
    assert "F_SET_RW_HINT" in source
    assert "arg_kind = fcntl_arg_kind(cmd);" in source
    assert "result = real_fcntl_fn(fd, cmd, int_arg);" in source
    assert "result = real_fcntl_fn(fd, cmd, ptr_arg);" in source


def test_fs_capture_shim_has_shell_finish_trigger() -> None:
    source = shim_source_path().read_text()

    assert "VCS_CORE_SHELL_FINISH_ACTIVE" in source
    assert "VCS_CORE_SHELL_FINISH_PATH" in source
    assert "emit_shell_command_finish_event" in source
    assert '\\"op\\":\\"shell_command_finish\\"' in source
    assert "maybe_emit_shell_finish_for_fd(fd)" in source
