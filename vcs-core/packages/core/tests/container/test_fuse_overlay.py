# under-test: vcs_core._fuse_overlay
"""Tests for the fuse-overlayfs backend."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

import pytest
from vcs_core import UnsupportedOverlayEntryError
from vcs_core._fuse_overlay import FuseOverlayBackend


def _make_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FuseOverlayBackend:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = tmp_path / "overlay-state"
    state_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FuseOverlayBackend, "_ensure_supported", lambda self: None)

    def fake_mount(self: FuseOverlayBackend, *, lowerdir: str, upperdir: Path, workdir: Path, merged: Path) -> None:
        del lowerdir, workdir
        if merged.exists():
            merged.rmdir()
        merged.symlink_to(upperdir, target_is_directory=True)

    def fake_unmount(self: FuseOverlayBackend, merged: Path) -> None:
        if merged.is_symlink():
            merged.unlink()
            merged.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FuseOverlayBackend, "_mount_overlay", fake_mount)
    monkeypatch.setattr(FuseOverlayBackend, "_unmount", fake_unmount)
    monkeypatch.setattr(FuseOverlayBackend, "_is_mounted", lambda self, path: path.is_symlink())

    return FuseOverlayBackend(workspace=workspace, state_root=state_root)


def _initialize_ground(backend: FuseOverlayBackend) -> None:
    backend.create_layer("ground", parent_scope_id=None)


def test_create_layer_write_and_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        backend.write_file("child", "src/example.py", b"print('hi')\n")

        assert backend.read_file("child", "src/example.py") == b"print('hi')\n"
        assert backend.diff_layer("child") == [("src/example.py", b"print('hi')\n", 0o100644)]
        assert backend.working_path("child").name == "merged"
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_delete_and_commit_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    _initialize_ground(backend)

    try:
        backend.write_file("ground", "remove.txt", b"before")
        backend.create_layer("child", parent_scope_id="ground")
        whiteout = backend._layer_paths("child").upper / ".wh.remove.txt"
        whiteout.write_text("")

        assert backend.diff_layer("child") == [("remove.txt", None, 0)]

        backend.commit_layer("child", into_scope_id="ground")

        with pytest.raises(FileNotFoundError):
            backend.read_file("ground", "remove.txt")
        assert not backend.has_layer("child")
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_commit_layer_resets_executable_to_regular(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    _initialize_ground(backend)

    try:
        backend.write_file("ground", "bin/run", b"old", mode=0o100755)
        backend.create_layer("child", parent_scope_id="ground")
        backend.write_file("child", "bin/run", b"new", mode=0o100644)

        backend.commit_layer("child", into_scope_id="ground")

        committed = backend.working_path("ground") / "bin" / "run"
        assert committed.read_bytes() == b"new"
        assert stat.S_IMODE(committed.stat().st_mode) == 0o644
        assert backend.diff_layer("ground") == [("bin/run", b"new", 0o100644)]
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_diff_layer_ignores_opaque_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        upper = backend._layer_paths("child").upper
        (upper / "src").mkdir(parents=True, exist_ok=True)
        (upper / "src" / ".wh..opq").write_text("")
        (upper / "src" / "example.py").write_text("print('hi')\n")

        assert backend.diff_layer("child") == [("src/example.py", b"print('hi')\n", 0o100644)]
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_diff_layer_rejects_symlink_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        upper = backend._layer_paths("child").upper
        (upper / "escape-link").symlink_to("/tmp")

        with pytest.raises(UnsupportedOverlayEntryError, match=r"escape-link.*symlink"):
            backend.diff_layer("child")
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_push_layer_materializes_and_resets_ground(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    _initialize_ground(backend)

    try:
        backend.write_file("ground", "nested/output.txt", b"content")
        backend.push_layer()

        assert (workspace / "nested" / "output.txt").read_bytes() == b"content"
        assert backend.diff_layer("ground") == []
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_push_layer_resets_executable_to_regular(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    executable = workspace / "nested" / "output.txt"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"old")
    executable.chmod(0o755)
    _initialize_ground(backend)

    try:
        backend.write_file("ground", "nested/output.txt", b"content", mode=0o100644)
        backend.push_layer()

        assert executable.read_bytes() == b"content"
        assert stat.S_IMODE(executable.stat().st_mode) == 0o644
        assert backend.diff_layer("ground") == []
    finally:
        backend.deactivate()
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)
