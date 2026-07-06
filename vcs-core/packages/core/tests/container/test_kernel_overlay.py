# under-test: vcs_core._kernel_overlay
"""Tests for the kernel overlay backend."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from vcs_core._kernel_overlay import KernelOverlayBackend

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="KernelOverlayBackend tests require Linux and root privileges.",
)


def _make_backend(tmp_path) -> KernelOverlayBackend:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    configured = os.environ.get("VCS_CORE_KERNEL_OVERLAY_STATE_ROOT")
    if configured:
        state_root = Path(configured) / f"vcs-core-{uuid.uuid4().hex[:8]}"
    else:
        state_root = tmp_path / "overlay-state"
    state_root.mkdir(parents=True, exist_ok=True)
    return KernelOverlayBackend(workspace=workspace, state_root=state_root)


def _initialize_ground(backend: KernelOverlayBackend) -> None:
    try:
        backend.create_layer("ground", parent_scope_id=None)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"Kernel overlayfs not available in this environment: {exc}")


def test_create_layer_write_and_diff(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        backend.write_file("child", "src/example.py", b"print('hi')\n")
        assert backend.read_file("child", "src/example.py") == b"print('hi')\n"
        assert backend.diff_layer("child") == [("src/example.py", b"print('hi')\n", 0o100644)]
    finally:
        backend.deactivate()
        shutil.rmtree(backend._state_root, ignore_errors=True)


def test_delete_and_commit_layer(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "remove.txt").write_text("before")
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        backend.delete_file("child", "remove.txt")
        assert backend.diff_layer("child") == [("remove.txt", None, 0)]

        backend.commit_layer("child", into_scope_id="ground")

        with pytest.raises(FileNotFoundError):
            backend.read_file("ground", "remove.txt")
        assert not backend.has_layer("child")
    finally:
        backend.deactivate()
        shutil.rmtree(backend._state_root, ignore_errors=True)


def test_commit_layer_resets_executable_to_regular(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "bin").mkdir(parents=True, exist_ok=True)
    (workspace / "bin" / "run").write_bytes(b"old")
    (workspace / "bin" / "run").chmod(0o755)
    _initialize_ground(backend)

    try:
        backend.create_layer("child", parent_scope_id="ground")
        backend.write_file("child", "bin/run", b"new", mode=0o100644)

        backend.commit_layer("child", into_scope_id="ground")

        committed = backend.working_path("ground") / "bin" / "run"
        assert committed.read_bytes() == b"new"
        assert stat.S_IMODE(committed.stat().st_mode) == 0o644
        assert backend.diff_layer("ground") == [("bin/run", b"new", 0o100644)]
    finally:
        backend.deactivate()
        shutil.rmtree(backend._state_root, ignore_errors=True)


def test_push_layer_materializes_and_resets_ground(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    workspace = tmp_path / "workspace"
    _initialize_ground(backend)

    try:
        backend.write_file("ground", "nested/output.txt", b"content")
        backend.push_layer()

        assert (workspace / "nested" / "output.txt").read_bytes() == b"content"
        assert backend.diff_layer("ground") == []
    finally:
        backend.deactivate()
        shutil.rmtree(backend._state_root, ignore_errors=True)
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)


def test_push_layer_resets_executable_to_regular(tmp_path) -> None:
    backend = _make_backend(tmp_path)
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
        shutil.rmtree(backend._state_root, ignore_errors=True)
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)
