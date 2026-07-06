# under-test: vcs_core._fuse_overlay
"""The read-only carrier mode — the syscall-EROFS enforcement tier.

Two halves (read-only-carrier-mode.md):

- **Cross-platform (mocked mount):** the in-process write-refusal — a
  ``read_only`` backend raises ``ReadOnlyCarrierError`` on write/delete, so a
  framework-internal write refuses symmetrically with the out-of-band one
  (no honor-system asymmetry), and ``diff_layer`` is empty (nothing landed).
- **Container (real fuse-overlayfs):** the EROFS proof — a real lowerdir-only
  mount denies an out-of-band ``write(2)`` with ``Read-only file system``.
  This is the tier the 2026-06-03 probe platform-proved; it now runs against
  vcs-core's own mount path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from vcs_core import ReadOnlyCarrierError
from vcs_core._fuse_overlay import FuseOverlayBackend

# --- cross-platform: the in-process write-refusal (mocked mount) -----------


def _read_only_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FuseOverlayBackend:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "seed.txt").write_text("seed")
    state_root = tmp_path / "overlay-state"
    state_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(FuseOverlayBackend, "_ensure_supported", lambda self: None)

    def fake_mount(self: FuseOverlayBackend, *, lowerdir: str, upperdir: Path, workdir: Path, merged: Path) -> None:
        # Mirror the real read-only mount: lowerdir-only, no upper given.
        del upperdir, workdir
        if merged.exists():
            merged.rmdir()
        merged.symlink_to(lowerdir.split(":", maxsplit=1)[0], target_is_directory=True)

    def fake_unmount(self: FuseOverlayBackend, merged: Path) -> None:
        if merged.is_symlink():
            merged.unlink()
            merged.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FuseOverlayBackend, "_mount_overlay", fake_mount)
    monkeypatch.setattr(FuseOverlayBackend, "_unmount", fake_unmount)
    monkeypatch.setattr(FuseOverlayBackend, "_is_mounted", lambda self, path: path.is_symlink())
    return FuseOverlayBackend(workspace=workspace, state_root=state_root, read_only=True)


def test_read_only_backend_refuses_in_process_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _read_only_backend(tmp_path, monkeypatch)
    backend.create_layer(backend.GROUND_SCOPE_ID, parent_scope_id=None)
    with pytest.raises(ReadOnlyCarrierError, match="read-only carrier"):
        backend.write_file(backend.GROUND_SCOPE_ID, "new.txt", b"nope")
    with pytest.raises(ReadOnlyCarrierError, match="read-only carrier"):
        backend.delete_file(backend.GROUND_SCOPE_ID, "seed.txt")


def test_read_only_backend_diff_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing can be written ⇒ nothing to capture (containment without reversibility)."""
    backend = _read_only_backend(tmp_path, monkeypatch)
    backend.create_layer(backend.GROUND_SCOPE_ID, parent_scope_id=None)
    assert backend.diff_layer(backend.GROUND_SCOPE_ID) == []


# --- container: the real EROFS proof ---------------------------------------

_FUSE_AVAILABLE = sys.platform == "linux" and shutil.which("fuse-overlayfs") is not None


@pytest.mark.container
@pytest.mark.skipif(not _FUSE_AVAILABLE, reason="needs Linux + fuse-overlayfs (the container gate)")
def test_real_read_only_mount_denies_out_of_band_write_with_erofs(tmp_path: Path) -> None:
    """A real lowerdir-only fuse-overlayfs mount denies an out-of-band write(2)
    with EROFS — the syscall-refusal tier, against vcs-core's own mount path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "seed.txt").write_text("seed")
    state_root = tmp_path / "state"
    state_root.mkdir()
    backend = FuseOverlayBackend(workspace=workspace, state_root=state_root, read_only=True)
    backend.create_layer(backend.GROUND_SCOPE_ID, parent_scope_id=None)
    try:
        merged = backend.working_path(backend.GROUND_SCOPE_ID)
        # The seed is readable through the read-only mount...
        assert (merged / "seed.txt").read_text() == "seed"
        # ...but an out-of-band write(2) (a subprocess, bypassing the backend)
        # is refused at the syscall with EROFS.
        proc = subprocess.run(
            ["sh", "-c", f"printf x > '{merged / 'oob.txt'}'"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert "read-only file system" in proc.stderr.lower(), proc.stderr
        assert not (merged / "oob.txt").exists()
    finally:
        backend.deactivate()
