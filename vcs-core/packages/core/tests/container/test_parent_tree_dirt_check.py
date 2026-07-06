# under-test: vcs_core._fuse_overlay
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from vcs_core import ParentWorkingTreeDivergedError
from vcs_core._fuse_overlay import FuseOverlayBackend

from ..support.builders import make_marker_filesystem_vcscore


def _mocked_fuse_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FuseOverlayBackend:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = tmp_path / "overlay-state"
    state_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FuseOverlayBackend, "_ensure_supported", lambda self: None)

    def fake_mount(self: FuseOverlayBackend, *, lowerdir: str, upperdir: Path, workdir: Path, merged: Path) -> None:
        del lowerdir, workdir
        if merged.exists():
            shutil.rmtree(merged)
        shutil.copytree(upperdir, merged, dirs_exist_ok=True)

    def fake_unmount(self: FuseOverlayBackend, merged: Path) -> None:
        if merged.is_symlink():
            merged.unlink()
            merged.mkdir(parents=True, exist_ok=True)
            return
        if merged.exists():
            shutil.rmtree(merged)
        merged.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FuseOverlayBackend, "_mount_overlay", fake_mount)
    monkeypatch.setattr(FuseOverlayBackend, "_unmount", fake_unmount)
    monkeypatch.setattr(FuseOverlayBackend, "_is_mounted", lambda self, path: path.is_dir())
    return FuseOverlayBackend(workspace=workspace, state_root=state_root)


@pytest.mark.container
def test_parent_tree_dirt_check_refuses_mocked_fuse_parent_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = _mocked_fuse_backend(tmp_path, monkeypatch)
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "fuse-parent-dirt-child", hints={"isolated": True})
        (backend.working_path("ground") / "unrecorded.txt").write_text("parent dirt\n", encoding="utf-8")

        with pytest.raises(ParentWorkingTreeDivergedError, match=r"unrecorded\.txt"):
            mg.merge(child, mg.ground)
    finally:
        mg.deactivate(warn_on_open_scopes=False)
        shutil.rmtree(tmp_path / "overlay-state", ignore_errors=True)
