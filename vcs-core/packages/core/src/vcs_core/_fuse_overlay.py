"""fuse-overlayfs backend for FilesystemSubstrate."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from vcs_core._errors import ReadOnlyCarrierError, UnsupportedOverlayEntryError
from vcs_core._overlay_entries import unsupported_overlay_entry_kind
from vcs_core._workspace_paths import normalize_workspace_relative_path
from vcs_core.types import FileState, normalize_git_filemode, posix_to_git_mode

_WHITEOUT_PREFIX = ".wh."


@dataclass(frozen=True)
class LayerPaths:
    """Filesystem paths that define a single overlay layer."""

    root: Path
    upper: Path
    work: Path
    merged: Path


class FuseOverlayBackend:
    """Linux fuse-overlayfs backend.

    Unlike the kernel backend, this backend does not require root, but it does
    require Linux, `/dev/fuse`, `fuse-overlayfs`, and `fusermount3`.
    """

    GROUND_SCOPE_ID = "ground"

    def __init__(
        self,
        workspace: Path,
        state_root: Path,
        *,
        base_lowerdir: Path | None = None,
        base_tree_oid: str | None = None,
        fuse_overlayfs_bin: str = "fuse-overlayfs",
        fusermount_bin: str = "fusermount3",
        read_only: bool = False,
    ) -> None:
        # read_only: mount lowerdir-only (no writable upper) — the EROFS
        # enforcement tier. Writes refuse symmetrically (in-process + syscall);
        # there is nothing to capture (read-only-carrier-mode.md).
        self._read_only = read_only
        self._workspace = workspace.resolve()
        self._base_lowerdir = (base_lowerdir or workspace).resolve()
        self._base_tree_oid = base_tree_oid
        self._state_root = state_root.resolve()
        self._ground_root = self._state_root / self.GROUND_SCOPE_ID
        self._scopes_root = self._state_root / "scopes"
        self._base_tree_oid_path = self._state_root / "base-tree-oid"
        self._parent_layers: dict[str, str | None] = {self.GROUND_SCOPE_ID: None}
        self._fuse_overlayfs_bin = fuse_overlayfs_bin
        self._fusermount_bin = fusermount_bin
        self._ensure_supported()
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._scopes_root.mkdir(parents=True, exist_ok=True)
        self._reset_if_base_changed()

    def create_layer(self, scope_id: str, *, parent_scope_id: str | None) -> None:
        if scope_id == self.GROUND_SCOPE_ID:
            parent_scope_id = None

        paths = self._layer_paths(scope_id)
        if scope_id != self.GROUND_SCOPE_ID and paths.root.exists():
            msg = f"Overlay layer already exists for scope {scope_id!r}."
            raise RuntimeError(msg)

        self._parent_layers[scope_id] = parent_scope_id

        paths.root.mkdir(parents=True, exist_ok=True)
        paths.upper.mkdir(parents=True, exist_ok=True)
        paths.work.mkdir(parents=True, exist_ok=True)
        paths.merged.mkdir(parents=True, exist_ok=True)

        if self._is_mounted(paths.merged):
            return

        lowerdir = self._lowerdir(scope_id, parent_scope_id)
        self._mount_overlay(lowerdir=lowerdir, upperdir=paths.upper, workdir=paths.work, merged=paths.merged)

    def has_layer(self, scope_id: str) -> bool:
        return self._layer_paths(scope_id).root.exists()

    def read_file(self, scope_id: str, path: str) -> bytes:
        return self.read_file_state(scope_id, path).content

    def read_file_state(self, scope_id: str, path: str) -> FileState:
        file_path = self._merged_file_path(scope_id, path)
        return FileState(file_path.read_bytes(), posix_to_git_mode(file_path.stat().st_mode))

    def write_file(self, scope_id: str, path: str, content: bytes, *, mode: int = 0o100644) -> None:
        self._refuse_if_read_only("write", path)
        file_path = self._merged_file_path(scope_id, path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        file_path.chmod(stat.S_IMODE(normalize_git_filemode(mode)))

    def delete_file(self, scope_id: str, path: str) -> None:
        self._refuse_if_read_only("delete", path)
        file_path = self._merged_file_path(scope_id, path)
        if file_path.exists():
            file_path.unlink()

    def _refuse_if_read_only(self, op: str, path: str) -> None:
        if self._read_only:
            msg = f"{op} {path!r} refused: read-only carrier (the EROFS tier — no writable layer)"
            raise ReadOnlyCarrierError(msg)

    def diff_layer(self, scope_id: str) -> list[tuple[str, bytes | None, int]]:
        paths = self._layer_paths(scope_id)
        if not paths.upper.exists():
            return []

        changes: list[tuple[str, bytes | None, int]] = []
        for candidate in sorted(paths.upper.rglob("*")):
            rel = candidate.relative_to(paths.upper).as_posix()
            if not rel:
                continue

            try:
                file_stat = os.lstat(candidate)
            except OSError as exc:
                raise UnsupportedOverlayEntryError(path=rel, kind=f"unreadable ({exc.strerror or exc})") from exc
            if stat.S_ISDIR(file_stat.st_mode):
                continue
            if self._is_opaque_marker(candidate.name):
                continue
            if self._is_whiteout(candidate, file_stat):
                if candidate.name.startswith(_WHITEOUT_PREFIX):
                    rel = (
                        candidate.relative_to(paths.upper)
                        .parent.joinpath(candidate.name[len(_WHITEOUT_PREFIX) :])
                        .as_posix()
                    )
                changes.append((rel, None, 0))
                continue
            if stat.S_ISREG(file_stat.st_mode):
                try:
                    content = candidate.read_bytes()
                except OSError as exc:
                    raise UnsupportedOverlayEntryError(path=rel, kind=f"unreadable ({exc.strerror or exc})") from exc
                changes.append((rel, content, posix_to_git_mode(file_stat.st_mode)))
                continue
            kind = unsupported_overlay_entry_kind(file_stat.st_mode) or "unsupported"
            raise UnsupportedOverlayEntryError(path=rel, kind=kind)
        return changes

    def commit_layer(self, scope_id: str, *, into_scope_id: str | None) -> None:
        if into_scope_id is None:
            msg = "commit_layer() requires an explicit destination layer."
            raise RuntimeError(msg)
        if scope_id == self.GROUND_SCOPE_ID:
            msg = "Ground layer cannot be committed into another layer."
            raise RuntimeError(msg)

        changes = self.diff_layer(scope_id)
        for path, content, mode in changes:
            if content is None:
                self.delete_file(into_scope_id, path)
            else:
                self.write_file(into_scope_id, path, content, mode=mode)

        self.discard_layer(scope_id)

    def discard_layer(self, scope_id: str) -> None:
        paths = self._layer_paths(scope_id)
        if not paths.root.exists():
            return

        self._unmount(paths.merged)

        if scope_id == self.GROUND_SCOPE_ID:
            return

        self._parent_layers.pop(scope_id, None)
        shutil.rmtree(paths.root, ignore_errors=True)

    def push_layer(self, scope_id: str | None = None) -> None:
        target_scope_id = scope_id or self.GROUND_SCOPE_ID
        if target_scope_id != self.GROUND_SCOPE_ID:
            msg = "Only the ground overlay layer can be materialized."
            raise RuntimeError(msg)

        changes = self.diff_layer(self.GROUND_SCOPE_ID)
        for path, content, mode in changes:
            workspace_path = self._workspace_file_path(path)
            if content is None:
                if workspace_path.exists():
                    workspace_path.unlink()
                continue
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.write_bytes(content)
            workspace_path.chmod(stat.S_IMODE(normalize_git_filemode(mode)))

        self._reset_ground_layer()

    def working_path(self, scope_id: str) -> Path:
        return self._layer_paths(scope_id).merged

    def deactivate(self) -> None:
        ground = self._layer_paths(self.GROUND_SCOPE_ID).merged
        self._unmount(ground)
        if self._scopes_root.exists():
            for root in sorted(self._scopes_root.iterdir(), reverse=True):
                self._unmount(root / "merged")
        self._parent_layers = {self.GROUND_SCOPE_ID: None}

    def _reset_if_base_changed(self) -> None:
        if self._base_tree_oid is None:
            return
        previous = self._base_tree_oid_path.read_text().strip() if self._base_tree_oid_path.exists() else None
        if previous == self._base_tree_oid:
            return
        self.deactivate()
        shutil.rmtree(self._ground_root, ignore_errors=True)
        shutil.rmtree(self._scopes_root, ignore_errors=True)
        self._scopes_root.mkdir(parents=True, exist_ok=True)
        self._parent_layers = {self.GROUND_SCOPE_ID: None}
        self._base_tree_oid_path.write_text(self._base_tree_oid)

    def _ensure_supported(self) -> None:
        if _platform_name() != "linux":
            msg = "FuseOverlayBackend requires Linux."
            raise RuntimeError(msg)
        if not Path("/dev/fuse").exists():
            msg = "FuseOverlayBackend requires /dev/fuse."
            raise RuntimeError(msg)
        if shutil.which(self._fuse_overlayfs_bin) is None:
            msg = f"FuseOverlayBackend requires {self._fuse_overlayfs_bin!r}."
            raise RuntimeError(msg)
        if shutil.which(self._fusermount_bin) is None:
            msg = f"FuseOverlayBackend requires {self._fusermount_bin!r}."
            raise RuntimeError(msg)

    def _reset_ground_layer(self) -> None:
        ground_paths = self._layer_paths(self.GROUND_SCOPE_ID)
        self._unmount(ground_paths.merged)
        shutil.rmtree(ground_paths.root, ignore_errors=True)
        self._parent_layers[self.GROUND_SCOPE_ID] = None
        self.create_layer(self.GROUND_SCOPE_ID, parent_scope_id=None)

    def _lowerdir(self, scope_id: str, parent_scope_id: str | None) -> str:
        if scope_id == self.GROUND_SCOPE_ID:
            return str(self._base_lowerdir)
        if parent_scope_id is None:
            msg = f"Overlay layer {scope_id!r} requires a parent layer."
            raise RuntimeError(msg)
        lowerdirs: list[str] = []
        current: str | None = parent_scope_id
        while current is not None:
            parent_paths = self._layer_paths(current)
            if not parent_paths.upper.exists():
                msg = f"Parent overlay layer {current!r} does not exist."
                raise RuntimeError(msg)
            lowerdirs.append(str(parent_paths.upper))
            current = self._parent_layers.get(current)
        lowerdirs.append(str(self._base_lowerdir))
        return ":".join(lowerdirs)

    def _mount_overlay(self, *, lowerdir: str, upperdir: Path, workdir: Path, merged: Path) -> None:
        if self._read_only:
            # Lowerdir-only ⇒ a read-only mount: every write through `merged`
            # returns EROFS (the syscall-refusal tier). No upper/work to give.
            options = f"lowerdir={lowerdir}"
        else:
            options = f"lowerdir={lowerdir},upperdir={upperdir},workdir={workdir}"
        self._run([self._fuse_overlayfs_bin, "-o", options, str(merged)])

    def _unmount(self, merged: Path) -> None:
        if not merged.exists() or not self._is_mounted(merged):
            return
        result = subprocess.run(
            [self._fusermount_bin, "-u", str(merged)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        lazy = subprocess.run(
            [self._fusermount_bin, "-u", "-z", str(merged)],
            check=False,
            capture_output=True,
            text=True,
        )
        if lazy.returncode != 0:
            raise RuntimeError(f"Failed to unmount {merged}: {lazy.stderr.strip() or result.stderr.strip()}")

    def _layer_paths(self, scope_id: str) -> LayerPaths:
        if scope_id == self.GROUND_SCOPE_ID:
            root = self._ground_root
        else:
            root = self._scopes_root / scope_id
        return LayerPaths(root=root, upper=root / "upper", work=root / "work", merged=root / "merged")

    def _merged_file_path(self, scope_id: str, path: str) -> Path:
        relative = self._normalize_relative_path(path)
        layer = self._layer_paths(scope_id)
        if not layer.merged.exists():
            msg = f"Overlay layer {scope_id!r} is not available."
            raise RuntimeError(msg)
        return layer.merged / relative

    def _workspace_file_path(self, path: str) -> Path:
        return self._workspace / self._normalize_relative_path(path)

    def _normalize_relative_path(self, path: str) -> Path:
        return normalize_workspace_relative_path(path)

    def _is_mounted(self, path: Path) -> bool:
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.exists():
            return False
        target = str(path.resolve())
        with mountinfo.open() as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 4 and fields[4] == target:
                    return True
        return False

    def _is_whiteout(self, path: Path, file_stat: os.stat_result | None = None) -> bool:
        stat_result = file_stat
        if stat_result is None:
            stat_result = os.lstat(path)
        if stat.S_ISCHR(stat_result.st_mode):
            return os.major(stat_result.st_rdev) == 0 and os.minor(stat_result.st_rdev) == 0
        return path.name.startswith(_WHITEOUT_PREFIX)

    def _is_opaque_marker(self, filename: str) -> bool:
        return filename in {".wh..opq", ".wh..wh..opq"}

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, check=True, capture_output=True, text=True)


def _platform_name() -> str:
    return sys.platform
