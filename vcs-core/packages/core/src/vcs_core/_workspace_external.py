"""Normalized views of physical and Git-backed workspace state."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import pygit2

from vcs_core._pygit2_helpers import require_blob, require_object, require_tree
from vcs_core._workspace_paths import normalize_workspace_relative_path
from vcs_core.types import posix_to_git_mode

ExternalSource = Literal["git-head", "worktree"]
ExactPhysicalKind = Literal["absent", "file", "symlink", "directory", "unsupported"]

_CONTROL_DIRS = {".git", ".vcscore"}
_SUPPORTED_FILE_MODES = {
    pygit2.GIT_FILEMODE_BLOB,
    pygit2.GIT_FILEMODE_BLOB_EXECUTABLE,
}


@dataclass(frozen=True)
class ExternalWorkspaceFile:
    """Content and Git mode for one workspace-relative external file."""

    content: bytes
    mode: int = pygit2.GIT_FILEMODE_BLOB


@dataclass(frozen=True)
class ExternalWorkspaceBlocker:
    """One external workspace path that prevents safe acknowledgement/use."""

    path: str
    reason: str


@dataclass(frozen=True)
class GitWorkspace:
    """The enclosing Git worktree plus the selected workspace subdirectory."""

    root: Path
    prefix: str


@dataclass(frozen=True)
class ExactPhysicalState:
    """Exact physical state for one workspace-relative path.

    Unlike adoption and admission enumeration, exact reads intentionally ignore
    Git ignore rules. Push preflight uses this for write-target safety.
    """

    path: str
    kind: ExactPhysicalKind
    content: bytes | None = None
    mode: int | None = None
    detail: str | None = None

    @property
    def file_tuple(self) -> tuple[bytes, int] | None:
        if self.kind != "file" or self.content is None or self.mode is None:
            return None
        return (self.content, self.mode)

    @property
    def is_unsupported(self) -> bool:
        return self.kind not in {"absent", "file"}


def validate_relative_path(path: str) -> str:
    pure = normalize_workspace_relative_path(path)
    if any(part in _CONTROL_DIRS for part in pure.parts):
        raise ValueError(f"Refusing to adopt control-plane path: {path!r}")
    return pure.as_posix()


def is_control_path(path: str | Path) -> bool:
    parts = PurePosixPath(os.fspath(path)).parts
    return any(part in _CONTROL_DIRS for part in parts)


def _mode_for_physical_file(path: Path) -> int:
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Cannot adopt unsupported filesystem entry: {path}")
    return posix_to_git_mode(file_stat.st_mode)


class ExternalWorkspace:
    """Policy boundary for physical and Git-backed workspace state."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.git_workspace = discover_git_workspace(workspace)

    def validate_relative_path(self, path: str) -> str:
        return validate_relative_path(path)

    def read_git_head_source(self) -> dict[str, ExternalWorkspaceFile]:
        return _read_git_head_workspace(self.workspace, self.git_workspace)

    def read_worktree_source(self) -> dict[str, ExternalWorkspaceFile]:
        return _read_worktree_workspace(self.workspace, self.git_workspace)

    def read_adoption_source(self, source: ExternalSource) -> dict[str, ExternalWorkspaceFile]:
        if source == "git-head":
            return self.read_git_head_source()
        if source == "worktree":
            return self.read_worktree_source()
        raise ValueError(f"Unknown external workspace source: {source!r}")

    def git_status_blockers(self, *, reason: str = "git-worktree-dirty") -> tuple[ExternalWorkspaceBlocker, ...]:
        return _git_status_blockers(self.workspace, self.git_workspace, reason=reason)

    def git_index_blockers(self, *, reason: str = "git-index-dirty") -> tuple[ExternalWorkspaceBlocker, ...]:
        return _git_index_blockers(self.workspace, self.git_workspace, reason=reason)

    def read_exact_physical(self, path: str) -> ExactPhysicalState:
        normalized = validate_relative_path(path)
        candidate = self.workspace.joinpath(*PurePosixPath(normalized).parts)
        try:
            file_stat = candidate.lstat()
        except FileNotFoundError:
            return ExactPhysicalState(path=normalized, kind="absent")
        except OSError as exc:
            return ExactPhysicalState(path=normalized, kind="unsupported", detail=str(exc))

        if stat.S_ISLNK(file_stat.st_mode):
            return ExactPhysicalState(path=normalized, kind="symlink")
        if stat.S_ISDIR(file_stat.st_mode):
            return ExactPhysicalState(path=normalized, kind="directory")
        if not stat.S_ISREG(file_stat.st_mode):
            return ExactPhysicalState(path=normalized, kind="unsupported")
        try:
            return ExactPhysicalState(
                path=normalized,
                kind="file",
                content=candidate.read_bytes(),
                mode=posix_to_git_mode(file_stat.st_mode),
            )
        except OSError as exc:
            return ExactPhysicalState(path=normalized, kind="unsupported", detail=str(exc))


def discover_git_workspace(workspace: Path) -> GitWorkspace | None:
    """Return enclosing Git worktree information, or None outside Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    root = Path(result.stdout.strip()).resolve()
    resolved_workspace = workspace.resolve()
    try:
        relative = resolved_workspace.relative_to(root)
    except ValueError:
        prefix_result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-prefix"],
            check=True,
            capture_output=True,
            text=True,
        )
        prefix = prefix_result.stdout.strip().rstrip("/")
    else:
        prefix = relative.as_posix() if relative.parts else ""
    return GitWorkspace(root=root, prefix=prefix)


def _workspace_relative_git_path(git_path: str, git_workspace: GitWorkspace) -> str | None:
    path = PurePosixPath(git_path).as_posix()
    prefix = git_workspace.prefix
    if prefix:
        if path == prefix:
            return None
        prefix_with_slash = f"{prefix}/"
        if not path.startswith(prefix_with_slash):
            return None
        path = path[len(prefix_with_slash) :]
    if not path or is_control_path(path):
        return None
    return validate_relative_path(path)


def _walk_git_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    *,
    git_workspace: GitWorkspace,
    prefix: str = "",
) -> dict[str, ExternalWorkspaceFile]:
    files: dict[str, ExternalWorkspaceFile] = {}
    for entry in tree:
        git_path = f"{prefix}{entry.name}"
        obj = require_object(repo, entry.id)
        if isinstance(obj, pygit2.Tree):
            files.update(_walk_git_tree(repo, obj, git_workspace=git_workspace, prefix=f"{git_path}/"))
            continue
        workspace_path = _workspace_relative_git_path(git_path, git_workspace)
        if workspace_path is None:
            continue
        if not isinstance(obj, pygit2.Blob):
            raise TypeError(f"Cannot adopt unsupported Git tree entry: {workspace_path}")
        mode = int(entry.filemode)
        if mode not in _SUPPORTED_FILE_MODES:
            raise ValueError(f"Cannot adopt unsupported Git file mode {mode:o} at {workspace_path!r}.")
        blob = require_blob(repo, entry.id, context=f"git-head workspace read {workspace_path}")
        files[workspace_path] = ExternalWorkspaceFile(content=bytes(blob.data), mode=mode)
    return files


def _read_git_head_workspace(
    workspace: Path,
    git_workspace: GitWorkspace | None,
) -> dict[str, ExternalWorkspaceFile]:
    """Read regular files from HEAD under the selected workspace root."""
    if git_workspace is None:
        raise ValueError(f"{workspace} does not have a readable Git worktree to adopt.")
    try:
        repo = pygit2.Repository(str(git_workspace.root))
        commit = repo.head.peel(pygit2.Commit)
    except (KeyError, ValueError, pygit2.GitError) as exc:
        raise ValueError(f"{workspace} does not have a readable Git HEAD to adopt.") from exc
    tree = require_tree(repo, commit.tree_id, context="git-head adoption HEAD tree")
    return _walk_git_tree(repo, tree, git_workspace=git_workspace)


def read_git_head_workspace(workspace: Path) -> dict[str, ExternalWorkspaceFile]:
    return ExternalWorkspace(workspace).read_git_head_source()


def _status_workspace_path(path: str, git_workspace: GitWorkspace) -> str | None:
    return _workspace_relative_git_path(path, git_workspace)


def _decode_git_path(raw_path: bytes) -> str:
    return raw_path.decode("utf-8", errors="surrogateescape")


def _iter_porcelain_z_entries(output: bytes) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tokens = output.split(b"\0")
    entries: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        status = token[:2].decode("ascii", errors="strict")
        path = _decode_git_path(token[3:])
        paths = [path]
        if "R" in status or "C" in status:
            if index >= len(tokens) or not tokens[index]:
                raise ValueError("Malformed Git porcelain rename/copy record.")
            paths.append(_decode_git_path(tokens[index]))
            index += 1
        entries.append((status, tuple(paths)))
    return tuple(entries)


def _git_status_blockers(
    workspace: Path,
    git_workspace: GitWorkspace | None,
    *,
    reason: str = "git-worktree-dirty",
) -> tuple[ExternalWorkspaceBlocker, ...]:
    """Return dirty/staged/untracked non-ignored paths under workspace."""
    if git_workspace is None:
        return ()
    result = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", "."],
        check=True,
        capture_output=True,
    )
    blockers: list[ExternalWorkspaceBlocker] = []
    for _status, raw_paths in _iter_porcelain_z_entries(result.stdout):
        for raw_path in raw_paths:
            path = _status_workspace_path(raw_path, git_workspace)
            if path is None:
                continue
            blockers.append(ExternalWorkspaceBlocker(path=path, reason=reason))
    return tuple(sorted(blockers, key=lambda blocker: (blocker.path, blocker.reason)))


def git_status_blockers(
    workspace: Path,
    *,
    reason: str = "git-worktree-dirty",
) -> tuple[ExternalWorkspaceBlocker, ...]:
    return ExternalWorkspace(workspace).git_status_blockers(reason=reason)


def _git_index_blockers(
    workspace: Path,
    git_workspace: GitWorkspace | None,
    *,
    reason: str = "git-index-dirty",
) -> tuple[ExternalWorkspaceBlocker, ...]:
    """Return staged/index dirty paths under workspace, excluding untracked paths."""
    if git_workspace is None:
        return ()
    result = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", "."],
        check=True,
        capture_output=True,
    )
    blockers: list[ExternalWorkspaceBlocker] = []
    for status, raw_paths in _iter_porcelain_z_entries(result.stdout):
        index_status = status[0]
        if index_status in {" ", "?"}:
            continue
        for raw_path in raw_paths:
            path = _status_workspace_path(raw_path, git_workspace)
            if path is None:
                continue
            blockers.append(ExternalWorkspaceBlocker(path=path, reason=reason))
    return tuple(sorted(blockers, key=lambda blocker: (blocker.path, blocker.reason)))


def git_index_blockers(
    workspace: Path,
    *,
    reason: str = "git-index-dirty",
) -> tuple[ExternalWorkspaceBlocker, ...]:
    return ExternalWorkspace(workspace).git_index_blockers(reason=reason)


def _git_nonignored_worktree_paths(workspace: Path, git_workspace: GitWorkspace | None) -> set[str] | None:
    if git_workspace is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "-z", "-c", "-o", "--exclude-standard", "--full-name", "--", "."],
        check=True,
        capture_output=True,
    )
    paths: set[str] = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        raw_path = _decode_git_path(raw)
        path = _workspace_relative_git_path(raw_path, git_workspace)
        if path is not None:
            paths.add(path)
    return paths


def _read_worktree_workspace(
    workspace: Path,
    git_workspace: GitWorkspace | None,
) -> dict[str, ExternalWorkspaceFile]:
    """Read physical workspace files, excluding control and ignored files."""
    allowed_git_paths = _git_nonignored_worktree_paths(workspace, git_workspace)
    files: dict[str, ExternalWorkspaceFile] = {}
    if allowed_git_paths is None:
        selected_paths = [
            candidate.relative_to(workspace).as_posix()
            for candidate in sorted(workspace.rglob("*"))
            if not candidate.is_dir()
        ]
    else:
        selected_paths = sorted(allowed_git_paths)

    for rel in selected_paths:
        if is_control_path(rel):
            continue
        path = validate_relative_path(rel)
        candidate = workspace.joinpath(*PurePosixPath(path).parts)
        if allowed_git_paths is not None and not candidate.exists():
            continue
        if candidate.is_symlink():
            raise ValueError(f"Cannot adopt symbolic link: {rel}")
        if not candidate.is_file():
            raise ValueError(f"Cannot adopt unsupported filesystem entry: {rel}")
        files[path] = ExternalWorkspaceFile(content=candidate.read_bytes(), mode=_mode_for_physical_file(candidate))
    return files


def read_worktree_workspace(workspace: Path) -> dict[str, ExternalWorkspaceFile]:
    return ExternalWorkspace(workspace).read_worktree_source()


def read_exact_physical_workspace_path(workspace: Path, path: str) -> ExactPhysicalState:
    return ExternalWorkspace(workspace).read_exact_physical(path)


def read_external_workspace_source(workspace: Path, source: ExternalSource) -> dict[str, ExternalWorkspaceFile]:
    return ExternalWorkspace(workspace).read_adoption_source(source)
