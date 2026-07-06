"""Admission checks for physical workspace state before sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vcs_core._errors import VcsCoreError
from vcs_core._workspace_external import (
    ExternalWorkspace,
    ExternalWorkspaceFile,
)
from vcs_core.store import MATERIALIZED_REF

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


class WorkspaceStateStore(Protocol):
    """Store operations needed to compare workspace files with a stored ref."""

    def list_workspace_files(self, ref: str) -> Iterable[tuple[str, str, int]]: ...

    def read_workspace_file(self, ref: str, path: str) -> bytes | None: ...


@dataclass(frozen=True)
class ExternalStateBlocker:
    """One physical workspace path that is not represented by materialized state."""

    path: str
    reason: str


class ExternalWorkspaceStateError(VcsCoreError, RuntimeError):
    """Expected admission failure for dirty or unadopted physical workspace state."""

    def __init__(self, blockers: tuple[ExternalStateBlocker, ...], *, action: str) -> None:
        if not blockers:
            msg = "External workspace state error requires at least one blocker."
            raise ValueError(msg)
        self.blockers = blockers
        self.action = action
        super().__init__(format_external_state_blockers(blockers, action=action))


def format_external_state_blockers(blockers: tuple[ExternalStateBlocker, ...], *, action: str) -> str:
    sample = ", ".join(f"{blocker.path} ({blocker.reason})" for blocker in blockers[:5])
    remainder = len(blockers) - min(len(blockers), 5)
    suffix = f", and {remainder} more" if remainder > 0 else ""
    return (
        f"Cannot {action} with {len(blockers)} unadopted or dirty physical workspace path(s): "
        f"{sample}{suffix}. Run `vcs-core init --adopt git-head --all` or "
        "`vcs-core init --adopt worktree --all` after making the workspace clean."
    )


def _store_snapshot(store: WorkspaceStateStore, ref: str) -> dict[str, WorkspaceFileState]:
    return {
        path: WorkspaceFileState(content=store.read_workspace_file(ref, path) or b"", mode=mode)
        for path, _oid, mode in store.list_workspace_files(ref)
    }


WorkspaceFileState = ExternalWorkspaceFile


def _states_differ(left: WorkspaceFileState | None, right: WorkspaceFileState | None) -> bool:
    if left is None or right is None:
        return left != right
    return left.content != right.content or left.mode != right.mode


def _workspace_has_git_head(workspace: Path) -> bool:
    external_workspace = ExternalWorkspace(workspace)
    if external_workspace.git_workspace is None:
        return False
    try:
        external_workspace.read_git_head_source()
    except ValueError:
        return False
    return True


def _path_is_admitted(
    workspace: Path,
    path: str,
    is_path_admitted: Callable[[Path], bool] | None,
) -> bool:
    return is_path_admitted is not None and is_path_admitted(workspace / path)


def external_workspace_blockers(
    store: WorkspaceStateStore,
    workspace: Path,
    *,
    is_path_admitted: Callable[[Path], bool] | None = None,
    reference: str = MATERIALIZED_REF,
) -> tuple[ExternalStateBlocker, ...]:
    """Return paths that would be visible physically but not in the reference state.

    Per SPI v0.1 spike 260524 (admission contract Q1), the worktree must
    be reachable from a substrate-aware reference. T4a adds the
    ``reference`` parameter so callers can compare against different
    refs based on the admission timing:

    - ``MATERIALIZED_REF`` (default): for session-start and re-
      materialization admission — the worktree should match what was
      last materialized.
    - ``GROUND_REF``: for push admission — the worktree should match
      what's about to be published. Pre-T4a push admission used the
      default (``MATERIALIZED_REF``) which fails after Python-tier
      capture commits to ground but before the next push advances
      ``MATERIALIZED_REF``. The push-admission bug from the parent
      spike (260523-python-tier-push-admission) is structurally
      resolved by callers passing ``reference=GROUND_REF`` here.

    Future tranches (Stage 2 query plane integration) may consult
    pending v2 substrate operations directly; in v0.1 the
    GROUND_REF-vs-worktree comparison captures the relevant projected
    post-push state because the v2 capture-reduction flow advances
    the scalar GROUND_REF as part of its merge.
    """
    external_workspace = ExternalWorkspace(workspace)
    materialized = _store_snapshot(store, reference)
    blockers: list[ExternalStateBlocker] = []
    if external_workspace.git_workspace is not None and _workspace_has_git_head(workspace):
        source = external_workspace.read_git_head_source()
        reason = "git-head-not-adopted"
        blockers.extend(
            ExternalStateBlocker(path=blocker.path, reason=blocker.reason)
            for blocker in external_workspace.git_index_blockers()
            if not _path_is_admitted(workspace, blocker.path, is_path_admitted)
        )
        indexed_paths = {blocker.path for blocker in blockers}
        blockers.extend(
            ExternalStateBlocker(path=blocker.path, reason=blocker.reason)
            for blocker in external_workspace.git_status_blockers()
            if blocker.path not in indexed_paths and not _path_is_admitted(workspace, blocker.path, is_path_admitted)
        )
    else:
        source = external_workspace.read_worktree_source()
        reason = "worktree-not-adopted"

    seen = {blocker.path for blocker in blockers}
    for path in sorted(set(materialized) | set(source)):
        if path in seen:
            continue
        if _path_is_admitted(workspace, path, is_path_admitted):
            continue
        if _states_differ(materialized.get(path), source.get(path)):
            blockers.append(ExternalStateBlocker(path=path, reason=reason))
    return tuple(sorted(blockers, key=lambda blocker: (blocker.path, blocker.reason)))


def assert_workspace_admissible(
    store: WorkspaceStateStore,
    workspace: Path,
    *,
    action: str = "start a session",
    is_path_admitted: Callable[[Path], bool] | None = None,
    reference: str = MATERIALIZED_REF,
) -> None:
    blockers = external_workspace_blockers(
        store,
        workspace,
        is_path_admitted=is_path_admitted,
        reference=reference,
    )
    if not blockers:
        return
    raise ExternalWorkspaceStateError(blockers, action=action)


def pending_workspace_ops_for_path(
    store: WorkspaceStateStore,
    workspace: Path,
    path: str,
) -> tuple[str, ...]:
    """Stage 1 query-plane helper (per 260524 Q5).

    Returns the operation ids (if any) that touched ``path`` and are
    pending publication. Today this walks the scalar store's
    GROUND_REF view; the Stage 2 implementation (per the query plane
    proposal) replaces this with a selector invocation over an
    InventorySnapshot. The signature is shaped so the Stage 2 swap-in
    is a one-line wrapper around the selector call.

    v0.1: stub implementation returning the empty tuple. The
    admission rewrite uses GROUND_REF-vs-worktree comparison
    directly; the per-path operation-id annotation is reserved for
    future diagnostic enrichment when blocker messages need to cite
    specific operations.
    """
    del store, workspace, path
    return ()
