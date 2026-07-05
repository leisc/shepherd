"""Build workspace-state manifests for scalar-to-driver bridge paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygit2

from vcs_core._world_substrate_adapters import workspace_state_revision_payload
from vcs_core._world_types import canonical_digest
from vcs_core.git_store import build_tree, walk_workspace_tree
from vcs_core.types import EffectRecord, WorkspaceChange, normalize_git_filemode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcs_core.store import Store
    from vcs_core.types import ScopeInfo

WORKSPACE_CAPTURE_REDUCER_VERSION = "scalar-effects/v1"


@dataclass(frozen=True)
class WorkspaceCaptureReduction:
    """Workspace-state payload plus reducer proof metadata.

    When ``workspace_tree_oid`` is set, the reduction is tree-backed: the payload
    manifest declares ``byte_authority="tree-backed"`` and the substrate revision
    will embed a ``workspace/`` Git tree pointing at that OID. The tree must be
    reachable from the substrate's ODB (alternates to the coord cover this).
    """

    payload: dict[str, object]
    reduced_state_proof: dict[str, object]
    workspace_tree_oid: str | None = None


def workspace_capture_reduction_from_effects(
    *,
    command_operation_id: str,
    effects: Sequence[EffectRecord],
    covered_paths: Sequence[str],
    event_count: int,
    failed_command_origin: dict[str, object] | None = None,
) -> WorkspaceCaptureReduction:
    """Build the v2 digest-only workspace capture payload from scalar effects.

    Effects alone do not pin a Git tree OID (no source commit), so this builder
    stays digest-only. Use :func:`workspace_capture_state_from_store` for the
    tree-backed path.
    """
    entries = tuple(_workspace_manifest_entries(effects))
    payload = workspace_state_revision_payload(entries)
    manifest = payload["state_manifest"]
    reduced_state_proof: dict[str, object] = {
        "command_operation_id": command_operation_id,
        "byte_authority": "digest-only",
        "manifest_digest": canonical_digest(manifest),
        "covered_paths": list(covered_paths),
        "event_count": event_count,
        "reduced_effect_count": len(effects),
        "reducer": WORKSPACE_CAPTURE_REDUCER_VERSION,
    }
    if failed_command_origin is not None:
        reduced_state_proof["failed_command_origin"] = dict(failed_command_origin)
    return WorkspaceCaptureReduction(
        payload=payload,
        reduced_state_proof=reduced_state_proof,
    )


def workspace_capture_state_from_store(
    *,
    store: Store,
    scope: ScopeInfo,
    command_operation_id: str,
    effects: Sequence[EffectRecord],
    covered_paths: Sequence[str],
    event_count: int,
    failed_command_origin: dict[str, object] | None = None,
    tree_backed: bool = False,
) -> WorkspaceCaptureReduction:
    """Build a full-state workspace payload from a finalized scalar scope tree.

    When ``tree_backed`` is true, the workspace tree OID from the scope's tip
    commit is recorded in the payload manifest (``byte_authority="tree-backed"``)
    and returned on the result. Callers wire this OID through the substrate
    driver so the resulting substrate revision embeds a ``workspace/`` Git tree
    pointing at the same content.
    """
    source_commit = store.resolve_to_commit(scope.ref)
    workspace_tree_oid: str | None = None
    if tree_backed:
        workspace_tree_oid = _effective_workspace_tree_oid(
            store=store,
            source_commit=source_commit,
            effects=effects,
        )
        entries = tuple(_workspace_state_entries_from_tree(store, workspace_tree_oid))
    else:
        entries = tuple(_workspace_state_entries(store, scope.ref))
    byte_authority = "tree-backed" if tree_backed else "digest-only"
    payload = workspace_state_revision_payload(entries, byte_authority=byte_authority)
    manifest = payload["state_manifest"]
    final_present_paths = {str(entry["path"]) for entry in entries if entry.get("state", "present") == "present"}
    reduced_state_proof: dict[str, object] = {
        "command_operation_id": command_operation_id,
        "byte_authority": byte_authority,
        "manifest_digest": canonical_digest(manifest),
        "scope_name": scope.name,
        "scope_instance_id": scope.instance_id,
        "state_source": "scalar-scope-tree",
        "state_source_commit": str(source_commit.id) if source_commit is not None else None,
        "covered_paths": list(covered_paths),
        "deleted_paths": _deleted_or_absent_covered_paths(final_present_paths, effects, covered_paths),
        "event_count": event_count,
        "reduced_effect_count": len(effects),
        "reducer": WORKSPACE_CAPTURE_REDUCER_VERSION,
        "state_derivation": _state_derivation(tree_backed=tree_backed, effects=effects),
    }
    if workspace_tree_oid is not None:
        reduced_state_proof["workspace_tree_oid"] = workspace_tree_oid
    if failed_command_origin is not None:
        reduced_state_proof["failed_command_origin"] = dict(failed_command_origin)
    return WorkspaceCaptureReduction(
        payload=payload,
        reduced_state_proof=reduced_state_proof,
        workspace_tree_oid=workspace_tree_oid,
    )


@dataclass(frozen=True)
class WorkspaceStatePayload:
    """Workspace-state payload plus the source workspace tree OID when known."""

    payload: dict[str, object]
    workspace_tree_oid: str | None = None


def workspace_state_payload_from_store(
    *,
    store: Store,
    scope: ScopeInfo,
    tree_backed: bool = False,
    effects: Sequence[EffectRecord] = (),
) -> WorkspaceStatePayload:
    """Build a full-state workspace payload from a scalar scope tree.

    When ``tree_backed`` is true, the manifest declares
    ``byte_authority="tree-backed"`` and the workspace tree OID from the scope's
    tip commit is returned alongside the payload. Runtime ``effects`` are
    applied over that tree first so nested/unmaterialized operation output can
    be represented without advancing the scalar scope ref.
    """
    if tree_backed:
        source_commit = store.resolve_to_commit(scope.ref)
        workspace_tree_oid = _effective_workspace_tree_oid(
            store=store,
            source_commit=source_commit,
            effects=effects,
        )
        entries = tuple(_workspace_state_entries_from_tree(store, workspace_tree_oid))
    else:
        workspace_tree_oid = None
        entries = tuple(_workspace_state_entries(store, scope.ref))
    byte_authority = "tree-backed" if tree_backed else "digest-only"
    payload = workspace_state_revision_payload(entries, byte_authority=byte_authority)
    return WorkspaceStatePayload(payload=payload, workspace_tree_oid=workspace_tree_oid)


def _workspace_tree_oid_from_commit(store: Store, source_commit: pygit2.Commit | None) -> str | None:
    """Return the hex Git oid of the ``workspace`` subtree on ``source_commit``."""
    if source_commit is None:
        return None
    tree_oid = store._get_workspace_tree_oid(str(source_commit.id))
    return None if tree_oid is None else str(tree_oid)


def _effective_workspace_tree_oid(
    *,
    store: Store,
    source_commit: pygit2.Commit | None,
    effects: Sequence[EffectRecord],
) -> str | None:
    source_tree_oid = _workspace_tree_oid_from_commit(store, source_commit)
    workspace_changes = _workspace_changes_from_effects(effects)
    if not workspace_changes:
        return source_tree_oid
    parent_tree_oid = None if source_tree_oid is None else pygit2.Oid(hex=source_tree_oid)
    return str(build_tree(store._repo, parent_tree_oid, workspace_changes))


def _workspace_changes_from_effects(effects: Sequence[EffectRecord]) -> tuple[WorkspaceChange, ...]:
    return tuple(change for effect in effects for change in effect.workspace_changes)


def _state_derivation(*, tree_backed: bool, effects: Sequence[EffectRecord]) -> str:
    if tree_backed and any(effect.workspace_changes for effect in effects):
        return "scalar-scope-tree+runtime-effects/v1"
    return "scalar-scope-tree/v1"


def _workspace_manifest_entries(effects: Sequence[EffectRecord]) -> list[dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    for effect in effects:
        for change in effect.workspace_changes:
            entry = _workspace_manifest_entry(change)
            by_path[str(entry["path"])] = entry
    return [by_path[path] for path in sorted(by_path)]


def _workspace_state_entries(store: Store, ref: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path, _blob_oid, mode in store.list_workspace_files(ref):
        content = store.read_workspace_file(ref, path)
        if content is None:
            continue
        entries.append(
            {
                "path": path,
                "state": "present",
                "mode": normalize_git_filemode(mode),
                "content_digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
            }
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def _workspace_state_entries_from_tree(
    store: Store,
    tree_oid: str | None,
) -> list[dict[str, object]]:
    if tree_oid is None:
        return []
    entries: list[dict[str, object]] = []
    for path, blob_oid, mode in walk_workspace_tree(store._repo, pygit2.Oid(hex=tree_oid)):
        blob = store._repo.get(blob_oid)
        if not isinstance(blob, pygit2.Blob):
            continue
        entries.append(
            {
                "path": path,
                "state": "present",
                "mode": normalize_git_filemode(mode),
                "content_digest": f"sha256:{hashlib.sha256(bytes(blob.data)).hexdigest()}",
            }
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def _deleted_or_absent_covered_paths(
    final_present_paths: set[str],
    effects: Sequence[EffectRecord],
    covered_paths: Sequence[str],
) -> list[str]:
    absent = {path for path in covered_paths if path not in final_present_paths}
    for effect in effects:
        for change in effect.workspace_changes:
            if change[1] is None and change[0] not in final_present_paths:
                absent.add(change[0])
    return sorted(absent)


def _workspace_manifest_entry(change: WorkspaceChange) -> dict[str, object]:
    path = change[0]
    content = change[1]
    if content is None:
        return {"path": path, "state": "deleted"}
    mode = normalize_git_filemode(change[2]) if len(change) == 3 else 0o100644
    return {
        "path": path,
        "state": "present",
        "mode": mode,
        "content_digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
    }
