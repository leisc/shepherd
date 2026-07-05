"""Inventory probes for first-cut scope and selected-world readiness facts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._pygit2_helpers import require_commit
from vcs_core._query_inventory import (
    AUTHORITY_REF_MISSING,
    AUTHORITY_REF_TARGET_MISSING_WORLD,
    AUTHORITY_REF_UNREADABLE,
    SCOPE_MISSING_REF,
    SCOPE_REF_UNREADABLE,
    WORLD_BINDING_INVALID,
    WORLD_BINDING_MISSING,
    WORLD_MISSING,
    WORLD_SELECTED_HEAD_DANGLING,
    WORLD_UNREADABLE,
    InventoryIssue,
    InventoryItem,
    expected,
    issue_id,
    missing,
    present_invalid,
    present_valid,
)
from vcs_core._world_refs import encode_ref_component
from vcs_core._world_storage_installation import (
    DEFAULT_WORLD_STORE_ID,
    default_world_storage_exists,
    default_world_storage_root,
    open_existing_default_world_storage,
)
from vcs_core.store import GROUND_REF

if TYPE_CHECKING:
    from vcs_core._world_storage_manager import WorldStorageManager
    from vcs_core._world_types import SubstrateHead, WorldCommit


@dataclass(frozen=True)
class RequiredBinding:
    """Shallow selected-head requirement for first-cut readiness."""

    binding: str
    head_kind: str | None = None
    role: str | None = None
    check: str = "selected_head"


@dataclass(frozen=True)
class _WorldManagerProbe:
    manager: WorldStorageManager | None
    open_error: str | None = None


def scope_ref_for_selector(selector: str | None) -> str:
    """Resolve the first-cut scope selector shape to a durable scope ref."""
    if selector in (None, "", "ground", GROUND_REF):
        return GROUND_REF
    selector_value = selector or "ground"
    if selector_value.startswith("refs/vcscore/"):
        return selector_value
    return f"refs/vcscore/scopes/{selector_value}"


def scope_name_for_ref(ref: str) -> str:
    if ref == GROUND_REF:
        return "ground"
    prefix = "refs/vcscore/scopes/"
    if ref.startswith(prefix):
        return ref.removeprefix(prefix)
    return ref.rsplit("/", 1)[-1]


def probe_scope(repo_path: str | Path, scope_selector: str | None) -> InventoryItem:
    """Classify one expected scalar scope ref without requiring activation."""
    ref = scope_ref_for_selector(scope_selector)
    scope_name = scope_name_for_ref(ref)
    item_id = f"scope:{ref}"
    fields: dict[str, object] = {
        "scope_name": scope_name,
        "scope_ref": ref,
        "requested_selector": scope_selector or "ground",
    }
    source_identity: dict[str, object] = {"ref": ref, "repo_path": str(Path(repo_path))}
    try:
        repo = pygit2.Repository(str(repo_path))
    except (KeyError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            domain="scope",
            kind="scope_ref",
            locator=ref,
            code=SCOPE_REF_UNREADABLE,
            message=f"scope repository could not be opened: {exc}",
            fields=fields,
            source_identity=source_identity,
            source_store="coordinator",
        )
    if ref not in repo.references:
        issue = _issue(item_id, SCOPE_MISSING_REF, f"scope ref is missing: {ref}", locator=ref)
        return InventoryItem(
            id=item_id,
            domain="scope",
            kind="scope_ref",
            locator=ref,
            source_kind="git_ref",
            source_store="coordinator",
            health=expected(issue_codes=(SCOPE_MISSING_REF,), authority_role="authoritative"),
            role=("authority",),
            fields=fields,
            source_identity=source_identity,
            issues=(issue,),
        )
    try:
        target_oid = str(repo.references[ref].target)
        source_identity["ref_target_oid"] = target_oid
        commit = require_commit(repo, pygit2.Oid(hex=target_oid), context=f"scope ref {ref}")
        source_identity["tree_oid"] = str(commit.tree_id)
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            domain="scope",
            kind="scope_ref",
            locator=ref,
            code=SCOPE_REF_UNREADABLE,
            message=str(exc),
            fields=fields,
            source_identity=source_identity,
            source_store="coordinator",
        )
    return InventoryItem(
        id=item_id,
        domain="scope",
        kind="scope_ref",
        locator=ref,
        source_kind="git_ref",
        source_store="coordinator",
        health=present_valid(authority_role="authoritative"),
        role=("authority",),
        fields=fields,
        source_identity=source_identity,
    )


def probe_authority_ref(repo_path: str | Path, authority_ref: str) -> InventoryItem:
    """Classify one expected v2 world authority ref."""
    probe = _open_existing_manager(repo_path)
    manager = probe.manager
    source_store = _world_source_store(repo_path, manager)
    item_id = f"authority_ref:{source_store}:{authority_ref}"
    fields: dict[str, object] = {
        "authority_ref": authority_ref,
        "scope_name": scope_name_for_ref(authority_ref),
        "world_store_id": DEFAULT_WORLD_STORE_ID if manager is None else manager.world_store.world_store_id,
    }
    source_identity: dict[str, object] = {
        "ref": authority_ref,
        "world_storage_root": str(default_world_storage_root(repo_path)),
    }
    if probe.open_error is not None:
        source_identity["world_storage_open_error"] = probe.open_error
        return _invalid_item(
            item_id=item_id,
            domain="authority_ref",
            kind="world_authority_ref",
            locator=authority_ref,
            code=AUTHORITY_REF_UNREADABLE,
            message=f"world storage could not be opened: {probe.open_error}",
            fields=fields,
            source_identity=source_identity,
            source_store=source_store,
        )
    if manager is None:
        issue = _issue(
            item_id,
            AUTHORITY_REF_MISSING,
            f"world storage is missing for authority ref: {authority_ref}",
            locator=authority_ref,
        )
        return InventoryItem(
            id=item_id,
            domain="authority_ref",
            kind="world_authority_ref",
            locator=authority_ref,
            source_kind="git_ref",
            source_store=source_store,
            health=expected(issue_codes=(AUTHORITY_REF_MISSING,), authority_role="authoritative"),
            role=("authority",),
            fields=fields,
            source_identity=source_identity,
            issues=(issue,),
        )
    source_identity["world_repo_path"] = manager.world_store.repo_path
    repo = manager.world_store.repo
    if authority_ref not in repo.references:
        issue = _issue(
            item_id,
            AUTHORITY_REF_MISSING,
            f"world authority ref is missing: {authority_ref}",
            locator=authority_ref,
        )
        return InventoryItem(
            id=item_id,
            domain="authority_ref",
            kind="world_authority_ref",
            locator=authority_ref,
            source_kind="git_ref",
            source_store=source_store,
            health=missing(issue_codes=(AUTHORITY_REF_MISSING,), authority_role="authoritative"),
            role=("authority",),
            fields=fields,
            source_identity=source_identity,
            issues=(issue,),
        )
    try:
        target_oid = str(repo.references[authority_ref].target)
        source_identity["ref_target_oid"] = target_oid
        require_commit(repo, pygit2.Oid(hex=target_oid), context=f"world authority ref {authority_ref}")
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            domain="authority_ref",
            kind="world_authority_ref",
            locator=authority_ref,
            code=AUTHORITY_REF_UNREADABLE,
            message=str(exc),
            fields=fields,
            source_identity=source_identity,
            source_store=source_store,
        )
    try:
        manager.read_world(target_oid)
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            domain="authority_ref",
            kind="world_authority_ref",
            locator=authority_ref,
            code=AUTHORITY_REF_TARGET_MISSING_WORLD,
            message=str(exc),
            fields={**fields, "world_oid": target_oid},
            source_identity=source_identity,
            source_store=source_store,
            primary_issue="dangling_dependency",
        )
    return InventoryItem(
        id=item_id,
        domain="authority_ref",
        kind="world_authority_ref",
        locator=authority_ref,
        source_kind="git_ref",
        source_store=source_store,
        health=present_valid(authority_role="authoritative"),
        role=("authority",),
        fields={**fields, "world_oid": target_oid},
        source_identity=source_identity,
    )


def probe_selected_world(
    repo_path: str | Path,
    authority_ref: str,
    *,
    required_bindings: tuple[RequiredBinding, ...] = (),
) -> tuple[InventoryItem, ...]:
    """Return selected-world and required-binding facts for one authority ref."""
    probe = _open_existing_manager(repo_path)
    manager = probe.manager
    source_store = _world_source_store(repo_path, manager)
    if probe.open_error is not None:
        open_error_source_identity: dict[str, object] = {
            "authority_ref": authority_ref,
            "world_storage_root": str(default_world_storage_root(repo_path)),
            "world_storage_open_error": probe.open_error,
        }
        return (
            _invalid_item(
                item_id=f"world:{source_store}:{authority_ref}",
                domain="world",
                kind="selected_world",
                locator=authority_ref,
                code=WORLD_UNREADABLE,
                message=f"world storage could not be opened: {probe.open_error}",
                fields={"authority_ref": authority_ref},
                source_identity=open_error_source_identity,
                source_store=source_store,
            ),
            *[
                _missing_binding_item(
                    binding,
                    authority_ref=authority_ref,
                    world_oid="unreadable",
                    source_store=source_store,
                    source_identity=open_error_source_identity,
                )
                for binding in required_bindings
            ],
        )
    if manager is None:
        return (
            _missing_world_item(
                repo_path,
                authority_ref=authority_ref,
                source_store=source_store,
                code=WORLD_MISSING,
                message="world storage is missing",
                is_expected=True,
            ),
            *[
                _missing_binding_item(
                    binding,
                    authority_ref=authority_ref,
                    world_oid="missing",
                    source_store=source_store,
                    source_identity={"authority_ref": authority_ref},
                    is_expected=True,
                )
                for binding in required_bindings
            ],
        )
    if authority_ref not in manager.world_store.repo.references:
        return (
            _missing_world_item(
                repo_path,
                authority_ref=authority_ref,
                source_store=source_store,
                code=WORLD_MISSING,
                message=f"selected world authority ref is missing: {authority_ref}",
            ),
            *[
                _missing_binding_item(
                    binding,
                    authority_ref=authority_ref,
                    world_oid="missing",
                    source_store=source_store,
                    source_identity={
                        "authority_ref": authority_ref,
                        "world_repo_path": manager.world_store.repo_path,
                    },
                )
                for binding in required_bindings
            ],
        )
    source_identity: dict[str, object] = {
        "authority_ref": authority_ref,
        "world_repo_path": manager.world_store.repo_path,
    }
    try:
        world_oid = str(manager.world_store.repo.references[authority_ref].target)
        source_identity["authority_ref_target_oid"] = world_oid
        commit = require_commit(manager.world_store.repo, pygit2.Oid(hex=world_oid), context="selected world")
        source_identity["world_tree_oid"] = str(commit.tree_id)
        world = manager.read_world(world_oid)
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        world_item = _invalid_item(
            item_id=f"world:{source_store}:{authority_ref}",
            domain="world",
            kind="selected_world",
            locator=authority_ref,
            code=WORLD_UNREADABLE,
            message=str(exc),
            fields={"authority_ref": authority_ref},
            source_identity=source_identity,
            source_store=source_store,
        )
        return (
            world_item,
            *[
                _missing_binding_item(
                    binding,
                    authority_ref=authority_ref,
                    world_oid="unreadable",
                    source_store=source_store,
                    source_identity=source_identity,
                )
                for binding in required_bindings
            ],
        )
    items = [
        _world_item(world, authority_ref=authority_ref, source_store=source_store, source_identity=source_identity)
    ]
    for binding in required_bindings:
        items.append(_binding_item(manager, world, binding, source_store=source_store, source_identity=source_identity))
    return tuple(items)


def _binding_item(
    manager: WorldStorageManager,
    world: WorldCommit,
    binding: RequiredBinding,
    *,
    source_store: str,
    source_identity: dict[str, object],
) -> InventoryItem:
    try:
        head = world.snapshot.head_for(binding.binding)
    except KeyError:
        return _missing_binding_item(
            binding,
            authority_ref=str(source_identity.get("authority_ref", "")),
            world_oid=world.oid,
            source_store=source_store,
            source_identity=source_identity,
        )
    fields = {
        "world_oid": world.oid,
        "snapshot_digest": world.snapshot.digest(),
        "binding": head.binding,
        "head_kind": head.kind,
        "role": head.role,
        "store_id": head.store_id,
        "store_scope": head.store_scope,
        "resource_id": head.resource_id,
        "head": head.head,
        "object_format": head.object_format,
        "required": True,
    }
    item_id = f"world_binding:{source_store}:{world.oid}:{encode_ref_component(binding.binding)}"
    mismatch = _binding_mismatch(head, binding)
    if mismatch is not None:
        return _invalid_item(
            item_id=item_id,
            domain="world",
            kind="selected_binding",
            locator=str(source_identity.get("authority_ref", "")),
            code=WORLD_BINDING_INVALID,
            message=mismatch,
            fields=fields,
            source_identity=source_identity,
            source_store=source_store,
            primary_issue="schema_mismatch",
        )
    try:
        store = manager.store(head.store_id)
        if not store.contains(head):
            raise InvalidRepositoryStateError(f"selected head does not exist in substrate store: {head.head}")
    except (InvalidRepositoryStateError, KeyError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            domain="world",
            kind="selected_binding",
            locator=str(source_identity.get("authority_ref", "")),
            code=WORLD_SELECTED_HEAD_DANGLING,
            message=str(exc),
            fields=fields,
            source_identity=source_identity,
            source_store=source_store,
            primary_issue="dangling_dependency",
        )
    return InventoryItem(
        id=item_id,
        domain="world",
        kind="selected_binding",
        locator=str(source_identity.get("authority_ref", "")),
        source_kind="substrate_store",
        source_store=source_store,
        health=present_valid(authority_role="authoritative"),
        role=("selected_state", "authority"),
        fields=fields,
        source_identity={**source_identity, "selected_head": head.head},
    )


def _binding_mismatch(head: SubstrateHead, required: RequiredBinding) -> str | None:
    if required.head_kind is not None and head.kind != required.head_kind:
        return f"selected binding {head.binding!r} kind {head.kind!r} does not match {required.head_kind!r}"
    if required.role is not None and head.role != required.role:
        return f"selected binding {head.binding!r} role {head.role!r} does not match {required.role!r}"
    return None


def _world_item(
    world: WorldCommit,
    *,
    authority_ref: str,
    source_store: str,
    source_identity: dict[str, object],
) -> InventoryItem:
    fields = {
        "authority_ref": authority_ref,
        "world_oid": world.oid,
        "snapshot_digest": world.snapshot.digest(),
        "operation_id": world.transition.get("operation_id"),
        "selected_binding_count": len(world.snapshot.heads),
    }
    return InventoryItem(
        id=f"world:{source_store}:{world.oid}",
        domain="world",
        kind="selected_world",
        locator=authority_ref,
        source_kind="git_ref",
        source_store=source_store,
        health=present_valid(authority_role="authoritative"),
        role=("selected_state", "authority"),
        fields=fields,
        source_identity=source_identity,
    )


def _missing_world_item(
    repo_path: str | Path,
    *,
    authority_ref: str,
    source_store: str,
    code: str,
    message: str,
    is_expected: bool = False,
) -> InventoryItem:
    item_id = f"world:{source_store}:{authority_ref}"
    issue = _issue(item_id, code, message, locator=authority_ref)
    return InventoryItem(
        id=item_id,
        domain="world",
        kind="selected_world",
        locator=authority_ref,
        source_kind="git_ref",
        source_store=source_store,
        health=(expected if is_expected else missing)(issue_codes=(code,), authority_role="authoritative"),
        role=("selected_state", "authority"),
        fields={"authority_ref": authority_ref},
        source_identity={
            "authority_ref": authority_ref,
            "world_storage_root": str(default_world_storage_root(repo_path)),
        },
        issues=(issue,),
    )


def _missing_binding_item(
    required: RequiredBinding,
    *,
    authority_ref: str,
    world_oid: str,
    source_store: str,
    source_identity: dict[str, object],
    is_expected: bool = False,
) -> InventoryItem:
    item_id = f"world_binding:{source_store}:{world_oid}:{encode_ref_component(required.binding)}"
    issue = _issue(
        item_id,
        WORLD_BINDING_MISSING,
        f"selected world is missing required binding: {required.binding}",
        locator=authority_ref,
    )
    return InventoryItem(
        id=item_id,
        domain="world",
        kind="selected_binding",
        locator=authority_ref,
        source_kind="substrate_store",
        source_store=source_store,
        health=(expected if is_expected else missing)(
            issue_codes=(WORLD_BINDING_MISSING,), authority_role="authoritative"
        ),
        role=("selected_state", "authority"),
        fields={
            "authority_ref": authority_ref,
            "world_oid": world_oid,
            "binding": required.binding,
            "head_kind": required.head_kind,
            "role": required.role,
            "required": True,
        },
        source_identity=dict(source_identity),
        issues=(issue,),
    )


def _open_existing_manager(repo_path: str | Path) -> _WorldManagerProbe:
    if not default_world_storage_exists(repo_path):
        return _WorldManagerProbe(manager=None)
    try:
        return _WorldManagerProbe(manager=open_existing_default_world_storage(repo_path))
    except (InvalidRepositoryStateError, KeyError, ValueError, pygit2.GitError) as exc:
        return _WorldManagerProbe(manager=None, open_error=str(exc))


def _world_source_store(repo_path: str | Path, manager: WorldStorageManager | None) -> str:
    if manager is None:
        return f"world-store:{DEFAULT_WORLD_STORE_ID}:{default_world_storage_root(repo_path)}"
    return f"world-store:{manager.world_store.world_store_id}:{manager.world_store.repo_path}"


def _invalid_item(
    *,
    item_id: str,
    domain: str,
    kind: str,
    locator: str | None,
    code: str,
    message: str,
    fields: dict[str, object],
    source_identity: dict[str, object],
    source_store: str,
    primary_issue: str = "unreadable",
) -> InventoryItem:
    issue = _issue(item_id, code, message, locator=locator)
    return InventoryItem(
        id=item_id,
        domain=domain,
        kind=kind,
        locator=locator,
        source_kind="git_ref",
        source_store=source_store,
        health=present_invalid(
            primary_issue=primary_issue,  # type: ignore[arg-type]
            issue_codes=(code,),
            authority_role="authoritative",
        ),
        role=("authority", "selected_state") if domain == "world" else ("authority",),
        fields=fields,
        source_identity=source_identity,
        issues=(issue,),
    )


def _issue(subject_id: str, code: str, message: str, *, locator: str | None) -> InventoryIssue:
    # No generic recovery_hint. The prior default ("run readiness/inspect ... before
    # retrying") was a circular non-hint on benign pre-first-push absences (issue 02:
    # it pointed at readiness, which reports healthy) and added nothing actionable
    # elsewhere. recovery_hint stays a per-issue field; specific conditions that have a
    # real recovery action set one explicitly.
    return InventoryIssue(
        id=issue_id(subject_id, code),
        code=code,
        message=message,
        subject_id=subject_id,
        locator=locator,
    )
