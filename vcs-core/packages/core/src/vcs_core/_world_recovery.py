"""Conservative private recovery helpers for v2 world storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygit2

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._world_operation_builder import PreparedWorldOperation
from vcs_core._world_publication_plan import PublicationPlan
from vcs_core._world_storage_records import DEFAULT_GROUND_REF

if TYPE_CHECKING:
    from vcs_core._world_storage_manager import WorldStorageManager


@dataclass(frozen=True)
class WorldRecoveryAction:
    """One action or blocker observed by the private recovery runner."""

    code: str
    message: str
    world_oid: str | None = None
    operation_id: str | None = None


@dataclass(frozen=True)
class WorldRecoveryReport:
    """Conservative recovery result."""

    actions: tuple[WorldRecoveryAction, ...]

    @property
    def ok(self) -> bool:
        return not any(action.code.endswith("_blocked") for action in self.actions)


def cleanup_orphan_pins(
    manager: WorldStorageManager,
    world_oid: str,
    *,
    authority_refs: tuple[str, ...] = (DEFAULT_GROUND_REF,),
) -> WorldRecoveryReport:
    """Delete orphan selected-head pins and retention receipts for an unpublished world."""
    deleted = manager.cleanup_orphan_pins(world_oid, authority_refs=authority_refs)
    return WorldRecoveryReport(tuple(_orphan_cleanup_action(world_oid, ref) for ref in deleted))


def cleanup_stale_publication_leases(
    manager: WorldStorageManager,
    *,
    authority_refs: tuple[str, ...] = (DEFAULT_GROUND_REF,),
    abandon_journalless: bool = False,
) -> WorldRecoveryReport:
    """Delete publication leases that no longer protect an active publication attempt."""
    deleted = manager.cleanup_stale_publication_leases(
        authority_refs=authority_refs,
        abandon_journalless=abandon_journalless,
    )
    # Recovery is off the hot path: reconcile the active-lease accelerator with the
    # authoritative lease refs so a crash-lagged or stale index converges.
    manager.rebuild_active_lease_index()
    return WorldRecoveryReport(tuple(_publication_lease_cleanup_action(ref) for ref in deleted))


def reconcile_open_operation_journal_index(manager: WorldStorageManager) -> WorldRecoveryReport:
    """Rebuild the open-journal accelerator from the authoritative open refs (out-of-model drift heal).

    The store-global backstop for the open-journal index, mirroring the lease rebuild above. Per-op
    stale-open cleanup already co-writes its tombstone atomically, so this is only needed to
    reconcile drift left by an out-of-model writer (manual/private-ref edit) that bypassed the
    co-write — the *stale* case ``verify_open_operation_journal_index`` surfaces on deep journal
    fsck. Off the hot path; the authority is unaffected.
    """
    manager.rebuild_open_operation_journal_index()
    return WorldRecoveryReport(
        (
            WorldRecoveryAction(
                code="open_operation_journal_index_reconciled",
                message="rebuilt the open-operation-journal accelerator from the authoritative open refs",
            ),
        )
    )


def archive_failed_operation(manager: WorldStorageManager, operation_id: str) -> WorldRecoveryReport:
    """Archive an open failed operation journal, or no-op if already archived."""
    try:
        manager.read_operation_journal(operation_id, family="archived")
    except InvalidRepositoryStateError:
        pass
    else:
        cleanup_actions = _cleanup_stale_open_actions(manager, operation_id, terminal_family="archived")
        return WorldRecoveryReport(
            (
                *cleanup_actions,
                WorldRecoveryAction(
                    code="operation_already_archived",
                    message="operation journal is already archived",
                    operation_id=operation_id,
                ),
            )
        )
    history = manager.read_operation_journal(operation_id)
    status = history.tip.payload.get("status")
    if status != "failed":
        return WorldRecoveryReport(
            (
                WorldRecoveryAction(
                    code="operation_archive_blocked",
                    message=f"operation status {status!r} is not failed",
                    operation_id=operation_id,
                ),
            )
        )
    manager.archive_operation_journal(operation_id)
    return WorldRecoveryReport(
        (
            WorldRecoveryAction(
                code="operation_archived",
                message="archived failed operation journal",
                operation_id=operation_id,
            ),
        )
    )


def complete_committed_operation(manager: WorldStorageManager, operation_id: str) -> WorldRecoveryReport:
    """Complete publication for an open finalized or committed journal when the authority ref still matches input."""
    try:
        closed = manager.read_operation_journal(operation_id, family="closed")
    except InvalidRepositoryStateError:
        pass
    else:
        cleanup_actions = _cleanup_stale_open_actions(manager, operation_id, terminal_family="closed")
        world_oid = closed.tip.payload.get("world_oid")
        return WorldRecoveryReport(
            (
                *cleanup_actions,
                WorldRecoveryAction(
                    code="operation_already_closed",
                    message="operation journal is already closed",
                    world_oid=world_oid if isinstance(world_oid, str) else None,
                    operation_id=operation_id,
                ),
            )
        )
    try:
        archived = manager.read_operation_journal(operation_id, family="archived")
    except InvalidRepositoryStateError:
        pass
    else:
        cleanup_actions = _cleanup_stale_open_actions(manager, operation_id, terminal_family="archived")
        world_oid = archived.tip.payload.get("world_oid")
        return WorldRecoveryReport(
            (
                *cleanup_actions,
                WorldRecoveryAction(
                    code="operation_complete_blocked",
                    message="operation journal is archived",
                    world_oid=world_oid if isinstance(world_oid, str) else None,
                    operation_id=operation_id,
                ),
            )
        )
    history = manager.read_operation_journal(operation_id)
    tip = history.tip.payload
    status = tip.get("status")
    if status not in {"finalized", "world_committed", "publishing", "published"}:
        return WorldRecoveryReport(
            (
                WorldRecoveryAction(
                    code="operation_complete_blocked",
                    message=f"operation status {status!r} is not finalized, world_committed, publishing, or published",
                    operation_id=operation_id,
                ),
            )
        )
    target_ref = _required_str(tip, "target_ref")
    input_world_oid = _nullable_str(tip, "input_world_oid")
    if status == "finalized":
        world_oid = _commit_finalized_world(manager, tip)
        manager.record_operation_world_committed(operation_id, world_oid=world_oid)
        status = "world_committed"
    else:
        world_oid = _required_str(tip, "world_oid")
    current_target = _current_ref_target(manager, target_ref)
    if status == "world_committed":
        if current_target == world_oid:
            return WorldRecoveryReport(
                (
                    WorldRecoveryAction(
                        code="operation_complete_blocked",
                        message="authority ref already equals operation world before publication intent was journaled",
                        world_oid=world_oid,
                        operation_id=operation_id,
                    ),
                )
            )
        if input_world_oid is None:
            if current_target is not None:
                return WorldRecoveryReport(
                    (
                        WorldRecoveryAction(
                            code="operation_complete_blocked",
                            message="authority ref already exists before root publication intent was journaled",
                            world_oid=world_oid,
                            operation_id=operation_id,
                        ),
                    )
                )
            publication_plan = manager.build_root_publication_plan(
                ref=target_ref,
                world_oid=world_oid,
            )
        elif current_target != input_world_oid:
            return WorldRecoveryReport(
                (
                    WorldRecoveryAction(
                        code="operation_complete_blocked",
                        message="authority ref no longer equals operation input world",
                        world_oid=world_oid,
                        operation_id=operation_id,
                    ),
                )
            )
        else:
            publication_plan = manager.build_advance_publication_plan(
                ref=target_ref,
                world_oid=world_oid,
                expected_oid=input_world_oid,
                input_world_oid=input_world_oid,
            )
        manager.record_operation_publishing(operation_id, world_oid=world_oid, publication_plan=publication_plan)
        status = "publishing"
    elif status == "publishing":
        publication_plan = _publication_plan_from_tip(tip)
    else:
        publication_plan = None
    if status == "publishing" and current_target == input_world_oid:
        if publication_plan is None:
            publication_plan = _publication_plan_from_tip(tip)
        prepared_publication = manager.prepare_publication(publication_plan)
        if not manager.advance_publication(prepared_publication):
            manager.complete_publication(prepared_publication)
            return WorldRecoveryReport(
                (
                    WorldRecoveryAction(
                        code="operation_complete_blocked",
                        message="authority ref CAS failed during recovery",
                        world_oid=world_oid,
                        operation_id=operation_id,
                    ),
                )
            )
        manager.complete_publication(prepared_publication)
        manager.record_operation_published(operation_id, world_oid=world_oid)
    elif _authority_ref_protects_world(manager, target_ref, world_oid):
        if status in {"world_committed", "publishing"}:
            manager.cleanup_stale_publication_leases()
            manager.record_operation_published(operation_id, world_oid=world_oid)
    else:
        return WorldRecoveryReport(
            (
                WorldRecoveryAction(
                    code="operation_complete_blocked",
                    message="authority ref no longer equals operation input world",
                    world_oid=world_oid,
                    operation_id=operation_id,
                ),
            )
        )
    manager.close_operation_journal(
        operation_id,
        world_oid=world_oid,
    )
    return WorldRecoveryReport(
        (
            WorldRecoveryAction(
                code="operation_completed",
                message="completed publication for committed operation",
                world_oid=world_oid,
                operation_id=operation_id,
            ),
        )
    )


def complete_journaled_operation(manager: WorldStorageManager, operation_id: str) -> WorldRecoveryReport:
    """Complete publication for a journaled finalized, committed, publishing, or published operation."""
    return complete_committed_operation(manager, operation_id)


def _publication_plan_from_tip(tip: object) -> PublicationPlan:
    if not isinstance(tip, dict):
        raise InvalidRepositoryStateError("operation journal payload must be an object")
    raw_plan = tip.get("publication_plan")
    if not isinstance(raw_plan, dict):
        raise InvalidRepositoryStateError("operation journal publishing state requires publication_plan")
    plan = PublicationPlan.from_json(raw_plan)
    if tip.get("publication_plan_digest") != plan.digest():
        raise InvalidRepositoryStateError("operation journal publication_plan_digest disagrees with plan")
    return plan


def _commit_finalized_world(manager: WorldStorageManager, tip: object) -> str:
    if not isinstance(tip, dict):
        raise InvalidRepositoryStateError("operation journal payload must be an object")
    raw_prepared = tip.get("prepared_world_operation")
    if not isinstance(raw_prepared, dict):
        raise InvalidRepositoryStateError("operation journal finalized state requires prepared_world_operation")
    try:
        prepared = PreparedWorldOperation.from_json(raw_prepared)
    except (TypeError, ValueError) as exc:
        raise InvalidRepositoryStateError(str(exc)) from exc
    return manager.create_world_from_prepared(prepared)


def _cleanup_stale_open_actions(
    manager: WorldStorageManager,
    operation_id: str,
    *,
    terminal_family: str,
) -> tuple[WorldRecoveryAction, ...]:
    deleted = manager.cleanup_stale_terminal_operation_open_ref(operation_id, terminal_family=terminal_family)
    if not deleted:
        return ()
    return (
        WorldRecoveryAction(
            code="stale_open_journal_deleted",
            message="deleted stale open operation journal ref",
            operation_id=operation_id,
        ),
    )


def _current_ref_target(manager: WorldStorageManager, ref: str) -> str | None:
    try:
        return str(manager.world_store.repo.references[ref].target)
    except KeyError:
        return None


def _authority_ref_protects_world(manager: WorldStorageManager, ref: str, world_oid: str) -> bool:
    target = _current_ref_target(manager, ref)
    if target is None:
        return False
    if target == world_oid:
        return True
    try:
        return bool(manager.world_store.repo.descendant_of(pygit2.Oid(hex=target), pygit2.Oid(hex=world_oid)))
    except (TypeError, ValueError, pygit2.GitError):
        return False


def _orphan_cleanup_action(world_oid: str, ref: str) -> WorldRecoveryAction:
    if ref.startswith("refs/vcscore/retention/receipts/"):
        return WorldRecoveryAction(
            code="orphan_retention_receipt_deleted",
            message=f"deleted orphan retention receipt {ref}",
            world_oid=world_oid,
        )
    return WorldRecoveryAction(
        code="orphan_pin_deleted",
        message=f"deleted orphan selected-head pin {ref}",
        world_oid=world_oid,
    )


def _publication_lease_cleanup_action(ref: str) -> WorldRecoveryAction:
    return WorldRecoveryAction(
        code="stale_publication_lease_deleted",
        message=f"deleted stale publication lease {ref}",
    )


def _required_str(value: object, key: str) -> str:
    if not isinstance(value, dict):
        raise InvalidRepositoryStateError("operation journal payload must be an object")
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise InvalidRepositoryStateError(f"operation journal field {key!r} must be a non-empty string")
    return raw


def _nullable_str(value: object, key: str) -> str | None:
    if not isinstance(value, dict):
        raise InvalidRepositoryStateError("operation journal payload must be an object")
    if key not in value:
        raise InvalidRepositoryStateError(f"operation journal field {key!r} is required")
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise InvalidRepositoryStateError(f"operation journal field {key!r} must be null or a non-empty string")
    return raw
