"""Private finalization helpers for bridge-era world authority transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._operation_journal_inventory import probe_operation_journal
from vcs_core._world_operation_runner import WorldOperationRunner
from vcs_core._world_recovery import archive_failed_operation, complete_journaled_operation

if TYPE_CHECKING:
    from collections.abc import Callable

    from vcs_core._world_operation_builder import PreparedWorldOperation
    from vcs_core._world_storage_manager import WorldStorageManager

MAX_AUTHORITY_RETRY_ATTEMPTS = 100


AuthorityOutcomeStatus = Literal["closed", "already_closed", "retry_required"]


@dataclass(frozen=True)
class AuthorityPublicationOutcome:
    """Terminal or retry result for one world-authority publication attempt."""

    operation_id: str
    status: AuthorityOutcomeStatus
    world_oid: str | None = None
    message: str | None = None


class WorldAuthorityFinalizer:
    """Centralize recovery and cleanup decisions for one world authority ref."""

    def __init__(self, manager: WorldStorageManager) -> None:
        self._manager = manager

    def publish_prepared(self, prepared: PreparedWorldOperation) -> AuthorityPublicationOutcome:
        """Publish a prepared operation and close recoverable post-publication states."""
        result = WorldOperationRunner(self._manager).publish_prepared_world(prepared)
        if result.status == "closed":
            if result.world_oid is None or not self.authority_ref_protects_world(prepared.target_ref, result.world_oid):
                raise InvalidRepositoryStateError(
                    f"Published world authority operation {result.operation_id!r} is not protected by target ref"
                )
            return AuthorityPublicationOutcome(
                operation_id=result.operation_id,
                status="closed",
                world_oid=result.world_oid,
            )
        if result.published and result.world_oid is not None:
            current_world_oid = self.current_world_oid(prepared.target_ref)
            if current_world_oid is not None and self.authority_ref_protects_world(
                prepared.target_ref,
                result.world_oid,
            ):
                outcome = self.complete_existing(
                    operation_id=result.operation_id,
                    target_ref=prepared.target_ref,
                    expected_input_world_oid=prepared.input_world_oid,
                )
                # complete_existing only returns None when missing_ok=True; we
                # call with the default missing_ok=False so a missing open
                # journal raises rather than returning None.
                if outcome is None:
                    raise InvalidRepositoryStateError(
                        f"complete_existing returned None for {result.operation_id!r} despite missing_ok=False"
                    )
                return outcome
        detail = f": {result.error}" if result.error else f": {result.status}"
        raise InvalidRepositoryStateError(
            f"Failed to publish v2 world authority operation {result.operation_id!r}{detail}"
        )

    def publish_or_recover(
        self,
        *,
        operation_id: str,
        prepared_factory: Callable[[str], PreparedWorldOperation],
        target_ref: str,
        expected_input_world_oid: str | None,
    ) -> AuthorityPublicationOutcome:
        """Complete an existing operation attempt chain, otherwise publish a new attempt."""
        saw_attempt = False
        next_retry_count = 0
        for retry_count in range(MAX_AUTHORITY_RETRY_ATTEMPTS + 1):
            attempt_id = operation_id if retry_count == 0 else self.retry_operation_id(operation_id, retry_count)
            existing = self.complete_existing(
                operation_id=attempt_id,
                target_ref=target_ref,
                expected_input_world_oid=expected_input_world_oid,
                missing_ok=True,
            )
            if existing is None:
                continue
            saw_attempt = True
            if existing.status != "retry_required":
                self.require_terminal_authority(existing, target_ref=target_ref)
                return existing
            next_retry_count = max(next_retry_count, retry_count + 1)

        if not saw_attempt:
            return self.publish_prepared(prepared_factory(operation_id))
        if next_retry_count > MAX_AUTHORITY_RETRY_ATTEMPTS:
            raise InvalidRepositoryStateError(
                f"Cannot publish world authority operation {operation_id!r}: retry limit exceeded"
            )
        next_operation_id = self.retry_operation_id(operation_id, next_retry_count)
        return self.publish_prepared(prepared_factory(next_operation_id))

    def complete_existing(
        self,
        *,
        operation_id: str,
        target_ref: str,
        expected_input_world_oid: str | None,
        missing_ok: bool = False,
    ) -> AuthorityPublicationOutcome | None:
        """Complete or classify one existing operation journal."""
        closed = self._read_journal_tip(operation_id, family="closed")
        if closed is not None:
            self._validate_authority_identity(closed, target_ref, expected_input_world_oid)
            return AuthorityPublicationOutcome(
                operation_id=operation_id,
                status="already_closed",
                world_oid=_optional_str(closed.get("world_oid")),
            )
        archived = self._read_journal_tip(operation_id, family="archived")
        if archived is not None:
            self._validate_authority_identity(archived, target_ref, expected_input_world_oid)
            return AuthorityPublicationOutcome(
                operation_id=operation_id,
                status="retry_required",
                world_oid=_optional_str(archived.get("world_oid")),
                message="operation journal is archived",
            )
        opened = self._read_journal_tip(operation_id, family="open")
        if opened is None:
            if missing_ok:
                return None
            raise InvalidRepositoryStateError(f"operation journal ref is missing for {operation_id!r}")
        self._validate_authority_identity(opened, target_ref, expected_input_world_oid)

        status = opened.get("status")
        if status in {"finalized", "world_committed", "publishing", "published"}:
            report = complete_journaled_operation(self._manager, operation_id)
            if not report.ok:
                message = "; ".join(action.message for action in report.actions) or "operation recovery blocked"
                raise InvalidRepositoryStateError(
                    f"Cannot complete world authority operation {operation_id!r}: {message}"
                )
            closed_after = self._read_journal_tip(operation_id, family="closed")
            world_oid = _optional_str(closed_after.get("world_oid")) if closed_after is not None else None
            return AuthorityPublicationOutcome(operation_id=operation_id, status="closed", world_oid=world_oid)
        if status in {"opened", "prepared"}:
            self._manager.fail_operation_journal(operation_id, error="operation superseded by authority recovery")
            archive_failed_operation(self._manager, operation_id)
            return AuthorityPublicationOutcome(
                operation_id=operation_id,
                status="retry_required",
                message=f"operation journal status {status!r} was archived for retry",
            )
        if status == "failed":
            archive_failed_operation(self._manager, operation_id)
            return AuthorityPublicationOutcome(
                operation_id=operation_id,
                status="retry_required",
                world_oid=_optional_str(opened.get("world_oid")),
                message="operation journal is failed",
            )
        raise InvalidRepositoryStateError(
            f"Cannot complete world authority operation {operation_id!r}: unsupported status {status!r}"
        )

    def current_world_oid(self, ref: str) -> str | None:
        if ref not in self._manager.world_store.repo.references:
            return None
        return str(self._manager.world_store.repo.references[ref].target)

    def authority_ref_protects_world(self, ref: str, world_oid: str) -> bool:
        current = self.current_world_oid(ref)
        if current is None:
            return False
        return self._world_reaches(current, world_oid)

    def _world_reaches(self, start_oid: str, target_oid: str) -> bool:
        pending = [start_oid]
        seen: set[str] = set()
        while pending:
            oid = pending.pop()
            if oid == target_oid:
                return True
            if oid in seen:
                continue
            seen.add(oid)
            try:
                world = self._manager.read_world(oid)
            except Exception as exc:
                raise InvalidRepositoryStateError(f"Cannot inspect world ancestor {oid!r}") from exc
            pending.extend(parent for parent in world.parent_oids if parent not in seen)
        return False

    def _read_journal_tip(self, operation_id: str, *, family: str) -> dict[str, object] | None:
        item = probe_operation_journal(self._manager.world_store.repo, operation_id, family=family)
        if item.health.presence == "absent":
            return None
        if item.health.validity != "valid":
            issue_codes = ", ".join(item.health.issue_codes) or item.health.primary_issue
            raise InvalidRepositoryStateError(f"operation journal inventory item {item.id!r} is invalid: {issue_codes}")
        return dict(item.fields)

    def require_terminal_authority(self, outcome: AuthorityPublicationOutcome, *, target_ref: str) -> None:
        """Require a terminal outcome to still be protected by its target authority ref."""
        if outcome.world_oid is None:
            raise InvalidRepositoryStateError(
                f"World authority operation {outcome.operation_id!r} closed without a world_oid"
            )
        if not self.authority_ref_protects_world(target_ref, outcome.world_oid):
            raise InvalidRepositoryStateError(
                f"World authority operation {outcome.operation_id!r} is not protected by target ref"
            )

    @staticmethod
    def _validate_authority_identity(
        payload: dict[str, object],
        target_ref: str,
        expected_input_world_oid: str | None,
    ) -> None:
        if payload.get("target_ref") != target_ref:
            raise InvalidRepositoryStateError("operation journal target_ref disagrees with authority finalizer")
        if payload.get("input_world_oid") != expected_input_world_oid:
            raise InvalidRepositoryStateError("operation journal input_world_oid disagrees with authority finalizer")

    @staticmethod
    def retry_operation_id(operation_id: str, retry_count: int) -> str:
        return f"{operation_id}_retry_{retry_count}"


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
