"""The operation-journal state machine, extracted from ``WorldStorageManager``.

The journal transitions (``open`` → ``prepared`` → ``finalized`` →
``world_committed`` → ``publishing`` → ``published`` → terminal) plus the
open-journal accelerator index and per-journal fsck live here as a self-contained
collaborator. ``WorldStorageManager`` constructs one and delegates to it; every
former ``manager.<journal_method>`` remains a delegation shim so callers and
monkeypatch sites are unaffected (260704-1410-plan.md V2.1, standing rule 2/9).

Dependencies are one-directional and injected: ``operation_journal`` /
``world_store`` / ``stores`` are the shared substrate handles, and the two
validators are passed as callables so the controller never holds a back-reference
to the manager. Pure record/issue helpers live in the shared leaf module
``_world_storage_records`` (V2.2b / P4 rider 3), which both this controller and
``_world_storage_manager`` import — so there is no controller<->manager module
edge for these helpers. ``_world_storage_manager`` still imports this controller
lazily in ``__init__`` (the construction edge), which forms no import-time cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._incremental import OpenOperationJournalIndex, atomic_co_write
from vcs_core._world_refs import is_open_operation_journal_ref, operation_journal_ref
from vcs_core._world_storage_records import (
    OperationJournalFsckReport,
    _candidate_revision_to_json,
    _candidate_tuple_matches_head,
    _extend_candidate_ref_issues,
    _extend_final_evidence_issues,
    _issue,
    _operation_final_evidence_from_world,
    _optional_payload_str,
    _prepared_operation_from_json,
    _required_payload_str,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vcs_core._incremental import Health
    from vcs_core._substrate_store import SubstrateStore
    from vcs_core._world_operation_builder import (
        PreparedCandidateTupleRecord,
        PreparedWorldOperation,
    )
    from vcs_core._world_operation_journal import (
        OperationJournalEntry,
        OperationJournalHistory,
        OperationJournalStore,
        OperationJournalSummary,
    )
    from vcs_core._world_publication_plan import PublicationPlan
    from vcs_core._world_storage_records import OperationFinalEvidence
    from vcs_core._world_store import WorldStore
    from vcs_core._world_types import StructuredIssue, SubstrateHead


class OperationJournalController:
    """Owns the operation-journal lifecycle and its open-journal accelerator index."""

    def __init__(
        self,
        *,
        operation_journal: OperationJournalStore,
        world_store: WorldStore,
        stores: Mapping[str, SubstrateStore],
        validate_prepared_operation_admission: Callable[[PreparedWorldOperation], None],
        validate_publication_plan: Callable[..., None],
    ) -> None:
        self._operation_journal = operation_journal
        self._world_store = world_store
        self._stores = stores
        # Injected validators — coordinator-owned admission and publication-home plan
        # validation — passed as callables so the controller holds no manager back-ref.
        self._validate_prepared_operation_admission = validate_prepared_operation_admission
        self._validate_publication_plan = validate_publication_plan

    def open_operation_journal(
        self,
        *,
        operation_id: str,
        operation_kind: str,
        target_ref: str,
        input_world_oid: str | None,
        parent_operation_id: str | None = None,
        causal_links: Mapping[str, object] | None = None,
    ) -> OperationJournalEntry:
        store = self._operation_journal
        index = self._open_journal_index()
        open_ref = operation_journal_ref("open", operation_id)
        # Open the journal and add it to the open-journal index in ONE atomic transaction (the
        # create-open ref + the index add), under the store's in-process lock.
        with store.mutation_transaction():
            entry, authority_moves = store.prepare_open(
                operation_id=operation_id,
                operation_kind=operation_kind,
                target_ref=target_ref,
                input_world_oid=input_world_oid,
                parent_operation_id=parent_operation_id,
                causal_links=causal_links,
            )
            atomic_co_write(
                self._world_store.repo,
                authority_moves=authority_moves,
                prepare=lambda: index.prepare_add(open_ref),
            )
            return entry

    def _open_journal_index(self) -> OpenOperationJournalIndex:
        return OpenOperationJournalIndex(
            self._world_store.repo,
            self._world_store.world_store_id,
            rebuild_source=self._scan_open_operation_journal_refs,
        )

    def _scan_open_operation_journal_refs(self) -> frozenset[str]:
        """Live open operation-journal ref set — the index's rebuild oracle (O(total refs), off hot path)."""
        return frozenset(ref for ref in self._world_store.repo.references if is_open_operation_journal_ref(ref))

    def verify_open_operation_journal_index(self) -> Health:
        """Deep health (fsck only): is the open-journal accelerator consistent with the authority?

        ``fresh`` iff the live index reproduces the authoritative open-ref scan; ``missing``
        (fallback exists, not a blocker), ``corrupt``, or ``stale`` otherwise — the last being
        drift from an out-of-model writer that bypassed the co-write. Performs the authoritative
        full ref scan, so it must NOT run on the admission hot path. Never mutates.
        """
        return self._open_journal_index().verify_against_authority()

    def rebuild_open_operation_journal_index(self) -> None:
        """Rebuild the open-journal accelerator from the authoritative open refs (recovery self-heal).

        Reconciles a missing, corrupt, OR stale index (the stale case being out-of-model drift),
        mirroring :meth:`rebuild_active_lease_index`. The authority is unaffected.
        """
        self._open_journal_index().rebuild_from_durable_history()

    def read_open_operation_journal_index(self) -> frozenset[str] | None:
        """The indexed open-journal ref set, or ``None`` when the record is missing (caller falls back).

        The bounded admission read: one blob, never a ref-namespace scan. Raises
        :class:`InvalidRepositoryStateError` (fail closed) if the present record is corrupt, so
        admission surfaces a blocking fact rather than silently falling back to an authority scan.
        """
        return self._open_journal_index().read_open_refs()

    def open_operation_journal_index_corruption(self) -> str | None:
        """Cheap, index-only corruption check (one blob; **no** authoritative ref scan).

        Returns the corruption detail if the present index is unreadable/corrupt, else ``None``
        (missing, or present-and-valid). Mirrors :meth:`active_lease_index_corruption`; stale-vs-
        authority detection needs the full scan (:meth:`verify_open_operation_journal_index`).
        """
        try:
            self.read_open_operation_journal_index()
        except InvalidRepositoryStateError as exc:
            return str(exc)
        return None

    def record_operation_prepared(
        self,
        operation_id: str,
        *,
        prepared: PreparedWorldOperation,
    ) -> OperationJournalEntry:
        if prepared.operation_id != operation_id:
            raise InvalidRepositoryStateError("operation journal operation_id disagrees with prepared operation")
        try:
            prepared.require_candidate_tuples()
        except ValueError as exc:
            raise InvalidRepositoryStateError(str(exc)) from exc
        self._validate_prepared_operation_admission(prepared)
        selected = dict(prepared.selected or {})
        candidate_outcomes = tuple(
            outcome.to_json(final_operation_id=operation_id) for outcome in prepared.candidate_outcomes
        )
        prepared_json = prepared.to_json()
        _prepared_operation_from_json(prepared_json)
        return self._operation_journal.append(
            operation_id,
            status="prepared",
            updates={
                "candidate_refs": [_candidate_revision_to_json(candidate) for candidate in prepared.candidate_refs],
                "candidate_outcomes": [dict(outcome) for outcome in candidate_outcomes],
                "selected": selected,
                "prepared_world_operation": prepared_json,
                "prepared_world_operation_digest": prepared.prepared_operation_digest(),
            },
        )

    def record_operation_finalized(
        self,
        operation_id: str,
    ) -> OperationJournalEntry:
        prepared = self._prepared_operation_from_journal_tip(operation_id)
        if prepared is None:
            raise InvalidRepositoryStateError("operation finalization requires a prepared operation")
        finalized = prepared.finalize()
        if finalized.operation_id != operation_id:
            raise InvalidRepositoryStateError("operation journal operation_id disagrees with finalized operation")
        return self._operation_journal.append(
            operation_id,
            status="finalized",
            updates={
                "candidate_refs": [_candidate_revision_to_json(candidate) for candidate in finalized.candidate_refs],
                "candidate_commits": list(finalized.operation_final.payload["candidate_commits"]),
                "candidate_outcomes": list(finalized.candidate_outcome_payloads),
                "selected": dict(finalized.selected or {}),
                "snapshot": finalized.snapshot.to_json(),
                "snapshot_digest": finalized.snapshot_digest,
                "transition": dict(finalized.transition),
                "parents": list(finalized.parents),
                "operation_final": finalized.operation_final.payload,
                "operation_final_digest": finalized.operation_final_digest,
            },
        )

    def _prepared_operation_from_journal_tip(self, operation_id: str) -> PreparedWorldOperation | None:
        history = self.read_operation_journal(operation_id)
        prepared_value = history.tip.payload.get("prepared_world_operation")
        if prepared_value is None:
            return None
        if not isinstance(prepared_value, dict):
            raise InvalidRepositoryStateError("operation journal prepared_world_operation must be an object")
        try:
            return _prepared_operation_from_json(prepared_value)
        except (TypeError, ValueError) as exc:
            raise InvalidRepositoryStateError(str(exc)) from exc

    def _prepared_operation_from_any_journal_tip(self, operation_id: str) -> PreparedWorldOperation:
        last_error: Exception | None = None
        for family in ("closed", "open", "archived"):
            try:
                history = self.read_operation_journal(operation_id, family=family)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            prepared_value = history.tip.payload.get("prepared_world_operation")
            if not isinstance(prepared_value, dict):
                raise InvalidRepositoryStateError(
                    f"operation journal {operation_id!r} has no prepared_world_operation payload"
                )
            try:
                return _prepared_operation_from_json(prepared_value)
            except (TypeError, ValueError) as exc:
                raise InvalidRepositoryStateError(str(exc)) from exc
        raise InvalidRepositoryStateError(f"operation journal is missing for {operation_id!r}") from last_error

    def _candidate_tuple_for_selected_head(
        self,
        *,
        operation_id: str,
        producer_operation_id: str,
        candidate_id: str,
        head: SubstrateHead,
    ) -> PreparedCandidateTupleRecord:
        prepared = self._prepared_operation_from_any_journal_tip(operation_id)
        matches = tuple(
            candidate_tuple
            for candidate_tuple in prepared.candidate_tuples
            if _candidate_tuple_matches_head(
                candidate_tuple,
                head,
                producer_operation_id=producer_operation_id,
                candidate_id=candidate_id,
            )
        )
        if len(matches) != 1:
            raise InvalidRepositoryStateError(
                "operation "
                f"{operation_id!r} has no unique prepared candidate tuple for "
                f"{head.binding}@{head.store_id}/{head.resource_id}:{head.head}"
            )
        return matches[0]

    def record_operation_world_committed(
        self,
        operation_id: str,
        *,
        world_oid: str,
    ) -> OperationJournalEntry:
        evidence = self._operation_final_evidence(operation_id, world_oid)
        return self._operation_journal.append(
            operation_id,
            status="world_committed",
            updates={
                "world_oid": world_oid,
                "operation_final_digest": evidence.operation_final_digest,
                "selected": evidence.selected,
                "candidate_outcomes": list(evidence.candidate_outcomes),
            },
        )

    def record_operation_publishing(
        self,
        operation_id: str,
        *,
        world_oid: str,
        publication_plan: PublicationPlan,
    ) -> OperationJournalEntry:
        history = self.read_operation_journal(operation_id)
        tip = history.tip.payload
        self._validate_publication_plan(
            publication_plan,
            expected_world_oid=world_oid,
            expected_authority_ref=_required_payload_str(tip, "operation journal", "target_ref"),
            expected_input_world_oid=_optional_payload_str(tip, "operation journal", "input_world_oid"),
        )
        evidence = self._operation_final_evidence(operation_id, world_oid)
        updates: dict[str, object] = {
            "world_oid": world_oid,
            "operation_final_digest": evidence.operation_final_digest,
            "selected": evidence.selected,
            "candidate_outcomes": list(evidence.candidate_outcomes),
        }
        updates["publication_plan"] = publication_plan.to_json()
        updates["publication_plan_digest"] = publication_plan.digest()
        return self._operation_journal.append(
            operation_id,
            status="publishing",
            updates=updates,
        )

    def record_operation_published(
        self,
        operation_id: str,
        *,
        world_oid: str,
    ) -> OperationJournalEntry:
        evidence = self._operation_final_evidence(operation_id, world_oid)
        return self._operation_journal.append(
            operation_id,
            status="published",
            updates={
                "world_oid": world_oid,
                "operation_final_digest": evidence.operation_final_digest,
                "selected": evidence.selected,
                "candidate_outcomes": list(evidence.candidate_outcomes),
            },
        )

    def close_operation_journal(
        self,
        operation_id: str,
        *,
        world_oid: str,
    ) -> OperationJournalEntry:
        evidence = self._operation_final_evidence(operation_id, world_oid)
        return self._terminal_operation_journal(
            operation_id,
            family="closed",
            status="closed",
            updates={
                "selected": evidence.selected,
                "candidate_outcomes": list(evidence.candidate_outcomes),
                "world_oid": world_oid,
                "operation_final_digest": evidence.operation_final_digest,
            },
        )

    def fail_operation_journal(self, operation_id: str, *, error: str) -> OperationJournalEntry:
        # A non-terminal append retargets the open ref; it does NOT change open-set membership, so it
        # stays off the index co-write.
        return self._operation_journal.append(operation_id, status="failed", updates={"error": error})

    def archive_operation_journal(self, operation_id: str, *, error: str | None = None) -> OperationJournalEntry:
        updates = {} if error is None else {"error": error}
        return self._terminal_operation_journal(operation_id, family="archived", status="archived", updates=updates)

    def _terminal_operation_journal(
        self,
        operation_id: str,
        *,
        family: str,
        status: str,
        updates: Mapping[str, object],
    ) -> OperationJournalEntry:
        """Publish a terminal journal + tombstone its open-index entry in one atomic transaction.

        Create terminal + delete open + index tombstone, all-or-none, under the store lock.
        """
        store = self._operation_journal
        index = self._open_journal_index()
        open_ref = operation_journal_ref("open", operation_id)
        with store.mutation_transaction():
            entry, authority_moves = store.prepare_terminal(operation_id, family=family, status=status, updates=updates)
            atomic_co_write(
                self._world_store.repo,
                authority_moves=authority_moves,
                prepare=lambda: index.prepare_remove(open_ref),
            )
            return entry

    def cleanup_stale_terminal_operation_open_ref(self, operation_id: str, *, terminal_family: str) -> bool:
        """Delete a stale open ref + tombstone its open-index entry in one atomic transaction.

        The THIRD ``ops/open/*`` membership writer, on the co-write like open/terminal. For an
        out-of-model stale ref the index never indexed, ``prepare_remove`` is an idempotent no-op,
        so the batch atomically deletes just the open ref; for a co-written ref it also tombstones.
        """
        store = self._operation_journal
        index = self._open_journal_index()
        with store.mutation_transaction():
            open_ref, authority_moves = store.prepare_cleanup_stale_open_ref(
                operation_id, terminal_family=terminal_family
            )
            if open_ref is None:
                return False
            atomic_co_write(
                self._world_store.repo,
                authority_moves=authority_moves,
                prepare=lambda: index.prepare_remove(open_ref),
            )
            return True

    def read_operation_journal(self, operation_id: str, *, family: str = "open") -> OperationJournalHistory:
        return self._operation_journal.read(operation_id, family=family)

    def list_operation_journals(self, *, family: str | None = None) -> tuple[OperationJournalSummary, ...]:
        return self._operation_journal.list(family=family)

    def fsck_operation_journal(self, operation_id: str, *, family: str = "open") -> OperationJournalFsckReport:
        issues: list[StructuredIssue] = []
        try:
            history = self.read_operation_journal(operation_id, family=family)
        except (InvalidRepositoryStateError, KeyError, TypeError, ValueError) as exc:
            return OperationJournalFsckReport(
                operation_id=operation_id,
                issue_details=(
                    _issue(
                        "journal_read_failed",
                        str(exc),
                        operation_id=operation_id,
                        recovery_hint="Inspect or archive the operation journal before retrying.",
                    ),
                ),
            )
        tip = history.tip.payload
        _extend_candidate_ref_issues(issues, tip.get("candidate_refs", []), stores=self._stores)
        world_oid = tip.get("world_oid")
        if isinstance(world_oid, str):
            try:
                world = self._world_store.read_world_commit(world_oid)
            except (InvalidRepositoryStateError, KeyError, TypeError, ValueError) as exc:
                issues.append(_issue("journal_world_invalid", str(exc), operation_id=operation_id, world_oid=world_oid))
            else:
                _extend_final_evidence_issues(issues, tip, world, operation_id=operation_id)
        return OperationJournalFsckReport(operation_id=operation_id, issue_details=tuple(issues))

    def _operation_final_evidence(self, operation_id: str, world_oid: str) -> OperationFinalEvidence:
        world = self._world_store.read_world_commit(world_oid)
        evidence = _operation_final_evidence_from_world(world)
        if evidence.operation_id != operation_id:
            raise InvalidRepositoryStateError("operation journal operation_id disagrees with world operation-final")
        return evidence
