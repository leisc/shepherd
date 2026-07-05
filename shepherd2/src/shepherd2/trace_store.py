"""SQLite-backed canonical trace store.

The store persists canonical trace envelopes directly. It does not backfill or
infer retained identity while reading.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

from .kernel.canonical import (
    ROOT_WITNESS_REF,
    ROOT_WITNESS_SCHEMA_REF,
    WITNESS_SCHEMA_REF,
    record_digest,
    root_witness_body,
    root_witness_record_id,
    validate_witness_body,
)
from .kernel.facts import (
    AppendBatch,
    AppendContext,
    AppendGroup,
    AppendIntentId,
    AppendLocalId,
    AppendReceipt,
    ClosurePolicy,
    CommitReceipt,
    ContextAnchor,
    ContextId,
    ExternalAnchor,
    Fact,
    FactBody,
    FactDraft,
    FactEnvelope,
    FactId,
    FactShape,
    FactView,
    FrontierId,
    ModeFilter,
    OperationContext,
    OperationKind,
    OwnerCutoff,
    OwnerCutoffSpec,
    OwnerOrdinal,
    ReadContext,
    RetainedContext,
    RetainedContextDraft,
    TraceOwnerId,
    TraceSlice,
    VisibilityProfile,
    VisibleFact,
    VisibleRecord,
    WitnessAnchor,
    WitnessBody,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TraceStoreError(RuntimeError):
    """Base error for trace store law violations."""


class AppendIntentConflictError(TraceStoreError):
    """Raised when an append intent is retried with a different batch."""


class UnknownFactError(TraceStoreError):
    """Raised when a retained fact references a missing fact."""


AppendIntentConflict = AppendIntentConflictError
UnknownFact = UnknownFactError

WITNESS_TRACE_OWNER_ID = "kernel:witness"


@dataclass(frozen=True)
class _WitnessPlan:
    record_id: FactId
    schema_ref: str
    kind_label: str
    body: dict[str, Any]
    witness_ref: FactId


@dataclass(frozen=True)
class _PathEntry:
    path_ref: TraceOwnerId
    path_ordinal: OwnerOrdinal
    record_id: FactId


class SQLiteTraceStore:
    """Canonical TraceStore implementation backed by SQLite."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def append(
        self,
        append_context: AppendContext | OperationContext,
        batch: AppendBatch,
    ) -> AppendReceipt:
        """Append a semantic batch or return the prior receipt for its intent."""
        operation_context = _coerce_write_context(append_context, "append")
        _ensure_append_authorized(operation_context)
        with self._lock:
            self._begin()
            try:
                receipt = self._append_in_tx(operation_context, batch)
            except Exception:
                self._db.rollback()
                raise
            self._db.commit()
            return receipt

    def preview_record_ids(
        self,
        append_context: AppendContext | OperationContext,
        batch: AppendBatch,
    ) -> tuple[FactId, ...]:
        """Return the durable record ids this batch would allocate if committed."""
        operation_context = _coerce_write_context(append_context, "append")
        _ensure_append_authorized(operation_context)
        _validate_batch_shape(batch, operation_context)
        return self._preview_fact_ids(batch, operation_context)

    def preview_fact_ids(
        self,
        append_context: AppendContext | OperationContext,
        batch: AppendBatch,
    ) -> tuple[FactId, ...]:
        """Compatibility alias for context-explicit record id preview."""
        return self.preview_record_ids(append_context, batch)

    def read_fact(
        self,
        read_context: ReadContext | OperationContext,
        fact_id: FactId,
        visibility: VisibilityProfile | None = None,
    ) -> VisibleFact | ExternalAnchor:
        """Read one retained fact."""
        context = _coerce_read_context(read_context, visibility)
        _ensure_read_authorized(context)
        return _visible_fact(self._read_fact(fact_id), context.visibility_profile)

    def read_context(self, context_id: ContextId) -> RetainedContext:
        """Read one retained context for tests and diagnostics."""
        return self._read_context(context_id)

    def read_owner_prefix(
        self,
        read_context: ReadContext | OperationContext,
        trace_owner_id: TraceOwnerId,
        through: OwnerOrdinal,
        mode_filter: ModeFilter = "both",
    ) -> TraceSlice:
        read_context = _coerce_read_context(read_context, None)
        _ensure_read_authorized(read_context)
        _ensure_mode_filter(mode_filter)
        rows = self._db.execute(
            """
            SELECT record_id, path_ref, path_ordinal FROM path_entries
            WHERE path_ref = ? AND path_ordinal <= ?
            ORDER BY path_ordinal ASC
            """,
            (trace_owner_id, through),
        ).fetchall()
        return self._trace_slice(
            path_entries=tuple(_path_entry_from_row(row) for row in rows),
            frontier=None,
            visibility=read_context.visibility_profile,
            mode_filter=mode_filter,
        )

    def read_path_prefix(
        self,
        read_context: ReadContext | OperationContext,
        trace_owner_id: TraceOwnerId,
        through: OwnerOrdinal,
        mode_filter: ModeFilter = "both",
    ) -> TraceSlice:
        return self.read_owner_prefix(read_context, trace_owner_id, through, mode_filter)

    def publish_frontier(self, append_context: AppendContext | OperationContext, spec: OwnerCutoffSpec) -> OwnerCutoff:
        """Publish a retained owner-prefix frontier through the append path."""
        operation_context = _coerce_write_context(append_context, "publish_cut")
        _ensure_append_authorized(operation_context)
        with self._lock:
            self._begin()
            try:
                frontier = self._publish_owner_cutoff_in_tx(operation_context, spec)
            except Exception:
                self._db.rollback()
                raise
            self._db.commit()
            return frontier

    def publish_cut(self, append_context: AppendContext | OperationContext, spec: OwnerCutoffSpec) -> OwnerCutoff:
        """Publish a retained owner-prefix cut through the append path."""
        return self.publish_frontier(append_context, spec)

    def resolve_frontier(
        self,
        read_context: ReadContext | OperationContext,
        frontier_id: FrontierId,
        visibility: VisibilityProfile | None = None,
        mode_filter: ModeFilter = "both",
    ) -> TraceSlice:
        """Resolve a frontier into a graph-shaped trace slice."""
        context = _coerce_read_context(read_context, visibility)
        _ensure_read_authorized(context)
        _ensure_mode_filter(mode_filter)
        frontier = self.read_owner_cutoff(frontier_id)
        through = self._read_fact_at_path(
            frontier.through_fact_id,
            frontier.target_trace_owner_id,
            frontier.through_owner_ordinal,
        )
        if through.trace_owner_id != frontier.target_trace_owner_id:
            raise TraceStoreError("frontier through fact owner disagrees with target trace owner")
        if through.owner_ordinal != frontier.through_owner_ordinal:
            raise TraceStoreError("frontier through fact ordinal changed")

        rows = self._db.execute(
            """
            SELECT record_id, path_ref, path_ordinal FROM path_entries
            WHERE path_ref = ? AND path_ordinal <= ?
            ORDER BY path_ordinal ASC
            """,
            (frontier.target_trace_owner_id, frontier.through_owner_ordinal),
        ).fetchall()
        return self._trace_slice(
            path_entries=tuple(_path_entry_from_row(row) for row in rows),
            frontier=frontier,
            visibility=context.visibility_profile,
            mode_filter=mode_filter,
        )

    def resolve_cut(
        self,
        read_context: ReadContext | OperationContext,
        cut_id: FrontierId,
        visibility: VisibilityProfile | None = None,
        mode_filter: ModeFilter = "both",
    ) -> TraceSlice:
        """Resolve a cut into a graph-shaped trace slice."""
        return self.resolve_frontier(read_context, cut_id, visibility, mode_filter)

    def read_owner_cutoff(self, frontier_id: FrontierId) -> OwnerCutoff:
        return self._verified_frontier_truth(frontier_id)

    def read_causal_closure(
        self,
        read_context: ReadContext | OperationContext,
        roots: tuple[FactId, ...],
        *,
        visibility: VisibilityProfile | None = None,
        mode_filter: ModeFilter = "both",
        closure_policy: ClosurePolicy = "include_external_anchors",
    ) -> TraceSlice:
        """Read the causal closure for one or more root facts."""
        context = _coerce_read_context(read_context, visibility)
        _ensure_read_authorized(context)
        _ensure_mode_filter(mode_filter)
        _ensure_closure_policy(closure_policy)

        pending = list(roots)
        seen: set[FactId] = set()
        while pending:
            fact_id = pending.pop()
            if fact_id in seen:
                continue
            seen.add(fact_id)
            fact = self._read_fact(fact_id)
            pending.extend(parent for parent in fact.envelope.caused_by_fact_ids if parent not in seen)

        return self._trace_slice(
            path_entries=self._canonical_fact_order(seen),
            frontier=None,
            visibility=context.visibility_profile,
            mode_filter=mode_filter,
            include_external_anchors=closure_policy == "include_external_anchors",
        )

    def fact_count(self) -> int:
        """Return retained fact count for tests and diagnostics."""
        return int(self._db.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def context_count(self) -> int:
        """Return retained context count for tests and diagnostics."""
        return int(self._db.execute("SELECT COUNT(*) FROM contexts").fetchone()[0])

    def _append_in_tx(self, append_context: AppendContext | OperationContext, batch: AppendBatch) -> AppendReceipt:
        _validate_batch_shape(batch, append_context)
        batch_digest = _batch_digest(batch, append_context)

        row = self._db.execute(
            "SELECT batch_digest, receipt_json FROM append_intents WHERE append_intent_id = ?",
            (batch.append_intent_id,),
        ).fetchone()
        if row is not None:
            if row["batch_digest"] != batch_digest:
                raise AppendIntentConflictError(
                    f"append intent {batch.append_intent_id!r} was already committed with different content"
                )
            return _receipt_from_json(row["receipt_json"])

        contexts, witness_plans, facts, receipt = self._prepare_append(batch, append_context)
        self._db.execute(
            """
            INSERT INTO append_intents(append_intent_id, batch_digest, receipt_json)
            VALUES (?, ?, ?)
            """,
            (batch.append_intent_id, batch_digest, _receipt_to_json(receipt)),
        )
        for context in contexts:
            self._insert_context(context, batch.append_intent_id)
        for witness_plan in witness_plans:
            self._insert_witness_record_if_missing(witness_plan, batch.append_intent_id)
        for fact, commit_receipt in zip(facts, receipt.commit_receipts, strict=True):
            self._insert_fact_row(fact, commit_receipt, batch.append_intent_id)

        for owner, (_start, end) in receipt.owner_ordinal_ranges.items():
            self._db.execute(
                """
                INSERT INTO owner_ordinals(trace_owner_id, next_ordinal)
                VALUES (?, ?)
                ON CONFLICT(trace_owner_id)
                DO UPDATE SET next_ordinal = excluded.next_ordinal
                """,
                (owner, end + 1),
            )

        if receipt.commit_receipts:
            self._set_next_commit_seq(_next_commit_seq(receipt.commit_receipts))
        return receipt

    def _publish_owner_cutoff_in_tx(
        self,
        append_context: AppendContext | OperationContext,
        spec: OwnerCutoffSpec,
    ) -> OwnerCutoff:
        if not spec.frontier_id:
            raise ValueError("frontier_id is required")
        through = self._read_latest_fact_on_path(spec.through_fact_id, spec.target_trace_owner_id)
        if through.trace_owner_id != spec.target_trace_owner_id:
            raise TraceStoreError("owner cutoff target disagrees with through fact owner")

        publisher = spec.publisher_trace_owner_id or spec.target_trace_owner_id
        through_ordinal = through.owner_ordinal
        intent = spec.append_intent_id or f"frontier:{spec.frontier_id}"
        payload = {
            "frontier_id": spec.frontier_id,
            "target_trace_owner_id": spec.target_trace_owner_id,
            "through_fact_id": spec.through_fact_id,
            "through_owner_ordinal": through_ordinal,
            "publisher_trace_owner_id": publisher,
        }
        causal = (spec.through_fact_id, *spec.caused_by)
        retained_context = RetainedContextDraft(
            capability_witness_refs=tuple(append_context.presented_witness_refs),
            semantic_environment_refs=(append_context.schema_version_set,),
        )
        batch = AppendBatch(
            append_intent_id=intent,
            groups=(
                AppendGroup(
                    trace_owner_id=publisher,
                    retained_context=retained_context,
                    causal_parents=causal,
                    fact_drafts=(
                        FactDraft(
                            mode="capture",
                            schema_ref="shepherd2.frontier.owner_cutoff.v1",
                            kind_label="frontier_published",
                            payload=payload,
                        ),
                    ),
                ),
            ),
        )
        receipt = self._append_in_tx(append_context, batch)
        created_by_fact_id = receipt.fact_ids[0]
        frontier = OwnerCutoff(
            frontier_id=spec.frontier_id,
            target_trace_owner_id=spec.target_trace_owner_id,
            through_fact_id=spec.through_fact_id,
            through_owner_ordinal=through_ordinal,
            publisher_trace_owner_id=publisher,
            created_by_fact_id=created_by_fact_id,
        )
        existing = self._db.execute("SELECT * FROM frontiers WHERE frontier_id = ?", (spec.frontier_id,)).fetchone()
        if existing is not None:
            existing_frontier = self.read_owner_cutoff(spec.frontier_id)
            if existing_frontier != frontier:
                raise TraceStoreError(f"frontier id {spec.frontier_id!r} already names a different cutoff")
            return existing_frontier

        self._db.execute(
            """
            INSERT INTO frontiers(
                frontier_id,
                target_trace_owner_id,
                through_fact_id,
                through_owner_ordinal,
                publisher_trace_owner_id,
                created_by_fact_id,
                append_intent_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frontier.frontier_id,
                frontier.target_trace_owner_id,
                frontier.through_fact_id,
                frontier.through_owner_ordinal,
                frontier.publisher_trace_owner_id,
                frontier.created_by_fact_id,
                intent,
            ),
        )
        return frontier

    def _prepare_append(
        self,
        batch: AppendBatch,
        append_context: AppendContext | OperationContext,
    ) -> tuple[tuple[RetainedContext, ...], tuple[_WitnessPlan, ...], tuple[Fact, ...], AppendReceipt]:
        staged_ids: set[FactId] = set()
        local_fact_ids: dict[AppendLocalId, FactId] = {}
        staged_next: dict[TraceOwnerId, int] = {}
        for group in batch.groups:
            owner = _group_owner(group)
            if owner not in staged_next:
                staged_next[owner] = self._next_owner_ordinal(owner)

        contexts: list[RetainedContext] = []
        witness_plans: dict[FactId, _WitnessPlan] = {}
        facts: list[Fact] = []
        commit_receipts: list[CommitReceipt] = []
        owner_ranges: dict[TraceOwnerId, tuple[OwnerOrdinal, OwnerOrdinal]] = {}
        causal_edges: list[tuple[FactId, FactId]] = []
        context_receipts: list[ContextId] = []
        next_commit = self._next_commit_seq()

        for group_index, group in enumerate(batch.groups):
            owner = _group_owner(group)
            drafts = _group_drafts(group)
            if not drafts:
                continue
            context = self._resolve_group_context(batch.append_intent_id, group_index, group, append_context)
            if not self._context_exists(context.context_id):
                contexts.append(context)
            context_receipts.append(context.context_id)
            witness_plan = _ordinary_witness_plan(context, append_context)
            witness_plans.setdefault(root_witness_record_id(), _root_witness_plan())
            witness_plans.setdefault(witness_plan.record_id, witness_plan)
            start = staged_next[owner]
            ordinal = start
            for draft in drafts:
                caused_by = _resolved_causes(group, draft, local_fact_ids)
                self._validate_causal_parents(caused_by, staged_ids=staged_ids)
                commit_receipt = f"commit:{next_commit + len(facts)}"
                schema_ref = _resolve_schema_ref(draft, append_context)
                fact_id = record_digest(
                    schema_ref=schema_ref,
                    mode=draft.mode,
                    body=draft.payload,
                    caused_by=caused_by,
                    witness=witness_plan.record_id,
                )
                if draft.append_local_id is not None:
                    local_fact_ids[draft.append_local_id] = fact_id
                envelope = FactEnvelope(
                    record_id=fact_id,
                    digest=fact_id,
                    schema_ref=schema_ref,
                    mode=draft.mode,
                    witness_ref=witness_plan.record_id,
                    caused_by_record_ids=caused_by,
                )
                view = FactView(
                    trace_owner_id=owner,
                    owner_ordinal=ordinal,
                    retained_context_ref=context.context_id,
                    kind_label=draft.kind_label or schema_ref,
                )
                facts.append(Fact(envelope=envelope, body=FactBody(payload=dict(draft.payload)), view=view))
                commit_receipts.append(commit_receipt)
                causal_edges.extend((parent, fact_id) for parent in caused_by)
                staged_ids.add(fact_id)
                ordinal += 1
            staged_next[owner] = ordinal
            existing_range = owner_ranges.get(owner)
            range_start = start if existing_range is None else existing_range[0]
            owner_ranges[owner] = (range_start, ordinal - 1)

        receipt = AppendReceipt(
            append_intent_id=batch.append_intent_id,
            fact_ids=tuple(fact.envelope.fact_id for fact in facts),
            commit_receipts=tuple(commit_receipts),
            owner_ordinal_ranges=owner_ranges,
            causal_edges=tuple(causal_edges),
            context_receipts=tuple(context_receipts),
        )
        return tuple(contexts), tuple(witness_plans.values()), tuple(facts), receipt

    def _preview_fact_ids(
        self, batch: AppendBatch, append_context: AppendContext | OperationContext
    ) -> tuple[FactId, ...]:
        local_fact_ids: dict[AppendLocalId, FactId] = {}
        fact_ids: list[FactId] = []
        for group_index, group in enumerate(batch.groups):
            drafts = _group_drafts(group)
            if not drafts:
                continue
            context = self._resolve_group_context(batch.append_intent_id, group_index, group, append_context)
            witness_plan = _ordinary_witness_plan(context, append_context)
            for draft in drafts:
                caused_by = _resolved_causes(group, draft, local_fact_ids)
                fact_id = record_digest(
                    schema_ref=_resolve_schema_ref(draft, append_context),
                    mode=draft.mode,
                    body=draft.payload,
                    caused_by=caused_by,
                    witness=witness_plan.record_id,
                )
                if draft.append_local_id is not None:
                    local_fact_ids[draft.append_local_id] = fact_id
                fact_ids.append(fact_id)
        return tuple(fact_ids)

    def _resolve_group_context(
        self,
        append_intent_id: AppendIntentId,
        group_index: int,
        group: AppendGroup,
        append_context: AppendContext | OperationContext,
    ) -> RetainedContext:
        draft = _context_draft_for_group(group)
        if draft.reuse_context_id is not None:
            return self._read_context(draft.reuse_context_id)
        payload = _context_payload(draft, append_context)
        context_id = _context_id(append_intent_id, group_index, payload)
        existing = self._db.execute("SELECT * FROM contexts WHERE context_id = ?", (context_id,)).fetchone()
        context = RetainedContext(context_id=context_id, **payload)
        if existing is not None and self._context_from_row(existing) != context:
            raise TraceStoreError(f"context id {context_id!r} already names a different context")
        return context

    def _validate_causal_parents(self, caused_by: Iterable[FactId], *, staged_ids: set[FactId]) -> None:
        for fact_id in caused_by:
            if fact_id not in staged_ids and not self._fact_exists(fact_id):
                raise UnknownFactError(f"causal parent does not exist: {fact_id}")

    def _witness_support_closure(self, selected_records: tuple[Fact, ...]) -> tuple[Fact, ...]:
        support_by_id: dict[FactId, Fact] = {}
        validated_to_root: set[FactId] = set()
        for record in selected_records:
            if not record.envelope.witness_ref and record.envelope.schema_ref != ROOT_WITNESS_SCHEMA_REF:
                raise TraceStoreError(f"non-root record has empty witness_ref: {record.envelope.record_id}")

        for record in selected_records:
            if record.envelope.witness_ref:
                self._validate_witness_chain(record.envelope.witness_ref, support_by_id, validated_to_root)
        return tuple(support_by_id.values())

    def _validate_witness_chain(
        self,
        start_ref: FactId,
        support_by_id: dict[FactId, Fact],
        validated_to_root: set[FactId],
    ) -> None:
        seen_in_chain: set[FactId] = set()
        witness_ref = start_ref
        while True:
            if witness_ref in validated_to_root:
                return
            if witness_ref in seen_in_chain:
                raise TraceStoreError(f"witness chain cycle before root witness: {witness_ref}")
            seen_in_chain.add(witness_ref)

            witness = support_by_id.get(witness_ref)
            if witness is None:
                witness = self._read_fact(witness_ref)
                if witness.envelope.schema_ref not in {ROOT_WITNESS_SCHEMA_REF, WITNESS_SCHEMA_REF}:
                    raise TraceStoreError(f"witness_ref does not resolve to a witness record: {witness_ref}")
                support_by_id[witness_ref] = witness

            if witness.envelope.schema_ref not in {ROOT_WITNESS_SCHEMA_REF, WITNESS_SCHEMA_REF}:
                raise TraceStoreError(f"witness_ref does not resolve to a witness record: {witness_ref}")
            if witness.envelope.schema_ref == ROOT_WITNESS_SCHEMA_REF:
                if witness.envelope.witness_ref != ROOT_WITNESS_REF:
                    raise TraceStoreError("root witness record must use the empty witness sentinel")
                validated_to_root.update(seen_in_chain)
                return
            if not witness.envelope.witness_ref:
                raise TraceStoreError(f"non-root witness has empty witness_ref: {witness_ref}")
            witness_ref = witness.envelope.witness_ref

    def _trace_slice(
        self,
        *,
        path_entries: tuple[_PathEntry, ...],
        frontier: OwnerCutoff | None,
        visibility: VisibilityProfile,
        mode_filter: ModeFilter,
        include_external_anchors: bool = True,
    ) -> TraceSlice:
        loaded = tuple(
            (entry, fact)
            for entry in path_entries
            if _mode_matches(
                (fact := self._read_fact_at_path(entry.record_id, entry.path_ref, entry.path_ordinal)), mode_filter
            )
        )
        selected = {entry.record_id for entry, _fact in loaded}
        facts_by_id: dict[FactId, VisibleFact] = {}
        contexts_by_id: dict[ContextId, RetainedContext] = {}
        owner_paths: dict[TraceOwnerId, list[FactId]] = {}
        causal_edges: list[tuple[FactId, FactId]] = []
        external_anchors: dict[FactId, ExternalAnchor] = {}
        context_anchors: dict[ContextId, ContextAnchor] = {}
        visible_witnesses_by_id: dict[FactId, VisibleRecord] = {}
        witness_anchors: dict[FactId, WitnessAnchor] = {}

        for entry, fact in loaded:
            fact_id = entry.record_id
            visible_fact = _visible_fact(fact, visibility)
            if isinstance(visible_fact, Fact | FactShape):
                facts_by_id[fact_id] = visible_fact
            owner_paths.setdefault(entry.path_ref, []).append(fact_id)
            for parent in fact.envelope.caused_by_fact_ids:
                if parent in selected:
                    causal_edges.append((parent, fact_id))
                elif include_external_anchors:
                    external_anchors.setdefault(parent, self._anchor_for_fact(parent, "outside_frontier"))
            context_id = fact.retained_context_ref
            if visibility == "shape_only":
                context_anchors.setdefault(
                    context_id,
                    ContextAnchor(
                        context_id=context_id,
                        visible_shape={"context_id": context_id},
                    ),
                )
            elif context_id:
                contexts_by_id[context_id] = self._read_context(context_id)

        for witness in self._witness_support_closure(tuple(fact for _entry, fact in loaded)):
            witness_ref = witness.envelope.record_id
            visible_witness = _visible_fact(witness, visibility)
            if visibility == "shape_only":
                witness_anchors.setdefault(witness_ref, _witness_anchor(witness))
            elif isinstance(visible_witness, Fact | FactShape):
                visible_witnesses_by_id[witness_ref] = visible_witness

        return TraceSlice(
            frontier=frontier,
            visibility_profile=visibility,
            mode_filter=mode_filter,
            facts_by_id=facts_by_id,
            contexts_by_id=contexts_by_id,
            owner_paths={owner: tuple(path) for owner, path in owner_paths.items()},
            causal_edges=tuple(causal_edges),
            external_anchors=tuple(external_anchors.values()),
            context_anchors=tuple(context_anchors.values()),
            visible_witnesses_by_id=visible_witnesses_by_id,
            witness_anchors=tuple(witness_anchors.values()),
        )

    def _anchor_for_fact(self, fact_id: FactId, hidden_reason: str) -> ExternalAnchor:
        try:
            fact = self._read_fact(fact_id)
        except UnknownFactError:
            return ExternalAnchor(ref=fact_id, hidden_reason="unknown")
        return ExternalAnchor(
            ref=fact_id,
            visible_shape={
                "kind_label": fact.fact_kind,
                "schema_ref": fact.envelope.schema_ref,
                "trace_owner_id": fact.trace_owner_id,
                "owner_ordinal": fact.owner_ordinal,
                "witness_ref": fact.envelope.witness_ref,
            },
            hidden_reason=hidden_reason,
        )

    def _read_fact(self, fact_id: FactId) -> Fact:
        row = self._db.execute(
            """
            SELECT
                records.*,
                path_entries.path_ref AS trace_owner_id,
                path_entries.path_ordinal AS owner_ordinal,
                path_entries.retained_context_ref AS retained_context_ref,
                path_entries.kind_label AS kind_label
            FROM records
            JOIN path_entries ON path_entries.record_id = records.record_id
            WHERE records.record_id = ?
            ORDER BY path_entries.path_ref ASC, path_entries.path_ordinal ASC
            LIMIT 1
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            raise UnknownFactError(f"unknown fact id: {fact_id}")
        return _node_from_row(row)

    def _read_fact_at_path(
        self,
        fact_id: FactId,
        trace_owner_id: TraceOwnerId,
        owner_ordinal: OwnerOrdinal,
    ) -> Fact:
        row = self._db.execute(
            """
            SELECT
                records.*,
                path_entries.path_ref AS trace_owner_id,
                path_entries.path_ordinal AS owner_ordinal,
                path_entries.retained_context_ref AS retained_context_ref,
                path_entries.kind_label AS kind_label
            FROM path_entries
            JOIN records ON records.record_id = path_entries.record_id
            WHERE path_entries.record_id = ?
              AND path_entries.path_ref = ?
              AND path_entries.path_ordinal = ?
            """,
            (fact_id, trace_owner_id, owner_ordinal),
        ).fetchone()
        if row is None:
            raise UnknownFactError(f"unknown fact path entry: {fact_id}")
        return _node_from_row(row)

    def _read_latest_fact_on_path(self, fact_id: FactId, trace_owner_id: TraceOwnerId) -> Fact:
        row = self._db.execute(
            """
            SELECT
                records.*,
                path_entries.path_ref AS trace_owner_id,
                path_entries.path_ordinal AS owner_ordinal,
                path_entries.retained_context_ref AS retained_context_ref,
                path_entries.kind_label AS kind_label
            FROM path_entries
            JOIN records ON records.record_id = path_entries.record_id
            WHERE path_entries.record_id = ?
              AND path_entries.path_ref = ?
            ORDER BY path_entries.path_ordinal DESC
            LIMIT 1
            """,
            (fact_id, trace_owner_id),
        ).fetchone()
        if row is None:
            raise UnknownFactError(f"unknown fact path entry: {fact_id}")
        return _node_from_row(row)

    def _read_record(self, fact_id: FactId) -> Fact:
        row = self._db.execute(
            """
            SELECT
                records.*,
                '' AS trace_owner_id,
                -1 AS owner_ordinal,
                '' AS retained_context_ref,
                '' AS kind_label
            FROM records
            WHERE records.record_id = ?
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            raise UnknownFactError(f"unknown fact id: {fact_id}")
        return _node_from_row(row)

    def _read_context(self, context_id: ContextId) -> RetainedContext:
        row = self._db.execute("SELECT * FROM contexts WHERE context_id = ?", (context_id,)).fetchone()
        if row is None:
            raise TraceStoreError(f"unknown context id: {context_id}")
        return self._context_from_row(row)

    def _insert_context(self, context: RetainedContext, append_intent_id: AppendIntentId) -> None:
        self._db.execute(
            """
            INSERT INTO contexts (
                context_id,
                active_binding_refs_json,
                capability_witness_refs_json,
                semantic_environment_refs_json,
                visibility_policy_refs_json,
                substrate_ref,
                containment,
                append_intent_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.context_id,
                _json_dumps(list(context.active_binding_refs)),
                _json_dumps(list(context.capability_witness_refs)),
                _json_dumps(list(context.semantic_environment_refs)),
                _json_dumps(list(context.visibility_policy_refs)),
                context.substrate_ref,
                context.containment,
                append_intent_id,
            ),
        )

    def _insert_witness_record_if_missing(self, plan: _WitnessPlan, append_intent_id: AppendIntentId) -> None:
        if self._fact_exists(plan.record_id):
            existing = self._read_fact(plan.record_id)
            if not _witness_fact_matches_plan(existing, plan):
                raise TraceStoreError(f"witness record id {plan.record_id!r} already names a different witness")
            return

        ordinal = self._next_owner_ordinal(WITNESS_TRACE_OWNER_ID)
        fact = Fact(
            envelope=FactEnvelope(
                record_id=plan.record_id,
                digest=plan.record_id,
                schema_ref=plan.schema_ref,
                mode="capture",
                witness_ref=plan.witness_ref,
            ),
            body=FactBody(payload=dict(plan.body)),
            view=FactView(
                trace_owner_id=WITNESS_TRACE_OWNER_ID,
                owner_ordinal=ordinal,
                kind_label=plan.kind_label,
            ),
        )
        self._insert_fact_row(fact, f"witness:{plan.record_id}", append_intent_id)
        self._set_next_owner_ordinal(WITNESS_TRACE_OWNER_ID, ordinal + 1)

    def _insert_fact_row(
        self,
        fact: Fact,
        commit_receipt: CommitReceipt,
        append_intent_id: AppendIntentId,
    ) -> None:
        if fact.view is None:
            raise TraceStoreError("path append requires record view metadata")
        self._insert_record_if_missing(fact, append_intent_id)
        self._db.execute(
            """
            INSERT INTO path_entries (
                path_ref,
                path_ordinal,
                record_id,
                retained_context_ref,
                kind_label,
                append_intent_id,
                commit_receipt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.view.trace_owner_id,
                fact.view.owner_ordinal,
                fact.envelope.fact_id,
                fact.view.retained_context_ref,
                fact.view.kind_label,
                append_intent_id,
                commit_receipt,
            ),
        )

    def _insert_record_if_missing(self, fact: Fact, append_intent_id: AppendIntentId) -> None:
        if fact.envelope.fact_id != fact.envelope.digest:
            raise TraceStoreError("record_id must equal digest")
        if self._fact_exists(fact.envelope.fact_id):
            existing = self._read_record(fact.envelope.fact_id)
            if not _record_content_matches_fact(existing, fact):
                raise TraceStoreError(f"record id {fact.envelope.fact_id!r} already names different content")
            return
        self._db.execute(
            """
            INSERT INTO records (
                record_id,
                digest,
                schema_ref,
                mode,
                witness_ref,
                caused_by_json,
                body_json,
                append_intent_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.envelope.fact_id,
                fact.envelope.digest,
                fact.envelope.schema_ref,
                fact.envelope.mode,
                fact.envelope.witness_ref,
                _json_dumps(list(fact.envelope.caused_by_fact_ids)),
                _body_to_json(fact.body),
                append_intent_id,
            ),
        )
        for position, parent in enumerate(fact.envelope.caused_by_fact_ids):
            self._db.execute(
                """
                INSERT INTO record_edges(parent_record_id, child_record_id, parent_position)
                VALUES (?, ?, ?)
                """,
                (parent, fact.envelope.fact_id, position),
            )

    def _context_from_row(self, row: sqlite3.Row) -> RetainedContext:
        return RetainedContext(
            context_id=str(row["context_id"]),
            active_binding_refs=_json_tuple(row["active_binding_refs_json"], "active_binding_refs_json"),
            capability_witness_refs=_json_tuple(row["capability_witness_refs_json"], "capability_witness_refs_json"),
            semantic_environment_refs=_json_tuple(
                row["semantic_environment_refs_json"], "semantic_environment_refs_json"
            ),
            visibility_policy_refs=_json_tuple(row["visibility_policy_refs_json"], "visibility_policy_refs_json"),
            substrate_ref=str(row["substrate_ref"]),
            containment=str(row["containment"]),  # type: ignore[arg-type]
        )

    def _fact_exists(self, fact_id: FactId) -> bool:
        row = self._db.execute("SELECT 1 FROM records WHERE record_id = ?", (fact_id,)).fetchone()
        return row is not None

    def _context_exists(self, context_id: ContextId) -> bool:
        row = self._db.execute("SELECT 1 FROM contexts WHERE context_id = ?", (context_id,)).fetchone()
        return row is not None

    def _canonical_fact_order(self, fact_ids: set[FactId]) -> tuple[_PathEntry, ...]:
        if not fact_ids:
            return ()
        placeholders = ",".join("?" for _ in fact_ids)
        query = f"""
            SELECT record_id, path_ref, path_ordinal FROM path_entries
            WHERE record_id IN ({placeholders})
            ORDER BY path_ref ASC, path_ordinal ASC, record_id ASC
            """  # noqa: S608 - placeholders are generated internally; values remain bound.
        rows = self._db.execute(query, tuple(fact_ids)).fetchall()
        entries_by_record: dict[FactId, _PathEntry] = {}
        for row in rows:
            entry = _path_entry_from_row(row)
            entries_by_record.setdefault(entry.record_id, entry)
        return tuple(entries_by_record.values())

    def _frontier_row(self, frontier_id: FrontierId) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM frontiers WHERE frontier_id = ?", (frontier_id,)).fetchone()
        if row is None:
            raise TraceStoreError(f"unknown frontier id: {frontier_id}")
        return cast("sqlite3.Row", row)

    def _verified_frontier_truth(self, frontier_id: FrontierId) -> OwnerCutoff:
        row = self._frontier_row(frontier_id)
        frontier_fact = self._read_fact(str(row["created_by_fact_id"]))
        owner_cutoff = _owner_cutoff_from_frontier_fact(frontier_fact)
        expected = OwnerCutoff(
            frontier_id=str(row["frontier_id"]),
            target_trace_owner_id=str(row["target_trace_owner_id"]),
            through_fact_id=str(row["through_fact_id"]),
            through_owner_ordinal=int(row["through_owner_ordinal"]),
            publisher_trace_owner_id=row["publisher_trace_owner_id"],
            created_by_fact_id=str(row["created_by_fact_id"]),
        )
        if owner_cutoff != expected:
            raise TraceStoreError("resolver index disagrees with retained frontier fact")
        return owner_cutoff

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO meta(key, value) VALUES ('next_commit_seq', '0');

            CREATE TABLE IF NOT EXISTS append_intents (
                append_intent_id TEXT PRIMARY KEY,
                batch_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS contexts (
                context_id TEXT PRIMARY KEY,
                active_binding_refs_json TEXT NOT NULL,
                capability_witness_refs_json TEXT NOT NULL,
                semantic_environment_refs_json TEXT NOT NULL,
                visibility_policy_refs_json TEXT NOT NULL,
                substrate_ref TEXT NOT NULL,
                containment TEXT NOT NULL,
                append_intent_id TEXT NOT NULL,
                FOREIGN KEY(append_intent_id) REFERENCES append_intents(append_intent_id)
            );

            CREATE TABLE IF NOT EXISTS owner_ordinals (
                trace_owner_id TEXT PRIMARY KEY,
                next_ordinal INTEGER NOT NULL CHECK(next_ordinal >= 0)
            );

            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                digest TEXT NOT NULL,
                schema_ref TEXT NOT NULL,
                mode TEXT NOT NULL,
                witness_ref TEXT NOT NULL,
                caused_by_json TEXT NOT NULL,
                body_json TEXT NOT NULL,
                append_intent_id TEXT NOT NULL,
                CHECK(record_id = digest),
                FOREIGN KEY(append_intent_id) REFERENCES append_intents(append_intent_id)
            );

            CREATE TABLE IF NOT EXISTS record_edges (
                parent_record_id TEXT NOT NULL,
                child_record_id TEXT NOT NULL,
                parent_position INTEGER NOT NULL CHECK(parent_position >= 0),
                PRIMARY KEY(child_record_id, parent_position),
                FOREIGN KEY(parent_record_id) REFERENCES records(record_id),
                FOREIGN KEY(child_record_id) REFERENCES records(record_id)
            );

            CREATE INDEX IF NOT EXISTS idx_record_edges_parent
                ON record_edges(parent_record_id);

            CREATE TABLE IF NOT EXISTS path_entries (
                path_ref TEXT NOT NULL,
                path_ordinal INTEGER NOT NULL CHECK(path_ordinal >= 0),
                record_id TEXT NOT NULL,
                retained_context_ref TEXT NOT NULL,
                kind_label TEXT NOT NULL,
                append_intent_id TEXT NOT NULL,
                commit_receipt TEXT NOT NULL UNIQUE,
                PRIMARY KEY(path_ref, path_ordinal),
                FOREIGN KEY(record_id) REFERENCES records(record_id),
                FOREIGN KEY(append_intent_id) REFERENCES append_intents(append_intent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_path_entries_record
                ON path_entries(record_id);

            CREATE TABLE IF NOT EXISTS frontiers (
                frontier_id TEXT PRIMARY KEY,
                target_trace_owner_id TEXT NOT NULL,
                through_fact_id TEXT NOT NULL,
                through_owner_ordinal INTEGER NOT NULL CHECK(through_owner_ordinal >= 0),
                publisher_trace_owner_id TEXT,
                created_by_fact_id TEXT NOT NULL UNIQUE,
                append_intent_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY(through_fact_id) REFERENCES records(record_id),
                FOREIGN KEY(created_by_fact_id) REFERENCES records(record_id),
                FOREIGN KEY(append_intent_id) REFERENCES append_intents(append_intent_id)
            );
            """
        )

    def _begin(self) -> None:
        self._db.execute("BEGIN IMMEDIATE")

    def _next_owner_ordinal(self, trace_owner_id: TraceOwnerId) -> int:
        row = self._db.execute(
            "SELECT next_ordinal FROM owner_ordinals WHERE trace_owner_id = ?",
            (trace_owner_id,),
        ).fetchone()
        return 0 if row is None else int(row["next_ordinal"])

    def _next_commit_seq(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'next_commit_seq'").fetchone()
        if row is None:
            raise TraceStoreError("trace store metadata is missing next_commit_seq")
        return int(row["value"])

    def _set_next_commit_seq(self, next_commit_seq: int) -> None:
        self._db.execute(
            "UPDATE meta SET value = ? WHERE key = 'next_commit_seq'",
            (str(next_commit_seq),),
        )

    def _set_next_owner_ordinal(self, trace_owner_id: TraceOwnerId, next_ordinal: int) -> None:
        self._db.execute(
            """
            INSERT INTO owner_ordinals(trace_owner_id, next_ordinal)
            VALUES (?, ?)
            ON CONFLICT(trace_owner_id)
            DO UPDATE SET next_ordinal = excluded.next_ordinal
            """,
            (trace_owner_id, next_ordinal),
        )


def _coerce_read_context(
    read_context: ReadContext | OperationContext,
    visibility: VisibilityProfile | None,
) -> OperationContext:
    if isinstance(read_context, OperationContext):
        if read_context.operation != "read":
            raise TraceStoreError("read operation requires OperationContext(operation='read')")
        context = read_context
    else:
        context = read_context.to_operation_context()
    if visibility is None or visibility == context.visibility_profile:
        return context
    return OperationContext(
        actor_ref=context.actor_ref,
        operation="read",
        presented_authority_refs=context.presented_authority_refs,
        visibility_profile=visibility,
    )


def _coerce_write_context(
    append_context: AppendContext | OperationContext,
    operation: OperationKind,
) -> OperationContext:
    if isinstance(append_context, OperationContext):
        if append_context.operation != operation:
            raise TraceStoreError(f"{operation} operation requires OperationContext(operation={operation!r})")
        return append_context
    return append_context.to_operation_context(operation)


def _ensure_append_authorized(context: OperationContext) -> None:
    if context.trust_mode == "internal" or "trusted:internal" in context.presented_witness_refs:
        return
    raise TraceStoreError("Slice A append requires trusted internal witness")


def _ensure_read_authorized(context: OperationContext) -> None:
    if context.visibility_profile not in {"shape_only", "payload", "full_internal"}:
        raise ValueError(f"unknown visibility profile: {context.visibility_profile}")
    if context.visibility_profile == "full_internal" and "trusted:internal" not in context.presented_witness_refs:
        raise TraceStoreError("full_internal reads require trusted internal witness")


def _ensure_mode_filter(mode_filter: ModeFilter) -> None:
    if mode_filter not in {"declarations_only", "captures_only", "both"}:
        raise ValueError(f"unknown mode filter: {mode_filter}")


def _ensure_closure_policy(closure_policy: ClosurePolicy) -> None:
    if closure_policy not in {"visible_only", "include_external_anchors"}:
        raise ValueError(f"unknown closure policy: {closure_policy}")


def _mode_matches(fact: Fact, mode_filter: ModeFilter) -> bool:
    if mode_filter == "both":
        return True
    if mode_filter == "declarations_only":
        return fact.envelope.mode == "declaration"
    if mode_filter == "captures_only":
        return fact.envelope.mode == "capture"
    raise ValueError(f"unknown mode filter: {mode_filter}")


def _validate_batch_shape(batch: AppendBatch, append_context: AppendContext | OperationContext) -> None:
    if not batch.append_intent_id:
        raise ValueError("append_intent_id is required")
    if batch.atomicity != "atomic":
        raise ValueError("SQLiteTraceStore only supports atomic append batches")
    seen_local_refs: set[AppendLocalId] = set()
    for group in batch.groups:
        owner = _group_owner(group)
        if not owner:
            raise ValueError("trace_owner_id is required")
        _context_draft_for_group(group)
        for draft in _group_drafts(group):
            if draft.mode not in {"capture", "declaration"}:
                raise ValueError("fact mode must be 'capture' or 'declaration'")
            if draft.append_local_id is not None:
                if draft.append_local_id in seen_local_refs:
                    raise ValueError(f"duplicate append-local id: {draft.append_local_id}")
                seen_local_refs.add(draft.append_local_id)
            _resolve_schema_ref(draft, append_context)
            _json_dumps(draft.payload)


def _group_owner(group: AppendGroup) -> TraceOwnerId:
    if not group.trace_owner_id:
        raise ValueError("trace_owner_id is required")
    return group.trace_owner_id


def _group_drafts(group: AppendGroup) -> tuple[FactDraft, ...]:
    return group.fact_drafts


def _resolved_causes(
    group: AppendGroup,
    draft: FactDraft,
    local_fact_ids: dict[AppendLocalId, FactId],
) -> tuple[FactId, ...]:
    local_causes: list[FactId] = []
    for local_ref in draft.caused_by_local_refs:
        fact_id = local_fact_ids.get(local_ref)
        if fact_id is None:
            raise TraceStoreError(f"append-local causal parent does not exist: {local_ref}")
        local_causes.append(fact_id)
    caused_by = (*group.causal_parents, *draft.caused_by_fact_ids, *local_causes)
    if len(set(caused_by)) != len(caused_by):
        raise ValueError("duplicate causal parent")
    return caused_by


def _context_draft_for_group(group: AppendGroup) -> RetainedContextDraft:
    if isinstance(group.retained_context, RetainedContextDraft):
        return group.retained_context
    if isinstance(group.retained_context, RetainedContext):
        return RetainedContextDraft(reuse_context_id=group.retained_context.context_id)
    if isinstance(group.retained_context, str):
        return RetainedContextDraft(reuse_context_id=group.retained_context)
    return RetainedContextDraft()


def _context_payload(
    draft: RetainedContextDraft,
    append_context: AppendContext | OperationContext,
) -> dict[str, Any]:
    semantic_environment_refs = tuple(
        dict.fromkeys((*draft.semantic_environment_refs, append_context.schema_version_set))
    )
    return {
        "active_binding_refs": tuple(draft.active_binding_refs),
        "capability_witness_refs": tuple(draft.capability_witness_refs),
        "semantic_environment_refs": semantic_environment_refs,
        "visibility_policy_refs": tuple(draft.visibility_policy_refs),
        "substrate_ref": draft.substrate_ref,
        "containment": draft.containment,
    }


def _resolve_schema_ref(draft: FactDraft, append_context: AppendContext | OperationContext) -> SchemaRef:
    del append_context
    if draft.schema_ref:
        return draft.schema_ref
    raise ValueError("schema_ref is required")


SchemaRef = str


def _root_witness_plan() -> _WitnessPlan:
    record_id = root_witness_record_id()
    return _WitnessPlan(
        record_id=record_id,
        schema_ref=ROOT_WITNESS_SCHEMA_REF,
        kind_label="witness_root",
        body=root_witness_body(),
        witness_ref=ROOT_WITNESS_REF,
    )


def _ordinary_witness_plan(context: RetainedContext, append_context: AppendContext | OperationContext) -> _WitnessPlan:
    body = WitnessBody(
        actor_ref=append_context.actor_ref,
        authority_refs=context.capability_witness_refs,
        active_binding_refs=context.active_binding_refs,
        semantic_environment_refs=context.semantic_environment_refs,
        visibility_policy_refs=context.visibility_policy_refs,
        substrate_ref=context.substrate_ref,
        containment=context.containment,
    ).to_payload()
    validate_witness_body(schema_ref=WITNESS_SCHEMA_REF, body=body)
    root_ref = root_witness_record_id()
    record_id = record_digest(
        schema_ref=WITNESS_SCHEMA_REF,
        mode="capture",
        body=body,
        witness=root_ref,
    )
    return _WitnessPlan(
        record_id=record_id,
        schema_ref=WITNESS_SCHEMA_REF,
        kind_label="witness",
        body=body,
        witness_ref=root_ref,
    )


def _witness_fact_matches_plan(fact: Fact, plan: _WitnessPlan) -> bool:
    return (
        fact.envelope.fact_id == plan.record_id
        and fact.envelope.digest == plan.record_id
        and fact.envelope.schema_ref == plan.schema_ref
        and fact.envelope.mode == "capture"
        and fact.envelope.witness_ref == plan.witness_ref
        and fact.fact_kind == plan.kind_label
        and fact.body.payload == plan.body
    )


def _context_id(append_intent_id: AppendIntentId, group_index: int, payload: dict[str, Any]) -> ContextId:
    digest = hashlib.sha256(
        f"{append_intent_id}\0{group_index}\0{_json_dumps(_context_payload_to_json(payload))}".encode()
    ).hexdigest()
    return f"context:{digest[:32]}"


def _batch_digest(batch: AppendBatch, append_context: AppendContext | OperationContext) -> str:
    payload = {
        "append_intent_id": batch.append_intent_id,
        "atomicity": batch.atomicity,
        "actor_ref": append_context.actor_ref,
        "schema_version_set": append_context.schema_version_set,
        "groups": [
            {
                "trace_owner_id": _group_owner(group),
                "retained_context": _context_draft_to_json(_context_draft_for_group(group)),
                "causal_parents": list(group.causal_parents),
                "fact_drafts": [
                    {
                        "append_local_id": draft.append_local_id,
                        "kind_label": draft.kind_label,
                        "mode": draft.mode,
                        "schema_ref": _resolve_schema_ref(draft, append_context),
                        "payload": draft.payload,
                        "caused_by_fact_ids": list(draft.caused_by_fact_ids),
                        "caused_by_local_refs": list(draft.caused_by_local_refs),
                    }
                    for draft in _group_drafts(group)
                ],
            }
            for group in batch.groups
        ],
    }
    return hashlib.sha256(_json_dumps(payload).encode()).hexdigest()


def _context_payload_to_json(payload: dict[str, Any]) -> dict[str, object]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


def _context_draft_to_json(draft: RetainedContextDraft) -> dict[str, object]:
    return {
        "active_binding_refs": list(draft.active_binding_refs),
        "capability_witness_refs": list(draft.capability_witness_refs),
        "semantic_environment_refs": list(draft.semantic_environment_refs),
        "visibility_policy_refs": list(draft.visibility_policy_refs),
        "substrate_ref": draft.substrate_ref,
        "containment": draft.containment,
        "reuse_context_id": draft.reuse_context_id,
    }


def _body_to_json(body: FactBody) -> str:
    return _json_dumps({"payload": body.payload})


def _body_from_json(payload: str) -> FactBody:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TraceStoreError("fact body must decode to an object")
    raw_payload = data.get("payload", data.get("body", {}))
    if not isinstance(raw_payload, dict):
        raise TraceStoreError("fact payload must decode to an object")
    return FactBody(payload=dict(raw_payload))


def _path_entry_from_row(row: sqlite3.Row) -> _PathEntry:
    return _PathEntry(
        path_ref=str(row["path_ref"]),
        path_ordinal=int(row["path_ordinal"]),
        record_id=str(row["record_id"]),
    )


def _record_content_matches_fact(existing: Fact, fact: Fact) -> bool:
    return (
        existing.envelope.fact_id == fact.envelope.fact_id
        and existing.envelope.digest == fact.envelope.digest
        and existing.envelope.schema_ref == fact.envelope.schema_ref
        and existing.envelope.mode == fact.envelope.mode
        and existing.envelope.witness_ref == fact.envelope.witness_ref
        and existing.envelope.caused_by_fact_ids == fact.envelope.caused_by_fact_ids
        and existing.body == fact.body
    )


def _node_from_row(row: sqlite3.Row) -> Fact:
    caused_by = json.loads(row["caused_by_json"])
    if not isinstance(caused_by, list):
        raise TraceStoreError("caused_by_json must decode to a list")
    record_id = str(row["record_id"])
    digest = str(row["digest"])
    if record_id != digest:
        raise TraceStoreError("record_id must equal digest")
    return Fact(
        envelope=FactEnvelope(
            record_id=record_id,
            digest=digest,
            schema_ref=str(row["schema_ref"]),
            mode=str(row["mode"]),  # type: ignore[arg-type]
            witness_ref=str(row["witness_ref"]),
            caused_by_record_ids=tuple(str(fact_id) for fact_id in caused_by),
        ),
        body=_body_from_json(row["body_json"]),
        view=FactView(
            trace_owner_id=str(row["trace_owner_id"]),
            owner_ordinal=int(row["owner_ordinal"]),
            retained_context_ref=str(row["retained_context_ref"]),
            kind_label=str(row["kind_label"]),
        ),
    )


def _owner_cutoff_from_frontier_fact(fact: Fact) -> OwnerCutoff:
    if fact.envelope.schema_ref != "shepherd2.frontier.owner_cutoff.v1":
        raise TraceStoreError(f"expected frontier_published fact, got {fact.envelope.schema_ref!r}")
    payload = fact.body.payload
    try:
        frontier_id = str(payload["frontier_id"])
        target_trace_owner_id = str(payload["target_trace_owner_id"])
        through_fact_id = str(payload["through_fact_id"])
        through_owner_ordinal = int(payload["through_owner_ordinal"])
    except KeyError as exc:
        raise TraceStoreError(f"frontier fact missing payload field: {exc.args[0]}") from exc
    raw_publisher = payload.get("publisher_trace_owner_id")
    return OwnerCutoff(
        frontier_id=frontier_id,
        target_trace_owner_id=target_trace_owner_id,
        through_fact_id=through_fact_id,
        through_owner_ordinal=through_owner_ordinal,
        publisher_trace_owner_id=str(raw_publisher) if raw_publisher is not None else None,
        created_by_fact_id=fact.envelope.fact_id,
    )


def _visible_fact(fact: Fact, visibility: VisibilityProfile) -> VisibleFact | ExternalAnchor:
    if visibility in {"payload", "full_internal"}:
        return fact
    if visibility == "shape_only":
        return FactShape(envelope=fact.envelope, view=fact.view)
    raise ValueError(f"unknown visibility profile: {visibility}")


def _witness_anchor(witness: Fact) -> WitnessAnchor:
    return WitnessAnchor(
        witness_ref=witness.envelope.record_id,
        visible_shape={
            "schema_ref": witness.envelope.schema_ref,
            "record_id": witness.envelope.record_id,
            "mode": witness.envelope.mode,
            "witness_ref": witness.envelope.witness_ref,
        },
    )


def _receipt_to_json(receipt: AppendReceipt) -> str:
    return _json_dumps(
        {
            "append_intent_id": receipt.append_intent_id,
            "fact_ids": list(receipt.fact_ids),
            "commit_receipts": list(receipt.commit_receipts),
            "owner_ordinal_ranges": {
                owner: list(ordinal_range) for owner, ordinal_range in receipt.owner_ordinal_ranges.items()
            },
            "causal_edges": [list(edge) for edge in receipt.causal_edges],
            "context_receipts": list(receipt.context_receipts),
        }
    )


def _receipt_from_json(payload: str) -> AppendReceipt:
    data = json.loads(payload)
    return AppendReceipt(
        append_intent_id=str(data["append_intent_id"]),
        fact_ids=tuple(str(fact_id) for fact_id in data["fact_ids"]),
        commit_receipts=tuple(str(commit) for commit in data["commit_receipts"]),
        owner_ordinal_ranges={
            str(owner): (int(rng[0]), int(rng[1])) for owner, rng in data["owner_ordinal_ranges"].items()
        },
        causal_edges=tuple((str(left), str(right)) for left, right in data["causal_edges"]),
        context_receipts=tuple(str(context_id) for context_id in data.get("context_receipts", ())),
    )


def _next_commit_seq(commit_receipts: tuple[CommitReceipt, ...]) -> int:
    next_seq = 0
    for receipt in commit_receipts:
        prefix, separator, value = receipt.partition(":")
        if prefix != "commit" or separator != ":":
            continue
        next_seq = max(next_seq, int(value) + 1)
    return next_seq


def _json_tuple(payload: str, field_name: str) -> tuple[str, ...]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise TraceStoreError(f"{field_name} must decode to a list")
    return tuple(str(value) for value in decoded)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
