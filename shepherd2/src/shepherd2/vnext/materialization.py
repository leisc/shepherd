"""vNext materialization orchestration over the Ring 0 trace store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..kernel.facts import (
    AppendBatch,
    AppendGroup,
    AppendIntentId,
    Fact,
    FactId,
    OperationContext,
    OwnerOrdinal,
    ReadContext,
    RetainedContextDraft,
    TraceOwnerId,
)
from ..trace_store import AppendIntentConflictError, SQLiteTraceStore, TraceStoreError
from .substrates import MaterializationReceipt, SubstrateRegistry

MAX_OWNER_ORDINAL = (2**63) - 1


@dataclass(frozen=True)
class MaterializationRequest:
    """Path-explicit request to materialize retained declaration records."""

    append_intent_id: AppendIntentId
    target_trace_owner_id: TraceOwnerId
    target_record_ids: tuple[FactId, ...]
    target_through_owner_ordinal: OwnerOrdinal = MAX_OWNER_ORDINAL
    capture_trace_owner_id: TraceOwnerId | None = None

    @property
    def capture_owner(self) -> TraceOwnerId:
        """Return the owner path where materialization captures should be appended."""
        return self.capture_trace_owner_id or self.target_trace_owner_id


# Phase 2 boundary (260621-1600-trace.md): materialize() stays bound to the concrete SQLiteTraceStore
# because it reaches the SQLite-private materialization-intent ledger (store._db / sqlite3). Extracting
# that ledger to TraceStore facts / vcs-core records is Phase 2; until then this is not protocol-seamed.
def materialize(
    store: SQLiteTraceStore,
    operation_context: OperationContext,
    request: MaterializationRequest,
    registry: SubstrateRegistry,
) -> MaterializationReceipt:
    """Materialize declarations through their witness-stamped substrate.

    This is intentionally a vNext orchestrator over the Ring 0 store, not a
    frozen TraceStore method.
    """
    _ensure_materialize_authorized(operation_context)
    _validate_request(request)
    request_digest = _request_digest(request, operation_context)

    existing = _read_completed_intent(store, request.append_intent_id)
    if existing is not None:
        existing_digest, receipt = existing
        if existing_digest != request_digest:
            raise AppendIntentConflictError(
                f"materialize intent {request.append_intent_id!r} was already committed with different content"
            )
        return receipt

    targets = _read_target_records(store, request)
    substrate_refs = tuple(dict.fromkeys(_substrate_ref_for_record(store, target) for target in targets))
    if len(substrate_refs) != 1:
        raise TraceStoreError("materialize batch must target exactly one substrate")
    substrate_ref = substrate_refs[0]
    substrate = registry.get(substrate_ref)

    unsupported = tuple(
        target.envelope.schema_ref
        for target in targets
        if target.envelope.schema_ref not in substrate.declaration_schemas
    )
    if unsupported:
        raise TraceStoreError(f"substrate {substrate_ref!r} does not accept schemas: {', '.join(unsupported)}")

    result = substrate.materialize(targets)
    if result.outcome not in {"success", "clean_failure", "split_state"}:
        raise TraceStoreError(f"unknown materialization outcome: {result.outcome}")
    for draft in result.capture_drafts:
        if draft.mode != "capture":
            raise TraceStoreError("materialize substrates may only emit capture drafts")
        if draft.schema_ref not in substrate.capture_schemas:
            raise TraceStoreError(
                f"substrate {substrate_ref!r} emitted unsupported capture schema {draft.schema_ref!r}"
            )

    produced_record_ids: tuple[FactId, ...] = ()
    if result.capture_drafts:
        append_context = _append_context_for_materialize(operation_context)
        append_receipt = store.append(
            append_context,
            AppendBatch(
                append_intent_id=request.append_intent_id,
                groups=(
                    AppendGroup(
                        trace_owner_id=request.capture_owner,
                        retained_context=RetainedContextDraft(
                            capability_witness_refs=operation_context.presented_authority_refs,
                            semantic_environment_refs=(operation_context.schema_environment_ref,),
                            substrate_ref=substrate.substrate_ref,
                            containment=substrate.containment,
                        ),
                        fact_drafts=result.capture_drafts,
                    ),
                ),
            ),
        )
        produced_record_ids = append_receipt.fact_ids

    receipt = MaterializationReceipt(
        outcome=result.outcome,
        substrate_ref=substrate_ref,
        target_record_ids=request.target_record_ids,
        produced_record_ids=produced_record_ids,
        failure_reason=result.failure_reason,
        world_side_anchors=result.world_side_anchors,
    )
    _record_completed_intent(store, request.append_intent_id, request_digest, receipt)
    return receipt


def _ensure_materialize_authorized(context: OperationContext) -> None:
    if context.operation != "materialize":
        raise TraceStoreError("materialize operation requires OperationContext(operation='materialize')")
    if context.trust_mode == "internal" or "trusted:internal" in context.presented_witness_refs:
        return
    raise TraceStoreError("materialize requires trusted internal witness")


def _validate_request(request: MaterializationRequest) -> None:
    if not request.append_intent_id:
        raise ValueError("append_intent_id is required")
    if not request.target_trace_owner_id:
        raise ValueError("target_trace_owner_id is required")
    if not request.target_record_ids:
        raise ValueError("materialize requires at least one target record id")
    if request.capture_trace_owner_id is not None and not request.capture_trace_owner_id:
        raise ValueError("capture_trace_owner_id cannot be empty")
    if request.target_through_owner_ordinal < 0:
        raise ValueError("target_through_owner_ordinal must be non-negative")


def _read_target_records(store: SQLiteTraceStore, request: MaterializationRequest) -> tuple[Fact, ...]:
    read_context = ReadContext(actor_ref="vnext:materialize")
    trace_slice = store.read_owner_prefix(
        read_context,
        request.target_trace_owner_id,
        request.target_through_owner_ordinal,
    )
    targets: list[Fact] = []
    for record_id in request.target_record_ids:
        visible = trace_slice.visible_facts_by_id.get(record_id)
        if not isinstance(visible, Fact):
            raise TraceStoreError(
                f"materialize target {record_id!r} is not present on owner path {request.target_trace_owner_id!r}"
            )
        if visible.envelope.mode != "declaration":
            raise TraceStoreError("materialize targets must be declaration records")
        targets.append(visible)
    return tuple(targets)


def _substrate_ref_for_record(store: SQLiteTraceStore, record: Fact) -> str:
    witness = store.read_fact(ReadContext(actor_ref="vnext:materialize"), record.envelope.witness_ref)
    if not isinstance(witness, Fact):
        raise TraceStoreError(f"record witness {record.envelope.witness_ref!r} is not payload-visible")
    substrate_ref = witness.body.payload.get("substrate_ref")
    if not isinstance(substrate_ref, str) or not substrate_ref:
        raise TraceStoreError(f"record witness {record.envelope.witness_ref!r} has no substrate_ref")
    return substrate_ref


def _append_context_for_materialize(context: OperationContext) -> OperationContext:
    return OperationContext(
        actor_ref=context.actor_ref,
        operation="append",
        presented_authority_refs=context.presented_authority_refs,
        schema_environment_ref=context.schema_environment_ref,
        visibility_profile=context.visibility_profile,
        trust_mode=context.trust_mode,
    )


def _ensure_ledger(store: SQLiteTraceStore) -> None:
    store._db.execute(
        """
        CREATE TABLE IF NOT EXISTS materialization_intents (
            materialize_intent_id TEXT PRIMARY KEY,
            request_digest TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _read_completed_intent(
    store: SQLiteTraceStore,
    append_intent_id: AppendIntentId,
) -> tuple[str, MaterializationReceipt] | None:
    with store._lock:
        _ensure_ledger(store)
        row = store._db.execute(
            """
            SELECT request_digest, receipt_json FROM materialization_intents
            WHERE materialize_intent_id = ?
            """,
            (append_intent_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["request_digest"]), _receipt_from_json(str(row["receipt_json"]))


def _record_completed_intent(
    store: SQLiteTraceStore,
    append_intent_id: AppendIntentId,
    request_digest: str,
    receipt: MaterializationReceipt,
) -> None:
    with store._lock:
        _ensure_ledger(store)
        try:
            store._db.execute(
                """
                INSERT INTO materialization_intents(materialize_intent_id, request_digest, receipt_json)
                VALUES (?, ?, ?)
                """,
                (append_intent_id, request_digest, _receipt_to_json(receipt)),
            )
        except sqlite3.IntegrityError as exc:
            existing = _read_completed_intent(store, append_intent_id)
            if existing is None or existing[0] != request_digest:
                raise AppendIntentConflictError(
                    f"materialize intent {append_intent_id!r} was already committed with different content"
                ) from exc


def _request_digest(request: MaterializationRequest, context: OperationContext) -> str:
    payload = {
        "append_intent_id": request.append_intent_id,
        "target_trace_owner_id": request.target_trace_owner_id,
        "target_record_ids": list(request.target_record_ids),
        "target_through_owner_ordinal": request.target_through_owner_ordinal,
        "capture_trace_owner_id": request.capture_trace_owner_id,
        "actor_ref": context.actor_ref,
        "authority_refs": list(context.presented_authority_refs),
        "schema_environment_ref": context.schema_environment_ref,
        "trust_mode": context.trust_mode,
    }
    return hashlib.sha256(_json_dumps(payload).encode()).hexdigest()


def _receipt_to_json(receipt: MaterializationReceipt) -> str:
    return _json_dumps(
        {
            "outcome": receipt.outcome,
            "substrate_ref": receipt.substrate_ref,
            "target_record_ids": list(receipt.target_record_ids),
            "produced_record_ids": list(receipt.produced_record_ids),
            "failure_reason": receipt.failure_reason,
            "world_side_anchors": list(receipt.world_side_anchors),
        }
    )


def _receipt_from_json(payload: str) -> MaterializationReceipt:
    data = json.loads(payload)
    anchors = data.get("world_side_anchors", ())
    return MaterializationReceipt(
        outcome=str(data["outcome"]),  # type: ignore[arg-type]
        substrate_ref=str(data["substrate_ref"]),
        target_record_ids=tuple(str(record_id) for record_id in data["target_record_ids"]),
        produced_record_ids=tuple(str(record_id) for record_id in data.get("produced_record_ids", ())),
        failure_reason=str(data.get("failure_reason", "")),
        world_side_anchors=tuple(dict(anchor) for anchor in anchors),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
