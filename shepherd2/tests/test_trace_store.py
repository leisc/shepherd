from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from shepherd2 import (
    TRUSTED_APPEND_CONTEXT,
    AppendBatch,
    AppendContext,
    AppendGroup,
    AppendIntentConflict,
    Fact,
    FactDraft,
    FactShape,
    OperationContext,
    OwnerCutoffSpec,
    ReadContext,
    RetainedContextDraft,
    SQLiteTraceStore,
    TraceStoreError,
    UnknownFact,
    root_witness_record_id,
)

if TYPE_CHECKING:
    from pathlib import Path


TRUSTED = AppendContext(
    actor_ref="runtime:test",
    presented_witness_refs=("trusted:internal",),
    schema_version_set="shepherd2-slice-a",
    trust_mode="internal",
)
READER = ReadContext(actor_ref="reader")


def _draft(kind: str, **payload: Any) -> FactDraft:
    caused_by = tuple(payload.pop("caused_by", ()))
    mode = str(payload.pop("mode", "capture"))
    return FactDraft(
        kind_label=kind,
        mode=mode,  # type: ignore[arg-type]
        schema_ref=f"shepherd2.trace.{kind}.v1",
        payload=payload,
        caused_by_fact_ids=caused_by,
    )


def _append_one(store: SQLiteTraceStore, intent: str, owner: str, *drafts: FactDraft) -> tuple[str, ...]:
    return store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id=intent,
            groups=(AppendGroup(trace_owner_id=owner, fact_drafts=drafts),),
        ),
    ).fact_ids


def _payload_facts(store: SQLiteTraceStore, owner: str, through: int = 99) -> tuple[Fact, ...]:
    trace_slice = store.read_owner_prefix(READER, owner, through)
    return tuple(
        fact
        for fact_id in trace_slice.fact_ids()
        if isinstance((fact := trace_slice.visible_facts_by_id[fact_id]), Fact)
    )


def test_append_intent_idempotency_survives_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    batch = AppendBatch(
        append_intent_id="intent:parent-start",
        groups=(
            AppendGroup(
                trace_owner_id="exec:parent",
                fact_drafts=(_draft("execution_started", execution_id="exec:parent"),),
            ),
        ),
    )

    with SQLiteTraceStore(db_path) as store:
        first = store.append(TRUSTED, batch)

    with SQLiteTraceStore(db_path) as restarted:
        second = restarted.append(TRUSTED, batch)
        assert second == first
        assert restarted.read_owner_prefix(READER, "exec:parent", 99).fact_ids() == first.fact_ids


def test_append_requires_explicit_context() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:explicit-context",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step"),)),),
    )

    with pytest.raises(TypeError):
        store.append(batch)  # type: ignore[call-arg]


def test_operation_context_is_operation_specific() -> None:
    store = SQLiteTraceStore()
    append_context = OperationContext(
        actor_ref="runtime:test",
        operation="append",
        presented_authority_refs=("trusted:internal",),
        schema_environment_ref="shepherd2-slice-a",
        trust_mode="internal",
    )
    read_context = OperationContext(actor_ref="reader", operation="read")
    batch = AppendBatch(
        append_intent_id="intent:operation-context",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step"),)),),
    )

    receipt = store.append(append_context, batch)
    assert store.read_owner_prefix(read_context, "exec:one", 99).fact_ids() == receipt.fact_ids

    with pytest.raises(TraceStoreError, match="append operation"):
        store.append(read_context, batch)

    with pytest.raises(TraceStoreError, match="publish_cut operation"):
        store.publish_cut(
            append_context,
            OwnerCutoffSpec(
                frontier_id="frontier:operation-context",
                target_trace_owner_id="exec:one",
                through_fact_id=receipt.fact_ids[-1],
            ),
        )

    publish_context = OperationContext(
        actor_ref="runtime:test",
        operation="publish_cut",
        presented_authority_refs=("trusted:internal",),
        schema_environment_ref="shepherd2-slice-a",
        trust_mode="internal",
    )
    cut = store.publish_cut(
        publish_context,
        OwnerCutoffSpec(
            frontier_id="frontier:operation-context",
            target_trace_owner_id="exec:one",
            through_fact_id=receipt.fact_ids[-1],
        ),
    )
    assert store.resolve_cut(read_context, cut.frontier_id).fact_ids() == receipt.fact_ids


def test_preview_record_ids_match_append_under_default_context() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:preview:default",
        groups=(AppendGroup(trace_owner_id="exec:preview", fact_drafts=(_draft("step", value=1),)),),
    )

    previewed = store.preview_record_ids(TRUSTED_APPEND_CONTEXT, batch)
    receipt = store.append(TRUSTED_APPEND_CONTEXT, batch)

    assert previewed == receipt.fact_ids
    assert store.preview_fact_ids(TRUSTED_APPEND_CONTEXT, batch) == receipt.fact_ids


def test_preview_record_ids_match_append_under_non_default_actor() -> None:
    store = SQLiteTraceStore()
    context = AppendContext(
        actor_ref="runtime:preview-actor",
        presented_witness_refs=("trusted:internal",),
        schema_version_set="shepherd2-slice-a",
        trust_mode="internal",
    )
    batch = AppendBatch(
        append_intent_id="intent:preview:actor",
        groups=(AppendGroup(trace_owner_id="exec:preview", fact_drafts=(_draft("step", value=1),)),),
    )

    previewed = store.preview_record_ids(context, batch)
    receipt = store.append(context, batch)

    assert previewed == receipt.fact_ids
    assert previewed != store.preview_record_ids(TRUSTED_APPEND_CONTEXT, batch)


def test_preview_record_ids_match_append_under_non_default_substrate() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:preview:substrate",
        groups=(
            AppendGroup(
                trace_owner_id="exec:preview",
                retained_context=RetainedContextDraft(substrate_ref="kv.sqlite.local.v1", containment="buffered"),
                fact_drafts=(_draft("step", value=1),),
            ),
        ),
    )

    previewed = store.preview_record_ids(TRUSTED, batch)
    receipt = store.append(TRUSTED, batch)

    assert previewed == receipt.fact_ids


def test_preview_record_ids_accepts_append_operation_context() -> None:
    store = SQLiteTraceStore()
    context = OperationContext(
        actor_ref="runtime:preview-operation",
        operation="append",
        presented_authority_refs=("trusted:internal",),
        schema_environment_ref="shepherd2-slice-a",
        trust_mode="internal",
    )
    batch = AppendBatch(
        append_intent_id="intent:preview:operation-context",
        groups=(AppendGroup(trace_owner_id="exec:preview", fact_drafts=(_draft("step", value=1),)),),
    )

    previewed = store.preview_record_ids(context, batch)
    receipt = store.append(context, batch)

    assert previewed == receipt.fact_ids


def test_preview_record_ids_rejects_read_operation_context() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:preview:read-context",
        groups=(AppendGroup(trace_owner_id="exec:preview", fact_drafts=(_draft("step", value=1),)),),
    )

    with pytest.raises(TraceStoreError, match="append operation"):
        store.preview_record_ids(OperationContext(actor_ref="reader", operation="read"), batch)

    with pytest.raises(TypeError):
        store.preview_fact_ids(batch)  # type: ignore[call-arg]


def test_same_append_intent_with_different_batch_is_rejected() -> None:
    store = SQLiteTraceStore()
    first = AppendBatch(
        append_intent_id="intent:once",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step", value=1),)),),
    )
    conflicting = AppendBatch(
        append_intent_id="intent:once",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step", value=2),)),),
    )

    store.append(TRUSTED, first)

    with pytest.raises(AppendIntentConflict):
        store.append(TRUSTED, conflicting)
    assert _payload_facts(store, "exec:one")[0].body.payload == {"value": 1}


def test_fact_digest_is_canonical_across_append_intents_and_storage_receipts(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    draft = FactDraft(
        kind_label="step",
        schema_ref="example.step.v1",
        mode="capture",
        payload={"value": 1},
    )

    with SQLiteTraceStore(db_path) as store:
        first = store.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:digest:first",
                groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(draft,)),),
            ),
        )
        second = store.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:digest:second",
                groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(draft,)),),
            ),
        )

    with SQLiteTraceStore(db_path) as restarted:
        first_fact = restarted.read_fact(READER, first.fact_ids[0])
        second_fact = restarted.read_fact(READER, second.fact_ids[0])

    assert isinstance(first_fact, Fact)
    assert isinstance(second_fact, Fact)
    assert first_fact.envelope.fact_id == first_fact.envelope.digest
    assert second_fact.envelope.fact_id == second_fact.envelope.digest
    assert first.fact_ids == second.fact_ids
    assert first.commit_receipts != second.commit_receipts
    assert first_fact.envelope.digest == second_fact.envelope.digest
    assert first_fact.envelope.digest.startswith("sha256:")


def test_content_addressed_record_can_have_multiple_owner_paths() -> None:
    store = SQLiteTraceStore()
    draft = FactDraft(
        kind_label="step",
        mode="capture",
        schema_ref="example.step.v1",
        payload={"value": 1},
    )

    first = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:path:first",
            groups=(AppendGroup(trace_owner_id="owner:first", fact_drafts=(draft,)),),
        ),
    )
    second = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:path:second",
            groups=(AppendGroup(trace_owner_id="owner:second", fact_drafts=(draft,)),),
        ),
    )

    first_view = store.read_owner_prefix(READER, "owner:first", 99)
    second_view = store.read_owner_prefix(READER, "owner:second", 99)
    first_fact = first_view.visible_facts_by_id[first.fact_ids[0]]
    second_fact = second_view.visible_facts_by_id[second.fact_ids[0]]

    assert first.fact_ids == second.fact_ids
    assert first.fact_ids[0].startswith("sha256:")
    assert first.commit_receipts != second.commit_receipts
    assert isinstance(first_fact, Fact)
    assert isinstance(second_fact, Fact)
    assert first_fact.envelope.fact_id == first_fact.envelope.digest
    assert second_fact.envelope.fact_id == second_fact.envelope.digest
    assert first_fact.trace_owner_id == "owner:first"
    assert second_fact.trace_owner_id == "owner:second"


def test_fact_digest_changes_when_mode_changes() -> None:
    store = SQLiteTraceStore()
    capture = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:digest:capture",
            groups=(
                AppendGroup(
                    trace_owner_id="exec:one",
                    fact_drafts=(
                        FactDraft(
                            kind_label="step",
                            schema_ref="example.step.v1",
                            mode="capture",
                            payload={"value": 1},
                        ),
                    ),
                ),
            ),
        ),
    )
    declaration = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:digest:declaration",
            groups=(
                AppendGroup(
                    trace_owner_id="exec:one",
                    fact_drafts=(
                        FactDraft(
                            kind_label="step",
                            schema_ref="example.step.v1",
                            mode="declaration",
                            payload={"value": 1},
                        ),
                    ),
                ),
            ),
        ),
    )

    capture_fact = store.read_fact(READER, capture.fact_ids[0])
    declaration_fact = store.read_fact(READER, declaration.fact_ids[0])

    assert isinstance(capture_fact, Fact)
    assert isinstance(declaration_fact, Fact)
    assert capture_fact.envelope.mode == "capture"
    assert declaration_fact.envelope.mode == "declaration"
    assert capture_fact.envelope.digest != declaration_fact.envelope.digest


def test_append_intent_conflict_includes_mode() -> None:
    store = SQLiteTraceStore()
    first = AppendBatch(
        append_intent_id="intent:mode-conflict",
        groups=(
            AppendGroup(
                trace_owner_id="exec:one",
                fact_drafts=(
                    FactDraft(
                        kind_label="step",
                        schema_ref="shepherd2.trace.step.v1",
                        mode="capture",
                        payload={"value": 1},
                    ),
                ),
            ),
        ),
    )
    conflicting = AppendBatch(
        append_intent_id="intent:mode-conflict",
        groups=(
            AppendGroup(
                trace_owner_id="exec:one",
                fact_drafts=(
                    FactDraft(
                        kind_label="step",
                        schema_ref="shepherd2.trace.step.v1",
                        mode="declaration",
                        payload={"value": 1},
                    ),
                ),
            ),
        ),
    )

    store.append(TRUSTED, first)

    with pytest.raises(AppendIntentConflict):
        store.append(TRUSTED, conflicting)


def test_append_intent_conflict_includes_actor_witness() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:actor-conflict",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step", value=1),)),),
    )
    first_actor = AppendContext(
        actor_ref="runtime:first",
        presented_witness_refs=("trusted:internal",),
        schema_version_set="shepherd2-slice-a",
        trust_mode="internal",
    )
    second_actor = AppendContext(
        actor_ref="runtime:second",
        presented_witness_refs=("trusted:internal",),
        schema_version_set="shepherd2-slice-a",
        trust_mode="internal",
    )

    store.append(first_actor, batch)

    with pytest.raises(AppendIntentConflict):
        store.append(second_actor, batch)


def test_append_rejects_unknown_mode() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:bad-mode",
        groups=(
            AppendGroup(
                trace_owner_id="exec:one",
                fact_drafts=(
                    FactDraft(
                        kind_label="step",
                        schema_ref="shepherd2.trace.step.v1",
                        mode="captured",
                        payload={},
                    ),
                ),  # type: ignore[arg-type]
            ),
        ),
    )

    with pytest.raises(ValueError, match="mode"):
        store.append(TRUSTED, batch)


def test_multi_owner_append_is_atomic_and_owner_ordered() -> None:
    store = SQLiteTraceStore()
    rejected = AppendBatch(
        append_intent_id="intent:bad-multi-owner",
        groups=(
            AppendGroup(trace_owner_id="exec:parent", fact_drafts=(_draft("parent_step"),)),
            AppendGroup(
                trace_owner_id="exec:child", fact_drafts=(FactDraft(mode="capture", schema_ref="", payload={}),)
            ),
        ),
    )

    with pytest.raises(ValueError, match="schema_ref"):
        store.append(TRUSTED, rejected)

    assert store.read_owner_prefix(READER, "exec:parent", 99).fact_ids() == ()
    assert store.read_owner_prefix(READER, "exec:child", 99).fact_ids() == ()

    accepted = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:good-multi-owner",
            groups=(
                AppendGroup(trace_owner_id="exec:parent", fact_drafts=(_draft("parent_step"),)),
                AppendGroup(trace_owner_id="exec:child", fact_drafts=(_draft("child_step"),)),
            ),
        ),
    )

    assert accepted.owner_ordinal_ranges == {"exec:parent": (0, 0), "exec:child": (0, 0)}
    assert _payload_facts(store, "exec:parent")[0].owner_ordinal == 0
    assert _payload_facts(store, "exec:child")[0].owner_ordinal == 0


def test_causal_parents_must_exist() -> None:
    store = SQLiteTraceStore()
    batch = AppendBatch(
        append_intent_id="intent:bad-cause",
        groups=(
            AppendGroup(
                trace_owner_id="exec:one",
                fact_drafts=(_draft("step", caused_by=("fact:missing",)),),
            ),
        ),
    )

    with pytest.raises(UnknownFact, match="causal parent"):
        store.append(TRUSTED, batch)
    assert store.fact_count() == 0


def test_append_retains_root_and_ordinary_witness_records(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    context = AppendContext(
        actor_ref="runtime:witness-test",
        presented_witness_refs=("trusted:internal", "cap:not-retained-implicitly"),
        schema_version_set="shepherd2-slice-a",
        trust_mode="internal",
    )

    with SQLiteTraceStore(db_path) as store:
        receipt = store.append(
            context,
            AppendBatch(
                append_intent_id="intent:witness-records",
                groups=(
                    AppendGroup(
                        trace_owner_id="owner:one",
                        retained_context=None,
                        fact_drafts=(_draft("step", value=1),),
                    ),
                ),
            ),
        )
        retained = store.read_fact(READER, receipt.fact_ids[0])
        assert isinstance(retained, Fact)
        witness_ref = retained.envelope.witness_ref
        ordinary_witness = store.read_fact(READER, witness_ref)
        root_witness = store.read_fact(READER, root_witness_record_id())

    with SQLiteTraceStore(db_path) as restarted:
        restored_root = restarted.read_fact(READER, root_witness_record_id())
        restored_retained = restarted.read_fact(READER, receipt.fact_ids[0])

    assert isinstance(ordinary_witness, Fact)
    assert isinstance(root_witness, Fact)
    assert isinstance(restored_retained, Fact)
    assert witness_ref
    assert retained.envelope.digest == restored_retained.envelope.digest
    assert ordinary_witness.envelope.schema_ref == "kernel.witness.v1"
    assert ordinary_witness.envelope.witness_ref == root_witness_record_id()
    assert ordinary_witness.body.payload["actor_ref"] == "runtime:witness-test"
    assert ordinary_witness.body.payload["substrate_ref"] == "sqlite.local.v1"
    assert ordinary_witness.body.payload["containment"] == "contained"
    assert "cap:not-retained-implicitly" not in ordinary_witness.body.payload["authority_refs"]
    assert root_witness.envelope.schema_ref == "kernel.witness.root.v1"
    assert root_witness.envelope.witness_ref == ""
    assert restored_root == root_witness


def test_causal_closure_crosses_parent_child_facts_with_visibility() -> None:
    store = SQLiteTraceStore()
    child_event = _append_one(store, "intent:child-file-created", "exec:child", _draft("file_created", path="note.txt"))
    observed = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:parent-observed-child",
            groups=(
                AppendGroup(
                    trace_owner_id="exec:parent",
                    causal_parents=child_event,
                    fact_drafts=(_draft("observed", binding="binding:parent"),),
                ),
            ),
        ),
    )
    step = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:parent-interpreter-step",
            groups=(
                AppendGroup(
                    trace_owner_id="exec:parent",
                    causal_parents=observed.fact_ids,
                    fact_drafts=(_draft("interpreter_step", decision="publish"),),
                ),
            ),
        ),
    )

    payload_view = store.read_causal_closure(READER, step.fact_ids)
    shape_view = store.read_causal_closure(
        ReadContext(actor_ref="reader", visibility_profile="shape_only"), step.fact_ids
    )

    assert payload_view.fact_ids() == (*child_event, *observed.fact_ids, *step.fact_ids)
    first = payload_view.visible_facts_by_id[child_event[0]]
    assert isinstance(first, Fact)
    assert first.body.payload == {"path": "note.txt"}
    assert all(isinstance(fact, FactShape) for fact in shape_view.visible_facts_by_id.values())


def test_restart_preserves_future_commit_and_owner_ordinal_allocation(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    first_batch = AppendBatch(
        append_intent_id="intent:first",
        groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step"),)),),
    )
    with SQLiteTraceStore(db_path) as store:
        first_fact_ids = store.append(TRUSTED, first_batch).fact_ids

    with SQLiteTraceStore(db_path) as restarted:
        receipt = restarted.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:second",
                groups=(AppendGroup(trace_owner_id="exec:one", fact_drafts=(_draft("step"),)),),
            ),
        )

        assert receipt.commit_receipts == ("commit:1",)
        assert restarted.read_owner_prefix(READER, "exec:one", 99).fact_ids() == (
            *first_fact_ids,
            *receipt.fact_ids,
        )
        assert _payload_facts(restarted, "exec:one")[-1].owner_ordinal == 1


def test_decode_does_not_allocate_retained_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(db_path) as store:
        _append_one(store, "intent:first", "exec:one", _draft("step"))
        before = store.fact_count()

        with pytest.raises(UnknownFact):
            store.read_fact(READER, "fact:missing")

        assert store.fact_count() == before


def test_corrupt_retained_row_is_rejected_during_decode(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(db_path) as store:
        fact_id = _append_one(store, "intent:first", "exec:one", _draft("step"))[0]

    db = sqlite3.connect(db_path)
    db.execute("UPDATE records SET caused_by_json = ? WHERE record_id = ?", ('{"not":"a-list"}', fact_id))
    db.commit()
    db.close()

    with SQLiteTraceStore(db_path) as restarted, pytest.raises(TraceStoreError, match="caused_by_json"):
        restarted.read_fact(READER, fact_id)


def test_record_id_digest_mismatch_is_rejected_during_decode(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(db_path) as store:
        fact_id = _append_one(store, "intent:digest-mismatch", "exec:one", _draft("step"))[0]

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA ignore_check_constraints = ON")
    db.execute("UPDATE records SET digest = ? WHERE record_id = ?", ("sha256:" + "0" * 64, fact_id))
    db.commit()
    db.close()

    with SQLiteTraceStore(db_path) as restarted, pytest.raises(TraceStoreError, match="record_id must equal digest"):
        restarted.read_fact(READER, fact_id)


def test_sqlite_records_table_rejects_record_id_digest_mismatch() -> None:
    store = SQLiteTraceStore()
    store._db.execute(
        """
        INSERT INTO append_intents(append_intent_id, batch_digest, receipt_json)
        VALUES (?, ?, ?)
        """,
        ("intent:bad-record-row", "batch:digest", "{}"),
    )

    with pytest.raises(sqlite3.IntegrityError):
        store._db.execute(
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
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
                "example.corrupt.v1",
                "capture",
                root_witness_record_id(),
                "[]",
                '{"payload":{}}',
                "intent:bad-record-row",
            ),
        )


def test_frontier_fact_payload_is_source_of_truth_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(db_path) as store:
        receipt = _append_one(store, "intent:frontier-truth-target", "owner:one", _draft("terminal"))
        expected = store.publish_frontier(
            TRUSTED,
            OwnerCutoffSpec(
                frontier_id="frontier:truth",
                target_trace_owner_id="owner:one",
                through_fact_id=receipt[-1],
            ),
        )

    with SQLiteTraceStore(db_path) as restarted:
        restored = restarted.read_owner_cutoff("frontier:truth")

    assert restored == expected
