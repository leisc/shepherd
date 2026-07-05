from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shepherd2 import (
    ABI_VERSION,
    WITNESS_SCHEMA_REF,
    AppendBatch,
    AppendGroup,
    Fact,
    FactDraft,
    FactShape,
    OperationContext,
    OwnerCutoffSpec,
    ProjectionModeError,
    ProjectionSpec,
    RetainedContextDraft,
    SQLiteTraceStore,
    TraceStoreError,
    canonical_record_input,
    create_execution_batch,
    ensure_projection_compatible,
    execution_id_for,
    project_execution_slice,
    root_witness_record_id,
    witness_body_digest,
)

if TYPE_CHECKING:
    from pathlib import Path


APPEND = OperationContext(
    actor_ref="runtime:laws",
    operation="append",
    presented_authority_refs=("trusted:internal",),
    schema_environment_ref="shepherd2-slice-a",
    trust_mode="internal",
)
PUBLISH_CUT = OperationContext(
    actor_ref="runtime:laws",
    operation="publish_cut",
    presented_authority_refs=("trusted:internal",),
    schema_environment_ref="shepherd2-slice-a",
    trust_mode="internal",
)
READ = OperationContext(actor_ref="reader:laws", operation="read")


def _draft(kind: str = "step", **payload: object) -> FactDraft:
    return FactDraft(
        kind_label=kind,
        mode="capture",
        schema_ref=f"law.{kind}.v1",
        payload=dict(payload),
    )


def test_kernel_abi_v0_marker_is_frozen() -> None:
    assert ABI_VERSION == "shepherd.kernel.abi.v0"


def test_record_id_is_digest_and_excludes_path_receipts() -> None:
    store = SQLiteTraceStore()
    draft = _draft(value=1)

    first = store.append(
        APPEND,
        AppendBatch("law:record:first", (AppendGroup("owner:first", fact_drafts=(draft,)),)),
    )
    second = store.append(
        APPEND,
        AppendBatch("law:record:second", (AppendGroup("owner:second", fact_drafts=(draft,)),)),
    )

    first_fact = store.read_owner_prefix(READ, "owner:first", 99).visible_facts_by_id[first.fact_ids[0]]
    second_fact = store.read_owner_prefix(READ, "owner:second", 99).visible_facts_by_id[second.fact_ids[0]]

    assert first.fact_ids == second.fact_ids
    assert first.commit_receipts != second.commit_receipts
    assert isinstance(first_fact, Fact)
    assert isinstance(second_fact, Fact)
    assert first_fact.envelope.fact_id == first_fact.envelope.digest
    assert second_fact.envelope.fact_id == second_fact.envelope.digest
    assert first_fact.trace_owner_id != second_fact.trace_owner_id


def test_record_digest_excludes_legacy_kind_label() -> None:
    store = SQLiteTraceStore()
    first = FactDraft(
        kind_label="friendly_name",
        mode="capture",
        schema_ref="law.same_schema.v1",
        payload={"value": 1},
    )
    second = FactDraft(
        kind_label="renamed_view_label",
        mode="capture",
        schema_ref="law.same_schema.v1",
        payload={"value": 1},
    )

    first_receipt = store.append(
        APPEND,
        AppendBatch("law:label:first", (AppendGroup("owner:label:first", fact_drafts=(first,)),)),
    )
    second_receipt = store.append(
        APPEND,
        AppendBatch("law:label:second", (AppendGroup("owner:label:second", fact_drafts=(second,)),)),
    )

    assert first_receipt.fact_ids == second_receipt.fact_ids
    assert (
        store.read_owner_prefix(READ, "owner:label:first", 99).visible_facts_by_id[first_receipt.fact_ids[0]].fact_kind
        == "friendly_name"
    )
    assert (
        store.read_owner_prefix(READ, "owner:label:second", 99)
        .visible_facts_by_id[second_receipt.fact_ids[0]]
        .fact_kind
        == "renamed_view_label"
    )


def test_witness_records_are_retained_and_non_root_records_have_witnesses() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch("law:witness", (AppendGroup("owner:witness", fact_drafts=(_draft(),)),)),
    )

    record = store.read_fact(READ, receipt.fact_ids[0])
    root = store.read_fact(READ, root_witness_record_id())

    assert isinstance(record, Fact)
    assert isinstance(root, Fact)
    assert record.envelope.witness_ref
    assert root.envelope.schema_ref == "kernel.witness.root.v1"
    assert root.envelope.witness_ref == ""
    ordinary_witness = store.read_fact(READ, record.envelope.witness_ref)
    assert isinstance(ordinary_witness, Fact)
    assert record.envelope.witness_ref != witness_body_digest(
        schema_ref=ordinary_witness.envelope.schema_ref,
        body=ordinary_witness.body.payload,
    )
    assert ordinary_witness.envelope.caused_by_fact_ids == ()
    assert root.envelope.caused_by_fact_ids == ()


def test_witness_body_validation_is_enforced_on_append_and_preview() -> None:
    store = SQLiteTraceStore()
    empty_substrate = AppendBatch(
        "law:witness:empty-substrate",
        (
            AppendGroup(
                "owner:witness-validation",
                retained_context=RetainedContextDraft(substrate_ref=""),
                fact_drafts=(_draft(),),
            ),
        ),
    )
    invalid_containment = AppendBatch(
        "law:witness:invalid-containment",
        (
            AppendGroup(
                "owner:witness-validation",
                retained_context=RetainedContextDraft(containment="partial"),  # type: ignore[arg-type]
                fact_drafts=(_draft(),),
            ),
        ),
    )
    unknown_substrate = AppendBatch(
        "law:witness:unknown-substrate",
        (
            AppendGroup(
                "owner:witness-validation",
                retained_context=RetainedContextDraft(substrate_ref="unknown.substrate.v1"),
                fact_drafts=(_draft(),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="substrate_ref"):
        store.append(APPEND, empty_substrate)
    with pytest.raises(ValueError, match="substrate_ref"):
        store.preview_record_ids(APPEND, empty_substrate)
    with pytest.raises(ValueError, match="containment"):
        store.append(APPEND, invalid_containment)
    with pytest.raises(ValueError, match="containment"):
        store.preview_record_ids(APPEND, invalid_containment)

    receipt = store.append(APPEND, unknown_substrate)
    retained = store.read_fact(READ, receipt.fact_ids[0])
    assert isinstance(retained, Fact)
    witness = store.read_fact(READ, retained.envelope.witness_ref)
    assert isinstance(witness, Fact)
    assert witness.body.payload["substrate_ref"] == "unknown.substrate.v1"


def test_non_root_record_with_empty_witness_ref_rejected() -> None:
    with pytest.raises(ValueError, match="root witness"):
        canonical_record_input(
            schema_ref="law.non_root.v1",
            mode="capture",
            body={},
            witness="",
        )

    with pytest.raises(ValueError, match="root witness"):
        canonical_record_input(
            schema_ref=WITNESS_SCHEMA_REF,
            mode="capture",
            body={},
            witness="",
        )


def test_ordered_causality_rejects_duplicate_parents() -> None:
    store = SQLiteTraceStore()
    parent = store.append(
        APPEND,
        AppendBatch("law:parent", (AppendGroup("owner:causal", fact_drafts=(_draft("parent"),)),)),
    )

    with pytest.raises(ValueError, match="duplicate causal parent"):
        store.append(
            APPEND,
            AppendBatch(
                "law:duplicate-parent",
                (
                    AppendGroup(
                        "owner:causal",
                        causal_parents=parent.fact_ids,
                        fact_drafts=(
                            FactDraft(
                                kind_label="child",
                                mode="capture",
                                schema_ref="law.child.v1",
                                caused_by_fact_ids=parent.fact_ids,
                            ),
                        ),
                    ),
                ),
            ),
        )


def test_append_local_refs_are_resolved_before_retention() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch(
            "law:local-refs",
            (
                AppendGroup(
                    "owner:local-refs",
                    fact_drafts=(
                        FactDraft(
                            kind_label="parent",
                            append_local_id="local:parent",
                            mode="capture",
                            schema_ref="law.parent.v1",
                        ),
                        FactDraft(
                            kind_label="child",
                            mode="capture",
                            schema_ref="law.child.v1",
                            caused_by_local_refs=("local:parent",),
                        ),
                    ),
                ),
            ),
        ),
    )

    child = store.read_fact(READ, receipt.fact_ids[1])

    assert isinstance(child, Fact)
    assert child.envelope.caused_by_fact_ids == (receipt.fact_ids[0],)
    assert "local:parent" not in child.envelope.caused_by_fact_ids


def test_shape_only_hides_payloads_and_preserves_context_anchor() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch(
            "law:shape-only",
            (
                AppendGroup(
                    "owner:shape-only",
                    retained_context=RetainedContextDraft(active_binding_refs=("binding:workspace",)),
                    fact_drafts=(_draft(secret=True),),
                ),
            ),
        ),
    )
    cut = store.publish_cut(
        PUBLISH_CUT,
        OwnerCutoffSpec(
            frontier_id="law:cut:shape-only",
            target_trace_owner_id="owner:shape-only",
            through_fact_id=receipt.fact_ids[-1],
        ),
    )

    view = store.resolve_cut(
        OperationContext(actor_ref="reader", operation="read", visibility_profile="shape_only"),
        cut.frontier_id,
    )
    visible = view.visible_facts_by_id[receipt.fact_ids[0]]

    assert isinstance(visible, FactShape)
    assert not isinstance(visible, Fact)
    assert view.contexts_by_id == {}
    assert view.context_anchors
    assert view.visible_witnesses_by_id == {}
    assert view.witness_anchors


def test_slice_exposes_visible_witness_records_under_payload_visibility() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch("law:visible-witnesses", (AppendGroup("owner:visible-witnesses", fact_drafts=(_draft(),)),)),
    )

    view = store.read_owner_prefix(READ, "owner:visible-witnesses", 99)
    visible = view.visible_facts_by_id[receipt.fact_ids[0]]

    assert isinstance(visible, Fact)
    witness = view.visible_witnesses_by_id[visible.envelope.witness_ref]
    assert isinstance(witness, Fact)
    assert witness.envelope.schema_ref == WITNESS_SCHEMA_REF
    assert witness.body.payload["actor_ref"] == "runtime:laws"
    root = view.visible_witnesses_by_id[root_witness_record_id()]
    assert isinstance(root, Fact)
    assert root.envelope.schema_ref == "kernel.witness.root.v1"
    assert view.witness_anchors == ()


def test_slice_witness_support_is_closed_to_root_under_shape_visibility() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch(
            "law:witness-closure-shape", (AppendGroup("owner:witness-closure-shape", fact_drafts=(_draft(),)),)
        ),
    )

    view = store.read_owner_prefix(
        OperationContext(actor_ref="reader", operation="read", visibility_profile="shape_only"),
        "owner:witness-closure-shape",
        99,
    )
    visible = view.visible_facts_by_id[receipt.fact_ids[0]]

    assert isinstance(visible, FactShape)
    assert view.visible_witnesses_by_id == {}
    assert {anchor.witness_ref for anchor in view.witness_anchors} == {
        visible.envelope.witness_ref,
        root_witness_record_id(),
    }


def test_witness_support_rejects_witness_cycles() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch("law:witness-cycle", (AppendGroup("owner:witness-cycle", fact_drafts=(_draft(),)),)),
    )
    record = store.read_fact(READ, receipt.fact_ids[0])
    assert isinstance(record, Fact)
    witness_ref = record.envelope.witness_ref
    store._db.execute(
        "UPDATE records SET witness_ref = ? WHERE record_id = ?",
        (witness_ref, witness_ref),
    )

    with pytest.raises(TraceStoreError, match=r"witness.*cycle.*root"):
        store.read_owner_prefix(READ, "owner:witness-cycle", 99)


def test_shape_only_witness_anchor_preserves_witness_ref_shape() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch(
            "law:witness-anchor-shape",
            (AppendGroup("owner:witness-anchor-shape", fact_drafts=(_draft(),)),),
        ),
    )

    view = store.read_owner_prefix(
        OperationContext(actor_ref="reader", operation="read", visibility_profile="shape_only"),
        "owner:witness-anchor-shape",
        99,
    )
    visible = view.visible_facts_by_id[receipt.fact_ids[0]]
    assert isinstance(visible, FactShape)

    ordinary_anchor = next(
        anchor for anchor in view.witness_anchors if anchor.witness_ref == visible.envelope.witness_ref
    )
    root_anchor = next(anchor for anchor in view.witness_anchors if anchor.witness_ref == root_witness_record_id())

    assert ordinary_anchor.visible_shape["witness_ref"] == root_witness_record_id()
    assert root_anchor.visible_shape["witness_ref"] == ""


def test_witness_support_ignores_mode_filter_and_does_not_change_selected_graph() -> None:
    store = SQLiteTraceStore()
    declaration = store.append(
        APPEND,
        AppendBatch(
            "law:witness-mode:declaration",
            (
                AppendGroup(
                    "owner:witness-mode",
                    fact_drafts=(FactDraft(kind_label="intent", mode="declaration", schema_ref="law.intent.v1"),),
                ),
            ),
        ),
    )
    capture = store.append(
        APPEND,
        AppendBatch(
            "law:witness-mode:capture",
            (
                AppendGroup(
                    "owner:witness-mode",
                    causal_parents=declaration.fact_ids,
                    fact_drafts=(_draft("capture"),),
                ),
            ),
        ),
    )

    view = store.read_owner_prefix(READ, "owner:witness-mode", 99, mode_filter="captures_only")
    visible = view.visible_facts_by_id[capture.fact_ids[0]]

    assert view.fact_ids() == capture.fact_ids
    assert view.owner_paths == {"owner:witness-mode": capture.fact_ids}
    assert view.causal_edges == ()
    assert isinstance(visible, Fact)
    assert set(view.visible_witnesses_by_id) == {visible.envelope.witness_ref, root_witness_record_id()}
    assert declaration.fact_ids[0] in {anchor.ref for anchor in view.external_anchors}


def test_witness_support_is_deduped_by_record_id() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch(
            "law:witness-dedupe",
            (
                AppendGroup(
                    "owner:witness-dedupe",
                    fact_drafts=(_draft("first"), _draft("second")),
                ),
            ),
        ),
    )

    view = store.read_owner_prefix(READ, "owner:witness-dedupe", 99)
    witnesses = {
        visible.envelope.witness_ref for visible in view.visible_facts_by_id.values() if isinstance(visible, Fact)
    }

    assert len(witnesses) == 1
    assert view.fact_ids() == receipt.fact_ids
    assert set(view.visible_witnesses_by_id) == {*witnesses, root_witness_record_id()}


def test_missing_witness_support_fails_loudly(tmp_path) -> None:
    db_path = tmp_path / "trace.sqlite"
    missing_witness_ref = "sha256:" + "0" * 64
    with SQLiteTraceStore(db_path) as store:
        receipt = store.append(
            APPEND,
            AppendBatch(
                "law:missing-witness",
                (AppendGroup("owner:missing-witness", fact_drafts=(_draft(),)),),
            ),
        )
        store._db.execute(
            "UPDATE records SET witness_ref = ? WHERE record_id = ?",
            (missing_witness_ref, receipt.fact_ids[0]),
        )

        with pytest.raises(TraceStoreError, match=missing_witness_ref):
            store.read_owner_prefix(READ, "owner:missing-witness", 99)


def test_external_anchor_preserves_out_of_cut_parent() -> None:
    store = SQLiteTraceStore()
    child = store.append(
        APPEND,
        AppendBatch("law:external-child", (AppendGroup("owner:child", fact_drafts=(_draft("child"),)),)),
    )
    parent = store.append(
        APPEND,
        AppendBatch(
            "law:external-parent",
            (
                AppendGroup(
                    "owner:parent",
                    causal_parents=child.fact_ids,
                    fact_drafts=(_draft("parent"),),
                ),
            ),
        ),
    )
    cut = store.publish_cut(
        PUBLISH_CUT,
        OwnerCutoffSpec(
            frontier_id="law:cut:external-parent",
            target_trace_owner_id="owner:parent",
            through_fact_id=parent.fact_ids[-1],
        ),
    )

    view = store.resolve_cut(READ, cut.frontier_id)

    assert view.fact_ids() == parent.fact_ids
    assert view.external_anchors[0].ref == child.fact_ids[0]
    assert view.external_anchors[0].visible_shape["schema_ref"] == "law.child.v1"


def test_causal_closure_policy_controls_filtered_parent_anchors() -> None:
    store = SQLiteTraceStore()
    declaration = store.append(
        APPEND,
        AppendBatch(
            "law:closure-policy:declaration",
            (
                AppendGroup(
                    "owner:closure-policy",
                    fact_drafts=(
                        FactDraft(
                            kind_label="intent",
                            mode="declaration",
                            schema_ref="law.intent.v1",
                        ),
                    ),
                ),
            ),
        ),
    )
    capture = store.append(
        APPEND,
        AppendBatch(
            "law:closure-policy:capture",
            (
                AppendGroup(
                    "owner:closure-policy",
                    causal_parents=declaration.fact_ids,
                    fact_drafts=(_draft("capture"),),
                ),
            ),
        ),
    )

    anchored = store.read_causal_closure(
        READ,
        capture.fact_ids,
        mode_filter="captures_only",
        closure_policy="include_external_anchors",
    )
    visible_only = store.read_causal_closure(
        READ,
        capture.fact_ids,
        mode_filter="captures_only",
        closure_policy="visible_only",
    )

    assert anchored.fact_ids() == capture.fact_ids
    assert anchored.external_anchors[0].ref == declaration.fact_ids[0]
    assert visible_only.fact_ids() == capture.fact_ids
    assert visible_only.external_anchors == ()

    with pytest.raises(ValueError, match="closure policy"):
        store.read_causal_closure(
            READ,
            capture.fact_ids,
            closure_policy="exact_set",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        store.read_causal_closure(READ, capture.fact_ids, "captures_only")  # type: ignore[misc]


def test_causal_closure_mode_filter_does_not_prune_traversal() -> None:
    store = SQLiteTraceStore()
    ancestor = store.append(
        APPEND,
        AppendBatch(
            "law:closure-mode:ancestor",
            (AppendGroup("owner:closure-mode", fact_drafts=(_draft("ancestor"),)),),
        ),
    )
    bridge = store.append(
        APPEND,
        AppendBatch(
            "law:closure-mode:bridge",
            (
                AppendGroup(
                    "owner:closure-mode",
                    causal_parents=ancestor.fact_ids,
                    fact_drafts=(
                        FactDraft(
                            kind_label="bridge",
                            mode="declaration",
                            schema_ref="law.bridge.v1",
                        ),
                    ),
                ),
            ),
        ),
    )
    root = store.append(
        APPEND,
        AppendBatch(
            "law:closure-mode:root",
            (
                AppendGroup(
                    "owner:closure-mode",
                    causal_parents=bridge.fact_ids,
                    fact_drafts=(_draft("root"),),
                ),
            ),
        ),
    )

    visible_only = store.read_causal_closure(
        READ,
        root.fact_ids,
        mode_filter="captures_only",
        closure_policy="visible_only",
    )

    assert visible_only.fact_ids() == (ancestor.fact_ids[0], root.fact_ids[0])
    assert visible_only.causal_edges == ()
    assert visible_only.external_anchors == ()


def test_full_internal_reads_require_trusted_authority() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch("law:full-internal", (AppendGroup("owner:internal", fact_drafts=(_draft(),)),)),
    )
    untrusted = OperationContext(actor_ref="reader", operation="read", visibility_profile="full_internal")
    trusted = OperationContext(
        actor_ref="reader",
        operation="read",
        presented_authority_refs=("trusted:internal",),
        visibility_profile="full_internal",
    )

    with pytest.raises(TraceStoreError, match="full_internal"):
        store.read_owner_prefix(untrusted, "owner:internal", 99)

    assert store.read_owner_prefix(trusted, "owner:internal", 99).fact_ids() == receipt.fact_ids


def test_cut_slice_mode_and_projection_laws() -> None:
    store = SQLiteTraceStore()
    execution_id = execution_id_for("law:execution:create")
    receipt = store.append(
        APPEND,
        create_execution_batch(
            append_intent_id="law:execution:create",
            execution_id=execution_id,
            task_ref="LawTask",
            inputs={"x": 1},
        ),
    )
    cut = store.publish_cut(
        PUBLISH_CUT,
        OwnerCutoffSpec(
            frontier_id="law:cut:execution",
            target_trace_owner_id=execution_id,
            through_fact_id=receipt.fact_ids[-1],
        ),
    )
    before = store.fact_count()

    both = store.resolve_cut(READ, cut.frontier_id)
    declarations = store.resolve_cut(READ, cut.frontier_id, mode_filter="declarations_only")
    captures = store.resolve_cut(READ, cut.frontier_id, mode_filter="captures_only")
    projected = project_execution_slice(both, execution_id)

    assert both.mode_filter == "both"
    assert declarations.fact_ids() == (receipt.fact_ids[0],)
    assert captures.fact_ids() == (receipt.fact_ids[1],)
    assert projected.task_ref == "LawTask"
    assert store.fact_count() == before


def test_later_records_do_not_change_published_cut() -> None:
    store = SQLiteTraceStore()
    first = store.append(
        APPEND,
        AppendBatch("law:cut-stability:first", (AppendGroup("owner:cut-stability", fact_drafts=(_draft("first"),)),)),
    )
    cut = store.publish_cut(
        PUBLISH_CUT,
        OwnerCutoffSpec(
            frontier_id="law:cut:stable",
            target_trace_owner_id="owner:cut-stability",
            through_fact_id=first.fact_ids[-1],
        ),
    )
    store.append(
        APPEND,
        AppendBatch("law:cut-stability:second", (AppendGroup("owner:cut-stability", fact_drafts=(_draft("second"),)),)),
    )

    assert store.resolve_cut(READ, cut.frontier_id).fact_ids() == first.fact_ids
    assert store.read_owner_prefix(READ, "owner:cut-stability", 99).fact_ids() != first.fact_ids


def test_projection_fails_on_incompatible_mode_filter() -> None:
    store = SQLiteTraceStore()
    execution_id = execution_id_for("law:projection-mode")
    receipt = store.append(
        APPEND,
        create_execution_batch(
            append_intent_id="law:projection-mode",
            execution_id=execution_id,
            task_ref="LawTask",
            inputs={},
        ),
    )
    cut = store.publish_cut(
        PUBLISH_CUT,
        OwnerCutoffSpec(
            frontier_id="law:cut:projection-mode",
            target_trace_owner_id=execution_id,
            through_fact_id=receipt.fact_ids[-1],
        ),
    )
    both = store.resolve_cut(READ, cut.frontier_id)

    with pytest.raises(ProjectionModeError, match="mode_filter"):
        ensure_projection_compatible(
            both,
            ProjectionSpec(name="declaration_projection", mode_requirement="declarations_only"),
        )


def test_operation_context_does_not_cross_authorize_cut_publication() -> None:
    store = SQLiteTraceStore()
    receipt = store.append(
        APPEND,
        AppendBatch("law:operation", (AppendGroup("owner:operation", fact_drafts=(_draft(),)),)),
    )

    with pytest.raises(TraceStoreError, match="publish_cut operation"):
        store.publish_cut(
            APPEND,
            OwnerCutoffSpec(
                frontier_id="law:cut:operation",
                target_trace_owner_id="owner:operation",
                through_fact_id=receipt.fact_ids[-1],
            ),
        )


def test_restart_preserves_records_cuts_and_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(db_path) as store:
        receipt = store.append(
            APPEND,
            AppendBatch("law:restart", (AppendGroup("owner:restart", fact_drafts=(_draft(),)),)),
        )
        cut = store.publish_cut(
            PUBLISH_CUT,
            OwnerCutoffSpec(
                frontier_id="law:cut:restart",
                target_trace_owner_id="owner:restart",
                through_fact_id=receipt.fact_ids[-1],
                publisher_trace_owner_id="owner:publisher",
            ),
        )

    with SQLiteTraceStore(db_path) as restarted:
        assert restarted.read_owner_prefix(READ, "owner:restart", 99).fact_ids() == receipt.fact_ids
        assert restarted.resolve_cut(READ, cut.frontier_id).fact_ids() == receipt.fact_ids
