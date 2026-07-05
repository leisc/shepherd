"""Backend-agnostic TraceStore conformance suite.

Every `TraceStore` backend must satisfy the same behavioral contract. The suite is parametrized over
a store *factory* so it runs against the SQLite reference now and a future `VcsCoreTraceStore` by
adding one fixture param — it is the durable "done" gate for that backend (Phase 0 of the cutover,
`260621-1600-trace.md`). Backend-specific storage tests (SQLite decode / corrupt-row) stay in
`test_trace_store.py`; this file asserts only protocol-surface behavior.

Lifted from the agnostic half of `test_trace_store.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from shepherd2.schemas.run_outputs import (
    RUN_OUTPUT_DESCRIPTOR_SCHEMA,
    RunOutputDescriptorLocator,
    project_run_output_descriptor_payloads,
    project_run_output_descriptors,
    resolve_run_output_descriptor,
    resolve_run_output_descriptor_from_store,
    run_output_descriptor_fact,
)

from shepherd2 import (
    AppendBatch,
    AppendContext,
    AppendGroup,
    AppendIntentConflict,
    Fact,
    FactDraft,
    OwnerCutoffSpec,
    ReadContext,
    SQLiteTraceStore,
    TraceStoreError,
)

if TYPE_CHECKING:
    from pathlib import Path

# A factory: each call (re)opens a durable store at a fixed backing location, so restart / durability
# tests can reopen the same store. A backend that cannot be reopened at a stable location fails here.
StoreFactory = Callable[[], Any]

TRUSTED = AppendContext(
    actor_ref="runtime:conformance",
    presented_witness_refs=("trusted:internal",),
    schema_version_set="shepherd2-slice-a",
    trust_mode="internal",
)
READER = ReadContext(actor_ref="reader")


@pytest.fixture(params=["sqlite"])
def make_store(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    if request.param == "sqlite":
        path = tmp_path / "conformance.sqlite"
        return lambda: SQLiteTraceStore(path)
    raise AssertionError(f"unknown backend {request.param!r}")


def _draft(kind: str, *, caused_by: tuple[str, ...] = (), mode: str = "capture", **payload: Any) -> FactDraft:
    return FactDraft(
        kind_label=kind,
        mode=mode,  # type: ignore[arg-type]
        schema_ref=f"shepherd2.conformance.{kind}.v1",
        payload=payload,
        caused_by_fact_ids=tuple(caused_by),
    )


def _append(store: Any, intent: str, owner: str, *drafts: FactDraft) -> Any:
    return store.append(
        TRUSTED,
        AppendBatch(append_intent_id=intent, groups=(AppendGroup(trace_owner_id=owner, fact_drafts=drafts),)),
    )


def test_append_then_read_owner_prefix(make_store: StoreFactory) -> None:
    with make_store() as store:
        receipt = _append(store, "intent:a", "exec:one", _draft("step", value=1))
        assert store.read_owner_prefix(READER, "exec:one", 99).fact_ids() == receipt.fact_ids


def test_append_intent_idempotent_across_restart(make_store: StoreFactory) -> None:
    with make_store() as store:
        first = _append(store, "intent:start", "exec:parent", _draft("execution_started", execution_id="exec:parent"))
    with make_store() as restarted:
        second = _append(
            restarted, "intent:start", "exec:parent", _draft("execution_started", execution_id="exec:parent")
        )
        assert second == first
        assert restarted.read_owner_prefix(READER, "exec:parent", 99).fact_ids() == first.fact_ids


def test_same_intent_different_batch_is_rejected(make_store: StoreFactory) -> None:
    with make_store() as store:
        _append(store, "intent:once", "exec:one", _draft("step", value=1))
        with pytest.raises(AppendIntentConflict):
            _append(store, "intent:once", "exec:one", _draft("step", value=2))
        only = store.read_owner_prefix(READER, "exec:one", 99)
        first_fact = only.visible_facts_by_id[only.fact_ids()[0]]
        assert isinstance(first_fact, Fact)
        assert first_fact.body.payload == {"value": 1}


def test_preview_record_ids_match_append(make_store: StoreFactory) -> None:
    with make_store() as store:
        batch = AppendBatch(
            append_intent_id="intent:preview",
            groups=(AppendGroup(trace_owner_id="exec:preview", fact_drafts=(_draft("step", value=1),)),),
        )
        previewed = store.preview_record_ids(TRUSTED, batch)
        receipt = store.append(TRUSTED, batch)
        assert previewed == receipt.fact_ids
        assert store.preview_fact_ids(TRUSTED, batch) == receipt.fact_ids


def test_fact_id_is_content_addressed_across_intents(make_store: StoreFactory) -> None:
    # Same content under two intents -> identical content-addressed fact ids, distinct commit receipts.
    with make_store() as store:
        draft = _draft("step", value=1)
        first = _append(store, "intent:first", "exec:one", draft)
        second = _append(store, "intent:second", "exec:one", draft)
        assert first.fact_ids == second.fact_ids
        assert first.fact_ids[0].startswith("sha256:")
        assert first.commit_receipts != second.commit_receipts


def test_content_addressed_fact_spans_multiple_owner_paths(make_store: StoreFactory) -> None:
    with make_store() as store:
        draft = _draft("step", value=1)
        first = _append(store, "intent:owner-a", "owner:a", draft)
        second = _append(store, "intent:owner-b", "owner:b", draft)
        assert first.fact_ids == second.fact_ids
        a_fact = store.read_owner_prefix(READER, "owner:a", 99).visible_facts_by_id[first.fact_ids[0]]
        b_fact = store.read_owner_prefix(READER, "owner:b", 99).visible_facts_by_id[second.fact_ids[0]]
        assert isinstance(a_fact, Fact)
        assert isinstance(b_fact, Fact)
        assert a_fact.trace_owner_id == "owner:a"
        assert b_fact.trace_owner_id == "owner:b"


def test_cut_publish_resolve_roundtrip(make_store: StoreFactory) -> None:
    with make_store() as store:
        receipt = _append(store, "intent:cut", "exec:one", _draft("a"), _draft("b"))
        cut = store.publish_cut(
            TRUSTED,
            OwnerCutoffSpec(
                frontier_id="frontier:cut",
                target_trace_owner_id="exec:one",
                through_fact_id=receipt.fact_ids[-1],
            ),
        )
        assert store.resolve_cut(READER, cut.frontier_id).fact_ids() == receipt.fact_ids


def test_read_owner_cutoff_roundtrips_a_published_cut(make_store: StoreFactory) -> None:
    # read_owner_cutoff is used by the run handles but was missing from the Protocol (Phase 1 adds it).
    with make_store() as store:
        receipt = _append(store, "intent:cutoff", "exec:one", _draft("a"))
        published = store.publish_cut(
            TRUSTED,
            OwnerCutoffSpec(
                frontier_id="frontier:cutoff",
                target_trace_owner_id="exec:one",
                through_fact_id=receipt.fact_ids[-1],
            ),
        )
        cutoff = store.read_owner_cutoff(published.frontier_id)
        assert cutoff.frontier_id == published.frontier_id
        assert cutoff.target_trace_owner_id == "exec:one"
        assert store.resolve_frontier(READER, cutoff.frontier_id).fact_ids() == receipt.fact_ids


def test_causal_closure_includes_parents(make_store: StoreFactory) -> None:
    with make_store() as store:
        parent = _append(store, "intent:parent", "exec:parent", _draft("parent"))
        parent_id = parent.fact_ids[0]
        child = _append(store, "intent:child", "exec:child", _draft("child", caused_by=(parent_id,)))
        child_id = child.fact_ids[0]
        closure = store.read_causal_closure(READER, (child_id,))
        ids = closure.fact_ids()
        assert child_id in ids
        assert parent_id in ids


def test_causal_parent_must_exist(make_store: StoreFactory) -> None:
    with make_store() as store, pytest.raises(TraceStoreError):
        _append(store, "intent:orphan", "exec:one", _draft("child", caused_by=("sha256:does-not-exist",)))


# --- Descriptor fail-closed exactness (Fork 5.2) --------------------------------------------------
# Run-output descriptor resolution is part of the semantic TraceStore ABI, not the future native
# backend. Any backend's resolve_frontier / read_owner_prefix must produce slices over which the
# descriptor projection/resolution functions stay fail-closed. Lifted from test_run_outputs_schema.py;
# the key one is visibility-stability (post-frontier / cross-owner facts are invisible), which a future
# VcsCoreTraceStore solves via the frozen append-head (260623-1530-plan.md Fork 5).


def _citation_payload(*, output_name: str = "workspace", binding: str = "workspace") -> dict[str, object]:
    return {
        "schema": "shepherd2.skeleton.run_output.v0",
        "output_name": output_name,
        "parent_scope_name": "ground",
        "parent_ref": "refs/vcscore/scopes/ground",
        "scope_name": "child",
        "scope_ref": "refs/vcscore/scopes/child",
        "scope_instance_id": "scope-instance",
        "binding": binding,
        "output_world_oid": "world-output",
        "handoff_ref": "refs/vcscore/retained/handoff",
        "candidate_id": "candidate-1",
        "candidate_ref": "refs/vcscore/candidates/1",
        "candidate_head": "sha256:candidate",
        "parent_basis_world_oid": "world-parent",
        "store_id": "store_workspace",
        "resource_id": "workspace",
        "materialization_kind": "tree",
        "retained_handle_head": "sha256:candidate",
        "changed_paths": ["candidate.txt"],
        "trace_run_id": "run-1",
        "trace_execution_id": "exec:run-1",
        "trace_frontier_id": "frontier:run-1",
    }


def _append_descriptor(
    store: Any,
    *,
    execution_id: str = "exec:run-1",
    output_name: str = "workspace",
    binding: str = "workspace",
    citation: dict[str, object] | None = None,
) -> str:
    descriptor = run_output_descriptor_fact(
        execution_id=execution_id,
        output_name=output_name,
        world_binding=binding,
        citation=citation or _citation_payload(output_name=output_name, binding=binding),
    )
    receipt = store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id=f"intent:run-output-descriptor:{execution_id}:{output_name}",
            groups=(AppendGroup(trace_owner_id=execution_id, fact_drafts=(descriptor,)),),
        ),
    )
    return receipt.fact_ids[0]


def _publish_descriptor_frontier(store: Any, *, execution_id: str = "exec:run-1", through_fact_id: str) -> None:
    store.publish_frontier(
        TRUSTED,
        OwnerCutoffSpec(
            frontier_id="frontier:run-1", target_trace_owner_id=execution_id, through_fact_id=through_fact_id
        ),
    )


def test_descriptor_projection_resolution_roundtrip(make_store: StoreFactory) -> None:
    with make_store() as store:
        fact_id = _append_descriptor(store)
        _publish_descriptor_frontier(store, through_fact_id=fact_id)
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        projected = project_run_output_descriptors(frontier_slice, "exec:run-1")
        assert "workspace" in projected
        resolved = resolve_run_output_descriptor(frontier_slice, projected["workspace"].locator)
        assert resolved is not None


def test_descriptor_resolution_rejects_fact_outside_frontier_or_owner_path(make_store: StoreFactory) -> None:
    # Visibility stability: a frontier-resolved slice must exclude facts appended after the frontier
    # and facts in other owners. This is the load-bearing exactness invariant.
    with make_store() as store:
        first = _append_descriptor(store)
        _publish_descriptor_frontier(store, through_fact_id=first)
        after_frontier = _append_descriptor(
            store,
            output_name="backend",
            binding="backend",
            citation=_citation_payload(output_name="backend", binding="backend"),
        )
        other_owner = _append_descriptor(store, execution_id="exec:other")
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        for hidden_fact, name in ((after_frontier, "backend"), (other_owner, "workspace")):
            with pytest.raises(ValueError, match="not visible"):
                resolve_run_output_descriptor(
                    frontier_slice,
                    RunOutputDescriptorLocator(
                        execution_id="exec:run-1",
                        output_name=name,
                        frontier_id="frontier:run-1",
                        descriptor_fact_id=hidden_fact,
                    ),
                )


def test_descriptor_resolution_rejects_output_name_mismatch(make_store: StoreFactory) -> None:
    with make_store() as store:
        fact_id = _append_descriptor(store)
        _publish_descriptor_frontier(store, through_fact_id=fact_id)
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        locator = project_run_output_descriptors(frontier_slice, "exec:run-1")["workspace"].locator
        with pytest.raises(ValueError, match="output_name disagrees"):
            resolve_run_output_descriptor(frontier_slice, replace(locator, output_name="patch"))


def test_descriptor_resolution_rejects_malformed_citation(make_store: StoreFactory) -> None:
    with make_store() as store:
        receipt = store.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:malformed-run-output-descriptor",
                groups=(
                    AppendGroup(
                        trace_owner_id="exec:run-1",
                        fact_drafts=(
                            FactDraft(
                                mode="capture",
                                schema_ref=RUN_OUTPUT_DESCRIPTOR_SCHEMA,
                                kind_label="run_output_descriptor",
                                payload={
                                    "execution_id": "exec:run-1",
                                    "output_name": "workspace",
                                    "world_binding": "workspace",
                                    "citation": [],
                                },
                            ),
                        ),
                    ),
                ),
            ),
        )
        _publish_descriptor_frontier(store, through_fact_id=receipt.fact_ids[0])
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        with pytest.raises(TypeError, match="citation must be an object"):
            resolve_run_output_descriptor(
                frontier_slice,
                RunOutputDescriptorLocator(
                    execution_id="exec:run-1",
                    output_name="workspace",
                    frontier_id="frontier:run-1",
                    descriptor_fact_id=receipt.fact_ids[0],
                ),
            )


def test_descriptor_projection_rejects_duplicate_output_names(make_store: StoreFactory) -> None:
    with make_store() as store:
        first = run_output_descriptor_fact(
            execution_id="exec:run-1", output_name="workspace", world_binding="workspace", citation=_citation_payload()
        )
        second = run_output_descriptor_fact(
            execution_id="exec:run-1", output_name="workspace", world_binding="workspace", citation=_citation_payload()
        )
        receipt = store.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:duplicate-run-output-descriptor",
                groups=(AppendGroup(trace_owner_id="exec:run-1", fact_drafts=(first, second)),),
            ),
        )
        trace_slice = store.read_owner_prefix(READER, "exec:run-1", len(receipt.fact_ids))
        with pytest.raises(ValueError, match="duplicate RunOutput descriptor"):
            project_run_output_descriptor_payloads(trace_slice, "exec:run-1")


def test_descriptor_resolution_rejects_frontier_and_owner_mismatch(make_store: StoreFactory) -> None:
    # The locator's frontier_id and execution_id (owner) must match the resolved slice. This exercises
    # the backend's resolve_frontier + projection, so a future backend could get it wrong (unlike the
    # frontier-less/locator-serialization cases, which reject before touching the backend).
    with make_store() as store:
        fact_id = _append_descriptor(store)
        _publish_descriptor_frontier(store, through_fact_id=fact_id)
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        locator = project_run_output_descriptors(frontier_slice, "exec:run-1")["workspace"].locator
        with pytest.raises(ValueError, match="frontier_id disagrees"):
            resolve_run_output_descriptor(frontier_slice, replace(locator, frontier_id="frontier:other"))
        with pytest.raises(ValueError, match="execution_id disagrees"):
            resolve_run_output_descriptor(
                frontier_slice,
                RunOutputDescriptorLocator(
                    execution_id="exec:other",
                    output_name="workspace",
                    frontier_id="frontier:run-1",
                    descriptor_fact_id=locator.descriptor_fact_id,
                ),
            )


def test_descriptor_resolution_rejects_duplicate_output_names(make_store: StoreFactory) -> None:
    # Duplicate detection through resolve over a frontier slice — distinct from the projection-duplicate
    # case above (a different resolution path the backend's resolve_frontier must keep fail-closed).
    with make_store() as store:
        first = run_output_descriptor_fact(
            execution_id="exec:run-1", output_name="workspace", world_binding="workspace", citation=_citation_payload()
        )
        second = run_output_descriptor_fact(
            execution_id="exec:run-1", output_name="workspace", world_binding="workspace", citation=_citation_payload()
        )
        receipt = store.append(
            TRUSTED,
            AppendBatch(
                append_intent_id="intent:duplicate-run-output-descriptor-frontier",
                groups=(AppendGroup(trace_owner_id="exec:run-1", fact_drafts=(first, second)),),
            ),
        )
        _publish_descriptor_frontier(store, through_fact_id=receipt.fact_ids[-1])
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        with pytest.raises(ValueError, match="duplicate RunOutput descriptor"):
            resolve_run_output_descriptor(
                frontier_slice,
                RunOutputDescriptorLocator(
                    execution_id="exec:run-1",
                    output_name="workspace",
                    frontier_id="frontier:run-1",
                    descriptor_fact_id=receipt.fact_ids[0],
                ),
            )


def test_descriptor_resolution_through_store_wrapper(make_store: StoreFactory) -> None:
    # Product code resolves via resolve_run_output_descriptor_from_store (the wrapper that does
    # resolve_frontier + resolve internally) -- the other descriptor tests call resolve on a slice they
    # resolved by hand. Pin the actual product entrypoint so a future backend satisfies it through the
    # path product code uses.
    with make_store() as store:
        fact_id = _append_descriptor(store)
        _publish_descriptor_frontier(store, through_fact_id=fact_id)
        frontier_slice = store.resolve_frontier(READER, "frontier:run-1")
        locator = project_run_output_descriptors(frontier_slice, "exec:run-1")["workspace"].locator
        resolved = resolve_run_output_descriptor_from_store(store, READER, locator)
        assert resolved.locator == locator
