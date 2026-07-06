# under-test: vcs_core._world_substrate_adapters
"""Capability tests for provider-neutral trace substrate heads in v2 worlds."""

from __future__ import annotations

from pathlib import Path

from vcs_core import WORLD_TRANSITION_SCHEMA, WorldSnapshot
from vcs_core._transition_kernel_records import CandidateOutcomeRecord
from vcs_core._world_refs import world_pin_ref
from vcs_core._world_substrate_adapters import TaskTraceSubstrateAdapter
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import (
    DEFAULT_GROUND_REF,
    CandidateSelection,
    OperationFinalBuilder,
    SubstrateStoreSpec,
    WorldStorageManager,
)


def _trace_manager(tmp_path: Path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_trace",
                    kind="shepherd.trace",
                    resource_id="shepherd-trace:parent",
                ),
                locator="substrates/trace.git",
            ),
        ),
    )


def test_trace_substrate_candidate_can_be_selected_and_pinned_by_parent_world(tmp_path: Path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)
    bundle = trace.create_candidate(
        operation_id="op-parent-select-trace",
        payload={
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:child",
            "frontier_id": "frontier:T11",
        },
    )
    selected_head = trace.head(bundle.candidate.head)
    plan = trace.plan_candidate_selection(bundle)

    prepared = (
        OperationFinalBuilder("op-parent-select-trace")
        .select_candidate_plan(plan=plan)
        .build_prepared(
            operation_kind="select-trace",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=WorldSnapshot((selected_head,)),
            transition={
                "schema": WORLD_TRANSITION_SCHEMA,
                "operation_id": "op-parent-select-trace",
                "parent_worlds": [],
            },
        )
    )
    world_oid = manager.create_world_from_prepared(prepared)

    assert manager.publish_root_world(ref=DEFAULT_GROUND_REF, world_oid=world_oid)
    world = manager.read_world(world_oid)
    pin_ref = world_pin_ref(manager.world_store.world_store_id, world_oid, "trace")

    assert world.snapshot.head_for("trace").kind == "shepherd.trace"
    assert world.snapshot.head_for("trace").role == "shepherd.TraceState"
    assert world.snapshot.head_for("trace").head == bundle.candidate.head
    assert str(manager.store("store_trace").repo.references[pin_ref].target) == bundle.candidate.head


def test_trace_candidate_can_be_archived_as_evidence_without_being_selected(tmp_path: Path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)
    t10 = trace.create_checkpoint(
        "refs/heads/parent",
        {
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T10",
        },
        operation_id="op-parent-initial-trace",
    )
    discarded = trace.create_candidate(
        operation_id="op-child-discard",
        payload={
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:child",
            "frontier_id": "frontier:T11",
        },
        parents=(t10,),
    )
    selected_head = trace.head(t10)
    existing_plan = manager.plan_existing_head_selection(
        operation_id="op-parent-archive-trace",
        head=selected_head,
        selection_kind="checkpoint",
    )

    prepared = (
        OperationFinalBuilder("op-parent-archive-trace")
        .select_existing(plan=existing_plan)
        .archive_candidate(selection=CandidateSelection.from_bundle(discarded))
        .build_prepared(
            operation_kind="archive-trace",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=WorldSnapshot((selected_head,)),
            transition={
                "schema": WORLD_TRANSITION_SCHEMA,
                "operation_id": "op-parent-archive-trace",
                "parent_worlds": [],
            },
        )
    )
    world_oid = manager.create_world_from_prepared(prepared)
    world = manager.read_world(world_oid)
    outcome = CandidateOutcomeRecord.from_operation_final_json(world.operation_final["candidate_outcomes"][0])

    assert world.snapshot.head_for("trace").head == t10
    assert world.snapshot.head_for("trace").head != discarded.candidate.head
    assert outcome.outcome == "archived"
    assert outcome.candidate == discarded.candidate.head
