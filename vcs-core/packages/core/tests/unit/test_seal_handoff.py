# under-test: vcs_core._seal_handoff
"""Unit coverage for durable seal handoff records."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from vcs_core import InvalidRepositoryStateError, Store
from vcs_core._operation_journal_controller import OperationJournalController
from vcs_core._seal_handoff import read_seal_handoff, seal_handoff_ref, write_seal_handoff
from vcs_core._world_operation_builder import PreparedCandidateTupleRecord
from vcs_core._world_types import SubstrateHead
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import SubstrateStoreSpec, WorldStorageManager
from vcs_core.types import ScopeInfo, SealCandidateHandoff

from .world_vectors_v2_helpers import create_prepared_candidate


def test_seal_handoff_round_trips_full_candidate_tuple(tmp_path: Path) -> None:
    store = Store(str(tmp_path / "coord" / ".vcscore"))
    scope = ScopeInfo(
        name="child",
        ref="refs/vcscore/scopes/child",
        instance_id="inst-1",
        creation_oid="0" * 40,
        world_id="world-child",
    )
    candidate_tuple = _candidate_tuple(tmp_path)
    handoff = _handoff(scope, candidate_tuple)

    loaded = write_seal_handoff(store, handoff=handoff, candidate_tuple=candidate_tuple)

    assert loaded.handoff == handoff
    assert loaded.candidate_tuple == candidate_tuple
    assert write_seal_handoff(store, handoff=handoff, candidate_tuple=candidate_tuple) == loaded
    assert read_seal_handoff(store, scope) == loaded


def test_seal_handoff_rejects_tuple_mismatch(tmp_path: Path) -> None:
    store = Store(str(tmp_path / "coord" / ".vcscore"))
    scope = ScopeInfo(
        name="child",
        ref="refs/vcscore/scopes/child",
        instance_id="inst-1",
        creation_oid="0" * 40,
        world_id="world-child",
    )
    candidate_tuple = _candidate_tuple(tmp_path)
    handoff = _handoff(scope, candidate_tuple)

    with pytest.raises(InvalidRepositoryStateError, match="candidate_ref disagrees"):
        write_seal_handoff(
            store,
            handoff=replace(handoff, candidate_ref="refs/vcscore/candidates/other"),
            candidate_tuple=candidate_tuple,
        )


def test_candidate_tuple_for_selected_head_matches_full_substrate_identity() -> None:
    head = SubstrateHead(
        binding="workspace",
        kind="filesystem",
        role="workspace",
        store_id="store_workspace",
        store_scope="repo",
        resource_id="fs:repo-main",
        head="a" * 40,
    )
    wrong_store = _candidate_tuple_double(
        binding=head.binding,
        store_id="store_other",
        resource_id=head.resource_id,
        candidate_head=head.head,
    )
    wrong_resource = _candidate_tuple_double(
        binding=head.binding,
        store_id=head.store_id,
        resource_id="fs:repo-other",
        candidate_head=head.head,
    )
    wrong_operation = _candidate_tuple_double(
        operation_id="op-other",
        binding=head.binding,
        store_id=head.store_id,
        resource_id=head.resource_id,
        candidate_head=head.head,
    )
    wrong_candidate_id = _candidate_tuple_double(
        candidate_id="secondary",
        binding=head.binding,
        store_id=head.store_id,
        resource_id=head.resource_id,
        candidate_head=head.head,
    )
    matching = _candidate_tuple_double(
        binding=head.binding,
        store_id=head.store_id,
        resource_id=head.resource_id,
        candidate_head=head.head,
    )
    manager = SimpleNamespace(
        _prepared_operation_from_any_journal_tip=lambda operation_id: SimpleNamespace(
            candidate_tuples=(wrong_store, wrong_resource, wrong_operation, wrong_candidate_id, matching)
        )
    )

    assert (
        OperationJournalController._candidate_tuple_for_selected_head(
            manager,
            operation_id="op-selector",
            producer_operation_id="op-child",
            candidate_id="primary",
            head=head,
        )
        is matching
    )


def _candidate_tuple(tmp_path: Path) -> PreparedCandidateTupleRecord:
    manager = WorldStorageManager.open_or_init(
        tmp_path / "world" / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_workspace",
                    kind="filesystem",
                    resource_id="fs:repo-main",
                ),
                locator="substrates/workspace.git",
            ),
        ),
    )
    candidate, candidate_commit = create_prepared_candidate(
        manager.store("store_workspace"),
        operation_id="op-child",
        binding="workspace",
        payload={"label": "child output"},
        world_store=manager.world_store,
    )
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    return PreparedCandidateTupleRecord(
        candidate=candidate,
        transition=provenance.transition,
        plan=provenance.plan,
        preparation=provenance.preparation,
        candidate_commit=candidate_commit,
    )


def _candidate_tuple_double(
    *,
    operation_id: str = "op-child",
    binding: str,
    store_id: str,
    resource_id: str,
    candidate_head: str,
    candidate_id: str = "primary",
) -> SimpleNamespace:
    return SimpleNamespace(
        candidate=SimpleNamespace(
            operation_id=operation_id,
            binding=binding,
            store_id=store_id,
            resource_id=resource_id,
            head=candidate_head,
            candidate_id=candidate_id,
        )
    )


def _handoff(scope: ScopeInfo, candidate_tuple: PreparedCandidateTupleRecord) -> SealCandidateHandoff:
    candidate = candidate_tuple.candidate
    return SealCandidateHandoff(
        seal_operation_id="seal-inst-1",
        producer_operation_id=candidate.operation_id,
        scope_name=scope.name,
        scope_ref=scope.ref,
        scope_instance_id=scope.instance_id,
        scope_world_id=scope.world_id,
        parent_ref="refs/vcscore/scopes/parent",
        parent_basis_world_oid="1" * 40,
        output_world_oid="2" * 40,
        binding=candidate.binding,
        store_id=candidate.store_id,
        resource_id=candidate.resource_id,
        candidate_id=candidate.candidate_id,
        candidate_ref=candidate.ref,
        candidate_head=candidate.head,
        candidate_tuple_digest=candidate_tuple.tuple_digest(),
        handoff_ref=seal_handoff_ref(scope),
        changed_paths=("child.txt",),
    )
