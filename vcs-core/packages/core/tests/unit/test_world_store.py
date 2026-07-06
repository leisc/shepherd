# under-test: vcs_core._world_store
"""Unit tests for the v2 WorldStore storage kernel."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pygit2
import pytest
from vcs_core import (
    WORLD_TRANSITION_SCHEMA,
    EvidenceRef,
    InvalidRepositoryStateError,
    WorldSnapshot,
    canonical_bytes,
    canonical_digest,
)
from vcs_core._substrate_store import SubstrateStore
from vcs_core._transition_kernel import JsonPayloadTransitionDriver
from vcs_core._transition_kernel_records import (
    CandidateCommitRecord,
    EvidenceRecord,
    RetentionPolicyRequirement,
    RevisionPreparationRecord,
    ValidatedPayloadDescriptor,
)
from vcs_core._world_refs import candidate_ref, evidence_record_ref, world_pin_ref
from vcs_core._world_store import WorldStore
from vcs_core._world_types import (
    OPERATION_FINAL_SCHEMA,
    WORLD_REF_SUBSTRATE_KIND,
    OperationFinalRecord,
    SubstrateHead,
    WorldRefPayload,
    compact_json_bytes,
)
from vcs_core.spi import RelationshipRequirement, SubstrateStoreIdentity

from .world_vectors_v2_helpers import (
    attach_selection_evidence_ref,
    candidate_outcome_for_commit,
    create_prepared_candidate,
    operation_final_with_head_selections,
    selection_evidence_ref,
)


def _workspace_store(tmp_path) -> SubstrateStore:
    return SubstrateStore.open_or_init(
        tmp_path / "substrates" / "workspace.git",
        SubstrateStoreIdentity(
            store_id="store_workspace",
            kind="filesystem",
            resource_id="fs:repo-main",
        ),
    )


def _session_store(tmp_path) -> SubstrateStore:
    return SubstrateStore.open_or_init(
        tmp_path / "substrates" / "session.git",
        SubstrateStoreIdentity(
            store_id="store_session",
            kind="shepherd.session_state",
            resource_id="shepherd-session:child-baseline",
        ),
    )


def _trace_store(tmp_path) -> SubstrateStore:
    return SubstrateStore.open_or_init(
        tmp_path / "substrates" / "trace.git",
        SubstrateStoreIdentity(
            store_id="store_trace",
            kind="shepherd.trace",
            resource_id="shepherd-trace:parent",
        ),
    )


def _world_ref_store(tmp_path) -> SubstrateStore:
    return SubstrateStore.open_or_init(
        tmp_path / "substrates" / "world-ref.git",
        SubstrateStoreIdentity(
            store_id="store_child_world_ref",
            kind=WORLD_REF_SUBSTRATE_KIND,
            resource_id="world-ref:child",
        ),
    )


def _world_store(tmp_path) -> WorldStore:
    return WorldStore.open_or_init(tmp_path / "world.git", world_store_id="store_world_test")


def _prepared_json_revision(
    world: WorldStore,
    store: SubstrateStore,
    ref: str,
    *,
    operation_id: str,
    binding: str,
    payload: dict[str, object],
    semantic_op: str,
    parents: tuple[str, ...] = (),
) -> str:
    prepared = JsonPayloadTransitionDriver(driver_id=f"test.{store.identity.kind}.json").prepare_candidate(
        store=store,
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        parents=parents,
        ingress_kind="command",
        semantic_op=semantic_op,
        relationship_requirements=(),
    )
    evidence_refs = tuple(world.store_evidence_record(record) for record in prepared.evidence_records)
    preparation = RevisionPreparationRecord(
        operation_id=operation_id,
        binding=binding,
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        transition_digest=prepared.transition.transition_digest(),
        revision_plan_digest=prepared.plan.revision_plan_digest(),
        content_digest=prepared.plan.content_digest,
        evidence_digests=prepared.transition.evidence_digests,
        evidence_refs=evidence_refs,
    )
    return store.create_revision_from_prepared(
        ref,
        transition=prepared.transition,
        plan=prepared.plan,
        preparation=preparation,
        payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(prepared.payload),
        payload=prepared.payload,
        parents=prepared.parents,
        evidence_resolver=world.resolve_evidence_ref,
    )


def test_world_store_persists_and_resolves_evidence_records(tmp_path) -> None:
    world = _world_store(tmp_path)
    record = EvidenceRecord(
        operation_id="op-evidence",
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
        ingress_kind="command",
        evidence_kind="command_envelope",
        payload_digest=canonical_digest({"payload": "workspace"}),
        stable_observation={"command": "write"},
    )

    evidence_ref = world.store_evidence_record(record)

    assert evidence_ref.ref == evidence_record_ref("op-evidence", record.record_digest())
    assert world.store_evidence_record(record) == evidence_ref
    assert world.resolve_evidence_ref(evidence_ref, expected_operation_id="op-evidence") == record

    with pytest.raises(InvalidRepositoryStateError, match="operation_id disagrees"):
        world.resolve_evidence_ref(evidence_ref, expected_operation_id="op-other")
    with pytest.raises(InvalidRepositoryStateError, match="evidence_digest disagrees"):
        world.resolve_evidence_ref(
            EvidenceRef(
                ref=evidence_ref.ref,
                evidence_digest=canonical_digest({"wrong": "evidence"}),
                record_digest=evidence_ref.record_digest,
                payload_digest=evidence_ref.payload_digest,
            )
        )
    with pytest.raises(InvalidRepositoryStateError, match="evidence ref is missing"):
        world.resolve_evidence_ref("refs/vcscore/evidence/missing/0")


def test_operation_final_record_canonicalizes_unordered_evidence_lists() -> None:
    outcome_a = {"binding": "a", "candidate": "1" * 40, "outcome": "archived"}
    outcome_b = {"binding": "b", "candidate": "2" * 40, "outcome": "archived"}
    base = {
        "schema": OPERATION_FINAL_SCHEMA,
        "operation_id": "op-canonical",
        "selected": {},
        "candidate_commits": [],
        "candidate_outcomes": [outcome_a, outcome_b],
        "head_selections": [],
        "selection_evidence": [],
    }
    reversed_payload = {**base, "candidate_outcomes": [outcome_b, outcome_a]}

    assert OperationFinalRecord(base).canonical_bytes() == OperationFinalRecord(reversed_payload).canonical_bytes()


def test_operation_final_record_rejects_unexpected_fields() -> None:
    with pytest.raises(ValueError, match="unexpected operation-final fields"):
        OperationFinalRecord(
            {
                "schema": OPERATION_FINAL_SCHEMA,
                "operation_id": "op-extra",
                "selected": {},
                "candidate_commits": [],
                "candidate_outcomes": [],
                "head_selections": [],
                "selection_evidence": [],
                "extra": "ignored-before-strict-validation",
            }
        )


def _operation_final(
    operation_id: str,
    selected: dict[str, str],
    *,
    outcomes: list[dict[str, object]] | None = None,
    candidate_commits=None,
    store_ids=None,
    resource_ids=None,
    selection_kinds=None,
    relationship_requirements=None,
    retention_policy_requirements=None,
):
    return operation_final_with_head_selections(
        operation_id,
        selected,
        outcomes=outcomes,
        candidate_commits=candidate_commits,
        store_ids=store_ids,
        resource_ids=resource_ids,
        selection_kinds=selection_kinds,
        relationship_requirements=relationship_requirements,
        retention_policy_requirements=retention_policy_requirements,
    )


def _bootstrap_json_revision(
    world: WorldStore,
    store: SubstrateStore,
    ref: str,
    *,
    operation_id: str,
    binding: str,
    payload: dict[str, object],
    parents: tuple[str, ...] = (),
) -> str:
    return _prepared_json_revision(
        world,
        store,
        ref,
        operation_id=operation_id,
        binding=binding,
        payload=payload,
        semantic_op="bootstrap",
        parents=parents,
    )


def _bootstrap_operation_final(
    world: WorldStore,
    operation_id: str,
    selected: dict[str, str],
    *,
    stores_by_binding: dict[str, SubstrateStore],
    outcomes: list[dict[str, object]] | None = None,
    candidate_commits=None,
    store_ids=None,
    resource_ids=None,
    relationship_requirements=None,
    retention_policy_requirements=None,
) -> dict[str, object]:
    final = _operation_final(
        operation_id,
        selected,
        outcomes=outcomes,
        candidate_commits=candidate_commits,
        store_ids=store_ids,
        resource_ids=resource_ids,
        selection_kinds=dict.fromkeys(selected, "bootstrap"),
        relationship_requirements=relationship_requirements,
        retention_policy_requirements=retention_policy_requirements,
    )
    for binding, head in selected.items():
        final = attach_selection_evidence_ref(
            final,
            binding=binding,
            evidence_ref=selection_evidence_ref(
                world,
                operation_id=operation_id,
                binding=binding,
                store=stores_by_binding[binding],
                head=head,
                evidence_kind="bootstrap",
            ),
        )
    return final


def _transition(operation_id: str, *, parents: list[str] | None = None, **extra):
    resolved_parents = parents or []
    if resolved_parents and "input_world" not in extra:
        extra["input_world"] = resolved_parents[0]
    return {
        "schema": WORLD_TRANSITION_SCHEMA,
        "operation_id": operation_id,
        "parent_worlds": resolved_parents,
        **extra,
    }


def _snapshot(*heads: SubstrateHead) -> WorldSnapshot:
    return WorldSnapshot(tuple(heads))


def _validate_pin_publish(
    world: WorldStore,
    *,
    ref: str,
    world_oid: str,
    expected_oid: str | None,
    bound_stores: dict[str, SubstrateStore],
) -> bool:
    world.validate_world_commit(world_oid, bound_stores)
    world.pin_selected_heads(world_oid, bound_stores)
    return world._publish_ref_unchecked(ref, world_oid, expected_oid)


def _raw_world_commit(
    world: WorldStore,
    *,
    snapshot: WorldSnapshot,
    transition: dict[str, object],
    operation_final_bytes: bytes,
    operation_final_path: str = "meta/operation-final.json",
    parents: tuple[str, ...] = (),
) -> str:
    transition_payload = {
        **transition,
        "operation_final": {
            "path": operation_final_path,
            "digest": f"sha256:{hashlib.sha256(operation_final_bytes).hexdigest()}",
        },
    }
    manifest = {
        "schema": "vcscore/world/v2",
        "snapshot": snapshot.to_json(),
        "locator_hints": {},
    }
    meta_builder = world.repo.TreeBuilder()
    meta_builder.insert("world.json", world.repo.create_blob(compact_json_bytes(manifest)), pygit2.GIT_FILEMODE_BLOB)
    meta_builder.insert(
        "transition.json",
        world.repo.create_blob(compact_json_bytes(transition_payload)),
        pygit2.GIT_FILEMODE_BLOB,
    )
    meta_builder.insert("operation-final.json", world.repo.create_blob(operation_final_bytes), pygit2.GIT_FILEMODE_BLOB)
    meta_tree = meta_builder.write()
    root_builder = world.repo.TreeBuilder()
    root_builder.insert("meta", meta_tree, pygit2.GIT_FILEMODE_TREE)
    root_tree = root_builder.write()
    signature = pygit2.Signature("vcs-core world store test", "test@example.invalid")
    oid = world.repo.create_commit(
        None,
        signature,
        signature,
        str(transition.get("operation_id", "raw world")),
        root_tree,
        [pygit2.Oid(hex=parent) for parent in parents],
    )
    return str(oid)


def test_world_store_publishes_mixed_snapshot_with_locator_free_identity_and_pins(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-workspace",
        binding="workspace",
        payload={"label": "workspace W43"},
    )
    session_head = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/main",
        operation_id="op-bootstrap-session",
        binding="session",
        payload={"label": "session S7"},
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=session_head, role="shepherd.SessionState"),
    )
    final = _bootstrap_operation_final(
        world,
        "op-initial",
        {"workspace": workspace_head, "session": session_head},
        stores_by_binding={"workspace": workspace, "session": session},
    )

    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-initial"),
        operation_final=final,
        locator_hints={
            "store_workspace": "substrates/workspace.git",
            "store_session": "substrates/session.git",
        },
    )

    assert _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=world_oid,
        expected_oid=None,
        bound_stores={"workspace": workspace, "session": session},
    )
    decoded = world.read_world_commit(world_oid)
    relocated_manifest = {
        **decoded.manifest,
        "locator_hints": {
            "store_workspace": "/imported/substrates/workspace.git",
            "store_session": "/imported/substrates/session.git",
        },
    }

    assert decoded.snapshot == snapshot
    assert decoded.snapshot.digest() == canonical_digest(relocated_manifest["snapshot"])
    assert decoded.manifest["locator_hints"]["store_workspace"] == "substrates/workspace.git"
    assert "store_locator" not in decoded.manifest["snapshot"]["workspace"]
    assert decoded.transition["operation_final"]["digest"] == OperationFinalRecord(final).digest()
    commit = world.repo[pygit2.Oid(hex=world_oid)]
    assert isinstance(commit, pygit2.Commit)
    with pytest.raises(KeyError):
        commit.tree["substrates"]
    assert workspace.repo.references[world_pin_ref(world.world_store_id, world_oid, "workspace")].target == pygit2.Oid(
        hex=workspace_head
    )
    assert session.repo.references[world_pin_ref(world.world_store_id, world_oid, "session")].target == pygit2.Oid(
        hex=session_head
    )


def test_world_store_publish_uses_cas_and_preserves_existing_ref_on_stale_expected(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-workspace",
        binding="workspace",
        payload={"label": "workspace"},
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    stores = {"workspace": workspace}
    p0 = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("p0"),
        operation_final=_bootstrap_operation_final(
            world,
            "p0",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
        ),
    )
    p1 = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("p1", parents=[p0]),
        operation_final=_operation_final("p1", {"workspace": workspace_head}),
        parents=(p0,),
    )
    p2 = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("p2", parents=[p0]),
        operation_final=_operation_final("p2", {"workspace": workspace_head}),
        parents=(p0,),
    )

    assert _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p0,
        expected_oid=None,
        bound_stores=stores,
    )
    assert not _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p2,
        expected_oid=p1,
        bound_stores=stores,
    )
    assert world.repo.references["refs/vcscore/ground"].target == pygit2.Oid(hex=p0)
    stale_pins = world.classify_world_pins(p2, stores, authority_refs=("refs/vcscore/ground",))
    assert stale_pins["orphaned"] == (world_pin_ref(world.world_store_id, p2, "workspace"),)
    assert world.delete_orphan_world_pins(p2, stores, authority_refs=("refs/vcscore/ground",)) == stale_pins["orphaned"]
    assert world.classify_world_pins(p2, stores, authority_refs=("refs/vcscore/ground",))["orphaned"] == ()
    assert _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p1,
        expected_oid=p0,
        bound_stores=stores,
    )
    assert not _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p2,
        expected_oid=p0,
        bound_stores=stores,
    )
    assert world.repo.references["refs/vcscore/ground"].target == pygit2.Oid(hex=p1)
    ancestor_pins = world.classify_world_pins(p0, stores, authority_refs=("refs/vcscore/ground",))
    assert ancestor_pins["published"] == (world_pin_ref(world.world_store_id, p0, "workspace"),)
    published_pins = world.classify_world_pins(p1, stores, authority_refs=("refs/vcscore/ground",))
    assert published_pins["published"] == (world_pin_ref(world.world_store_id, p1, "workspace"),)
    workspace.repo.references[world_pin_ref(world.world_store_id, p1, "workspace")].delete()
    assert world.classify_world_pins(p1, stores, authority_refs=("refs/vcscore/ground",))[
        "missing_for_published_world"
    ] == (world_pin_ref(world.world_store_id, p1, "workspace"),)


def test_world_store_unchecked_publish_ref_raises_for_invalid_refs_and_targets(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-publish"),
        operation_final=_operation_final("op-publish", {"workspace": workspace_head}),
    )

    with pytest.raises(InvalidRepositoryStateError, match="invalid world ref name"):
        world._publish_ref_unchecked("not a ref", world_oid, None)
    with pytest.raises(InvalidRepositoryStateError, match="existing commit"):
        world._publish_ref_unchecked("refs/vcscore/missing-target", "0" * 40, None)


def test_world_store_emits_explicit_sha1_gitlinks_as_inspection_index(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-child-workspace",
        binding="workspace",
        payload={"label": "workspace W1"},
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-gitlinks"),
        operation_final=_operation_final("op-gitlinks", {"workspace": workspace_head}),
        include_gitlinks=True,
    )

    decoded = world.read_world_commit(world_oid)
    commit = world.repo[pygit2.Oid(hex=world_oid)]
    assert isinstance(commit, pygit2.Commit)
    substrates = world.repo[commit.tree["substrates"].id]
    assert isinstance(substrates, pygit2.Tree)
    assert substrates["workspace"].filemode == 0o160000
    assert str(substrates["workspace"].id) == decoded.snapshot.head_for("workspace").head


def test_world_store_omits_or_rejects_gitlinks_that_cannot_represent_object_format(tmp_path) -> None:
    world = _world_store(tmp_path)
    sha256_head = "2" * 64
    head = SubstrateHead(
        binding="artifact",
        kind="shepherd.artifact",
        role="shepherd.ArtifactState",
        store_id="store_artifact",
        store_scope="resource",
        resource_id="artifact:main",
        head=sha256_head,
        object_format="sha256",
    )
    snapshot = _snapshot(head)

    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-manifest-only-sha256"),
        operation_final=_operation_final("op-manifest-only-sha256", {"artifact": sha256_head}),
        include_gitlinks=True,
    )
    commit = world.repo[pygit2.Oid(hex=world_oid)]
    assert isinstance(commit, pygit2.Commit)
    with pytest.raises(KeyError):
        commit.tree["substrates"]

    with pytest.raises(ValueError, match="cannot represent substrate object format"):
        world.create_world_commit(
            snapshot=snapshot,
            transition=_transition("op-bad-gitlink-sha256"),
            operation_final=_operation_final("op-bad-gitlink-sha256", {"artifact": sha256_head}),
            include_gitlinks=True,
            gitlink_heads={"artifact": "1" * 40},
        )


def test_world_store_rejects_optional_gitlink_mismatch(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-retention-workspace",
        binding="workspace",
        payload={"label": "workspace W1"},
    )
    wrong_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/wrong", {"label": "workspace W0"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-mismatched"),
        operation_final=_operation_final("op-mismatched", {"workspace": workspace_head}),
        include_gitlinks=True,
        gitlink_heads={"workspace": wrong_head},
    )

    with pytest.raises(InvalidRepositoryStateError, match="disagrees with manifest"):
        world.read_world_commit(world_oid)


def test_world_store_validates_store_identity_and_alias_policy(tmp_path) -> None:
    world = _world_store(tmp_path)
    session = _session_store(tmp_path)
    head_a = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/a",
        operation_id="op-bootstrap-session-a",
        binding="session_main",
        payload={"label": "session A"},
    )
    head_b = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/b",
        operation_id="op-bootstrap-session-b",
        binding="session_alias",
        payload={"label": "session B"},
    )
    aliased_snapshot = _snapshot(
        session.substrate_head(binding="session_main", head=head_a, role="shepherd.SessionState"),
        session.substrate_head(binding="session_alias", head=head_b, role="shepherd.SessionState"),
    )
    aliased_world = world.create_world_commit(
        snapshot=aliased_snapshot,
        transition=_transition("op-aliased"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-aliased",
            {"session_main": head_a, "session_alias": head_b},
            stores_by_binding={"session_main": session, "session_alias": session},
            store_ids={"session_main": "store_session", "session_alias": "store_session"},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="same-resource aliases"):
        world.validate_world_commit(aliased_world, {"session": session})
    world.validate_world_commit(aliased_world, {"session": session}, allow_same_resource_alias=True)

    mismatched_snapshot = _snapshot(
        SubstrateHead(
            binding="session",
            kind="shepherd.session_state",
            role="shepherd.SessionState",
            store_id="store_session",
            store_scope="resource",
            resource_id="shepherd-session:other",
            head=head_a,
        )
    )
    mismatched_world = world.create_world_commit(
        snapshot=mismatched_snapshot,
        transition=_transition("op-mismatched-identity"),
        operation_final=_operation_final("op-mismatched-identity", {"session": head_a}),
        include_gitlinks=False,
    )
    with pytest.raises(InvalidRepositoryStateError, match="identity mismatch"):
        world.validate_world_commit(mismatched_world, {"session": session})


def test_world_store_validates_world_ref_substrate(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    world_ref_store = _world_ref_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    child_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    child_world = world.create_world_commit(
        snapshot=child_snapshot,
        transition=_transition("op-child-world"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-child-world",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
        ),
    )
    world_ref_head = _bootstrap_json_revision(
        world,
        world_ref_store,
        "refs/heads/child",
        operation_id="op-bootstrap-world-ref",
        binding="child",
        payload=WorldRefPayload(
            world_store_id=world.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
    )
    parent_snapshot = _snapshot(
        world_ref_store.substrate_head(binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    parent_world = world.create_world_commit(
        snapshot=parent_snapshot,
        transition=_transition("op-parent-world"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-parent-world",
            {"child": world_ref_head},
            stores_by_binding={"child": world_ref_store},
            store_ids={"child": world_ref_store.identity.store_id},
            resource_ids={"child": world_ref_store.identity.resource_id},
            retention_policy_requirements={
                "child": (
                    RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
                    RetentionPolicyRequirement(
                        kind="child-world-retention",
                        target=f"world:{child_world}",
                        digest=child_snapshot.digest(),
                    ),
                )
            },
        ),
    )

    world.validate_world_commit(parent_world, {"child": world_ref_store})

    bad_ref_head = world_ref_store.create_unsafe_unprepared_json_revision(
        "refs/heads/bad-child",
        WorldRefPayload(
            world_store_id=world.world_store_id,
            world_oid=child_world,
            snapshot_digest=canonical_digest({"wrong": "snapshot"}),
        ).to_json(),
    )
    bad_snapshot = _snapshot(
        world_ref_store.substrate_head(binding="child", head=bad_ref_head, role="vcscore.WorldRef")
    )
    bad_world = world.create_world_commit(
        snapshot=bad_snapshot,
        transition=_transition("op-parent-bad-world-ref"),
        operation_final=_operation_final(
            "op-parent-bad-world-ref",
            {"child": bad_ref_head},
            store_ids={"child": world_ref_store.identity.store_id},
            resource_ids={"child": world_ref_store.identity.resource_id},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="snapshot_digest disagrees"):
        world.validate_world_commit(bad_world, {"child": world_ref_store})


def test_world_store_validates_operation_final_selected_heads_and_operation_id(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    wrong_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/wrong", {"label": "workspace W0"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    mismatched_selected = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-selected"),
        operation_final=_operation_final("op-selected", {"workspace": wrong_head}),
    )
    with pytest.raises(InvalidRepositoryStateError, match="selected heads disagree"):
        world.validate_world_commit(mismatched_selected, {"workspace": workspace})

    mismatched_operation = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-transition"),
        operation_final=_operation_final("op-final", {"workspace": workspace_head}),
    )
    with pytest.raises(InvalidRepositoryStateError, match="operation_id disagrees"):
        world.validate_world_commit(mismatched_operation, {"workspace": workspace})

    missing_operation = _raw_world_commit(
        world,
        snapshot=snapshot,
        transition=_transition("op-missing-final-id"),
        operation_final_bytes=canonical_bytes(
            {
                "schema": OPERATION_FINAL_SCHEMA,
                "candidate_commits": [],
                "candidate_outcomes": [],
                "head_selections": [],
                "selection_evidence": [],
                "selected": {"workspace": workspace_head},
            }
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="operation_id is required"):
        world.validate_world_commit(missing_operation, {"workspace": workspace})


def test_world_store_requires_typed_head_selection_evidence(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    missing_selection_records = _raw_world_commit(
        world,
        snapshot=snapshot,
        transition=_transition("op-missing-selection"),
        operation_final_bytes=canonical_bytes(
            {
                "schema": OPERATION_FINAL_SCHEMA,
                "operation_id": "op-missing-selection",
                "candidate_commits": [],
                "candidate_outcomes": [],
                "selected": {"workspace": workspace_head},
                "selection_evidence": [],
            }
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="head_selections must be a list"):
        world.validate_world_commit(missing_selection_records, {"workspace": workspace})

    missing_candidate_evidence = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-candidate-selection"),
        operation_final=operation_final_with_head_selections(
            "op-candidate-selection",
            {"workspace": workspace_head},
            selection_kinds={"workspace": "new-candidate"},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="candidate-backed selection requires"):
        world.validate_world_commit(missing_candidate_evidence, {"workspace": workspace})


def test_world_store_validates_selection_retention_policy_requirements(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    world_ref_store = _world_ref_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-retention-workspace",
        binding="workspace",
        payload={"label": "workspace W1"},
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    selected_pin = RetentionPolicyRequirement(kind="selected-head-pin", target=workspace_head)

    wrong_pin_target = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-wrong-pin-target"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-wrong-pin-target",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
            retention_policy_requirements={
                "workspace": (RetentionPolicyRequirement(kind="selected-head-pin", target="0" * 40),)
            },
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="must match selected head"):
        world.validate_world_commit(wrong_pin_target, {"workspace": workspace})

    duplicate_pin = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-duplicate-pin"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-duplicate-pin",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
            retention_policy_requirements={"workspace": (selected_pin, selected_pin)},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="exactly one selected-head-pin"):
        world.validate_world_commit(duplicate_pin, {"workspace": workspace})

    unknown_policy = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-unknown-retention"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-unknown-retention",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
            retention_policy_requirements={
                "workspace": (selected_pin, RetentionPolicyRequirement(kind="unknown-retention", target=workspace_head))
            },
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="unsupported retention policy kind"):
        world.validate_world_commit(unknown_policy, {"workspace": workspace})

    generic_policies = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-generic-retention"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-generic-retention",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
            retention_policy_requirements={
                "workspace": (
                    selected_pin,
                    RetentionPolicyRequirement(kind="candidate-ref", target="candidate:workspace"),
                    RetentionPolicyRequirement(kind="archive-ref", target="candidate:workspace"),
                    RetentionPolicyRequirement(
                        kind="evidence-ref",
                        target="evidence:workspace",
                        digest=canonical_digest({"evidence": "workspace"}),
                    ),
                    RetentionPolicyRequirement(kind="materialization-receipt", target="target:checkout-main"),
                )
            },
        ),
    )
    world.validate_world_commit(generic_policies, {"workspace": workspace})

    child_on_workspace = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-child-retention-on-workspace"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-child-retention-on-workspace",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
            retention_policy_requirements={
                "workspace": (
                    selected_pin,
                    RetentionPolicyRequirement(
                        kind="child-world-retention",
                        target=f"world:{workspace_head}",
                        digest=canonical_digest({"child": "snapshot"}),
                    ),
                )
            },
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match=r"requires a vcscore\.world_ref"):
        world.validate_world_commit(child_on_workspace, {"workspace": workspace})

    child_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    child_world = world.create_world_commit(
        snapshot=child_snapshot,
        transition=_transition("op-retained-child"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-retained-child",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
        ),
    )
    world_ref_head = _bootstrap_json_revision(
        world,
        world_ref_store,
        "refs/heads/child",
        operation_id="op-bootstrap-retention-world-ref",
        binding="child",
        payload=WorldRefPayload(
            world_store_id=world.world_store_id,
            world_oid=child_world,
            snapshot_digest=child_snapshot.digest(),
        ).to_json(),
    )
    world_ref_snapshot = _snapshot(
        world_ref_store.substrate_head(binding="child", head=world_ref_head, role="vcscore.WorldRef")
    )
    missing_child_retention = world.create_world_commit(
        snapshot=world_ref_snapshot,
        transition=_transition("op-world-ref-without-child-retention"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-world-ref-without-child-retention",
            {"child": world_ref_head},
            stores_by_binding={"child": world_ref_store},
            store_ids={"child": world_ref_store.identity.store_id},
            resource_ids={"child": world_ref_store.identity.resource_id},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="world-ref selection requires child-world-retention"):
        world.validate_world_commit(missing_child_retention, {"child": world_ref_store})

    missing_child_digest = world.create_world_commit(
        snapshot=world_ref_snapshot,
        transition=_transition("op-child-retention-missing-digest"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-child-retention-missing-digest",
            {"child": world_ref_head},
            stores_by_binding={"child": world_ref_store},
            store_ids={"child": world_ref_store.identity.store_id},
            resource_ids={"child": world_ref_store.identity.resource_id},
            retention_policy_requirements={
                "child": (
                    RetentionPolicyRequirement(kind="selected-head-pin", target=world_ref_head),
                    RetentionPolicyRequirement(kind="child-world-retention", target=f"world:{child_world}"),
                )
            },
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="requires referenced world snapshot digest"):
        world.validate_world_commit(missing_child_digest, {"child": world_ref_store})


def test_world_store_resolves_head_selection_evidence_refs(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    missing_ref = EvidenceRef(
        ref="refs/vcscore/evidence/missing/0",
        evidence_digest=canonical_digest({"missing": "evidence"}),
        record_digest=canonical_digest({"missing": "record"}),
        payload_digest=canonical_digest({"missing": "payload"}),
    )
    final = attach_selection_evidence_ref(
        _operation_final("op-selection-evidence-ref", {"workspace": workspace_head}),
        binding="workspace",
        evidence_ref=missing_ref,
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-selection-evidence-ref"),
        operation_final=final,
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence ref is missing"):
        world.validate_world_commit(world_oid, {"workspace": workspace})


def test_world_store_requires_checkpoint_selection_evidence(tmp_path) -> None:
    world = _world_store(tmp_path)
    session = _session_store(tmp_path)
    session_head = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    snapshot = _snapshot(session.substrate_head(binding="session", head=session_head, role="shepherd.SessionState"))

    missing_evidence = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-checkpoint-missing-evidence"),
        operation_final=_operation_final(
            "op-checkpoint-missing-evidence",
            {"session": session_head},
            selection_kinds={"session": "checkpoint"},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="checkpoint selection requires checkpoint evidence"):
        world.validate_world_commit(missing_evidence, {"session": session})

    wildcard_record = EvidenceRecord(
        operation_id="op-checkpoint-wildcard",
        evidence_kind="checkpoint",
        payload_digest=canonical_digest({"session": session_head, "kind": "checkpoint"}),
        stable_observation={"session": session_head, "kind": "checkpoint"},
        observed_head=session_head,
    )
    wildcard_final = attach_selection_evidence_ref(
        _operation_final(
            "op-checkpoint-wildcard",
            {"session": session_head},
            selection_kinds={"session": "checkpoint"},
        ),
        binding="session",
        evidence_ref=world.store_evidence_record(wildcard_record),
    )
    wildcard_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-checkpoint-wildcard"),
        operation_final=wildcard_final,
    )
    with pytest.raises(InvalidRepositoryStateError, match="checkpoint selection evidence must exactly observe"):
        world.validate_world_commit(wildcard_world, {"session": session})

    stable_observation = {
        "binding": "session",
        "store_id": session.identity.store_id,
        "resource_id": session.identity.resource_id,
        "substrate_kind": session.identity.kind,
        "head": session_head,
        "kind": "checkpoint",
    }
    command_record = EvidenceRecord(
        operation_id="op-checkpoint-command-evidence",
        binding="session",
        store_id=session.identity.store_id,
        substrate_kind=session.identity.kind,
        ingress_kind="command",
        evidence_kind="checkpoint",
        payload_digest=canonical_digest(stable_observation),
        stable_observation=stable_observation,
        observed_head=session_head,
    )
    command_final = attach_selection_evidence_ref(
        _operation_final(
            "op-checkpoint-command-evidence",
            {"session": session_head},
            selection_kinds={"session": "checkpoint"},
        ),
        binding="session",
        evidence_ref=world.store_evidence_record(command_record),
    )
    command_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-checkpoint-command-evidence"),
        operation_final=command_final,
    )
    with pytest.raises(InvalidRepositoryStateError, match="checkpoint selection evidence must exactly observe"):
        world.validate_world_commit(command_world, {"session": session})

    valid_final = attach_selection_evidence_ref(
        _operation_final("op-checkpoint", {"session": session_head}, selection_kinds={"session": "checkpoint"}),
        binding="session",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-checkpoint",
            binding="session",
            store=session,
            head=session_head,
            evidence_kind="checkpoint",
        ),
    )
    valid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-checkpoint"),
        operation_final=valid_final,
    )
    world.validate_world_commit(valid, {"session": session})

    wrong_head = session.create_unsafe_unprepared_json_revision(
        "refs/checkpoints/S8", {"label": "session S8"}, parents=(session_head,)
    )
    wrong_head_final = attach_selection_evidence_ref(
        _operation_final(
            "op-checkpoint-wrong-head", {"session": session_head}, selection_kinds={"session": "checkpoint"}
        ),
        binding="session",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-checkpoint-wrong-head",
            binding="session",
            store=session,
            head=wrong_head,
            evidence_kind="checkpoint",
        ),
    )
    wrong_head_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-checkpoint-wrong-head"),
        operation_final=wrong_head_final,
    )
    with pytest.raises(InvalidRepositoryStateError, match="checkpoint selection evidence must exactly observe"):
        world.validate_world_commit(wrong_head_world, {"session": session})


def test_world_store_requires_import_selection_evidence(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _prepared_json_revision(
        world,
        workspace,
        "refs/heads/imported",
        operation_id="op-create-import",
        binding="workspace",
        payload={"label": "workspace imported"},
        semantic_op="import",
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    missing_evidence = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-import-missing-evidence"),
        operation_final=_operation_final(
            "op-import-missing-evidence",
            {"workspace": workspace_head},
            selection_kinds={"workspace": "import"},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="import selection requires"):
        world.validate_world_commit(missing_evidence, {"workspace": workspace})

    valid_final = attach_selection_evidence_ref(
        _operation_final("op-import", {"workspace": workspace_head}, selection_kinds={"workspace": "import"}),
        binding="workspace",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-import",
            binding="workspace",
            store=workspace,
            head=workspace_head,
            evidence_kind="import",
        ),
    )
    valid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-import"),
        operation_final=valid_final,
    )
    world.validate_world_commit(valid, {"workspace": workspace})


def test_world_store_requires_bootstrap_for_root_selected_heads(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _prepared_json_revision(
        world,
        workspace,
        "refs/heads/bootstrap",
        operation_id="op-create-bootstrap",
        binding="workspace",
        payload={"label": "workspace bootstrap"},
        semantic_op="bootstrap",
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )

    unchanged_root = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-root-unchanged"),
        operation_final=_operation_final("op-root-unchanged", {"workspace": workspace_head}),
    )
    with pytest.raises(InvalidRepositoryStateError, match="root unchanged selection requires bootstrap"):
        world.validate_world_commit(unchanged_root, {"workspace": workspace})

    missing_evidence = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-bootstrap-missing-evidence"),
        operation_final=_operation_final(
            "op-bootstrap-missing-evidence",
            {"workspace": workspace_head},
            selection_kinds={"workspace": "bootstrap"},
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="bootstrap selection requires bootstrap evidence"):
        world.validate_world_commit(missing_evidence, {"workspace": workspace})

    valid_final = attach_selection_evidence_ref(
        _operation_final(
            "op-bootstrap",
            {"workspace": workspace_head},
            selection_kinds={"workspace": "bootstrap"},
        ),
        binding="workspace",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-bootstrap",
            binding="workspace",
            store=workspace,
            head=workspace_head,
            evidence_kind="bootstrap",
        ),
    )
    valid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-bootstrap"),
        operation_final=valid_final,
    )
    world.validate_world_commit(valid, {"workspace": workspace})


def test_world_store_validates_revert_selection_ancestry(tmp_path) -> None:
    world = _world_store(tmp_path)
    session = _session_store(tmp_path)
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-revert-target",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="revert",
    )
    s8 = session.create_unsafe_unprepared_json_revision("refs/heads/S8", {"label": "session S8"}, parents=(s7,))
    unrelated = session.create_unsafe_unprepared_json_revision("refs/heads/unrelated", {"label": "session unrelated"})
    evidence_ref = selection_evidence_ref(
        world,
        operation_id="op-revert",
        binding="session",
        store=session,
        head=s7,
        evidence_kind="revert",
        selected_from=s8,
    )
    valid_final = attach_selection_evidence_ref(
        _operation_final("op-revert", {"session": s7}, selection_kinds={"session": "revert"}),
        binding="session",
        evidence_ref=evidence_ref,
        selected_from=s8,
    )
    valid = world.create_world_commit(
        snapshot=_snapshot(session.substrate_head(binding="session", head=s7, role="shepherd.SessionState")),
        transition=_transition("op-revert"),
        operation_final=valid_final,
    )
    world.validate_world_commit(valid, {"session": session})

    invalid_final = attach_selection_evidence_ref(
        _operation_final("op-revert", {"session": s7}, selection_kinds={"session": "revert"}),
        binding="session",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-revert",
            binding="session",
            store=session,
            head=s7,
            evidence_kind="revert",
            selected_from=unrelated,
        ),
        selected_from=unrelated,
    )
    invalid = world.create_world_commit(
        snapshot=_snapshot(session.substrate_head(binding="session", head=s7, role="shepherd.SessionState")),
        transition=_transition("op-revert"),
        operation_final=invalid_final,
    )
    with pytest.raises(InvalidRepositoryStateError, match="selected_from must descend"):
        world.validate_world_commit(invalid, {"session": session})


def test_world_store_validates_head_selection_snapshot_identity(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W43"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    wrong_store = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-wrong-selection-store"),
        operation_final=_operation_final(
            "op-wrong-selection-store",
            {"workspace": workspace_head},
            store_ids={"workspace": "store_other"},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="store_id disagrees with world snapshot"):
        world.validate_world_commit(wrong_store, {"workspace": workspace})

    wrong_resource = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-wrong-selection-resource"),
        operation_final=_operation_final(
            "op-wrong-selection-resource",
            {"workspace": workspace_head},
            resource_ids={"workspace": "fs:other"},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="resource_id disagrees with world snapshot"):
        world.validate_world_commit(wrong_resource, {"workspace": workspace})


def test_world_store_rejects_selected_candidate_with_non_candidate_selection(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-base-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    candidate, candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-select",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef")
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    candidate_commit,
                    final_operation_id="op-select",
                    world_store=world,
                )
            ],
            candidate_commits=[candidate_commit],
            selection_kinds={"workspace": "unchanged"},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="non-candidate selection"):
        world.validate_world_commit(world_oid, {"workspace": workspace})


def test_world_store_rejects_selected_head_declared_as_archived_candidate(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    candidate, candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-archive-selected",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=candidate.head, role="shepherd.WorkspaceRef")
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-archive-selected"),
        operation_final=_operation_final(
            "op-archive-selected",
            {"workspace": candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    candidate_commit,
                    final_operation_id="op-archive-selected",
                    world_store=world,
                    outcome="archived",
                )
            ],
            candidate_commits=[candidate_commit],
            selection_kinds={"workspace": "unchanged"},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="must not name selected head"):
        world.validate_world_commit(world_oid, {"workspace": workspace})


def test_world_store_validates_transition_schema_operation_id_and_parents(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    p0 = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-parent"),
        operation_final=_operation_final("op-parent", {"workspace": workspace_head}),
    )

    bad_schema = world.create_world_commit(
        snapshot=snapshot,
        transition={"schema": "vcscore/transition/v1", "operation_id": "op-bad-schema", "parent_worlds": []},
        operation_final=_operation_final("op-bad-schema", {"workspace": workspace_head}),
    )
    with pytest.raises(InvalidRepositoryStateError, match="unsupported world transition schema"):
        world.validate_world_commit(bad_schema, {"workspace": workspace})

    missing_operation = world.create_world_commit(
        snapshot=snapshot,
        transition={"schema": WORLD_TRANSITION_SCHEMA, "parent_worlds": []},
        operation_final=_operation_final("op-missing-transition-id", {"workspace": workspace_head}),
    )
    with pytest.raises(InvalidRepositoryStateError, match="world transition operation_id is required"):
        world.validate_world_commit(missing_operation, {"workspace": workspace})

    parent_mismatch = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-parent-mismatch", parents=[]),
        operation_final=_operation_final("op-parent-mismatch", {"workspace": workspace_head}),
        parents=(p0,),
    )
    with pytest.raises(InvalidRepositoryStateError, match="parent_worlds disagree"):
        world.validate_world_commit(parent_mismatch, {"workspace": workspace})

    missing_input_world = world.create_world_commit(
        snapshot=snapshot,
        transition={"schema": WORLD_TRANSITION_SCHEMA, "operation_id": "op-missing-input", "parent_worlds": [p0]},
        operation_final=_operation_final("op-missing-input", {"workspace": workspace_head}),
        parents=(p0,),
    )
    with pytest.raises(InvalidRepositoryStateError, match="input_world is required"):
        world.validate_world_commit(missing_input_world, {"workspace": workspace})

    input_not_parent = world.create_world_commit(
        snapshot=snapshot,
        transition={
            "schema": WORLD_TRANSITION_SCHEMA,
            "operation_id": "op-input-not-parent",
            "parent_worlds": [p0],
            "input_world": "f" * 40,
        },
        operation_final=_operation_final("op-input-not-parent", {"workspace": workspace_head}),
        parents=(p0,),
    )
    with pytest.raises(InvalidRepositoryStateError, match="input_world must be one of parent_worlds"):
        world.validate_world_commit(input_not_parent, {"workspace": workspace})


def test_world_store_rejects_unchanged_selection_that_moves_from_input_world(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-base-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    w43 = workspace.create_unsafe_unprepared_json_revision(
        "refs/heads/main", {"label": "workspace W43"}, parents=(w42,)
    )
    p0 = world.create_world_commit(
        snapshot=_snapshot(workspace.substrate_head(binding="workspace", head=w42, role="shepherd.WorkspaceRef")),
        transition=_transition("op-base"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-base",
            {"workspace": w42},
            stores_by_binding={"workspace": workspace},
        ),
    )
    moved = world.create_world_commit(
        snapshot=_snapshot(workspace.substrate_head(binding="workspace", head=w43, role="shepherd.WorkspaceRef")),
        transition=_transition("op-move-as-unchanged", parents=[p0]),
        operation_final=_operation_final(
            "op-move-as-unchanged",
            {"workspace": w43},
            selection_kinds={"workspace": "unchanged"},
        ),
        parents=(p0,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="unchanged selection must match input world head"):
        world.validate_world_commit(moved, {"workspace": workspace})


def test_world_store_rejects_unchanged_selection_with_forged_head_identity(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-identity-workspace",
        binding="workspace",
        payload={"label": "workspace"},
    )
    input_head = workspace.substrate_head(
        binding="workspace",
        head=workspace_head,
        role="shepherd.WorkspaceRef",
    )
    p0 = world.create_world_commit(
        snapshot=_snapshot(input_head),
        transition=_transition("op-base-identity"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-base-identity",
            {"workspace": workspace_head},
            stores_by_binding={"workspace": workspace},
        ),
    )
    forged = world.create_world_commit(
        snapshot=_snapshot(replace(input_head, role="shepherd.OtherWorkspaceRef")),
        transition=_transition("op-forged-unchanged-identity", parents=[p0]),
        operation_final=_operation_final(
            "op-forged-unchanged-identity",
            {"workspace": workspace_head},
            selection_kinds={"workspace": "unchanged"},
        ),
        parents=(p0,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="input world head identity"):
        world.validate_world_commit(forged, {"workspace": workspace})


def test_world_store_rejects_noncanonical_operation_final_bytes(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    noncanonical_final = (
        b'vcscore.canonical.v2\n{"selected":{"workspace":"'
        + workspace_head.encode("ascii")
        + b'"},"candidate_outcomes":[],"operation_id":"op-noncanonical","schema":"vcscore/operation-final/v2"}'
    )
    world_oid = _raw_world_commit(
        world,
        snapshot=snapshot,
        transition=_transition("op-noncanonical"),
        operation_final_bytes=noncanonical_final,
    )

    with pytest.raises(ValueError, match="not byte-canonical"):
        world.read_world_commit(world_oid)


def test_world_store_rejects_semantically_noncanonical_operation_final_lists(tmp_path) -> None:
    world = _world_store(tmp_path)
    outcome_a = {"binding": "a", "candidate": "1" * 40, "outcome": "archived"}
    outcome_b = {"binding": "b", "candidate": "2" * 40, "outcome": "archived"}
    canonical_order = OperationFinalRecord(
        {
            "schema": OPERATION_FINAL_SCHEMA,
            "operation_id": "op-list-order",
            "selected": {},
            "candidate_commits": [],
            "candidate_outcomes": [outcome_a, outcome_b],
            "head_selections": [],
            "selection_evidence": [],
        }
    ).payload["candidate_outcomes"]
    raw_final = {
        "schema": OPERATION_FINAL_SCHEMA,
        "operation_id": "op-list-order",
        "selected": {},
        "candidate_commits": [],
        "candidate_outcomes": list(reversed(canonical_order)),
        "head_selections": [],
        "selection_evidence": [],
    }
    world_oid = _raw_world_commit(
        world,
        snapshot=WorldSnapshot(()),
        transition=_transition("op-list-order"),
        operation_final_bytes=canonical_bytes(raw_final),
    )

    with pytest.raises(InvalidRepositoryStateError, match="semantically canonical"):
        world.read_world_commit(world_oid)


def test_world_store_rejects_operation_final_paths_outside_embedded_record(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    workspace_head = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W1"})
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=workspace_head, role="shepherd.WorkspaceRef")
    )
    final = _operation_final("op-bad-path", {"workspace": workspace_head})
    world_oid = _raw_world_commit(
        world,
        snapshot=snapshot,
        transition=_transition("op-bad-path"),
        operation_final_bytes=canonical_bytes(final),
        operation_final_path="meta/elsewhere.json",
    )

    with pytest.raises(InvalidRepositoryStateError, match=r"operation_final\.path must be"):
        world.read_world_commit(world_oid)


def test_world_store_validates_archived_candidate_outcome_has_durable_ref(tmp_path) -> None:
    world = _world_store(tmp_path)
    trace = _trace_store(tmp_path)
    t10 = _bootstrap_json_revision(
        world,
        trace,
        "refs/heads/main",
        operation_id="op-bootstrap-parent-trace",
        binding="trace",
        payload={"label": "parent trace"},
    )
    t11, t11_commit = create_prepared_candidate(
        trace,
        operation_id="op-child-archive",
        binding="trace",
        payload={"label": "unselected trace"},
        parents=(t10,),
        world_store=world,
    )
    trace.repo.references[t11.ref].delete()
    snapshot = _snapshot(trace.substrate_head(binding="trace", head=t10, role="shepherd.TraceState"))
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-archive"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-archive",
            {"trace": t10},
            stores_by_binding={"trace": trace},
            outcomes=[
                candidate_outcome_for_commit(
                    trace,
                    t11_commit,
                    final_operation_id="op-archive",
                    world_store=world,
                    outcome="archived",
                )
            ],
            candidate_commits=[t11_commit],
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="lacks a durable candidate or archive ref"):
        world.validate_world_commit(world_oid, {"trace": trace})
    with pytest.raises(InvalidRepositoryStateError, match="lacks a durable candidate or archive ref"):
        world.validate_world_commit(world_oid, {"trace": trace}, require_selected_candidate_refs=False)
    trace.archive_candidate(operation_id="op-archive", binding="trace", head=t11.head)
    world.validate_world_commit(world_oid, {"trace": trace})


def test_world_store_validates_selected_candidate_outcomes_have_durable_refs(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-select",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
    )
    valid_selected = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-select",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    world.validate_world_commit(valid_selected, {"workspace": workspace})

    valid_outcome = candidate_outcome_for_commit(
        workspace,
        selected_candidate_commit,
        final_operation_id="op-select",
        world_store=world,
    )
    required_provenance_fields = {
        "store_id": "must include store_id",
        "resource_id": "must include resource_id",
        "transition_digest": "must include transition_digest",
        "revision_plan_digest": "must include revision_plan_digest",
        "content_digest": "must include content_digest",
        "revision_preparation_digest": "must include revision_preparation_digest",
        "candidate_commit_digest": "must include candidate_commit_digest",
        "evidence_digests": "must include evidence_digests",
        "evidence_refs": "must include evidence_refs",
    }
    for field, message in required_provenance_fields.items():
        missing_field_outcome = dict(valid_outcome)
        del missing_field_outcome[field]
        missing_field_world = world.create_world_commit(
            snapshot=snapshot,
            transition=_transition("op-select"),
            operation_final=_operation_final(
                "op-select",
                {"workspace": selected_candidate.head},
                outcomes=[missing_field_outcome],
                candidate_commits=[selected_candidate_commit],
            ),
        )
        with pytest.raises(InvalidRepositoryStateError, match=message):
            world.validate_world_commit(missing_field_world, {"workspace": workspace})

    forged_plan_outcome = {**valid_outcome, "revision_plan_digest": canonical_digest({"forged": "plan"})}
    forged_plan_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": selected_candidate.head},
            outcomes=[forged_plan_outcome],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="revision_plan_digest disagrees"):
        world.validate_world_commit(forged_plan_world, {"workspace": workspace})

    forged_commit_outcome = {**valid_outcome, "candidate_commit_digest": canonical_digest({"forged": "commit"})}
    forged_commit_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select"),
        operation_final=_operation_final(
            "op-select",
            {"workspace": selected_candidate.head},
            outcomes=[forged_commit_outcome],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="candidate_commit_digest disagrees"):
        world.validate_world_commit(forged_commit_world, {"workspace": workspace})

    workspace.repo.references[selected_candidate.ref].delete()
    with pytest.raises(InvalidRepositoryStateError, match="selected candidate outcome lacks a durable candidate ref"):
        world.validate_world_commit(valid_selected, {"workspace": workspace})
    world.validate_world_commit(
        valid_selected,
        {"workspace": workspace},
        require_selected_candidate_refs=False,
    )

    bad_producer = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-bad-producer"),
        operation_final=_operation_final(
            "op-bad-producer",
            {"workspace": selected_candidate.head},
            outcomes=[
                {
                    "binding": "workspace",
                    "candidate": selected_candidate.head,
                    "outcome": "selected",
                    "producer_operation_id": 123,
                }
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="producer_operation_id"):
        world.validate_world_commit(bad_producer, {"workspace": workspace})

    unprotected_head = workspace.create_unsafe_unprepared_json_revision(
        "refs/heads/unprotected",
        {"label": "workspace W44"},
        parents=(w42,),
    )
    unprotected_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=unprotected_head, role="shepherd.WorkspaceRef")
    )
    missing_candidate_ref = world.create_world_commit(
        snapshot=unprotected_snapshot,
        transition=_transition("op-unprotected-select"),
        operation_final=_operation_final(
            "op-unprotected-select",
            {"workspace": unprotected_head},
            outcomes=[{"binding": "workspace", "candidate": unprotected_head, "outcome": "selected"}],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="lacks matching candidate commit record"):
        world.validate_world_commit(missing_candidate_ref, {"workspace": workspace})

    unknown_status = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-unknown-candidate-status"),
        operation_final=_operation_final(
            "op-unknown-candidate-status",
            {"workspace": selected_candidate.head},
            outcomes=[{"binding": "workspace", "candidate": selected_candidate.head, "outcome": "deferred"}],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="unknown candidate outcome status"):
        world.validate_world_commit(unknown_status, {"workspace": workspace})


def test_world_store_uses_candidate_id_to_resolve_candidate_outcomes(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-select-alternate",
        binding="workspace",
        candidate_id="alternate",
        payload={"label": "workspace W43 alternate"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select-alternate"),
        operation_final=_operation_final(
            "op-select-alternate",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-select-alternate",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )

    world.validate_world_commit(world_oid, {"workspace": workspace})

    missing_candidate_id = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-select-alternate-missing-id"),
        operation_final=_operation_final(
            "op-select-alternate-missing-id",
            {"workspace": selected_candidate.head},
            outcomes=[{"binding": "workspace", "candidate": selected_candidate.head, "outcome": "selected"}],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="lacks matching candidate commit record"):
        world.validate_world_commit(missing_candidate_id, {"workspace": workspace})


def test_world_store_enforces_selection_relationship_requirements(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)
    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-related-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-parent-merge-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    s8 = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/child",
        operation_id="op-bootstrap-related-session-child",
        binding="session",
        payload={"label": "session S8"},
        parents=(s7,),
    )
    requirement = RelationshipRequirement(
        binding="workspace",
        relation="exact",
        target_binding="session",
        target_head=s7,
    )
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-related",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
        relationship_requirements=(requirement,),
    )
    final = attach_selection_evidence_ref(
        _operation_final(
            "op-related",
            {"workspace": selected_candidate.head, "session": s8},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-related",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
            selection_kinds={"session": "bootstrap"},
            relationship_requirements={"workspace": (requirement,)},
        ),
        binding="session",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-related",
            binding="session",
            store=session,
            head=s8,
            evidence_kind="bootstrap",
        ),
    )
    mismatched_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s8, role="shepherd.SessionState"),
    )
    mismatched_world = world.create_world_commit(
        snapshot=mismatched_snapshot,
        transition=_transition("op-related"),
        operation_final=final,
    )

    with pytest.raises(InvalidRepositoryStateError, match="target head disagrees"):
        world.validate_world_commit(mismatched_world, {"workspace": workspace, "session": session})

    matched_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s7, role="shepherd.SessionState"),
    )
    matched_world = world.create_world_commit(
        snapshot=matched_snapshot,
        transition=_transition("op-related"),
        operation_final=attach_selection_evidence_ref(
            _operation_final(
                "op-related",
                {"workspace": selected_candidate.head, "session": s7},
                outcomes=[
                    candidate_outcome_for_commit(
                        workspace,
                        selected_candidate_commit,
                        final_operation_id="op-related",
                        world_store=world,
                    )
                ],
                candidate_commits=[selected_candidate_commit],
                selection_kinds={"session": "checkpoint"},
                relationship_requirements={"workspace": (requirement,)},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                world,
                operation_id="op-related",
                binding="session",
                store=session,
                head=s7,
                evidence_kind="checkpoint",
            ),
        ),
    )
    world.validate_world_commit(matched_world, {"workspace": workspace, "session": session})


def test_world_store_enforces_descends_from_selection_relationships(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)
    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-descends-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-loop-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    s8 = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/child",
        operation_id="op-bootstrap-descends-session-child",
        binding="session",
        payload={"label": "session S8"},
        parents=(s7,),
    )
    unrelated = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/unrelated",
        operation_id="op-bootstrap-descends-session-unrelated",
        binding="session",
        payload={"label": "session unrelated"},
    )
    requirement = RelationshipRequirement(
        binding="workspace",
        relation="descends-from",
        target_binding="session",
        target_head=s7,
    )
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-descends-related",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
        relationship_requirements=(requirement,),
    )
    final = attach_selection_evidence_ref(
        _operation_final(
            "op-descends-related",
            {"workspace": selected_candidate.head, "session": s8},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-descends-related",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
            selection_kinds={"session": "bootstrap"},
            relationship_requirements={"workspace": (requirement,)},
        ),
        binding="session",
        evidence_ref=selection_evidence_ref(
            world,
            operation_id="op-descends-related",
            binding="session",
            store=session,
            head=s8,
            evidence_kind="bootstrap",
        ),
    )
    matched_world = world.create_world_commit(
        snapshot=_snapshot(
            workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef"),
            session.substrate_head(binding="session", head=s8, role="shepherd.SessionState"),
        ),
        transition=_transition("op-descends-related"),
        operation_final=final,
    )
    world.validate_world_commit(matched_world, {"workspace": workspace, "session": session})

    mismatched_world = world.create_world_commit(
        snapshot=_snapshot(
            workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef"),
            session.substrate_head(binding="session", head=unrelated, role="shepherd.SessionState"),
        ),
        transition=_transition("op-descends-related"),
        operation_final=attach_selection_evidence_ref(
            _operation_final(
                "op-descends-related",
                {"workspace": selected_candidate.head, "session": unrelated},
                outcomes=[
                    candidate_outcome_for_commit(
                        workspace,
                        selected_candidate_commit,
                        final_operation_id="op-descends-related",
                        world_store=world,
                    )
                ],
                candidate_commits=[selected_candidate_commit],
                selection_kinds={"session": "bootstrap"},
                relationship_requirements={"workspace": (requirement,)},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                world,
                operation_id="op-descends-related",
                binding="session",
                store=session,
                head=unrelated,
                evidence_kind="bootstrap",
            ),
        ),
    )
    with pytest.raises(InvalidRepositoryStateError, match="does not descend"):
        world.validate_world_commit(mismatched_world, {"workspace": workspace, "session": session})


def test_world_store_requires_candidate_relationship_requirements_to_match_selection(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-parent-merge-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    requirement = RelationshipRequirement(
        binding="workspace",
        relation="exact",
        target_binding="session",
        target_head=s7,
    )
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-related-mismatch",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s7, role="shepherd.SessionState"),
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-related-mismatch"),
        operation_final=_operation_final(
            "op-related-mismatch",
            {"workspace": selected_candidate.head, "session": s7},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-related-mismatch",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
            relationship_requirements={"workspace": (requirement,)},
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="requirements disagree with preparation"):
        world.validate_world_commit(world_oid, {"workspace": workspace, "session": session})


def test_world_store_requires_child_produced_selection_to_name_parent_producer_world(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-child",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    child_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
    )
    child_world = world.create_world_commit(
        snapshot=child_snapshot,
        transition=_transition("op-child"),
        operation_final=_operation_final(
            "op-child",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-child",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    parent_world = world.create_world_commit(
        snapshot=child_snapshot,
        transition=_transition("op-parent"),
        operation_final=_operation_final(
            "op-parent",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-parent",
                    world_store=world,
                    producer_world_oid=child_world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="must be a parent world"):
        world.validate_world_commit(parent_world, {"workspace": workspace})


def test_world_store_accepts_child_produced_proof_without_head_selection_producer_index(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-child-no-selection-index",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
    )
    child_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-child-no-selection-index"),
        operation_final=_operation_final(
            "op-child-no-selection-index",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-child-no-selection-index",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
    )
    parent_final = _operation_final(
        "op-parent-no-selection-index",
        {"workspace": selected_candidate.head},
        outcomes=[
            candidate_outcome_for_commit(
                workspace,
                selected_candidate_commit,
                final_operation_id="op-parent-no-selection-index",
                world_store=world,
                producer_world_oid=child_world,
            )
        ],
        candidate_commits=[selected_candidate_commit],
    )
    parent_world = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-parent-no-selection-index", parents=[child_world]),
        operation_final=parent_final,
        parents=(child_world,),
    )

    world.validate_world_commit(parent_world, {"workspace": workspace})


def test_world_store_rejects_child_produced_selection_from_invalid_producer_world(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-invalid-child-base",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-child-invalid",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    base_world = world.create_world_commit(
        snapshot=_snapshot(workspace.substrate_head(binding="workspace", head=w42, role="shepherd.WorkspaceRef")),
        transition=_transition("op-base"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-base",
            {"workspace": w42},
            stores_by_binding={"workspace": workspace},
        ),
    )
    invalid_child = world.create_world_commit(
        snapshot=_snapshot(
            workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-child-invalid", parents=[base_world]),
        operation_final=_operation_final(
            "op-child-invalid",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-child-invalid",
                    world_store=world,
                )
            ],
            candidate_commits=[selected_candidate_commit],
            selection_kinds={"workspace": "unchanged"},
        ),
        parents=(base_world,),
    )
    parent_world = world.create_world_commit(
        snapshot=_snapshot(
            workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
        ),
        transition=_transition("op-parent", parents=[base_world, invalid_child]),
        operation_final=_operation_final(
            "op-parent",
            {"workspace": selected_candidate.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    selected_candidate_commit,
                    final_operation_id="op-parent",
                    world_store=world,
                    producer_world_oid=invalid_child,
                )
            ],
            candidate_commits=[selected_candidate_commit],
        ),
        parents=(base_world, invalid_child),
    )

    with pytest.raises(InvalidRepositoryStateError, match="non-candidate selection must not carry candidate evidence"):
        world.validate_world_commit(parent_world, {"workspace": workspace})


def test_world_store_rejects_forged_candidate_producer_operation(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    w42 = workspace.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "workspace W42"})
    selected_candidate, selected_candidate_commit = create_prepared_candidate(
        workspace,
        operation_id="op-real-child",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    forged_ref = candidate_ref("op-forged-child", "workspace")
    workspace.repo.references.create(forged_ref, pygit2.Oid(hex=selected_candidate.head), force=True)
    forged_commit = CandidateCommitRecord(
        operation_id="op-forged-child",
        binding=selected_candidate_commit.binding,
        store_id=selected_candidate_commit.store_id,
        resource_id=selected_candidate_commit.resource_id,
        candidate_head=selected_candidate_commit.candidate_head,
        candidate_ref=forged_ref,
        revision_preparation_digest=selected_candidate_commit.revision_preparation_digest,
    )
    snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=selected_candidate.head, role="shepherd.WorkspaceRef")
    )
    world_oid = world.create_world_commit(
        snapshot=snapshot,
        transition=_transition("op-parent-merge"),
        operation_final=_operation_final(
            "op-parent-merge",
            {"workspace": selected_candidate.head},
            outcomes=[
                {
                    **candidate_outcome_for_commit(
                        workspace,
                        selected_candidate_commit,
                        final_operation_id="op-parent-merge",
                        world_store=world,
                        producer_world_oid="child-world-placeholder",
                    ),
                    "producer_operation_id": "op-forged-child",
                    "candidate_commit_digest": forged_commit.candidate_commit_digest(),
                }
            ],
            candidate_commits=[forged_commit],
        ),
    )

    with pytest.raises(InvalidRepositoryStateError, match="operation_id disagrees"):
        world.validate_world_commit(world_oid, {"workspace": workspace})


def test_parent_merge_can_select_child_workspace_and_pin_prior_session(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)

    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-parent-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    w43, w43_commit = create_prepared_candidate(
        workspace,
        operation_id="op-child",
        binding="workspace",
        payload={"label": "workspace W43"},
        parents=(w42,),
        world_store=world,
    )
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-loop-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    s19 = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/parent",
        operation_id="op-bootstrap-parent-session",
        binding="session",
        payload={"label": "session S19"},
        parents=(s7,),
    )
    s8, s8_commit = create_prepared_candidate(
        session,
        operation_id="op-child",
        binding="session",
        payload={"label": "session S8", "requires": [{"binding": "workspace", "head": w42}]},
        parents=(s7,),
        world_store=world,
    )

    def checkpoint_final(operation_id: str, selected: dict[str, str]) -> dict[str, object]:
        return attach_selection_evidence_ref(
            _operation_final(operation_id, selected, selection_kinds={"session": "checkpoint"}),
            binding="session",
            evidence_ref=selection_evidence_ref(
                world,
                operation_id=operation_id,
                binding="session",
                store=session,
                head=selected["session"],
                evidence_kind="checkpoint",
            ),
        )

    p0_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s19, role="shepherd.SessionState"),
    )
    p0 = world.create_world_commit(
        snapshot=p0_snapshot,
        transition=_transition("op-parent-initial"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-parent-initial",
            {"workspace": w42, "session": s19},
            stores_by_binding={"workspace": workspace, "session": session},
        ),
    )
    c0_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s7, role="shepherd.SessionState"),
    )
    c0 = world.create_world_commit(
        snapshot=c0_snapshot,
        transition=_transition("op-child-fork", parents=[p0]),
        operation_final=checkpoint_final("op-child-fork", {"workspace": w42, "session": s7}),
        parents=(p0,),
    )
    c1_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s8.head, role="shepherd.SessionState"),
    )
    c1 = world.create_world_commit(
        snapshot=c1_snapshot,
        transition=_transition("op-child", parents=[c0]),
        operation_final=_operation_final(
            "op-child",
            {"workspace": w43.head, "session": s8.head},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    w43_commit,
                    final_operation_id="op-child",
                    world_store=world,
                ),
                candidate_outcome_for_commit(
                    session,
                    s8_commit,
                    final_operation_id="op-child",
                    world_store=world,
                ),
            ],
            candidate_commits=[w43_commit, s8_commit],
        ),
        parents=(c0,),
    )
    p1_snapshot = _snapshot(
        workspace.substrate_head(binding="workspace", head=w43.head, role="shepherd.WorkspaceRef"),
        session.substrate_head(binding="session", head=s7, role="shepherd.SessionState"),
    )
    p1 = world.create_world_commit(
        snapshot=p1_snapshot,
        transition=_transition(
            "op-parent-merge",
            parents=[p0, c1],
            input_world=p0,
            changes={
                "workspace": {"from": w42, "to": w43.head, "policy": "take-child"},
                "session": {"from": s19, "to": s7, "policy": "pin-prior"},
            },
        ),
        operation_final=attach_selection_evidence_ref(
            _operation_final(
                "op-parent-merge",
                {"workspace": w43.head, "session": s7},
                outcomes=[
                    candidate_outcome_for_commit(
                        workspace,
                        w43_commit,
                        final_operation_id="op-parent-merge",
                        world_store=world,
                        producer_world_oid=c1,
                    )
                ],
                candidate_commits=[w43_commit],
                selection_kinds={"session": "checkpoint"},
            ),
            binding="session",
            evidence_ref=selection_evidence_ref(
                world,
                operation_id="op-parent-merge",
                binding="session",
                store=session,
                head=s7,
                evidence_kind="checkpoint",
            ),
        ),
        parents=(p0, c1),
    )

    assert _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p0,
        expected_oid=None,
        bound_stores={"workspace": workspace, "session": session},
    )
    assert _validate_pin_publish(
        world,
        ref="refs/vcscore/ground",
        world_oid=p1,
        expected_oid=p0,
        bound_stores={"workspace": workspace, "session": session},
    )
    decoded = world.read_world_commit(p1)

    assert decoded.snapshot.head_for("workspace").head == w43.head
    assert decoded.snapshot.head_for("session").head == s7
    assert decoded.snapshot.head_for("session").head not in {s8.head, s19}


def test_parent_subtask_loop_advances_workspace_and_reuses_session_checkpoint(tmp_path) -> None:
    world = _world_store(tmp_path)
    workspace = _workspace_store(tmp_path)
    session = _session_store(tmp_path)

    def checkpoint_final(operation_id: str, selected: dict[str, str]) -> dict[str, object]:
        return attach_selection_evidence_ref(
            _operation_final(operation_id, selected, selection_kinds={"session": "checkpoint"}),
            binding="session",
            evidence_ref=selection_evidence_ref(
                world,
                operation_id=operation_id,
                binding="session",
                store=session,
                head=selected["session"],
                evidence_kind="checkpoint",
            ),
        )

    def run_child_iteration(
        *,
        iteration: int,
        parent_world: str,
        workspace_base: str,
        session_checkpoint: str,
    ) -> tuple[str, str, str]:
        fork_operation = f"op-child-loop-fork-{iteration}"
        child_operation = f"op-child-loop-{iteration}"
        parent_operation = f"op-parent-loop-merge-{iteration}"
        fork_world = world.create_world_commit(
            snapshot=_snapshot(
                workspace.substrate_head(binding="workspace", head=workspace_base, role="shepherd.WorkspaceRef"),
                session.substrate_head(binding="session", head=session_checkpoint, role="shepherd.SessionState"),
            ),
            transition=_transition(fork_operation, parents=[parent_world]),
            operation_final=checkpoint_final(
                fork_operation,
                {"workspace": workspace_base, "session": session_checkpoint},
            ),
            parents=(parent_world,),
        )
        next_workspace, workspace_commit = create_prepared_candidate(
            workspace,
            operation_id=child_operation,
            binding="workspace",
            payload={"label": f"workspace W{42 + iteration}"},
            parents=(workspace_base,),
            world_store=world,
        )
        child_session, child_session_commit = create_prepared_candidate(
            session,
            operation_id=child_operation,
            binding="session",
            payload={"label": f"session child {iteration}"},
            parents=(session_checkpoint,),
            world_store=world,
        )
        child_world = world.create_world_commit(
            snapshot=_snapshot(
                workspace.substrate_head(binding="workspace", head=next_workspace.head, role="shepherd.WorkspaceRef"),
                session.substrate_head(binding="session", head=child_session.head, role="shepherd.SessionState"),
            ),
            transition=_transition(child_operation, parents=[fork_world]),
            operation_final=_operation_final(
                child_operation,
                {"workspace": next_workspace.head, "session": child_session.head},
                outcomes=[
                    candidate_outcome_for_commit(
                        workspace,
                        workspace_commit,
                        final_operation_id=child_operation,
                        world_store=world,
                    ),
                    candidate_outcome_for_commit(
                        session,
                        child_session_commit,
                        final_operation_id=child_operation,
                        world_store=world,
                    ),
                ],
                candidate_commits=[workspace_commit, child_session_commit],
            ),
            parents=(fork_world,),
        )
        merge_final = _operation_final(
            parent_operation,
            {"workspace": next_workspace.head, "session": session_checkpoint},
            outcomes=[
                candidate_outcome_for_commit(
                    workspace,
                    workspace_commit,
                    final_operation_id=parent_operation,
                    world_store=world,
                    producer_world_oid=child_world,
                )
            ],
            candidate_commits=[workspace_commit],
            selection_kinds={"session": "checkpoint"},
        )
        parent_merge = world.create_world_commit(
            snapshot=_snapshot(
                workspace.substrate_head(binding="workspace", head=next_workspace.head, role="shepherd.WorkspaceRef"),
                session.substrate_head(binding="session", head=session_checkpoint, role="shepherd.SessionState"),
            ),
            transition=_transition(parent_operation, parents=[parent_world, child_world]),
            operation_final=attach_selection_evidence_ref(
                merge_final,
                binding="session",
                evidence_ref=selection_evidence_ref(
                    world,
                    operation_id=parent_operation,
                    binding="session",
                    store=session,
                    head=session_checkpoint,
                    evidence_kind="checkpoint",
                ),
            ),
            parents=(parent_world, child_world),
        )
        return parent_merge, next_workspace.head, child_session.head

    w42 = _bootstrap_json_revision(
        world,
        workspace,
        "refs/heads/main",
        operation_id="op-bootstrap-loop-workspace",
        binding="workspace",
        payload={"label": "workspace W42"},
    )
    s7 = _prepared_json_revision(
        world,
        session,
        "refs/checkpoints/S7",
        operation_id="op-create-loop-checkpoint",
        binding="session",
        payload={"label": "session S7"},
        semantic_op="checkpoint",
    )
    s19 = _bootstrap_json_revision(
        world,
        session,
        "refs/heads/parent",
        operation_id="op-bootstrap-loop-session",
        binding="session",
        payload={"label": "session S19"},
        parents=(s7,),
    )
    p0 = world.create_world_commit(
        snapshot=_snapshot(
            workspace.substrate_head(binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            session.substrate_head(binding="session", head=s19, role="shepherd.SessionState"),
        ),
        transition=_transition("op-parent-loop-initial"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-parent-loop-initial",
            {"workspace": w42, "session": s19},
            stores_by_binding={"workspace": workspace, "session": session},
        ),
    )

    p1, w43, child_session_1 = run_child_iteration(
        iteration=1,
        parent_world=p0,
        workspace_base=w42,
        session_checkpoint=s7,
    )
    p2, w44, child_session_2 = run_child_iteration(
        iteration=2,
        parent_world=p1,
        workspace_base=w43,
        session_checkpoint=s7,
    )

    world.validate_world_commit(p2, {"workspace": workspace, "session": session})
    decoded = world.read_world_commit(p2)
    assert decoded.snapshot.head_for("workspace").head == w44
    assert decoded.snapshot.head_for("session").head == s7
    assert decoded.snapshot.head_for("session").head not in {child_session_1, child_session_2, s19}


def test_trace_candidate_can_be_archived_as_evidence_without_being_selected(tmp_path) -> None:
    world = _world_store(tmp_path)
    trace = _trace_store(tmp_path)
    t10 = _bootstrap_json_revision(
        world,
        trace,
        "refs/heads/main",
        operation_id="op-bootstrap-parent-trace",
        binding="trace",
        payload={
            "schema": "vcscore/trace-revision/v1",
            "kind": "shepherd.trace",
            "label": "parent trace",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T10",
        },
    )
    t11, t11_commit = create_prepared_candidate(
        trace,
        operation_id="op-child-discard",
        binding="trace",
        payload={
            "schema": "vcscore/trace-revision/v1",
            "kind": "shepherd.trace",
            "label": "discarded child trace",
            "trace_owner_id": "shepherd-run:child",
            "frontier_id": "frontier:T11",
        },
        parents=(t10,),
        world_store=world,
    )
    p0_snapshot = _snapshot(trace.substrate_head(binding="trace", head=t10, role="shepherd.TraceState"))
    p0 = world.create_world_commit(
        snapshot=p0_snapshot,
        transition=_transition("op-parent-initial"),
        operation_final=_bootstrap_operation_final(
            world,
            "op-parent-initial",
            {"trace": t10},
            stores_by_binding={"trace": trace},
        ),
    )
    c1_snapshot = _snapshot(trace.substrate_head(binding="trace", head=t11.head, role="shepherd.TraceState"))
    c1 = world.create_world_commit(
        snapshot=c1_snapshot,
        transition=_transition("op-child-discard", parents=[p0]),
        operation_final=_operation_final("op-child-discard", {"trace": t11.head}),
        parents=(p0,),
    )
    final = _operation_final(
        "op-parent-archive-trace",
        {"trace": t10},
        outcomes=[
            candidate_outcome_for_commit(
                trace,
                t11_commit,
                final_operation_id="op-parent-archive-trace",
                world_store=world,
                outcome="archived",
            )
        ],
        candidate_commits=[t11_commit],
    )
    p1 = world.create_world_commit(
        snapshot=p0_snapshot,
        transition=_transition("op-parent-archive-trace", parents=[p0, c1], input_world=p0),
        operation_final=final,
        parents=(p0, c1),
    )

    archive_ref = trace.archive_candidate(operation_id="op-parent-archive-trace", binding="trace", head=t11.head)
    world.validate_world_commit(p1, {"trace": trace})
    decoded = world.read_world_commit(p1)

    assert decoded.snapshot.head_for("trace").head == t10
    assert decoded.snapshot.head_for("trace").head != t11.head
    assert decoded.operation_final["candidate_outcomes"][0]["outcome"] == "archived"
    assert trace.repo.references[archive_ref].target == pygit2.Oid(hex=t11.head)


def test_world_store_rejects_existing_non_repo_paths(tmp_path) -> None:
    path = tmp_path / "world.git"
    path.mkdir()
    (path / "README").write_text("not a git repository", encoding="utf-8")

    with pytest.raises(InvalidRepositoryStateError, match="not a Git repository"):
        WorldStore.open_or_init(path, world_store_id="store_world_test")
