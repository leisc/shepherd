# under-test: vcs_core._substrate_store
"""Unit tests for the v2 Git-backed SubstrateStore."""

from __future__ import annotations

from dataclasses import replace

import pygit2
import pytest
from vcs_core import EvidenceRef, InvalidRepositoryStateError, canonical_bytes, canonical_digest
from vcs_core._substrate_store import SubstrateStore
from vcs_core._transition_kernel_records import (
    EvidenceRecord,
    LogicalTransition,
    PreparedRevisionPlan,
    RevisionPreparationRecord,
    ValidatedPayloadDescriptor,
)
from vcs_core._world_refs import (
    candidate_archive_ref,
    candidate_ref,
    encode_ref_component,
    is_ref_safe_component,
)
from vcs_core._world_types import SubstrateRevisionMetadata, compact_json_bytes, load_canonical_json
from vcs_core.git_store import build_tree, create_commit_with_recovery, insert_tree_entry
from vcs_core.spi import KeyedJsonPut, KeyedJsonTreeDraft, SubstrateStoreIdentity


def _identity(
    *,
    store_id: str = "store_workspace",
    kind: str = "filesystem",
    resource_id: str = "fs:repo-main",
) -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id=store_id, kind=kind, resource_id=resource_id)


def test_substrate_store_persists_identity_and_rejects_mismatch(tmp_path) -> None:
    path = tmp_path / "workspace.git"
    store = SubstrateStore.open_or_init(path, _identity())

    reopened = SubstrateStore.open_or_init(path, _identity())

    assert reopened.read_identity() == store.identity
    with pytest.raises(InvalidRepositoryStateError, match="identity mismatch"):
        SubstrateStore.open_or_init(path, _identity(resource_id="fs:other"))


def test_substrate_store_rejects_existing_non_repo_paths(tmp_path) -> None:
    path = tmp_path / "workspace.git"
    path.mkdir()
    (path / "README").write_text("not a git repository", encoding="utf-8")

    with pytest.raises(InvalidRepositoryStateError, match="not a Git repository"):
        SubstrateStore.open_or_init(path, _identity())


def test_substrate_store_candidate_creation_publishes_durable_candidate_ref(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    parent = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "parent"})

    candidate = store.create_unsafe_unprepared_candidate(
        operation_id="op child/unsafe",
        binding="workspace/main",
        payload={"label": "candidate"},
        parents=(parent,),
    )

    assert candidate.ref == candidate_ref("op child/unsafe", "workspace/main")
    assert store.repo.references[candidate.ref].target == pygit2.Oid(hex=candidate.head)
    store.validate_candidate_ref(
        operation_id="op child/unsafe",
        binding="workspace/main",
        expected_head=candidate.head,
    )
    with pytest.raises(InvalidRepositoryStateError, match="disagrees"):
        store.validate_candidate_ref(
            operation_id="op child/unsafe",
            binding="workspace/main",
            expected_head=parent,
        )
    with pytest.raises(InvalidRepositoryStateError, match="without a durable candidate ref"):
        store.validate_candidate_ref(
            operation_id="missing",
            binding="workspace/main",
            expected_head=candidate.head,
        )
    with pytest.raises(InvalidRepositoryStateError, match="Candidate ref already exists"):
        store.create_unsafe_unprepared_candidate(
            operation_id="op child/unsafe",
            binding="workspace/main",
            payload={"label": "duplicate"},
            parents=(parent,),
        )


def test_substrate_store_allows_multiple_candidate_ids_for_one_operation_binding(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    parent = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "parent"})

    primary = store.create_unsafe_unprepared_candidate(
        operation_id="op-multi",
        binding="workspace",
        payload={"label": "primary"},
        parents=(parent,),
    )
    alternate = store.create_unsafe_unprepared_candidate(
        operation_id="op-multi",
        binding="workspace",
        candidate_id="alternate",
        payload={"label": "alternate"},
        parents=(parent,),
    )

    assert primary.candidate_id == "primary"
    assert alternate.candidate_id == "alternate"
    assert primary.ref == candidate_ref("op-multi", "workspace", "primary")
    assert alternate.ref == candidate_ref("op-multi", "workspace", "alternate")
    store.validate_candidate_ref(
        operation_id="op-multi",
        binding="workspace",
        candidate_id="alternate",
        expected_head=alternate.head,
    )
    archive_ref = store.archive_candidate(
        operation_id="op-multi",
        binding="workspace",
        candidate_id="alternate",
        head=alternate.head,
    )
    assert archive_ref == candidate_archive_ref("op-multi", "workspace", "alternate")


def test_substrate_store_archives_unselected_candidate_idempotently(tmp_path) -> None:
    store = SubstrateStore.open_or_init(
        tmp_path / "trace.git", _identity(store_id="store_trace", kind="shepherd.trace")
    )
    candidate = store.create_unsafe_unprepared_candidate(
        operation_id="op trace",
        binding="trace",
        payload={"schema": "vcscore/trace-revision/v1", "label": "trace"},
    )

    archive_ref = store.archive_candidate(operation_id="op trace", binding="trace", head=candidate.head)

    assert archive_ref == candidate_archive_ref("op trace", "trace")
    assert store.archive_candidate(operation_id="op trace", binding="trace", head=candidate.head) == archive_ref
    assert store.repo.references[archive_ref].target == pygit2.Oid(hex=candidate.head)


def test_substrate_store_persists_canonical_revision_metadata_for_all_json_revisions(tmp_path) -> None:
    store = SubstrateStore.open_or_init(
        tmp_path / "session.git",
        _identity(store_id="store_session", kind="shepherd.session_state", resource_id="shepherd-session:child"),
    )
    base = store.create_unsafe_unprepared_json_revision("refs/checkpoints/base", {"label": "base"})
    child = store.create_unsafe_unprepared_candidate(
        operation_id="op-session",
        binding="session",
        payload={"schema": "example/session", "label": "child"},
        parents=(base,),
    )

    base_metadata = store.read_revision_metadata(base)
    child_metadata = store.read_revision_metadata(child.head)

    assert base_metadata.payload_digest.startswith("sha256:")
    assert base_metadata.kind == "shepherd.session_state"
    assert base_metadata.resource_id == "shepherd-session:child"
    assert base_metadata.materialization_class == "noop"
    assert base_metadata.parent_heads == ()
    assert child_metadata.parent_heads == (base,)
    assert child_metadata.produced_by_operation_id == "op-session"
    commit = store.repo[pygit2.Oid(hex=child.head)]
    assert "meta" in commit.tree
    assert "substrate-revision.json" in store.repo[commit.tree["meta"].id]


def test_substrate_store_creates_prepared_json_candidate_with_kernel_metadata(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    evidence_record = _evidence_record("op-prepared")
    evidence = _evidence_ref("op-prepared")
    transition = LogicalTransition(
        binding="workspace",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=(base,),
        ingress_kind="command",
        semantic_op="FilePatch",
        payload_digest=canonical_digest(payload),
        evidence_digests=(evidence.evidence_digest,),
    )
    plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=transition.base_heads,
        content_digest=canonical_digest(payload),
        materialization_class="noop",
        entries=({"path": "revision.json", "payload_digest": canonical_digest(payload)},),
    )
    preparation = RevisionPreparationRecord(
        operation_id="op-prepared",
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence,),
    )
    payload_descriptor = ValidatedPayloadDescriptor.for_json_payload(payload)

    with pytest.raises(InvalidRepositoryStateError, match="evidence_refs require a resolver"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            parents=(base,),
        )

    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
        payload=payload,
        parents=(base,),
        evidence_resolver=lambda _ref: evidence_record,
    )

    assert candidate.ref == candidate_ref("op-prepared", "workspace")
    metadata = store.validate_candidate(
        candidate.head,
        expected_transition_digest=transition.transition_digest(),
        expected_revision_plan_digest=plan.revision_plan_digest(),
        expected_revision_preparation_digest=preparation.revision_preparation_digest(),
    )
    assert metadata.content_digest == canonical_digest(payload)
    assert metadata.evidence_digests == (evidence.evidence_digest,)
    assert metadata.materialization_class == plan.materialization_class
    assert metadata.ingress_kind == "command"
    assert metadata.semantic_op == "FilePatch"
    commit = store.repo[pygit2.Oid(hex=candidate.head)]
    assert (
        load_canonical_json(_read_blob_bytes(store, commit.tree, "meta/logical-transition.json"))
        == transition.to_json()
    )
    assert (
        load_canonical_json(_read_blob_bytes(store, commit.tree, "meta/prepared-revision-plan.json")) == plan.to_json()
    )
    assert (
        load_canonical_json(_read_blob_bytes(store, commit.tree, "meta/revision-preparation.json"))
        == preparation.to_json()
    )
    assert (
        load_canonical_json(_read_blob_bytes(store, commit.tree, "meta/payload-descriptor.json"))
        == ValidatedPayloadDescriptor.for_json_payload(payload).to_json()
    )
    provenance = store.validate_prepared_candidate(candidate.head, evidence_resolver=lambda _ref: evidence_record)
    assert provenance.payload_descriptor == ValidatedPayloadDescriptor.for_json_payload(payload)

    with pytest.raises(InvalidRepositoryStateError, match="transition_digest disagrees"):
        store.validate_candidate(candidate.head, expected_transition_digest=canonical_digest({"wrong": True}))


def test_substrate_store_creates_prepared_keyed_json_tree_candidate(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "runs.git", _identity(kind="shepherd.runs", resource_id="runs:main"))
    manifest = {
        "schema": "shepherd.workspace_control.runs.v2",
        "storage_shape": "keyed-json-tree",
        "record_count": 1,
        "latest_run_ref": "run-0001",
    }
    row = {"run_ref": "run-0001", "status": "succeeded"}
    content = KeyedJsonTreeDraft(
        manifest=manifest,
        base_head=None,
        puts=(
            KeyedJsonPut(
                key="run-0001",
                path="runs/by-ref/ru/run-0001.json",
                payload=row,
            ),
        ),
    )
    payload_digest = canonical_digest(manifest)
    content_digest, plan_entries = store.plan_revision_content(content, payload_digest=payload_digest, parents=())
    evidence_record = EvidenceRecord(
        operation_id="op-keyed",
        binding="shepherd.runs",
        store_id=store.identity.store_id,
        substrate_kind=store.identity.kind,
        ingress_kind="command",
        evidence_kind="command:run-ledger-publish",
        payload_digest=payload_digest,
        stable_observation={"payload_digest": payload_digest},
    )
    evidence = _evidence_ref_for_record(evidence_record, operation_id="op-keyed")
    transition = LogicalTransition(
        binding="shepherd.runs",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="shepherd.run_ledger",
        driver_version="test",
        base_heads=(),
        ingress_kind="command",
        semantic_op="run-ledger-publish",
        payload_digest=payload_digest,
        evidence_digests=(evidence.evidence_digest,),
    )
    plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=(),
        expected_parent_heads=(),
        content_digest=content_digest,
        materialization_class="noop",
        entries=plan_entries,
    )
    preparation = RevisionPreparationRecord(
        operation_id="op-keyed",
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence,),
    )

    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(manifest),
        payload=manifest,
        content=content,
        evidence_resolver=lambda _ref: evidence_record,
    )

    provenance = store.validate_prepared_candidate(candidate.head, evidence_resolver=lambda _ref: evidence_record)
    assert provenance.metadata.byte_authority == "structured-tree"
    assert provenance.metadata.payload_digest == payload_digest
    assert provenance.metadata.content_digest == content_digest
    assert store.read_revision_manifest(candidate.head) == manifest
    assert store.read_revision_json_entry(candidate.head, "data/runs/by-ref/ru/run-0001.json") == row


def test_keyed_json_tree_content_digest_names_resulting_state(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "runs.git", _identity(kind="shepherd.runs", resource_id="runs:main"))
    row_a = {"run_ref": "run-0001", "status": "succeeded"}
    row_b = {"run_ref": "run-0002", "status": "running"}
    manifest_a = _run_manifest(record_count=1, latest_run_ref="run-0001")
    manifest_ab = _run_manifest(record_count=2, latest_run_ref="run-0002")

    head_a, _plan_a = _create_keyed_revision(
        store,
        operation_id="op-a",
        manifest=manifest_a,
        content=KeyedJsonTreeDraft(
            manifest=manifest_a,
            base_head=None,
            puts=(_run_put("run-0001", row_a),),
        ),
    )
    incremental_head, incremental_plan = _create_keyed_revision(
        store,
        operation_id="op-b",
        manifest=manifest_ab,
        content=KeyedJsonTreeDraft(
            manifest=manifest_ab,
            base_head=head_a,
            puts=(_run_put("run-0002", row_b),),
        ),
        parents=(head_a,),
    )
    one_shot_head, one_shot_plan = _create_keyed_revision(
        store,
        operation_id="op-one-shot",
        manifest=manifest_ab,
        content=KeyedJsonTreeDraft(
            manifest=manifest_ab,
            base_head=None,
            puts=(_run_put("run-0001", row_a), _run_put("run-0002", row_b)),
        ),
    )

    assert (
        store.read_revision_metadata(incremental_head).content_digest
        == store.read_revision_metadata(one_shot_head).content_digest
    )
    assert {entry["path"] for entry in incremental_plan.entries} == {
        "revision.json",
        "meta/structured-content.json",
        "data/runs/by-ref/ru/run-0002.json",
    }
    assert {entry["path"] for entry in one_shot_plan.entries} == {
        "revision.json",
        "meta/structured-content.json",
        "data/runs/by-ref/ru/run-0001.json",
        "data/runs/by-ref/ru/run-0002.json",
    }


def test_keyed_json_tree_entry_reads_reject_torn_content_tree(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "runs.git", _identity(kind="shepherd.runs", resource_id="runs:main"))
    manifest = _run_manifest(record_count=1, latest_run_ref="run-0001")
    row = {"run_ref": "run-0001", "status": "succeeded"}
    head, _plan = _create_keyed_revision(
        store,
        operation_id="op-keyed",
        manifest=manifest,
        content=KeyedJsonTreeDraft(
            manifest=manifest,
            base_head=None,
            puts=(_run_put("run-0001", row),),
        ),
    )
    original_commit = store.repo[pygit2.Oid(hex=head)]
    assert isinstance(original_commit, pygit2.Commit)
    tampered_tree = build_tree(
        store.repo,
        original_commit.tree.id,
        (
            (
                "data/runs/by-ref/ru/run-0001.json",
                compact_json_bytes({"run_ref": "run-0001", "status": "tampered"}),
            ),
        ),
    )
    signature = pygit2.Signature("test", "test@example.invalid")
    tampered_head = create_commit_with_recovery(
        store.repo,
        None,
        signature,
        signature,
        "tampered structured revision",
        tampered_tree,
        [],
    )

    with pytest.raises(InvalidRepositoryStateError, match="content root tree disagrees"):
        store.read_revision_json_entry(str(tampered_head), "data/runs/by-ref/ru/run-0001.json")


def test_substrate_store_rejects_prepared_candidate_with_missing_sidecars(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    payload = {"schema": "example/workspace", "label": "candidate"}
    metadata = SubstrateRevisionMetadata(
        kind=store.identity.kind,
        resource_id=store.identity.resource_id,
        materialization_class="external",
        payload_digest=canonical_digest(payload),
        transition_digest=canonical_digest({"transition": "missing"}),
        revision_plan_digest=canonical_digest({"plan": "missing"}),
        content_digest=canonical_digest(payload),
        revision_preparation_digest=canonical_digest({"preparation": "missing"}),
        evidence_digests=(canonical_digest({"evidence": "missing"}),),
    )
    candidate_oid = _write_manual_revision(store, payload, canonical_bytes(metadata.to_json()))

    with pytest.raises(InvalidRepositoryStateError, match="missing transition-kernel sidecar records"):
        store.validate_prepared_candidate(candidate_oid)


def test_substrate_store_rejects_inconsistent_prepared_candidate_records(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    evidence = _evidence_ref("op-prepared")
    transition = LogicalTransition(
        binding="workspace",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=(base,),
        ingress_kind="command",
        semantic_op="FilePatch",
        payload_digest=canonical_digest(payload),
        evidence_digests=(evidence.evidence_digest,),
    )
    bad_plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=transition.base_heads,
        content_digest=canonical_digest({"different": "payload"}),
        materialization_class="external",
        entries=(),
    )
    preparation = RevisionPreparationRecord(
        operation_id="op-prepared",
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=bad_plan.revision_plan_digest(),
        content_digest=bad_plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="content_digest disagrees with JSON payload"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=bad_plan,
            preparation=preparation,
            payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(payload),
            payload=payload,
            parents=(base,),
        )


def test_substrate_store_rejects_prepared_candidate_with_mismatched_evidence_refs(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    claimed_evidence = _evidence_ref("op-prepared", label="claimed")
    unrelated_evidence = _evidence_ref("op-prepared", label="unrelated")
    transition = LogicalTransition(
        binding="workspace",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=(base,),
        ingress_kind="command",
        semantic_op="FilePatch",
        payload_digest=canonical_digest(payload),
        evidence_digests=(claimed_evidence.evidence_digest,),
    )
    plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=transition.base_heads,
        content_digest=canonical_digest(payload),
        materialization_class="external",
        entries=({"path": "revision.json", "payload_digest": canonical_digest(payload)},),
    )
    preparation = RevisionPreparationRecord(
        operation_id="op-prepared",
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(unrelated_evidence,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="evidence_refs disagree with evidence_digests"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(payload),
            payload=payload,
            parents=(base,),
        )


@pytest.mark.parametrize(
    ("record_overrides", "match"),
    [
        ({"binding": "other"}, "binding disagrees"),
        ({"store_id": "store_other"}, "store_id disagrees"),
        ({"substrate_kind": "shepherd.session_state"}, "substrate kind disagrees"),
    ],
)
def test_substrate_store_rejects_prepared_candidate_with_wrong_evidence_scope(
    tmp_path,
    record_overrides: dict[str, object],
    match: str,
) -> None:
    """Evidence must agree with the preparation on binding / store / kind.

    Note: ``operation_id`` is deliberately not asserted to match. Evidence is
    content-addressed and can be cited across operations (e.g. raw capture
    evidence persisted under a command op, then cited by reducer ops). The
    cross-operation acceptance is pinned in
    ``test_substrate_store_accepts_cross_operation_evidence_citation``.
    """
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    evidence_record = replace(_evidence_record("op-b"), **record_overrides)
    evidence = _evidence_ref_for_record(evidence_record, operation_id=evidence_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-b",
        payload=payload,
        parents=(base,),
        evidence=evidence,
    )

    with pytest.raises(InvalidRepositoryStateError, match=match):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            parents=(base,),
            evidence_resolver=lambda _ref: evidence_record,
        )


def test_substrate_store_accepts_explicit_cross_operation_cited_evidence(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    evidence_record = _evidence_record("op-command")
    evidence = _evidence_ref_for_record(evidence_record, operation_id=evidence_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-reduce",
        payload=payload,
        parents=(base,),
        evidence=evidence,
        cited_evidence_refs=(evidence,),
    )

    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
        payload=payload,
        parents=(base,),
        evidence_resolver=lambda _ref: evidence_record,
    )

    assert candidate


def test_substrate_store_accepts_prepared_candidate_with_local_first_cited_suffix(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    local_record = _evidence_record("op-reduce", label="local")
    cited_record = _evidence_record("op-command", label="cited")
    local_evidence = _evidence_ref_for_record(local_record, operation_id=local_record.operation_id)
    cited_evidence = _evidence_ref_for_record(cited_record, operation_id=cited_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-reduce",
        payload=payload,
        parents=(base,),
        evidence=local_evidence,
        evidence_refs=(local_evidence, cited_evidence),
        cited_evidence_refs=(cited_evidence,),
    )

    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
        payload=payload,
        parents=(base,),
        evidence_resolver={local_evidence: local_record, cited_evidence: cited_record}.get,
    )

    assert candidate


def test_substrate_store_rejects_prepared_candidate_with_cited_ref_before_local_ref(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    local_record = _evidence_record("op-reduce", label="local")
    cited_record = _evidence_record("op-command", label="cited")
    local_evidence = _evidence_ref_for_record(local_record, operation_id=local_record.operation_id)
    cited_evidence = _evidence_ref_for_record(cited_record, operation_id=cited_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-reduce",
        payload=payload,
        parents=(base,),
        evidence=local_evidence,
        evidence_refs=(cited_evidence, local_evidence),
        cited_evidence_refs=(cited_evidence,),
    )

    with pytest.raises(InvalidRepositoryStateError, match="cited_evidence_refs must be a suffix"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            parents=(base,),
            evidence_resolver={local_evidence: local_record, cited_evidence: cited_record}.get,
        )


def test_substrate_store_accepts_cross_operation_evidence_citation(tmp_path) -> None:
    """A preparation may cite evidence produced by a different operation.

    Cross-operation citation is the basis of the capture-shadow flow: raw
    capture evidence is persisted under the command operation, and downstream
    reducer operations cite that evidence by content-addressed digest. The
    substrate-store validator must accept this on both the write and read sides.
    """
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "candidate"}
    # Evidence belongs to op-a (e.g. a command); preparation is op-b (e.g. a
    # reducer). Cross-operation citation is admitted only when the evidence is
    # explicitly listed in cited_evidence_refs (the capture-shadow flow routes
    # it there via the coordinator's ReductionBatch). Other scope fields
    # (binding / store / substrate_kind) still match.
    evidence_record = _evidence_record("op-a")
    evidence = _evidence_ref_for_record(evidence_record, operation_id="op-a")
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-b",
        payload=payload,
        parents=(base,),
        evidence=evidence,
        cited_evidence_refs=(evidence,),
    )

    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
        payload=payload,
        parents=(base,),
        evidence_resolver=lambda _ref: evidence_record,
    )
    # Read-side validation must also accept cross-operation evidence.
    provenance = store.validate_prepared_candidate(candidate.head, evidence_resolver=lambda _ref: evidence_record)
    assert provenance.preparation.operation_id == "op-b"
    assert provenance.preparation.evidence_refs[0].evidence_digest == evidence.evidence_digest


def test_substrate_store_rejects_stored_prepared_revision_with_cited_ref_before_local_ref(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "revision"}
    local_record = _evidence_record("op-reduce", label="local")
    cited_record = _evidence_record("op-command", label="cited")
    local_evidence = _evidence_ref_for_record(local_record, operation_id=local_record.operation_id)
    cited_evidence = _evidence_ref_for_record(cited_record, operation_id=cited_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-reduce",
        payload=payload,
        parents=(base,),
        evidence=local_evidence,
        evidence_refs=(cited_evidence, local_evidence),
        cited_evidence_refs=(cited_evidence,),
    )
    revision = _write_manual_prepared_revision(
        store,
        payload,
        parents=(base,),
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
    )

    with pytest.raises(InvalidRepositoryStateError, match="cited_evidence_refs must be a suffix"):
        store.validate_prepared_revision(
            revision,
            evidence_resolver={local_evidence: local_record, cited_evidence: cited_record}.get,
        )


def test_substrate_store_rejects_legacy_preparation_sidecar(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    base = store.create_unsafe_unprepared_json_revision("refs/heads/main", {"label": "base"})
    payload = {"schema": "example/workspace", "label": "revision"}
    evidence_record = _evidence_record("op-local")
    evidence = _evidence_ref_for_record(evidence_record, operation_id=evidence_record.operation_id)
    transition, plan, preparation, payload_descriptor = _prepared_records(
        store,
        operation_id="op-local",
        payload=payload,
        parents=(base,),
        evidence=evidence,
    )
    legacy_preparation_json = _legacy_v1_preparation_json(preparation)
    revision = _write_manual_prepared_revision(
        store,
        payload,
        parents=(base,),
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=payload_descriptor,
        preparation_json=legacy_preparation_json,
        revision_preparation_digest=str(legacy_preparation_json["revision_preparation_digest"]),
    )

    with pytest.raises(InvalidRepositoryStateError, match="invalid transition-kernel sidecar"):
        store.validate_prepared_revision(revision, evidence_resolver=lambda _ref: evidence_record)


def test_substrate_store_rejects_noncanonical_or_inconsistent_revision_metadata(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    bad_metadata = SubstrateRevisionMetadata(
        kind="filesystem",
        resource_id="fs:repo-main",
        materialization_class="external",
        payload_digest="sha256:" + "0" * 64,
    )
    bad_oid = _write_manual_revision(
        store, {"label": "payload"}, canonical_bytes({**bad_metadata.to_json(), "extra": "x"})
    )
    noncanonical_oid = _write_manual_revision(store, {"label": "payload"}, compact_json_bytes(bad_metadata.to_json()))

    with pytest.raises(ValueError, match="unexpected substrate revision metadata fields"):
        store.read_revision_metadata(bad_oid)
    with pytest.raises(ValueError, match="canonical record is missing"):
        store.read_revision_metadata(noncanonical_oid)


def test_substrate_store_without_shared_repo_does_not_configure_alternates(tmp_path) -> None:
    path = tmp_path / "workspace.git"
    SubstrateStore.open_or_init(path, _identity())
    alternates = path / "objects" / "info" / "alternates"
    assert not alternates.exists()


def test_substrate_store_with_shared_repo_configures_alternates_idempotently(tmp_path) -> None:
    coord_path = tmp_path / "worlds.git"
    pygit2.init_repository(str(coord_path), bare=True)
    sub_path = tmp_path / "workspace.git"
    SubstrateStore.open_or_init(sub_path, _identity(), shared_object_repo_path=coord_path)

    alternates_path = sub_path / "objects" / "info" / "alternates"
    expected_line = str((coord_path / "objects").resolve())
    assert alternates_path.read_text(encoding="utf-8").splitlines() == [expected_line]

    # Re-open with the same coord path should not duplicate the entry.
    SubstrateStore.open_existing(sub_path, _identity(), shared_object_repo_path=coord_path)
    assert alternates_path.read_text(encoding="utf-8").splitlines() == [expected_line]


def test_substrate_store_alternates_make_foreign_tree_visible(tmp_path) -> None:
    coord_path = tmp_path / "worlds.git"
    coord = pygit2.init_repository(str(coord_path), bare=True)
    blob_oid = coord.create_blob(b"hello\n")
    builder = coord.TreeBuilder()
    builder.insert("greeting.txt", blob_oid, pygit2.GIT_FILEMODE_BLOB)
    tree_oid = str(builder.write())

    sub_path = tmp_path / "workspace.git"
    sub = SubstrateStore.open_or_init(sub_path, _identity(), shared_object_repo_path=coord_path)

    tree_obj = sub.repo[pygit2.Oid(hex=tree_oid)]
    assert isinstance(tree_obj, pygit2.Tree)
    # Walk it to confirm blobs are also visible through the alternate.
    for entry in tree_obj:
        if entry.filemode == pygit2.GIT_FILEMODE_BLOB:
            blob = sub.repo[entry.id]
            assert isinstance(blob, pygit2.Blob)


def test_substrate_store_shared_repo_without_objects_dir_fails_closed(tmp_path) -> None:
    coord_path = tmp_path / "not-a-repo"
    coord_path.mkdir()
    with pytest.raises(InvalidRepositoryStateError, match="objects directory"):
        SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity(), shared_object_repo_path=coord_path)


def test_substrate_store_alternates_append_does_not_clobber_existing(tmp_path) -> None:
    sub_path = tmp_path / "workspace.git"
    sub = pygit2.init_repository(str(sub_path), bare=True)
    # Pre-seed alternates with an unrelated entry so the helper has something to preserve.
    existing_alt = tmp_path / "extra-objects"
    (existing_alt / "info").mkdir(parents=True)
    alternates_path = sub_path / "objects" / "info" / "alternates"
    alternates_path.parent.mkdir(parents=True, exist_ok=True)
    alternates_path.write_text(f"{existing_alt}\n", encoding="utf-8")
    del sub

    coord_path = tmp_path / "worlds.git"
    pygit2.init_repository(str(coord_path), bare=True)
    SubstrateStore.open_or_init(sub_path, _identity(), shared_object_repo_path=coord_path)

    lines = alternates_path.read_text(encoding="utf-8").splitlines()
    assert lines == [str(existing_alt), str((coord_path / "objects").resolve())]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        ".hidden",
        "workspace/main",
        "contains whitespace",
        "control-\x01-char",
        "component.lock",
        "has@{selector",
        "unicode-\u2603",
        "x" * 256,
    ],
)
def test_ref_component_encoding_handles_pathological_values(raw: str) -> None:
    encoded = encode_ref_component(raw)

    assert is_ref_safe_component(encoded)
    assert "/" not in encoded
    assert not encoded.startswith(".")
    assert encoded != raw or is_ref_safe_component(raw)
    if len(raw.encode("utf-8")) > 96:
        assert encoded.startswith("sha256_")


def _write_manual_revision(store: SubstrateStore, payload: dict[str, object], metadata_bytes: bytes) -> str:
    repo = store.repo
    tree_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        tree_builder,
        "revision.json",
        repo.create_blob(compact_json_bytes(payload)),
        pygit2.GIT_FILEMODE_BLOB,
    )
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        "substrate-revision.json",
        repo.create_blob(metadata_bytes),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(repo, tree_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("test", "test@example.invalid")
    oid = create_commit_with_recovery(repo, None, signature, signature, "manual revision", tree_builder.write(), [])
    return str(oid)


def _write_manual_prepared_revision(
    store: SubstrateStore,
    payload: dict[str, object],
    *,
    parents: tuple[str, ...],
    transition: LogicalTransition,
    plan: PreparedRevisionPlan,
    preparation: RevisionPreparationRecord,
    payload_descriptor: ValidatedPayloadDescriptor,
    preparation_json: dict[str, object] | None = None,
    revision_preparation_digest: str | None = None,
) -> str:
    repo = store.repo
    tree_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        tree_builder,
        "revision.json",
        repo.create_blob(compact_json_bytes(payload)),
        pygit2.GIT_FILEMODE_BLOB,
    )
    metadata = SubstrateRevisionMetadata(
        kind=store.identity.kind,
        resource_id=store.identity.resource_id,
        materialization_class=plan.materialization_class,
        payload_digest=canonical_digest(payload),
        parent_heads=parents,
        produced_by_operation_id=preparation.operation_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        revision_preparation_digest=revision_preparation_digest or preparation.revision_preparation_digest(),
        evidence_digests=preparation.evidence_digests,
        ingress_kind=transition.ingress_kind,
        semantic_op=transition.semantic_op,
        driver=transition.driver,
        driver_version=transition.driver_version,
    )
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        "substrate-revision.json",
        repo.create_blob(canonical_bytes(metadata.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(
        repo,
        meta_builder,
        "logical-transition.json",
        repo.create_blob(canonical_bytes(transition.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(
        repo,
        meta_builder,
        "prepared-revision-plan.json",
        repo.create_blob(canonical_bytes(plan.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(
        repo,
        meta_builder,
        "revision-preparation.json",
        repo.create_blob(canonical_bytes(preparation_json or preparation.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(
        repo,
        meta_builder,
        "payload-descriptor.json",
        repo.create_blob(canonical_bytes(payload_descriptor.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    insert_tree_entry(repo, tree_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("test", "test@example.invalid")
    oid = create_commit_with_recovery(
        repo,
        None,
        signature,
        signature,
        "manual prepared revision",
        tree_builder.write(),
        [pygit2.Oid(hex=parent) for parent in parents],
    )
    return str(oid)


def _run_manifest(*, record_count: int, latest_run_ref: str | None) -> dict[str, object]:
    return {
        "schema": "shepherd.workspace_control.runs.v2",
        "storage_shape": "keyed-json-tree",
        "record_count": record_count,
        "latest_run_ref": latest_run_ref,
    }


def _run_put(run_ref: str, payload: dict[str, object]) -> KeyedJsonPut:
    return KeyedJsonPut(
        key=run_ref,
        path=f"runs/by-ref/{run_ref[:2]}/{run_ref}.json",
        payload=payload,
    )


def _create_keyed_revision(
    store: SubstrateStore,
    *,
    operation_id: str,
    manifest: dict[str, object],
    content: KeyedJsonTreeDraft,
    parents: tuple[str, ...] = (),
) -> tuple[str, PreparedRevisionPlan]:
    payload_digest = canonical_digest(manifest)
    evidence_record = EvidenceRecord(
        operation_id=operation_id,
        binding="shepherd.runs",
        store_id=store.identity.store_id,
        substrate_kind=store.identity.kind,
        ingress_kind="command",
        evidence_kind="command:run-ledger-publish",
        payload_digest=payload_digest,
        stable_observation={"payload_digest": payload_digest},
    )
    evidence = _evidence_ref_for_record(evidence_record, operation_id=operation_id)
    transition = LogicalTransition(
        binding="shepherd.runs",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="shepherd.run_ledger",
        driver_version="test",
        base_heads=parents,
        ingress_kind="command",
        semantic_op="run-ledger-publish",
        payload_digest=payload_digest,
        evidence_digests=(evidence.evidence_digest,),
    )
    content_digest, entries = store.plan_revision_content(content, payload_digest=payload_digest, parents=parents)
    plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=parents,
        expected_parent_heads=parents,
        content_digest=content_digest,
        materialization_class="noop",
        entries=entries,
    )
    preparation = RevisionPreparationRecord(
        operation_id=operation_id,
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence,),
    )
    head = store.create_revision_from_prepared(
        f"refs/tests/{operation_id}",
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(manifest),
        payload=manifest,
        content=content,
        parents=parents,
        evidence_resolver=lambda _ref: evidence_record,
    )
    return head, plan


def _evidence_ref(operation_id: str, *, label: str = "workspace") -> EvidenceRef:
    record = _evidence_record(operation_id, label=label)
    return _evidence_ref_for_record(record, operation_id=operation_id)


def _evidence_ref_for_record(record: EvidenceRecord, *, operation_id: str) -> EvidenceRef:
    return EvidenceRef(
        ref=f"refs/vcscore/evidence/{operation_id}/1",
        evidence_digest=record.evidence_digest(),
        record_digest=record.record_digest(),
        payload_digest=record.payload_digest,
    )


def _evidence_record(operation_id: str, *, label: str = "workspace") -> EvidenceRecord:
    payload_digest = canonical_digest({"payload": label})
    return EvidenceRecord(
        operation_id=operation_id,
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
        ingress_kind="command",
        evidence_kind="command_envelope",
        payload_digest=payload_digest,
        stable_observation={"command": "write", "label": label},
    )


def _prepared_records(
    store: SubstrateStore,
    *,
    operation_id: str,
    payload: dict[str, object],
    parents: tuple[str, ...],
    evidence: EvidenceRef,
    evidence_refs: tuple[EvidenceRef, ...] | None = None,
    cited_evidence_refs: tuple[EvidenceRef, ...] = (),
) -> tuple[LogicalTransition, PreparedRevisionPlan, RevisionPreparationRecord, ValidatedPayloadDescriptor]:
    all_evidence_refs = evidence_refs or (evidence,)
    transition = LogicalTransition(
        binding="workspace",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=parents,
        ingress_kind="command",
        semantic_op="FilePatch",
        payload_digest=canonical_digest(payload),
        evidence_digests=tuple(ref.evidence_digest for ref in all_evidence_refs),
    )
    plan = PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=transition.base_heads,
        content_digest=canonical_digest(payload),
        materialization_class="external",
        entries=({"path": "revision.json", "payload_digest": canonical_digest(payload)},),
    )
    preparation = RevisionPreparationRecord(
        operation_id=operation_id,
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=all_evidence_refs,
        cited_evidence_refs=cited_evidence_refs,
    )
    return transition, plan, preparation, ValidatedPayloadDescriptor.for_json_payload(payload)


def _legacy_v1_preparation_json(preparation: RevisionPreparationRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "vcscore/revision-preparation/v1",
        "operation_id": preparation.operation_id,
        "binding": preparation.binding,
        "store_id": preparation.store_id,
        "resource_id": preparation.resource_id,
        "transition_digest": preparation.transition_digest,
        "revision_plan_digest": preparation.revision_plan_digest,
        "content_digest": preparation.content_digest,
        "evidence_digests": sorted(preparation.evidence_digests),
        "evidence_refs": sorted((ref.to_json() for ref in preparation.evidence_refs), key=canonical_digest),
        "relationship_requirements": sorted(
            (requirement.to_json() for requirement in preparation.relationship_requirements),
            key=canonical_digest,
        ),
    }
    payload["revision_preparation_digest"] = canonical_digest(payload)
    return payload


def _read_blob_bytes(store: SubstrateStore, tree: pygit2.Tree, path: str) -> bytes:
    obj: pygit2.Object = tree
    for component in path.split("/"):
        if not isinstance(obj, pygit2.Tree):
            raise TypeError(f"{path!r} did not resolve to a blob")
        obj = store.repo[obj[component].id]
    if not isinstance(obj, pygit2.Blob):
        raise TypeError(f"{path!r} did not resolve to a blob")
    return bytes(obj.data)
