"""Contract tests for tree-backed workspace substrate revisions (Track A / Tranche 1).

These tests pin the schema-enablement boundary between digest-only and tree-backed
substrate revisions. They do not exercise driver-side capture (Tranche 2) or
materialization preference (Tranche 3): the substrate store accepts a
``PreparedRevisionPlan.git_tree_oid`` referencing a tree already in its own ODB,
validates the manifest/tree correspondence, embeds the tree at ``workspace/`` in
the resulting commit, and records ``byte_authority``/``git_tree_oid`` in
``SubstrateRevisionMetadata``.

Cross-store object availability (Open Question 1 from the plan) is intentionally
out of scope here: every tree the validator inspects is authored directly in the
substrate store under test.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pygit2
import pytest
from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._substrate_store import SubstrateStore
from vcs_core._transition_kernel_records import (
    EvidenceRecord,
    EvidenceRef,
    LogicalTransition,
    PreparedRevisionPlan,
    RevisionPreparationRecord,
    ValidatedPayloadDescriptor,
)
from vcs_core._world_substrate_adapters import (
    WORKSPACE_MANIFEST_BYTE_AUTHORITY_MODES,
    workspace_state_revision_payload,
)
from vcs_core._world_types import (
    SubstrateRevisionMetadata,
    SubstrateStoreIdentity,
    canonical_digest,
)


def _identity(
    *,
    store_id: str = "store_workspace",
    kind: str = "filesystem",
    resource_id: str = "fs:repo-main",
) -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id=store_id, kind=kind, resource_id=resource_id)


def _content_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _build_workspace_tree(
    store: SubstrateStore,
    contents: dict[str, tuple[bytes, int]],
) -> str:
    """Author a workspace tree directly in the substrate ODB.

    ``contents`` maps forward-slash separated paths to ``(blob_bytes, filemode)``.
    Returns the hex Git oid of the assembled tree.
    """
    repo = store.repo
    groups: dict[tuple[str, ...], dict[str, tuple[bytes, int]]] = {}
    for path, value in contents.items():
        parts = tuple(path.split("/"))
        groups.setdefault(parts[:-1], {})[parts[-1]] = value

    def build(prefix: tuple[str, ...]) -> pygit2.Oid:
        builder = repo.TreeBuilder()
        child_prefixes = sorted({g for g in groups if len(g) > len(prefix) and g[: len(prefix)] == prefix})
        immediate: dict[str, tuple[str, ...]] = {}
        for child in child_prefixes:
            name = child[len(prefix)]
            immediate.setdefault(name, (*prefix, name))
        for name, full_prefix in sorted(immediate.items()):
            builder.insert(name, build(full_prefix), pygit2.GIT_FILEMODE_TREE)
        if prefix in groups:
            for name, (data, mode) in sorted(groups[prefix].items()):
                blob_oid = repo.create_blob(data)
                builder.insert(name, blob_oid, mode)
        return builder.write()

    return str(build(()))


def _manifest_entries(contents: dict[str, tuple[bytes, int]]) -> tuple[dict[str, object], ...]:
    """Return canonical manifest entries describing ``contents``."""
    entries: list[dict[str, object]] = []
    for path, (data, mode) in contents.items():
        entries.append(
            {
                "path": path,
                "state": "present",
                "mode": mode,
                "content_digest": _content_digest(data),
            }
        )
    return tuple(entries)


def _evidence_record(operation_id: str, store: SubstrateStore) -> EvidenceRecord:
    payload_digest = canonical_digest({"observation": operation_id})
    return EvidenceRecord(
        operation_id=operation_id,
        binding="workspace",
        store_id=store.identity.store_id,
        substrate_kind=store.identity.kind,
        ingress_kind="command",
        evidence_kind="command_envelope",
        payload_digest=payload_digest,
        stable_observation={"observation": operation_id},
    )


def _evidence_ref_for(record: EvidenceRecord, *, operation_id: str) -> EvidenceRef:
    return EvidenceRef(
        ref=f"refs/vcscore/evidence/{operation_id}/1",
        evidence_digest=record.evidence_digest(),
        record_digest=record.record_digest(),
        payload_digest=record.payload_digest,
    )


def _build_records(
    store: SubstrateStore,
    *,
    operation_id: str,
    payload: dict[str, object],
    parents: tuple[str, ...],
    git_tree_oid: str | None,
    evidence: EvidenceRef | None = None,
) -> tuple[LogicalTransition, PreparedRevisionPlan, RevisionPreparationRecord, ValidatedPayloadDescriptor]:
    evidence_digests = (evidence.evidence_digest,) if evidence is not None else ()
    evidence_refs = (evidence,) if evidence is not None else ()
    transition = LogicalTransition(
        binding="workspace",
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=parents,
        ingress_kind="command",
        semantic_op="workspace-tree-backed",
        payload_digest=canonical_digest(payload),
        evidence_digests=evidence_digests,
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
        git_tree_oid=git_tree_oid,
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
        evidence_refs=evidence_refs,
    )
    return transition, plan, preparation, ValidatedPayloadDescriptor.for_json_payload(payload)


# --- Schema-level cases (no commit creation needed) ---


def test_workspace_byte_authority_modes_admit_tree_backed() -> None:
    """The frozenset is the single source of truth for accepted modes."""
    assert frozenset({"digest-only", "tree-backed"}) == WORKSPACE_MANIFEST_BYTE_AUTHORITY_MODES


def test_metadata_tree_backed_requires_git_tree_oid() -> None:
    with pytest.raises(ValueError, match="tree-backed substrate revisions require git_tree_oid"):
        SubstrateRevisionMetadata(
            kind="filesystem",
            resource_id="fs:test",
            materialization_class="external",
            payload_digest=canonical_digest({}),
            byte_authority="tree-backed",
        )


def test_metadata_digest_only_forbids_git_tree_oid() -> None:
    with pytest.raises(ValueError, match="digest-only substrate revisions must not carry git_tree_oid"):
        SubstrateRevisionMetadata(
            kind="filesystem",
            resource_id="fs:test",
            materialization_class="external",
            payload_digest=canonical_digest({}),
            byte_authority="digest-only",
            git_tree_oid="0" * 40,
        )


def test_metadata_rejects_unsupported_byte_authority() -> None:
    with pytest.raises(ValueError, match="unsupported substrate revision byte_authority"):
        SubstrateRevisionMetadata(
            kind="filesystem",
            resource_id="fs:test",
            materialization_class="external",
            payload_digest=canonical_digest({}),
            byte_authority="external",
        )


def test_metadata_rejects_malformed_git_tree_oid() -> None:
    with pytest.raises(ValueError, match="must be a 40-char hex Git oid"):
        SubstrateRevisionMetadata(
            kind="filesystem",
            resource_id="fs:test",
            materialization_class="external",
            payload_digest=canonical_digest({}),
            byte_authority="tree-backed",
            git_tree_oid="not-hex",
        )


def test_metadata_round_trip_preserves_byte_authority_and_tree_oid() -> None:
    tree_oid = "0123456789abcdef0123456789abcdef01234567"
    metadata = SubstrateRevisionMetadata(
        kind="filesystem",
        resource_id="fs:test",
        materialization_class="external",
        payload_digest=canonical_digest({}),
        byte_authority="tree-backed",
        git_tree_oid=tree_oid,
    )
    serialized = metadata.to_json()
    assert serialized["byte_authority"] == "tree-backed"
    assert serialized["git_tree_oid"] == tree_oid
    restored = SubstrateRevisionMetadata.from_json(serialized)
    assert restored == metadata


def test_metadata_default_remains_digest_only_with_no_tree_oid() -> None:
    """Existing digest-only metadata fields keep working unchanged."""
    metadata = SubstrateRevisionMetadata(
        kind="filesystem",
        resource_id="fs:test",
        materialization_class="external",
        payload_digest=canonical_digest({}),
    )
    assert metadata.byte_authority == "digest-only"
    assert metadata.git_tree_oid is None
    serialized = metadata.to_json()
    assert serialized["byte_authority"] == "digest-only"
    assert "git_tree_oid" not in serialized
    restored = SubstrateRevisionMetadata.from_json(serialized)
    assert restored == metadata


# --- PreparedRevisionPlan digest stability ---


def test_plan_digest_excludes_git_tree_oid(tmp_path) -> None:
    """The plan digest is computed before tree assembly and must not include git_tree_oid."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="tree-backed")
    _transition, plan, _preparation, _descriptor = _build_records(
        store,
        operation_id="op-plan-digest",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    plan_without_tree = replace(plan, git_tree_oid=None)
    plan_with_different_tree = replace(
        plan,
        git_tree_oid="1111111111111111111111111111111111111111",
    )
    assert plan.revision_plan_digest() == plan_without_tree.revision_plan_digest()
    assert plan.revision_plan_digest() == plan_with_different_tree.revision_plan_digest()
    # Round-trip through JSON preserves the field but not the digest input.
    restored = PreparedRevisionPlan.from_json(plan.to_json())
    assert restored == plan
    assert restored.git_tree_oid == tree_oid


def test_plan_round_trip_handles_absent_git_tree_oid(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    payload = {"label": "no-tree"}
    _transition, plan, _preparation, _descriptor = _build_records(
        store,
        operation_id="op-no-tree",
        payload=payload,
        parents=(),
        git_tree_oid=None,
    )
    restored = PreparedRevisionPlan.from_json(plan.to_json())
    assert restored.git_tree_oid is None


# --- Commit-time positive cases ---


def test_tree_backed_revision_embeds_workspace_tree(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {
        "README.md": (b"# project\n", 0o100644),
        "src/main.py": (b"print('hi')\n", 0o100644),
        "scripts/run.sh": (b"#!/bin/sh\nexit 0\n", 0o100755),
    }
    tree_oid = _build_workspace_tree(store, contents)
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="tree-backed")
    evidence_record = _evidence_record("op-tb", store)
    evidence = _evidence_ref_for(evidence_record, operation_id="op-tb")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
        evidence=evidence,
    )
    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=descriptor,
        payload=payload,
        parents=(),
        evidence_resolver=lambda _ref: evidence_record,
    )
    commit = store.repo[pygit2.Oid(hex=candidate.head)]
    tree = commit.tree
    assert "workspace" in [entry.name for entry in tree]
    workspace_entry = tree["workspace"]
    assert workspace_entry.filemode == pygit2.GIT_FILEMODE_TREE
    assert str(workspace_entry.id) == tree_oid
    metadata = store.read_revision_metadata(candidate.head)
    assert metadata.byte_authority == "tree-backed"
    assert metadata.git_tree_oid == tree_oid
    # The deep validator should re-validate the tree-walk path on read.
    provenance = store.validate_prepared_candidate(candidate.head, evidence_resolver=lambda _ref: evidence_record)
    assert provenance.metadata.byte_authority == "tree-backed"
    assert provenance.plan.git_tree_oid == tree_oid


def test_digest_only_revision_round_trip_unchanged(tmp_path) -> None:
    """Existing digest-only revisions must keep working with no behavior change."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="digest-only")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-do",
        payload=payload,
        parents=(),
        git_tree_oid=None,
    )
    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=descriptor,
        payload=payload,
        parents=(),
    )
    commit = store.repo[pygit2.Oid(hex=candidate.head)]
    # No workspace/ entry on a digest-only revision.
    assert "workspace" not in [entry.name for entry in commit.tree]
    metadata = store.read_revision_metadata(candidate.head)
    assert metadata.byte_authority == "digest-only"
    assert metadata.git_tree_oid is None


def test_tree_backed_revision_handles_deleted_manifest_entries(tmp_path) -> None:
    """Deleted manifest entries must not appear in the tree."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"keeper.txt": (b"keep\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    entries = list(_manifest_entries(contents))
    entries.append({"path": "removed.txt", "state": "deleted"})
    payload = workspace_state_revision_payload(tuple(entries), byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-deleted",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=descriptor,
        payload=payload,
        parents=(),
    )
    metadata = store.read_revision_metadata(candidate.head)
    assert metadata.byte_authority == "tree-backed"


# --- Commit-time negative cases ---


def test_tree_backed_rejects_content_digest_mismatch(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    # Construct a manifest with the wrong content_digest.
    entries = (
        {
            "path": "README.md",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _content_digest(b"# different\n"),
        },
    )
    payload = workspace_state_revision_payload(entries, byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-bad-digest",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(InvalidRepositoryStateError, match=r"content_digest mismatch at 'README\.md'"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_mode_mismatch(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"scripts/run.sh": (b"#!/bin/sh\n", 0o100755)}
    tree_oid = _build_workspace_tree(store, contents)
    # Manifest says regular file, tree has executable.
    entries = (
        {
            "path": "scripts/run.sh",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _content_digest(b"#!/bin/sh\n"),
        },
    )
    payload = workspace_state_revision_payload(entries, byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-bad-mode",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(InvalidRepositoryStateError, match=r"mode mismatch at 'scripts/run\.sh'"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_blob_without_manifest_entry(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {
        "README.md": (b"# project\n", 0o100644),
        "extra.txt": (b"unmentioned\n", 0o100644),
    }
    tree_oid = _build_workspace_tree(store, contents)
    # Manifest covers only README.md, but the tree has extra.txt.
    entries = (
        {
            "path": "README.md",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _content_digest(b"# project\n"),
        },
    )
    payload = workspace_state_revision_payload(entries, byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-extra",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(
        InvalidRepositoryStateError,
        match=r"tree contains blob without a manifest entry: 'extra\.txt'",
    ):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_manifest_entry_without_blob(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    # Manifest claims a second file, but the tree only has README.md.
    entries = (
        {
            "path": "README.md",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _content_digest(b"# project\n"),
        },
        {
            "path": "missing.txt",
            "state": "present",
            "mode": 0o100644,
            "content_digest": _content_digest(b"nope\n"),
        },
    )
    payload = workspace_state_revision_payload(entries, byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-missing",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(InvalidRepositoryStateError, match=r"manifest entry has no tree blob: 'missing\.txt'"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_deleted_entry_present_in_tree(tmp_path) -> None:
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"removed.txt": (b"oops\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    # Manifest says removed.txt is deleted, but the tree contains it.
    entries = ({"path": "removed.txt", "state": "deleted"},)
    payload = workspace_state_revision_payload(entries, byte_authority="tree-backed")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-deleted-blob",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(
        InvalidRepositoryStateError,
        match=r"contains blob for deleted manifest entry: 'removed\.txt'",
    ):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_manifest_byte_authority_disagreement(tmp_path) -> None:
    """If the plan carries a tree but the manifest says digest-only, fail closed."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="digest-only")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-mismatch",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(InvalidRepositoryStateError, match="manifest byte_authority='tree-backed'"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_unknown_tree_oid(tmp_path) -> None:
    """libgit2 enforces local visibility, but our pre-walk error gives a clearer message."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="tree-backed")
    bogus_tree_oid = "deadbeef" * 5  # 40 hex chars, but unknown object.
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-unknown",
        payload=payload,
        parents=(),
        git_tree_oid=bogus_tree_oid,
    )
    with pytest.raises(InvalidRepositoryStateError, match="references unknown tree"):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


# --- Determinism ---


def test_tree_build_deterministic_for_identical_contents(tmp_path) -> None:
    """Identical workspace contents must produce the same git_tree_oid.

    Retry stability of materialization receipts depends on this property.
    """
    store_a = SubstrateStore.open_or_init(tmp_path / "a.git", _identity())
    store_b = SubstrateStore.open_or_init(tmp_path / "b.git", _identity())
    contents = {
        "a.txt": (b"alpha\n", 0o100644),
        "nested/b.txt": (b"bravo\n", 0o100644),
        "nested/deep/c.txt": (b"charlie\n", 0o100644),
    }
    oid_a = _build_workspace_tree(store_a, contents)
    oid_b = _build_workspace_tree(store_b, contents)
    reversed_contents = dict(reversed(list(contents.items())))
    oid_reordered = _build_workspace_tree(store_a, reversed_contents)
    assert oid_a == oid_b
    assert oid_a == oid_reordered


# --- Read-side metadata/sidecar consistency ---


def test_read_side_rejects_metadata_byte_authority_disagreement(tmp_path) -> None:
    """If the on-disk metadata blob disagrees with the plan sidecar, read-side fails closed."""
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    contents = {"README.md": (b"# project\n", 0o100644)}
    tree_oid = _build_workspace_tree(store, contents)
    payload = workspace_state_revision_payload(_manifest_entries(contents), byte_authority="tree-backed")
    evidence_record = _evidence_record("op-tb-read", store)
    evidence = _evidence_ref_for(evidence_record, operation_id="op-tb-read")
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-tb-read",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
        evidence=evidence,
    )
    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=descriptor,
        payload=payload,
        parents=(),
        evidence_resolver=lambda _ref: evidence_record,
    )
    # Surgically rewrite the on-disk metadata blob to claim digest-only while the
    # plan sidecar still names a tree. This is an attacker-shaped check.
    from vcs_core._world_types import canonical_bytes
    from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry

    commit = store.repo[pygit2.Oid(hex=candidate.head)]
    old_tree = commit.tree
    meta_tree = old_tree["meta"]
    new_meta_builder = store.repo.TreeBuilder(meta_tree.id)
    bad_metadata = SubstrateRevisionMetadata(
        kind=store.identity.kind,
        resource_id=store.identity.resource_id,
        materialization_class=plan.materialization_class,
        payload_digest=canonical_digest(payload),
        parent_heads=(),
        produced_by_operation_id=preparation.operation_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        revision_preparation_digest=preparation.revision_preparation_digest(),
        evidence_digests=preparation.evidence_digests,
        ingress_kind=transition.ingress_kind,
        semantic_op=transition.semantic_op,
        driver=transition.driver,
        driver_version=transition.driver_version,
        # Deliberately disagree with the plan sidecar.
        byte_authority="digest-only",
        git_tree_oid=None,
    )
    new_meta_builder.insert(
        "substrate-revision.json",
        store.repo.create_blob(canonical_bytes(bad_metadata.to_json())),
        pygit2.GIT_FILEMODE_BLOB,
    )
    new_tree_builder = store.repo.TreeBuilder(old_tree.id)
    insert_tree_entry(store.repo, new_tree_builder, "meta", new_meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("rewriter", "test@example.invalid")
    rewritten = create_commit_with_recovery(
        store.repo,
        None,
        signature,
        signature,
        "rewritten",
        new_tree_builder.write(),
        [],
    )
    # Metadata cannot self-construct with mismatched byte_authority/git_tree_oid
    # (the dataclass __post_init__ enforces internal consistency), so the
    # detectable on-disk disagreement is between the metadata blob and the plan
    # sidecar. The git_tree_oid check fires first.
    with pytest.raises(InvalidRepositoryStateError, match="git_tree_oid disagrees with sidecar"):
        store.validate_prepared_candidate(str(rewritten), evidence_resolver=lambda _ref: evidence_record)


# --- File-mode scope (pinning Tranche 1 boundary) ---


def test_tree_backed_rejects_symlink_mode(tmp_path) -> None:
    """Workspace trees may not contain symlinks (mode 0o120000) in Tranche 1.

    The manifest schema restricts file modes to {0o100644, 0o100755}. This
    test pins the matching restriction on the tree-walker so a future symlink
    mode is an explicit schema decision rather than an accidental admission
    that slips through because no test exercised it.
    """
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    # Author a tree directly in the substrate whose sole entry is a symlink.
    target_blob = store.repo.create_blob(b"../target\n")
    builder = store.repo.TreeBuilder()
    builder.insert("link.sym", target_blob, 0o120000)
    tree_oid = str(builder.write())

    # The manifest claims a regular blob at link.sym so we reach the walker.
    payload = workspace_state_revision_payload(
        (
            {
                "path": "link.sym",
                "state": "present",
                "mode": 0o100644,
                "content_digest": _content_digest(b"../target\n"),
            },
        ),
        byte_authority="tree-backed",
    )
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-symlink",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(
        InvalidRepositoryStateError,
        match=r"unsupported file mode at 'link\.sym'",
    ):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )


def test_tree_backed_rejects_gitlink_mode(tmp_path) -> None:
    """Workspace trees may not contain submodule / gitlink entries in Tranche 1.

    Gitlink entries (mode 0o160000) are deliberately unsupported: they would
    smuggle commit pointers into the substrate's payload tree, bypassing the
    explicit recursive-world model (vcscore.world_ref). Pin the rejection so a
    future submodule shape lands deliberately rather than silently.
    """
    store = SubstrateStore.open_or_init(tmp_path / "workspace.git", _identity())
    # Author a real commit to use as the gitlink target. libgit2 rejects a
    # null-OID gitlink at TreeBuilder.insert time, so the gitlink mode rejection
    # we want to pin happens at the walker layer once the tree exists. The
    # commit's content is irrelevant; only its mode-0o160000 reference matters.
    empty_tree_oid = store.repo.TreeBuilder().write()
    signature = pygit2.Signature("test", "test@example.invalid")
    submodule_commit_oid = store.repo.create_commit(None, signature, signature, "submodule head", empty_tree_oid, [])
    builder = store.repo.TreeBuilder()
    builder.insert("vendor", submodule_commit_oid, 0o160000)
    tree_oid = str(builder.write())

    payload = workspace_state_revision_payload(
        (
            {
                "path": "vendor",
                "state": "present",
                "mode": 0o100644,
                "content_digest": _content_digest(b""),
            },
        ),
        byte_authority="tree-backed",
    )
    transition, plan, preparation, descriptor = _build_records(
        store,
        operation_id="op-gitlink",
        payload=payload,
        parents=(),
        git_tree_oid=tree_oid,
    )
    with pytest.raises(
        InvalidRepositoryStateError,
        match=r"unsupported file mode at 'vendor'",
    ):
        store.create_candidate_from_prepared(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=descriptor,
            payload=payload,
            parents=(),
        )
