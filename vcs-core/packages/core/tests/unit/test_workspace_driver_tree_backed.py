# under-test: vcs_core._world_substrate_adapters
"""Tranche 2 integration tests for tree-backed workspace driver flows.

These tests exercise the full driver -> coordinator -> substrate path with a
``git_tree_oid`` threaded through, so they cover:

- TransitionDraft carries git_tree_oid;
- coordinator propagates it to PreparedRevisionPlan.git_tree_oid;
- SubstrateStore embeds a workspace/ tree entry and validates manifest/tree
  correspondence on the way in;
- substrate metadata reports byte_authority="tree-backed";
- alternates configured by WorldStorageManager make the source tree
  visible to the substrate ODB.

The companion contract tests in test_tree_backed_revisions.py author the
tree directly in the substrate; these tests author it in the source repo
(simulating the scalar Store path) and let alternates carry it across.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pygit2
from vcs_core._world_substrate_adapters import (
    WorkspaceSubstrateAdapter,
    workspace_state_revision_payload,
)
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import SubstrateStoreSpec, WorldStorageManager


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _build_workspace_tree(repo: pygit2.Repository, contents: dict[str, bytes]) -> str:
    """Build a single-level workspace tree in ``repo`` and return its hex oid."""
    builder = repo.TreeBuilder()
    for name in sorted(contents):
        blob_oid = repo.create_blob(contents[name])
        builder.insert(name, blob_oid, pygit2.GIT_FILEMODE_BLOB)
    return str(builder.write())


def _shared_manager(tmp_path: Path, source_repo: Path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / "world-vectors",
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
        substrate_shared_object_repo_path=source_repo,
    )


def test_workspace_driver_propagates_git_tree_oid_into_transition_draft(tmp_path) -> None:
    """A driver command that receives ``git_tree_oid`` in params propagates it
    to the resulting TransitionDraft. This is the boundary that Tranche 2A's
    coordinator lowering relies on to populate ``PreparedRevisionPlan``."""
    source = pygit2.init_repository(str(tmp_path / "source.git"), bare=True)
    tree_oid = _build_workspace_tree(source, {"a.txt": b"hi\n"})

    manager = _shared_manager(tmp_path, tmp_path / "source.git")
    adapter = WorkspaceSubstrateAdapter(manager)

    payload = workspace_state_revision_payload(
        ({"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"hi\n")},),
        byte_authority="tree-backed",
    )
    from vcs_core.spi import ScanRequest

    result = adapter.driver.prepare(
        adapter._context(operation_id="op-scan", parents=()),
        ScanRequest(
            scan_kind="workspace-scan",
            external_state={"payload": payload, "git_tree_oid": tree_oid},
        ),
    )
    assert len(result.transitions) == 1
    assert result.transitions[0].git_tree_oid == tree_oid
    assert result.transitions[0].semantic_op == "workspace-scan"


def test_workspace_scan_candidate_with_tree_oid_produces_tree_backed_revision(tmp_path) -> None:
    """End-to-end: a scan candidate created with workspace_tree_oid lands as a
    substrate revision whose metadata declares ``byte_authority="tree-backed"``
    and whose commit tree contains a ``workspace/`` entry pointing at the
    supplied tree oid."""
    source_path = tmp_path / "source.git"
    source = pygit2.init_repository(str(source_path), bare=True)
    contents = {"a.txt": b"alpha\n", "run.sh": b"#!/bin/sh\necho hi\n"}
    tree_oid = _build_workspace_tree(source, contents)

    manager = _shared_manager(tmp_path, source_path)
    workspace = WorkspaceSubstrateAdapter(manager)

    payload = workspace_state_revision_payload(
        (
            {"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(contents["a.txt"])},
            {"path": "run.sh", "state": "present", "mode": 0o100644, "content_digest": _digest(contents["run.sh"])},
        ),
        byte_authority="tree-backed",
    )
    bundle = workspace.create_scan_candidate(
        operation_id="op-scan",
        payload=payload,
        parents=(),
        workspace_tree_oid=tree_oid,
    )

    substrate = manager.store("store_workspace")
    provenance = substrate.validate_prepared_candidate(
        bundle.candidate.head, evidence_resolver=manager.world_store.resolve_evidence_ref
    )
    assert provenance.metadata.byte_authority == "tree-backed"
    assert provenance.metadata.git_tree_oid == tree_oid
    assert provenance.plan.git_tree_oid == tree_oid

    # The commit tree contains a workspace/ entry pointing at the source tree.
    commit = substrate.repo[pygit2.Oid(hex=bundle.candidate.head)]
    workspace_entry = commit.tree["workspace"]
    assert workspace_entry.filemode == pygit2.GIT_FILEMODE_TREE
    assert str(workspace_entry.id) == tree_oid


def test_workspace_adoption_and_overlay_candidates_carry_tree_oid(tmp_path) -> None:
    """The adopt-baseline and overlay-merge driver paths share the same
    plumbing as scan; this test pins their behavior so a future regression in
    one path is caught even when the others pass."""
    source_path = tmp_path / "source.git"
    source = pygit2.init_repository(str(source_path), bare=True)
    tree_oid = _build_workspace_tree(source, {"baseline.txt": b"baseline\n"})

    manager = _shared_manager(tmp_path, source_path)
    workspace = WorkspaceSubstrateAdapter(manager)

    payload = workspace_state_revision_payload(
        (
            {
                "path": "baseline.txt",
                "state": "present",
                "mode": 0o100644,
                "content_digest": _digest(b"baseline\n"),
            },
        ),
        byte_authority="tree-backed",
    )
    adoption = workspace.create_adoption_candidate(
        operation_id="op-adopt", payload=payload, parents=(), workspace_tree_oid=tree_oid
    )
    overlay = workspace.create_overlay_merge_candidate(
        operation_id="op-overlay", payload=payload, parents=(), workspace_tree_oid=tree_oid
    )

    substrate = manager.store("store_workspace")
    for bundle in (adoption, overlay):
        provenance = substrate.validate_prepared_candidate(
            bundle.candidate.head, evidence_resolver=manager.world_store.resolve_evidence_ref
        )
        assert provenance.metadata.byte_authority == "tree-backed"
        assert provenance.metadata.git_tree_oid == tree_oid


def test_workspace_candidate_without_tree_oid_remains_digest_only(tmp_path) -> None:
    """Backward compatibility: omitting workspace_tree_oid keeps revisions
    digest-only, which is what existing callers and tests rely on."""
    pygit2.init_repository(str(tmp_path / "source.git"), bare=True)
    manager = _shared_manager(tmp_path, tmp_path / "source.git")
    workspace = WorkspaceSubstrateAdapter(manager)

    payload = workspace_state_revision_payload(
        ({"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"a")},)
    )
    bundle = workspace.create_scan_candidate(operation_id="op-scan", payload=payload, parents=())

    substrate = manager.store("store_workspace")
    provenance = substrate.validate_prepared_candidate(
        bundle.candidate.head, evidence_resolver=manager.world_store.resolve_evidence_ref
    )
    assert provenance.metadata.byte_authority == "digest-only"
    assert provenance.metadata.git_tree_oid is None
    assert provenance.plan.git_tree_oid is None
    # No workspace/ entry on a digest-only revision.
    commit = substrate.repo[pygit2.Oid(hex=bundle.candidate.head)]
    assert "workspace" not in {entry.name for entry in commit.tree}


def test_workspace_tree_oid_outside_alternates_fails_closed(tmp_path) -> None:
    """When the substrate cannot see the supplied tree (no alternates), the
    write fails closed before producing a commit. This exercises the
    libgit2-enforced cross-store boundary from the 260523 capability spike."""
    # Build a tree in a repo the substrate does NOT have alternates to.
    unreachable = pygit2.init_repository(str(tmp_path / "unreachable.git"), bare=True)
    tree_oid = _build_workspace_tree(unreachable, {"a.txt": b"a"})

    # Substrate has no shared_object_repo_path: it cannot see the tree.
    manager = WorldStorageManager.open_or_init(
        tmp_path / "world-vectors",
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
    workspace = WorkspaceSubstrateAdapter(manager)
    payload = workspace_state_revision_payload(
        ({"path": "a.txt", "state": "present", "mode": 0o100644, "content_digest": _digest(b"a")},),
        byte_authority="tree-backed",
    )
    import pytest
    from vcs_core import InvalidRepositoryStateError

    # The Tranche 1 manifest/tree correspondence validator catches this first
    # at write time: it tries to resolve the tree from the substrate's ODB and
    # finds it missing because no alternates were configured.
    with pytest.raises(InvalidRepositoryStateError, match="references unknown tree"):
        workspace.create_scan_candidate(
            operation_id="op-scan",
            payload=payload,
            parents=(),
            workspace_tree_oid=tree_oid,
        )
