"""End-to-end integration: tree-backed v2 workspace from filesystem command to materialization.

Each layer of the tree-backed migration has unit coverage:

- ``test_substrate_store.py`` — alternates wiring (Tranche 2A);
- ``test_workspace_driver_tree_backed.py`` — driver -> coordinator -> substrate
  flow with ``git_tree_oid`` threaded through (Tranche 2B);
- ``test_substrate_tree_materialization.py`` — substrate-tree-first byte source
  and graceful fallback (Tranche 3).

This test walks the production ``mg.exec("filesystem", "write", ...)`` path,
which exercises every layer together: scalar effect record, capture-derived
tree-backed v2 candidate, world publication, and the materialization byte
source. The assertions are scoped to layering invariants — the lower-level
behaviors (digest correspondence, tree shape, alternates visibility) are
covered exhaustively in the unit suites.

The test is an insurance smoke check before Phase E narrows the scalar
fallback. A future refactor that breaks composition without breaking any
individual layer would fail here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from vcs_core.vcscore import VcsCore

from ...support.builders import make_marker_filesystem_vcscore


def _content_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_filesystem_write_publishes_tree_backed_v2_workspace(mg: VcsCore) -> None:
    """``mg.exec("filesystem", "write", ...)`` publishes a tree-backed v2 workspace.

    Walks the full layering for one write:

    - the scalar effect record lands on the scope's ref (existing v1 behavior);
    - the workspace driver produces a tree-backed substrate revision selected
      onto the scope's v2 authority ref;
    - the substrate revision's metadata reports ``byte_authority="tree-backed"``
      and carries the scalar workspace tree's Git oid;
    - the substrate revision commit embeds a ``workspace/`` tree entry pointing
      at that same oid (proving the alternates configuration carried the tree
      across stores);
    - the materialization byte source can serve the file content from the
      substrate tree, independent of the scalar ``Store.read_workspace_file``
      path.
    """
    task = mg.fork(mg.ground, "task-tree-backed-e2e")
    mg.exec("filesystem", "write", scope=task, path="hello.txt", content=b"hello\n")

    # The scope's v2 authority ref selects a tree-backed substrate revision.
    manager = mg._world_storage()
    selected_world = manager.read_world(task.ref)
    selected_head = selected_world.snapshot.head_for("workspace").head
    substrate = manager.store("store_workspace")
    metadata = substrate.read_revision_metadata(selected_head)
    assert metadata.byte_authority == "tree-backed", f"expected tree-backed workspace, got {metadata.byte_authority!r}"
    assert metadata.git_tree_oid is not None
    assert len(metadata.git_tree_oid) == 40

    # The substrate commit has a workspace/ tree entry pointing at the same oid
    # the metadata claims - the alternates configuration carried the tree
    # across the coord/substrate boundary at write time.
    import pygit2

    commit = substrate.repo[pygit2.Oid(hex=selected_head)]
    workspace_entry = commit.tree["workspace"]
    assert workspace_entry.filemode == pygit2.GIT_FILEMODE_TREE
    assert str(workspace_entry.id) == metadata.git_tree_oid

    # The materialization byte source serves the file from the substrate tree.
    # This proves the layering above is wired through to where it matters: the
    # filesystem substrate's materialize_workspace loop.
    #
    # NB: the byte source reads from the GROUND world. The write above lives on
    # task.ref, so we merge first to publish the tree-backed selection to ground.
    mg.merge(task, mg.ground)
    assert mg._ground_workspace_is_tree_backed() is True
    served = mg._read_v2_workspace_file_for_materialization("hello.txt")
    assert served is not None, "byte source must serve tree-backed paths on ground"
    content, mode = served
    assert content == b"hello\n"
    assert mode == 0o100644

    # A path not in the manifest returns None - the byte source signals a miss
    # cleanly so the materializer falls back to scalar.
    assert mg._read_v2_workspace_file_for_materialization("not-in-manifest.txt") is None


def test_tree_backed_v2_workspace_survives_repush(mg: VcsCore) -> None:
    """A second filesystem write on the same scope keeps the v2 ground tree-backed.

    Confirms that subsequent writes don't regress the byte_authority mode -
    i.e., the production tree-backed selection path is sticky, not a one-shot
    bootstrap.
    """
    task = mg.fork(mg.ground, "task-tree-backed-second-write")
    mg.exec("filesystem", "write", scope=task, path="a.txt", content=b"a\n")
    mg.exec("filesystem", "write", scope=task, path="b.txt", content=b"b\n")
    mg.merge(task, mg.ground)

    manager = mg._world_storage()
    selected_world = manager.read_world(mg.ground.ref)
    selected_head = selected_world.snapshot.head_for("workspace").head
    metadata = manager.store("store_workspace").read_revision_metadata(selected_head)
    assert metadata.byte_authority == "tree-backed"
    assert metadata.git_tree_oid is not None

    # Both files served from substrate tree.
    a_served = mg._read_v2_workspace_file_for_materialization("a.txt")
    b_served = mg._read_v2_workspace_file_for_materialization("b.txt")
    assert a_served == (b"a\n", 0o100644)
    assert b_served == (b"b\n", 0o100644)


def test_discard_clears_v2_authority_when_tree_backed(mg: VcsCore) -> None:
    """Discarding a scope removes its v2 authority ref even when tree-backed.

    Tree-backed scopes have a ``workspace/`` tree entry alongside the
    selected substrate head; discard must clean up the authority ref the
    same way it does for digest-only scopes. The substrate revision remains
    in the store as evidence (per the design — discard does not erase
    evidence).
    """
    task = mg.fork(mg.ground, "task-discard-tree-backed")
    mg.exec("filesystem", "write", scope=task, path="ephemeral.txt", content=b"discarded\n")

    manager = mg._world_storage()
    # Confirm the tree-backed selection landed on task.ref before discard.
    pre_discard = manager.read_world(task.ref)
    pre_head = pre_discard.snapshot.head_for("workspace").head
    pre_metadata = manager.store("store_workspace").read_revision_metadata(pre_head)
    assert pre_metadata.byte_authority == "tree-backed"

    mg.discard(task)

    # Authority ref is gone after discard.
    assert task.ref not in manager.world_store.repo.references, (
        f"discard must remove {task.ref!r} from v2 authority refs"
    )
    # The substrate revision is retained as evidence; reading its metadata
    # still works and still reports tree-backed.
    post_metadata = manager.store("store_workspace").read_revision_metadata(pre_head)
    assert post_metadata.byte_authority == "tree-backed"
    assert post_metadata.git_tree_oid == pre_metadata.git_tree_oid


def test_tree_backed_view_persists_across_vcscore_reload(workspace: Path) -> None:
    """Tree-backed v2 ground survives tearing down and re-opening the VcsCore
    instance from the same on-disk repo.

    This is a "persistence is durable, not in-memory" assertion: a fresh
    VcsCore handle constructed against the same ``.vcscore`` directory must
    discover the same tree-backed selection, with the same ``git_tree_oid``,
    and serve the same bytes from the substrate tree. Catches regressions
    where any tree-backed-ness depended on transient in-process state.

    (Recovery under publication failure is covered at the unit level by
    ``test_world_operation_runner_keeps_published_world_recoverable_after_bookkeeping_failure``;
    this end-to-end test scopes to durable persistence rather than re-testing
    recovery internals.)
    """
    first = make_marker_filesystem_vcscore(workspace, activate=True)
    try:
        task = first.fork(first.ground, "task-tree-backed-persist")
        first.exec("filesystem", "write", scope=task, path="persist.txt", content=b"durable\n")
        first.merge(task, first.ground)

        manager = first._world_storage()
        selected_head = manager.read_world(first.ground.ref).snapshot.head_for("workspace").head
        first_metadata = manager.store("store_workspace").read_revision_metadata(selected_head)
        assert first_metadata.byte_authority == "tree-backed"
        first_git_tree_oid = first_metadata.git_tree_oid
        assert first_git_tree_oid is not None
    finally:
        first.deactivate()

    # Re-open from disk; the second VcsCore instance has fresh in-memory state
    # but observes the same on-disk repo.
    second = make_marker_filesystem_vcscore(workspace, activate=True)
    try:
        assert second._ground_workspace_is_tree_backed() is True
        manager = second._world_storage()
        selected_head = manager.read_world(second.ground.ref).snapshot.head_for("workspace").head
        second_metadata = manager.store("store_workspace").read_revision_metadata(selected_head)
        assert second_metadata.byte_authority == "tree-backed"
        assert second_metadata.git_tree_oid == first_git_tree_oid
        # And the byte source still serves the file from the substrate tree.
        served = second._read_v2_workspace_file_for_materialization("persist.txt")
        assert served == (b"durable\n", 0o100644)
    finally:
        second.deactivate()


def test_sequential_writes_serve_latest_substrate_tree_bytes(mg: VcsCore) -> None:
    """A second write to the same path on ground updates the substrate tree.

    The byte source must serve the latest content, not a stale prior tree.
    This catches a class of regressions where the substrate-tree byte source
    is cached against the first published revision and misses later
    selections.
    """
    mg.exec("filesystem", "write", scope=mg.ground, path="evolving.txt", content=b"first\n")
    first_served = mg._read_v2_workspace_file_for_materialization("evolving.txt")
    assert first_served == (b"first\n", 0o100644)
    first_metadata_head = mg._world_storage().read_world(mg.ground.ref).snapshot.head_for("workspace").head

    mg.exec("filesystem", "write", scope=mg.ground, path="evolving.txt", content=b"second\n")
    second_served = mg._read_v2_workspace_file_for_materialization("evolving.txt")
    assert second_served == (b"second\n", 0o100644), (
        "byte source must serve the latest substrate-tree content, not a cached prior revision"
    )
    second_metadata_head = mg._world_storage().read_world(mg.ground.ref).snapshot.head_for("workspace").head
    # The selected substrate head must have advanced; otherwise the byte
    # source served the same revision and the assertion above proves
    # nothing about the freshness of the read.
    assert second_metadata_head != first_metadata_head, (
        "ground workspace head did not advance between writes; the freshness assertion is vacuous"
    )


# NB: SP3.2 originally scoped two additional tests beyond the existing
# `test_discard_clears_v2_authority_when_tree_backed` (which already covers
# Phase C / smoke #1). Both deferred per the pre-flight discipline in
# `vcs-core/design/roadmap/substrate-framework/260523-2200-spi/FOLLOW-UP-PASS.md`
# SP3.2 ("no clean mock target → defer"):
#
#   1. `test_recovery_preserves_tree_backed_after_publication_failure` —
#      simulating "publication failure between candidate creation and authority
#      advance" requires intercepting `WorldStorageManager._publish_world` or a
#      similar deep coordinator method mid-flight. The recovery flow itself is
#      exercised by the authority-finalizer mocking in
#      `tests/unit/test_world_recovery.py`, but lifting that into an integration
#      test that produces an interrupted-then-recovered tree-backed candidate
#      requires deeper recovery-internals knowledge than SP3.2's tranche budget
#      affords. Tree-backed metadata preservation across reload is already
#      covered by `test_tree_backed_view_persists_across_vcscore_reload`.
#
#   2. `test_materialization_falls_back_when_alternates_path_missing` —
#      VcsCore's `open_or_init` always re-establishes the substrate alternates
#      link via `_ensure_alternates` (idempotent write). The "alternates
#      missing" state is therefore unreachable through VcsCore in steady
#      state — manually unlinking the file and re-opening VcsCore just
#      re-creates it before any byte-source read sees the missing state.
#      Unit-level coverage exists at
#      `tests/unit/test_substrate_tree_materialization.py::test_read_substrate_workspace_file_returns_none_when_alternates_missing`,
#      which uses a direct `pygit2.Repository(substrate.repo.path)` handle
#      (no VcsCore) to bypass the re-establish behavior. Lifting to
#      integration without bypassing `_ensure_alternates` would require
#      either disabling the re-establish (production behavior change) or
#      monkey-patching the manager (deep coupling); both are out of scope
#      for SP3.2.
#
# Both deferred to follow-on commits if the cases become reproducible without
# the structural workarounds above. The existing N=5 tests in this file plus
# the unit coverage in `test_substrate_tree_materialization.py` cover the
# load-bearing invariants.
