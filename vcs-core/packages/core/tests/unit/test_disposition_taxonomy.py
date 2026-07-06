# under-test: vcs_core._query_inventory
"""Part C characterization: the typed Disposition (Tier 2 of the control-plane taxonomy).

Pins the behavior the taxonomy names (260622-control-plane-taxonomy.md): a fact's disposition
DRIVES (for the new open-journal-index facts) or DESCRIBES (for the retrofitted worked examples) how
`_derive_blockers` treats it — blocking gates the mutation, diagnostic never blocks and is never a
RecoveryKind, recoverable is the RecoveryKind set. Migrating every domain onto disposition (and
deleting the legacy fallback branches) is an explicit follow-on; this pins the worked examples.
"""

from __future__ import annotations

from typing import get_args

import pygit2
from vcs_core._operation_journal_inventory import (
    _open_operation_journal_index_corrupt_item,
    probe_operation_journal_ref,
)
from vcs_core._query_inventory import InventoryIssue, InventoryItem, RecoveryKind, present_invalid
from vcs_core._query_readiness import ReadinessRequest, _derive_blockers, _policy_for
from vcs_core._recovery_inventory import _recovery_item
from vcs_core._world_refs import world_publication_lease_index_ref
from vcs_core._world_storage_installation import open_or_init_default_world_storage
from vcs_core.testing import operation_journal_ref
from vcs_core.vcscore import VcsCore


def _mutating_blockers(items: tuple[InventoryItem, ...]):
    request = ReadinessRequest.create(command="shepherd.run", requested_freshness="locked", allow_best_effort=False)
    blockers, _ = _derive_blockers(request, _policy_for("shepherd.run"), items)
    return blockers


def _operation_journal_fact(disposition: str | None) -> InventoryItem:
    # A present-INVALID operation_journal fact: without a disposition it would block via the generic
    # validity fallback; the disposition is what classifies it (blocking vs diagnostic).
    issue = InventoryIssue(
        id="i", code="open_operation_journal_index_corrupt", message="m", subject_id="x", locator="r"
    )
    return InventoryItem(
        id="x",
        domain="operation_journal",
        kind="open_operation_journal_index",
        locator="r",
        source_kind="git_ref",
        source_store="coordinator",
        health=present_invalid(primary_issue="corrupt", issue_codes=("c",), lifecycle="recoverable"),
        issues=(issue,),
        disposition=disposition,  # type: ignore[arg-type]
    )


def test_blocking_disposition_produces_a_blocker() -> None:
    assert len(_mutating_blockers((_operation_journal_fact("blocking"),))) == 1


def test_diagnostic_disposition_suppresses_the_validity_block() -> None:
    # The SAME present-invalid fact, but diagnostic → no blocker (diagnostic is never a blocker, even
    # though validity=="invalid" would otherwise auto-block via the generic fallback).
    assert _mutating_blockers((_operation_journal_fact("diagnostic"),)) == ()


def test_to_json_emits_disposition_only_when_declared() -> None:
    assert _operation_journal_fact("blocking").to_json()["disposition"] == "blocking"
    assert "disposition" not in _operation_journal_fact(None).to_json()  # un-migrated fact: unchanged serialization


def test_corrupt_index_fact_declares_blocking() -> None:
    item = _open_operation_journal_index_corrupt_item(store_id="s", ref="refs/vcscore/publishing/x", detail="d")
    assert item.disposition == "blocking"


def test_open_journal_declares_blocking_terminal_declares_none(mg: VcsCore) -> None:
    manager = open_or_init_default_world_storage(mg._repo_path)
    manager.open_operation_journal(
        operation_id="op-open", operation_kind="shepherd.task", target_ref="refs/vcscore/ground", input_world_oid=None
    )
    open_item = probe_operation_journal_ref(
        manager.world_store.repo, operation_journal_ref("open", "op-open"), expected_family="open"
    )
    assert open_item.disposition == "blocking"  # the canonical blocking admission fact

    sig = pygit2.Signature("t", "t@e.invalid")
    closed = manager.world_store.repo.create_commit(
        None, sig, sig, "x", manager.world_store.repo.TreeBuilder().write(), []
    )
    manager.world_store.repo.references.create(operation_journal_ref("closed", "op-term"), closed)
    closed_item = probe_operation_journal_ref(
        manager.world_store.repo, operation_journal_ref("closed", "op-term"), expected_family="closed"
    )
    assert closed_item.disposition is None  # a terminal journal is not a blocking admission fact


def test_recovery_items_declare_recoverable() -> None:
    item = _recovery_item(
        item_id="recovery:orphaned_operation_ref:op",
        kind="orphaned_operation_ref",
        locator=operation_journal_ref("open", "op"),
        source_kind="git_ref",
        health_status="present_orphaned",
        issue_code="recovery_orphaned_operation_ref",
        message="orphaned",
        fields={},
        source_identity={},
    )
    assert item.disposition == "recoverable"  # == today's RecoveryKind set
    assert item.kind in get_args(RecoveryKind)


def test_active_lease_index_corruption_is_diagnostic_and_never_blocks(mg: VcsCore) -> None:
    manager = open_or_init_default_world_storage(mg._repo_path)
    repo = manager.world_store.repo
    sig = pygit2.Signature("t", "t@e.invalid")
    corrupt = repo.create_commit(None, sig, sig, "corrupt", repo.TreeBuilder().write(), [])
    repo.references.create(world_publication_lease_index_ref(manager.world_store.world_store_id), corrupt, force=True)

    lease_items = [item for item in mg.recovery_inventory().items if item.kind == "active_lease_index"]
    assert len(lease_items) == 1
    assert lease_items[0].disposition == "diagnostic"
    assert "active_lease_index" not in get_args(RecoveryKind)  # diagnostic is never a RecoveryKind
    assert _mutating_blockers((lease_items[0],)) == ()  # ...and never produces a blocker
