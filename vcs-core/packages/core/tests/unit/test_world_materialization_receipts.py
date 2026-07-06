# under-test: vcs_core._world_materialization_receipts
"""Unit tests for private v2 materialization receipt storage."""

from __future__ import annotations

import pygit2
import pytest
from vcs_core import InvalidRepositoryStateError, canonical_bytes
from vcs_core._world_materialization_receipts import MaterializationReceiptStore, materialization_receipt_ref
from vcs_core._world_store import WorldStore
from vcs_core._world_types import MaterializationReceipt
from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry


def _store(tmp_path) -> MaterializationReceiptStore:
    world = WorldStore.open_or_init(tmp_path / "worlds.git", world_store_id="store_world_test")
    return MaterializationReceiptStore(world.repo)


def test_materialization_receipt_store_round_trips_closed_receipt(tmp_path) -> None:
    store = _store(tmp_path)
    receipt = MaterializationReceipt(
        materialization_id="mat-1",
        unit_id="unit-1",
        binding="workspace",
        target_identity="file:///repo",
        status="completed",
        idempotency_key="unit-1@world",
        payload_digest="sha256:" + "0" * 64,
        world_oid="1" * 40,
    )

    entry = store.write(receipt, family="closed")
    reopened = store.read(family="closed", materialization_id="mat-1", unit_id="unit-1")

    assert reopened.oid == entry.oid
    assert reopened.receipt == receipt
    assert reopened.receipt.digest().startswith("sha256:")
    assert store.fsck(family="closed", materialization_id="mat-1", unit_id="unit-1").ok
    assert store.write(receipt, family="closed").oid == entry.oid


def test_materialization_receipt_store_rejects_conflicting_rewrite(tmp_path) -> None:
    store = _store(tmp_path)
    receipt = MaterializationReceipt(
        materialization_id="mat-1",
        unit_id="unit-1",
        binding="workspace",
        target_identity="file:///repo",
        status="completed",
    )
    store.write(receipt, family="closed")

    with pytest.raises(InvalidRepositoryStateError, match="different content"):
        store.write(
            MaterializationReceipt(
                materialization_id="mat-1",
                unit_id="unit-1",
                binding="workspace",
                target_identity="file:///other",
                status="completed",
            ),
            family="closed",
        )


def test_materialization_receipt_fsck_reports_noncanonical_record(tmp_path) -> None:
    world = WorldStore.open_or_init(tmp_path / "worlds.git", world_store_id="store_world_test")
    store = MaterializationReceiptStore(world.repo)
    meta = world.repo.TreeBuilder()
    insert_tree_entry(
        world.repo,
        meta,
        "materialization-receipt.json",
        world.repo.create_blob(b'{"schema":"vcscore/materialization-receipt/v1"}'),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root = world.repo.TreeBuilder()
    insert_tree_entry(world.repo, root, "meta", meta.write(), pygit2.GIT_FILEMODE_TREE)
    sig = pygit2.Signature("test", "test@example.invalid")
    oid = create_commit_with_recovery(world.repo, None, sig, sig, "bad receipt", root.write(), [])
    ref = materialization_receipt_ref("closed", "mat-1", "unit-1")
    world.repo.references.create(ref, oid)

    report = store.fsck(family="closed", materialization_id="mat-1", unit_id="unit-1")

    assert not report.ok
    assert report.issues[0].code == "materialization_receipt_invalid"


def test_materialization_receipt_fsck_reports_unexpected_fields(tmp_path) -> None:
    world = WorldStore.open_or_init(tmp_path / "worlds.git", world_store_id="store_world_test")
    store = MaterializationReceiptStore(world.repo)
    receipt = MaterializationReceipt(
        materialization_id="mat-1",
        unit_id="unit-1",
        binding="workspace",
        target_identity="file:///repo",
        status="completed",
    )
    meta = world.repo.TreeBuilder()
    insert_tree_entry(
        world.repo,
        meta,
        "materialization-receipt.json",
        world.repo.create_blob(canonical_bytes({**receipt.to_json(), "extra": "x"})),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root = world.repo.TreeBuilder()
    insert_tree_entry(world.repo, root, "meta", meta.write(), pygit2.GIT_FILEMODE_TREE)
    sig = pygit2.Signature("test", "test@example.invalid")
    oid = create_commit_with_recovery(world.repo, None, sig, sig, "bad receipt", root.write(), [])
    ref = materialization_receipt_ref("closed", "mat-1", "unit-1")
    world.repo.references.create(ref, oid)

    report = store.fsck(family="closed", materialization_id="mat-1", unit_id="unit-1")

    assert not report.ok
    assert "unexpected materialization receipt fields" in report.issues[0].message
