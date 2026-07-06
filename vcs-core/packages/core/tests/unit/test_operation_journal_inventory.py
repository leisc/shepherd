# under-test: vcs_core._operation_journal_inventory
from __future__ import annotations

import pygit2
import pytest
from vcs_core import InvalidRepositoryStateError, canonical_bytes
from vcs_core._operation_journal_inventory import probe_operation_journal, probe_operation_journals
from vcs_core._query_inventory import (
    OPERATION_JOURNAL_IDENTITY_MISMATCH,
    OPERATION_JOURNAL_MISSING_REF,
    OPERATION_JOURNAL_PAYLOAD_CORRUPT,
    OPERATION_JOURNAL_UNSUPPORTED_FAMILY,
)
from vcs_core._world_authority_finalizer import WorldAuthorityFinalizer
from vcs_core._world_operation_journal import OPERATION_JOURNAL_PATH, OPERATION_JOURNAL_SCHEMA
from vcs_core._world_refs import encode_ref_component
from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import DEFAULT_GROUND_REF, SubstrateStoreSpec, WorldStorageManager, operation_journal_ref


def _workspace_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="fs:repo-main")


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="world-store",
        stores=(SubstrateStoreSpec(identity=_workspace_identity(), locator="workspace-store.git"),),
    )


def test_probe_operation_journal_reports_targeted_absence(tmp_path) -> None:
    manager = _manager(tmp_path)

    item = probe_operation_journal(manager.world_store.repo, "op-missing", family="open")

    assert item.health.presence == "absent"
    assert item.health.issue_codes == (OPERATION_JOURNAL_MISSING_REF,)
    assert item.locator == operation_journal_ref("open", "op-missing")


def test_probe_operation_journal_reports_valid_journal(tmp_path) -> None:
    manager = _manager(tmp_path)
    manager.open_operation_journal(
        operation_id="op-valid",
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=None,
    )

    item = probe_operation_journal(manager.world_store.repo, "op-valid", family="open")

    assert item.health.status == "present_valid"
    assert item.fields["operation_id"] == "op-valid"
    assert item.fields["identity_match"] is True
    assert item.source_identity["ref_target_oid"]


def test_probe_operation_journals_keeps_malformed_payload_visible(tmp_path) -> None:
    manager = _manager(tmp_path)
    _write_manual_journal_commit(
        manager,
        payload_bytes=b"not json",
        ref=operation_journal_ref("open", "op-corrupt"),
    )

    items = probe_operation_journals(manager.world_store.repo)

    assert len(items) == 1
    assert items[0].health.status == "present_corrupt"
    assert items[0].health.issue_codes == (OPERATION_JOURNAL_PAYLOAD_CORRUPT,)


def test_probe_operation_journal_reports_payload_locator_identity_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-payload",
        "operation_kind": "shepherd.task",
        "status": "opened",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": None,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", "op-locator"),
    )

    item = probe_operation_journal(manager.world_store.repo, "op-locator", family="open")

    assert item.health.status == "identity_mismatch"
    assert item.health.issue_codes == (OPERATION_JOURNAL_IDENTITY_MISMATCH,)
    assert item.fields["payload_operation_id"] == "op-payload"
    assert item.fields["locator_operation_id"] == "op-locator"


def test_probe_operation_journal_preserves_targeted_identity_for_hashed_locator(tmp_path) -> None:
    manager = _manager(tmp_path)
    operation_id = f"op-{'x' * 120}"
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-payload",
        "operation_kind": "shepherd.task",
        "status": "opened",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": None,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
    )

    item = probe_operation_journal(manager.world_store.repo, operation_id, family="open")

    assert item.health.status == "identity_mismatch"
    assert item.health.issue_codes == (OPERATION_JOURNAL_IDENTITY_MISMATCH,)
    assert item.fields["expected_operation_id"] == operation_id
    assert item.fields["payload_operation_id"] == "op-payload"
    assert item.fields["locator_operation_encoding"] == "sha256"


def test_probe_operation_journals_ignores_legacy_ops_prefix_refs(tmp_path) -> None:
    manager = _manager(tmp_path)
    _write_manual_journal_commit(
        manager,
        payload_bytes=b"not json",
        ref="refs/vcscore/ops/legacy-open-operation",
    )

    assert probe_operation_journals(manager.world_store.repo) == ()


def test_probe_operation_journals_keeps_unknown_v2_family_visible(tmp_path) -> None:
    manager = _manager(tmp_path)
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-unknown-family",
        "operation_kind": "shepherd.task",
        "status": "opened",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": None,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        payload_bytes=canonical_bytes(payload),
        ref=(f"refs/vcscore/ops/{encode_ref_component('retrying')}/{encode_ref_component('op-unknown-family')}"),
    )

    items = probe_operation_journals(manager.world_store.repo)

    assert len(items) == 1
    assert items[0].health.status == "unsupported_schema"
    assert items[0].health.issue_codes == (OPERATION_JOURNAL_UNSUPPORTED_FAMILY,)


def test_world_authority_finalizer_blocks_on_present_invalid_journal(tmp_path) -> None:
    manager = _manager(tmp_path)
    _write_manual_journal_commit(
        manager,
        payload_bytes=b"not json",
        ref=operation_journal_ref("open", "op-corrupt"),
    )
    finalizer = WorldAuthorityFinalizer(manager)

    with pytest.raises(InvalidRepositoryStateError, match="operation_journal_payload_corrupt"):
        finalizer.complete_existing(
            operation_id="op-corrupt",
            target_ref=DEFAULT_GROUND_REF,
            expected_input_world_oid=None,
            missing_ok=True,
        )


def test_world_authority_finalizer_blocks_on_hashed_locator_identity_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    operation_id = f"op-{'x' * 120}"
    payload = {
        "schema": OPERATION_JOURNAL_SCHEMA,
        "operation_id": "op-payload",
        "operation_kind": "shepherd.task",
        "status": "opened",
        "seq": 0,
        "target_ref": DEFAULT_GROUND_REF,
        "input_world_oid": None,
        "candidate_refs": [],
        "candidate_outcomes": [],
        "selected": {},
        "created_at_unix_ns": 1,
        "updated_at_unix_ns": 1,
    }
    _write_manual_journal_commit(
        manager,
        payload_bytes=canonical_bytes(payload),
        ref=operation_journal_ref("open", operation_id),
    )
    finalizer = WorldAuthorityFinalizer(manager)

    with pytest.raises(InvalidRepositoryStateError, match="operation_journal_identity_mismatch"):
        finalizer.complete_existing(
            operation_id=operation_id,
            target_ref=DEFAULT_GROUND_REF,
            expected_input_world_oid=None,
            missing_ok=True,
        )


def _write_manual_journal_commit(
    manager: WorldStorageManager,
    *,
    payload_bytes: bytes,
    ref: str,
) -> str:
    repo = manager.world_store.repo
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        OPERATION_JOURNAL_PATH.split("/")[-1],
        repo.create_blob(payload_bytes),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root_builder = repo.TreeBuilder()
    insert_tree_entry(repo, root_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("vcs-core operation journal", "vcs-core@example.invalid")
    oid = create_commit_with_recovery(
        repo,
        None,
        signature,
        signature,
        "manual journal",
        root_builder.write(),
        [],
    )
    repo.references.create(ref, oid, force=True)
    return str(oid)
