# under-test: vcs_core._sibling_groups
"""Deferred sibling-group control-ref recovery projection tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from vcs_core import InvalidRepositoryStateError, _sibling_groups
from vcs_core._sibling_groups import (
    CarrierLeaseRecord,
    SiblingGroupRecord,
    SiblingHandleRecord,
    canonical_sibling_group_json,
    sibling_machine_scope_name,
)
from vcs_core.git_store import build_tree, create_signature
from vcs_core.store import Store


def _parent_oid(store: Store) -> str:
    return store.log(ref=Store.GROUND_REF, max_count=1)[0].oid


def _sibling(
    store: Store,
    *,
    group_id: str,
    ordinal: int,
) -> SiblingHandleRecord:
    machine_scope_name = sibling_machine_scope_name(group_id, ordinal)
    return SiblingHandleRecord(
        world_id=f"{group_id}-world-{ordinal}",
        machine_scope_name=machine_scope_name,
        display_label=f"attempt-{ordinal}",
        scope_ref=f"refs/vcscore/scopes/{machine_scope_name}",
        parent_ref=Store.GROUND_REF,
        creation_oid=_parent_oid(store),
        state="admitted",
        instance_id=f"inst-{ordinal}",
    )


def _record(store: Store, *, group_id: str = "sg-111111111111", status: str = "admitting") -> SiblingGroupRecord:
    siblings = (_sibling(store, group_id=group_id, ordinal=0), _sibling(store, group_id=group_id, ordinal=1))
    return SiblingGroupRecord(
        group_id=group_id,
        parent_ref=Store.GROUND_REF,
        parent_world_id="ground-world",
        admitted_parent_oid=_parent_oid(store),
        status=status,  # type: ignore[arg-type]
        siblings=siblings,
        leases=(
            CarrierLeaseRecord(
                lease_id=f"{group_id}-lease-0",
                world_id=siblings[0].world_id,
                substrate="filesystem",
                target_id="workspace",
                mode="writable_carrier",
                resource_key="workspace",
                state="planned",
                carrier_ref=siblings[0].scope_ref,
            ),
        ),
        created_at=1.0,
        updated_at=2.0,
    )


def _write_raw_sibling_group_payload(store: Store, *, group_id: str, payload: bytes) -> None:
    _write_raw_sibling_group_ref(store, ref=Store.sibling_group_ref(group_id), group_id=group_id, payload=payload)


def _write_raw_sibling_group_ref(store: Store, *, ref: str, group_id: str, payload: bytes) -> None:
    tree_oid = build_tree(store._repo, None, [("meta/sibling-group.json", payload)])
    sig = create_signature("sibling-group")
    commit_oid = store._repo.create_commit(
        None,
        sig,
        sig,
        f"sibling-group:{group_id}",
        tree_oid,
        [],
    )
    if ref in store._repo.references:
        store._repo.references[ref].set_target(commit_oid)
    else:
        store._repo.references.create(ref, commit_oid)


def _write_blob_sibling_group_ref(store: Store, *, group_id: str, payload: bytes) -> None:
    blob_oid = store._repo.create_blob(payload)
    store._repo.references.create(Store.sibling_group_ref(group_id), blob_oid)


def test_publish_and_load_sibling_group_round_trip(store: Store) -> None:
    record = _record(store)

    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None) is True

    snapshot = store.load_sibling_group(record.group_id)
    assert snapshot is not None
    assert snapshot.record == record
    assert snapshot.head_oid


def test_publish_sibling_group_for_recovery_test_updates_with_matching_head(store: Store) -> None:
    record = _record(store)
    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None)
    first = store.load_sibling_group(record.group_id)
    assert first is not None

    updated = replace(record, status="admitted", updated_at=3.0)

    assert store._publish_sibling_group_for_recovery_test(updated, expected_head_oid=first.head_oid) is True
    second = store.load_sibling_group(record.group_id)
    assert second is not None
    assert second.record == updated
    assert second.head_oid != first.head_oid


def test_publish_sibling_group_for_recovery_test_rejects_conflicting_head(store: Store) -> None:
    record = _record(store)
    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None)
    first = store.load_sibling_group(record.group_id)
    assert first is not None
    assert store._publish_sibling_group_for_recovery_test(
        replace(record, status="admitted"),
        expected_head_oid=first.head_oid,
    )

    conflicting = replace(record, status="failed", updated_at=4.0)

    assert store._publish_sibling_group_for_recovery_test(conflicting, expected_head_oid=first.head_oid) is False
    assert store.load_sibling_group(record.group_id).record.status == "admitted"  # type: ignore[union-attr]


def test_publish_sibling_group_for_recovery_test_rechecks_head_before_ref_update(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(store)
    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None)
    first = store.load_sibling_group(record.group_id)
    assert first is not None
    desired = replace(record, status="admitted", updated_at=3.0)
    competing = replace(record, status="failed", updated_at=4.0)
    original_create_commit = _sibling_groups.create_commit_with_recovery
    raced = False

    def create_commit_with_race(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal raced
        oid = original_create_commit(*args, **kwargs)
        if not raced:
            raced = True
            _write_raw_sibling_group_payload(
                store,
                group_id=competing.group_id,
                payload=canonical_sibling_group_json(competing),
            )
        return oid

    monkeypatch.setattr(_sibling_groups, "create_commit_with_recovery", create_commit_with_race)

    assert store._publish_sibling_group_for_recovery_test(desired, expected_head_oid=first.head_oid) is False
    current = store.load_sibling_group(record.group_id)
    assert current is not None
    assert current.record == competing


def test_publish_sibling_group_for_recovery_test_is_idempotent_for_same_payload(store: Store) -> None:
    record = _record(store)

    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None)
    first = store.load_sibling_group(record.group_id)
    assert first is not None

    assert store._publish_sibling_group_for_recovery_test(record, expected_head_oid=None) is True
    second = store.load_sibling_group(record.group_id)
    assert second is not None
    assert second.head_oid == first.head_oid


def test_load_sibling_group_detects_corrupt_payload(store: Store) -> None:
    _write_raw_sibling_group_payload(store, group_id="sg-222222222222", payload=b"not json")

    with pytest.raises(InvalidRepositoryStateError, match="missing a readable payload"):
        store.load_sibling_group("sg-222222222222")


def test_load_sibling_group_detects_mismatched_group_id(store: Store) -> None:
    record = _record(store, group_id="sg-333333333333")
    payload = json.loads(canonical_sibling_group_json(record))
    payload["group_id"] = "sg-444444444444"
    _write_raw_sibling_group_payload(
        store,
        group_id="sg-333333333333",
        payload=json.dumps(payload, sort_keys=True).encode("utf-8"),
    )

    with pytest.raises(InvalidRepositoryStateError, match="reports group_id"):
        store.load_sibling_group("sg-333333333333")


def test_load_sibling_group_detects_non_commit_ref(store: Store) -> None:
    _write_blob_sibling_group_ref(store, group_id="sg-444444444444", payload=b"not a commit")

    with pytest.raises(InvalidRepositoryStateError, match="readable commit"):
        store.load_sibling_group("sg-444444444444")


def test_list_sibling_groups_returns_readable_and_unreadable_entries(store: Store) -> None:
    readable = _record(store, group_id="sg-555555555555")
    assert store._publish_sibling_group_for_recovery_test(readable, expected_head_oid=None)
    _write_raw_sibling_group_payload(store, group_id="sg-666666666666", payload=b"not json")

    listing = store.list_sibling_groups()

    assert [snapshot.record.group_id for snapshot in listing.groups] == ["sg-555555555555"]
    assert [(entry.group_id, entry.ref) for entry in listing.unreadable] == [
        ("sg-666666666666", Store.sibling_group_ref("sg-666666666666"))
    ]


def test_list_sibling_groups_reports_malformed_ref_as_unreadable(store: Store) -> None:
    ref = "refs/vcscore/sibling-groups/SG-UPPER"
    _write_raw_sibling_group_ref(store, ref=ref, group_id="SG-UPPER", payload=b"not json")

    listing = store.list_sibling_groups()

    assert listing.groups == ()
    assert [(entry.group_id, entry.ref) for entry in listing.unreadable] == [("SG-UPPER", ref)]
    assert "group_id" in listing.unreadable[0].reason


def test_list_sibling_groups_reports_non_commit_ref_as_unreadable(store: Store) -> None:
    _write_blob_sibling_group_ref(store, group_id="sg-777777777777", payload=b"not a commit")

    listing = store.list_sibling_groups()

    assert listing.groups == ()
    assert [(entry.group_id, entry.ref) for entry in listing.unreadable] == [
        ("sg-777777777777", Store.sibling_group_ref("sg-777777777777"))
    ]
    assert "readable commit" in listing.unreadable[0].reason
