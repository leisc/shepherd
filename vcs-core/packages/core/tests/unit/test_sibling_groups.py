# under-test: vcs_core._sibling_groups
"""Deferred sibling-group DTO and canonical serialization tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from vcs_core._sibling_groups import (
    CarrierLeaseRecord,
    SiblingGroupRecord,
    SiblingHandleRecord,
    _record_from_json,
    canonical_sibling_group_json,
    sibling_group_ref,
    sibling_machine_scope_name,
)
from vcs_core.store import Store

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "vcs_core"


def _sibling(
    *,
    ordinal: int,
    group_id: str = "sg-abcdef123456",
    world_id: str | None = None,
    display_label: str | None = None,
    parent_ref: str = Store.GROUND_REF,
    creation_oid: str = "0" * 40,
) -> SiblingHandleRecord:
    machine_scope_name = sibling_machine_scope_name(group_id, ordinal)
    return SiblingHandleRecord(
        world_id=world_id or f"world-{ordinal}",
        machine_scope_name=machine_scope_name,
        display_label=display_label or f"attempt-{ordinal}",
        scope_ref=f"refs/vcscore/scopes/{machine_scope_name}",
        parent_ref=parent_ref,
        creation_oid=creation_oid,
        state="admitted",
        instance_id=f"inst-{ordinal}",
    )


def _record() -> SiblingGroupRecord:
    siblings = (_sibling(ordinal=0), _sibling(ordinal=1))
    return SiblingGroupRecord(
        group_id="sg-abcdef123456",
        parent_ref=Store.GROUND_REF,
        parent_world_id="ground-world",
        admitted_parent_oid="0" * 40,
        status="admitting",
        siblings=siblings,
        leases=(
            CarrierLeaseRecord(
                lease_id="lease-0",
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


def test_product_code_does_not_publish_deferred_sibling_groups() -> None:
    violations: list[str] = []
    allowed_fragments = {
        "def _publish_sibling_group_for_recovery_test(",
        "def publish_sibling_group_snapshot(",
        "publish_sibling_group_snapshot,",
        "return publish_sibling_group_snapshot(",
    }
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "publish_sibling_group" not in line:
                continue
            if any(fragment in line for fragment in allowed_fragments):
                continue
            violations.append(f"{path.relative_to(SOURCE_ROOT.parent).as_posix()}:{line_number}: {line.strip()}")

    assert violations == []


def test_sibling_group_canonical_json_is_stable() -> None:
    record = _record()

    encoded = canonical_sibling_group_json(record)

    assert encoded == canonical_sibling_group_json(record)
    assert json.loads(encoded) == {
        "version": 1,
        "group_id": "sg-abcdef123456",
        "parent_ref": Store.GROUND_REF,
        "parent_world_id": "ground-world",
        "admitted_parent_oid": "0" * 40,
        "status": "admitting",
        "siblings": [
            {
                "world_id": "world-0",
                "machine_scope_name": "sib-abcdef123456-0",
                "display_label": "attempt-0",
                "scope_ref": "refs/vcscore/scopes/sib-abcdef123456-0",
                "parent_ref": Store.GROUND_REF,
                "creation_oid": "0" * 40,
                "state": "admitted",
                "operation_ids": [],
                "carrier_refs": [],
                "instance_id": "inst-0",
            },
            {
                "world_id": "world-1",
                "machine_scope_name": "sib-abcdef123456-1",
                "display_label": "attempt-1",
                "scope_ref": "refs/vcscore/scopes/sib-abcdef123456-1",
                "parent_ref": Store.GROUND_REF,
                "creation_oid": "0" * 40,
                "state": "admitted",
                "operation_ids": [],
                "carrier_refs": [],
                "instance_id": "inst-1",
            },
        ],
        "leases": [
            {
                "lease_id": "lease-0",
                "world_id": "world-0",
                "substrate": "filesystem",
                "target_id": "workspace",
                "mode": "writable_carrier",
                "resource_key": "workspace",
                "state": "planned",
                "carrier_ref": "refs/vcscore/scopes/sib-abcdef123456-0",
            }
        ],
        "created_at": 1.0,
        "updated_at": 2.0,
    }


def test_constructible_record_round_trips_through_canonical_json() -> None:
    record = replace(
        _record(),
        siblings=(
            replace(
                _record().siblings[0],
                archive_ref="refs/vcscore/archive/sib-abcdef123456-0-inst-0",
                operation_ids=("op-0",),
                carrier_refs=("refs/vcscore/scopes/sib-abcdef123456-0",),
                branch_scope_ref="shepherd-branch-0",
            ),
            _record().siblings[1],
        ),
        leases=(
            replace(
                _record().leases[0],
                carrier_ref="refs/vcscore/scopes/sib-abcdef123456-0",
                reason="isolated overlay layer",
            ),
        ),
    )

    decoded = _record_from_json(json.loads(canonical_sibling_group_json(record)))

    assert decoded == record


def test_generated_machine_scope_names_strip_sg_prefix() -> None:
    assert sibling_machine_scope_name("sg-abcdef123456", 2) == "sib-abcdef123456-2"


def test_group_ref_rejects_unsafe_group_ids() -> None:
    with pytest.raises(ValueError, match="group_id"):
        sibling_group_ref("sg/bad")

    with pytest.raises(ValueError, match="group_id"):
        sibling_group_ref("SG-UPPER")


def test_machine_scope_name_rejects_negative_ordinals() -> None:
    with pytest.raises(ValueError, match="ordinal"):
        sibling_machine_scope_name("sg-abcdef123456", -1)


def test_sibling_rejects_scope_ref_that_does_not_match_machine_name() -> None:
    with pytest.raises(ValueError, match="scope_ref"):
        replace(_sibling(ordinal=0), scope_ref="refs/vcscore/scopes/other")


def test_sibling_rejects_constructible_values_that_loader_would_reject() -> None:
    sibling = _sibling(ordinal=0)

    with pytest.raises(ValueError, match="archive_ref"):
        replace(sibling, archive_ref="")

    with pytest.raises(ValueError, match="operation_ids"):
        replace(sibling, operation_ids=("",))

    with pytest.raises(ValueError, match="carrier_refs"):
        replace(sibling, carrier_refs=("",))

    with pytest.raises(ValueError, match="instance_id"):
        replace(sibling, instance_id="")

    with pytest.raises(ValueError, match="branch_scope_ref"):
        replace(sibling, branch_scope_ref="")


def test_lease_rejects_constructible_values_that_loader_would_reject() -> None:
    lease = _record().leases[0]

    with pytest.raises(ValueError, match="carrier_ref"):
        replace(lease, carrier_ref="")

    with pytest.raises(ValueError, match="reason"):
        replace(lease, reason="")


def test_group_rejects_duplicate_sibling_identity_fields() -> None:
    first = _sibling(ordinal=0)
    duplicate_world = replace(_sibling(ordinal=1), world_id=first.world_id)

    with pytest.raises(ValueError, match="world_ids"):
        replace(_record(), siblings=(first, duplicate_world))

    duplicate_label = replace(_sibling(ordinal=1), display_label=first.display_label)
    with pytest.raises(ValueError, match="display_labels"):
        replace(_record(), siblings=(first, duplicate_label))


def test_group_rejects_lease_for_unknown_world() -> None:
    bad_lease = replace(_record().leases[0], world_id="missing-world")

    with pytest.raises(ValueError, match="unknown sibling world_id"):
        replace(_record(), leases=(bad_lease,))


def test_group_rejects_singleton_sibling_groups() -> None:
    with pytest.raises(ValueError, match="at least two siblings"):
        replace(_record(), siblings=(_sibling(ordinal=0),))


def test_group_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unknown sibling group status"):
        replace(_record(), status="bogus")  # type: ignore[arg-type]


def test_group_rejects_non_finite_timestamps() -> None:
    with pytest.raises(ValueError, match="created_at"):
        replace(_record(), created_at=float("nan"))

    with pytest.raises(ValueError, match="updated_at"):
        replace(_record(), updated_at=float("inf"))
