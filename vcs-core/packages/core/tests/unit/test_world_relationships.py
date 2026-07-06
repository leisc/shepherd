# under-test: vcs_core._world_relationships
"""Unit tests for private v2 world relationship validation."""

from __future__ import annotations

from vcs_core import WorldSnapshot
from vcs_core._world_relationships import validate_relationships
from vcs_core.spi import RelationshipRequirement, SubstrateStoreIdentity
from vcs_core.testing import SubstrateStoreSpec, WorldStorageManager


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity("store_workspace", "filesystem", "fs:repo-main"),
                locator="substrates/workspace.git",
            ),
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity("store_session", "shepherd.session_state", "shepherd-session:child"),
                locator="substrates/session.git",
            ),
        ),
    )


def test_relationship_validation_accepts_exact_and_descends_from(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 42}
    )
    w43 = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/main",
        {"schema": "example/workspace", "n": 43},
        parents=(w42,),
    )
    s7 = manager.create_unsafe_unprepared_json_revision(
        "store_session", "refs/checkpoints/S7", {"schema": "example/session", "n": 7}
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head("store_workspace", binding="workspace", head=w43, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        )
    )

    report = validate_relationships(
        snapshot,
        manager.stores,
        (
            RelationshipRequirement("session", "descends-from", "workspace", w42),
            RelationshipRequirement("session", "exact", "session", s7),
        ),
    )

    assert report.ok


def test_relationship_validation_reports_mismatches(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 42}
    )
    w_other = manager.create_unsafe_unprepared_json_revision(
        "store_workspace",
        "refs/heads/other",
        {"schema": "example/workspace", "n": 99},
    )
    s7 = manager.create_unsafe_unprepared_json_revision(
        "store_session", "refs/checkpoints/S7", {"schema": "example/session", "n": 7}
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head("store_workspace", binding="workspace", head=w_other, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        )
    )

    report = validate_relationships(
        snapshot,
        manager.stores,
        (RelationshipRequirement("session", "descends-from", "workspace", w42),),
    )

    assert not report.ok
    assert report.issues[0].code == "relationship_descends_from_mismatch"


def test_relationship_validation_reports_malformed_and_missing_heads(tmp_path) -> None:
    manager = _manager(tmp_path)
    w42 = manager.create_unsafe_unprepared_json_revision(
        "store_workspace", "refs/heads/main", {"schema": "example/workspace", "n": 42}
    )
    s7 = manager.create_unsafe_unprepared_json_revision(
        "store_session", "refs/checkpoints/S7", {"schema": "example/session", "n": 7}
    )
    snapshot = WorldSnapshot(
        (
            manager.substrate_head("store_workspace", binding="workspace", head=w42, role="shepherd.WorkspaceRef"),
            manager.substrate_head("store_session", binding="session", head=s7, role="shepherd.SessionState"),
        )
    )

    report = validate_relationships(
        snapshot,
        manager.stores,
        (
            RelationshipRequirement("session", "descends-from", "workspace", "not-an-oid"),
            RelationshipRequirement("session", "descends-from", "workspace", "f" * 40),
        ),
    )

    assert [issue.code for issue in report.issues] == ["relationship_malformed_head", "relationship_missing_head"]
