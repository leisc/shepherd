# under-test: vcs_core._projection_store
"""Store scope-registry projection tests."""

from __future__ import annotations

import json

from vcs_core._projection_store import (
    SCOPE_REGISTRY_CURRENT_REF,
    ScopeRegistryEntry,
    load_scope_registry_snapshot,
)
from vcs_core.git_store import build_tree, create_signature
from vcs_core.store import Store


def _live_entry(task, *, parent_ref: str = Store.GROUND_REF) -> ScopeRegistryEntry:
    assert task.world_id is not None
    return ScopeRegistryEntry(
        name=task.name,
        ref=task.ref,
        instance_id=task.instance_id,
        creation_oid=task.creation_oid,
        parent_ref=parent_ref,
        world_id=task.world_id,
        isolation_mode="shared",
        status="live",
    )


def _write_invalid_scope_registry_projection(store: Store, manifest: dict[str, object]) -> None:
    tree_oid = build_tree(
        store._repo,
        None,
        [("meta/projection.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))],
    )
    sig = create_signature("projection")
    commit_oid = store._repo.create_commit(
        None,
        sig,
        sig,
        "projection:scope-registry",
        tree_oid,
        [],
    )
    if SCOPE_REGISTRY_CURRENT_REF in store._repo.references:
        store._repo.references[SCOPE_REGISTRY_CURRENT_REF].set_target(commit_oid)
    else:
        store._repo.references.create(SCOPE_REGISTRY_CURRENT_REF, commit_oid)


def test_scope_registry_publish_and_load_round_trip(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task")
    base_snapshot = store.require_scope_registry_projection()

    published = store.publish_scope_registry_projection(
        entries=(_live_entry(task),), expected_head_oid=base_snapshot.head_oid
    )

    assert published is True
    snapshot = store.load_scope_registry_projection()
    assert snapshot is not None
    assert snapshot.entries == (_live_entry(task),)
    assert snapshot.entries_by_name["task"] == _live_entry(task)
    assert snapshot.entries_by_ref[task.ref] == _live_entry(task)
    assert load_scope_registry_snapshot(store._repo) == snapshot


def test_create_root_commit_seeds_empty_scope_registry_projection(tmp_repo) -> None:
    store = Store(str(tmp_repo))
    store.create_root_commit()

    snapshot = store.require_scope_registry_projection()

    assert snapshot.entries == ()


def test_scope_registry_reports_live_ref_missing(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task")
    base_snapshot = store.require_scope_registry_projection()
    assert store.publish_scope_registry_projection(
        entries=(_live_entry(task),), expected_head_oid=base_snapshot.head_oid
    )

    store.discard(task)

    mismatches = store.scope_registry_projection_mismatches()

    assert [mismatch.kind for mismatch in mismatches] == ["registry_live_ref_missing"]
    assert mismatches[0].scope_name == "task"


def test_scope_registry_reports_ref_exists_registry_non_live(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task")
    entry = _live_entry(task)
    base_snapshot = store.require_scope_registry_projection()
    assert store.publish_scope_registry_projection(
        entries=(entry.__class__(**{**entry.__dict__, "status": "discarded"}),),
        expected_head_oid=base_snapshot.head_oid,
    )

    mismatches = store.scope_registry_projection_mismatches()

    assert [mismatch.kind for mismatch in mismatches] == ["ref_exists_registry_non_live"]
    assert mismatches[0].ref == task.ref


def test_scope_registry_reports_parentage_disagreement(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task")
    bad_entry = _live_entry(task, parent_ref="refs/vcscore/scopes/missing-parent")
    base_snapshot = store.require_scope_registry_projection()
    assert store.publish_scope_registry_projection(entries=(bad_entry,), expected_head_oid=base_snapshot.head_oid)

    mismatches = store.scope_registry_projection_mismatches()

    assert [mismatch.kind for mismatch in mismatches] == ["parentage_disagrees"]
    assert mismatches[0].scope_name == "task"


def test_scope_registry_reports_unreadable_projection(store: Store) -> None:
    _write_invalid_scope_registry_projection(
        store,
        manifest={
            "family": "scope-registry",
            "version": 1,
            "completeness": "complete",
            "source": "not-a-list",
            "source_digest": "invalid",
        },
    )

    mismatches = store.scope_registry_projection_mismatches()

    assert [mismatch.kind for mismatch in mismatches] == ["registry_format_unreadable"]


def test_scope_registry_publish_rejects_stale_expected_head(store: Store) -> None:
    task = store.fork(Store.GROUND_REF, "task")
    base_snapshot = store.require_scope_registry_projection()

    assert store.publish_scope_registry_projection(
        entries=(_live_entry(task),), expected_head_oid=base_snapshot.head_oid
    )
    assert (
        store.publish_scope_registry_projection(
            entries=(_live_entry(task),),
            expected_head_oid=base_snapshot.head_oid,
        )
        is False
    )
