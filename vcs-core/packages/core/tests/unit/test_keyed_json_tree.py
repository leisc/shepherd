from __future__ import annotations

import pytest
from vcs_core.keyed_json_tree import KeyedJsonRecord, KeyedJsonTreeStore, ShardingSpec


def test_keyed_json_tree_store_maps_keys_to_stable_sharded_paths() -> None:
    store = KeyedJsonTreeStore("runs/by-ref", sharding=ShardingSpec(prefix_chars=3))

    assert store.shard_for_key("run-123") == "run"
    assert store.shard_for_key("a") == "a__"
    assert store.path_for_key("run-123") == "runs/by-ref/run/run-123.json"
    assert store.entry_path_for_key("run-123") == "data/runs/by-ref/run/run-123.json"
    assert store.prefix_path() == "data/runs/by-ref"


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../run",
        "run/one",
        ".hidden",
        "space key",
    ],
)
def test_keyed_json_tree_store_rejects_invalid_keys(key: str) -> None:
    store = KeyedJsonTreeStore("runs/by-ref")

    with pytest.raises(ValueError, match="keyed JSON key"):
        store.path_for_key(key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_root", ""),
        ("record_root", "/records"),
        ("record_root", "records/"),
        ("record_root", "records//by-ref"),
        ("record_root", "records/../by-ref"),
        ("record_root", "meta/records"),
        ("content_root", "workspace"),
    ],
)
def test_keyed_json_tree_store_rejects_invalid_roots(field: str, value: str) -> None:
    kwargs = {"record_root": "runs/by-ref", "content_root": "data"}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        KeyedJsonTreeStore(**kwargs)


def test_keyed_json_tree_store_draft_normalizes_records() -> None:
    store = KeyedJsonTreeStore("runs/by-ref")
    draft = store.draft(
        manifest={"schema": "test/keyed", "record_count": 1},
        base_head="a" * 40,
        puts=(KeyedJsonRecord("run-1", {"run_ref": "run-1"}),),
        deletes=("run-2",),
    )

    assert draft.base_head == "a" * 40
    assert draft.manifest == {"schema": "test/keyed", "record_count": 1}
    assert draft.puts[0].key == "run-1"
    assert draft.puts[0].path == "runs/by-ref/ru/run-1.json"
    assert draft.puts[0].payload == {"run_ref": "run-1"}
    assert draft.deletes == ("runs/by-ref/ru/run-2.json",)
    assert draft.content_root == "data"


def test_keyed_json_tree_store_draft_rejects_duplicate_put_keys() -> None:
    store = KeyedJsonTreeStore("runs/by-ref")

    with pytest.raises(ValueError, match="duplicate keyed JSON record"):
        store.draft(
            manifest={"schema": "test/keyed"},
            base_head=None,
            puts=(
                KeyedJsonRecord("run-1", {"side": "left"}),
                KeyedJsonRecord("run-1", {"side": "right"}),
            ),
        )


def test_keyed_json_tree_store_draft_rejects_put_delete_conflict() -> None:
    store = KeyedJsonTreeStore("runs/by-ref")

    with pytest.raises(ValueError, match="put and delete"):
        store.draft(
            manifest={"schema": "test/keyed"},
            base_head=None,
            puts=(KeyedJsonRecord("run-1", {"run_ref": "run-1"}),),
            deletes=("run-1",),
        )


def test_keyed_json_tree_store_selected_read_helpers_delegate_to_vcscore_api() -> None:
    class FakeVcsCore:
        def read_selected_binding_json_entry(
            self, binding: str, path: str, *, scope: object = None
        ) -> dict[str, object]:
            assert binding == "shepherd.runs"
            assert path == "data/runs/by-ref/ru/run-1.json"
            assert scope == "ground"
            return {"run_ref": "run-1"}

        def read_selected_binding_json_entries(
            self,
            binding: str,
            prefix: str,
            *,
            scope: object = None,
        ) -> tuple[tuple[str, dict[str, object]], ...]:
            assert binding == "shepherd.runs"
            assert prefix == "data/runs/by-ref"
            assert scope == "ground"
            return (("data/runs/by-ref/ru/run-1.json", {"run_ref": "run-1"}),)

    store = KeyedJsonTreeStore("runs/by-ref")

    assert store.read_selected(FakeVcsCore(), "shepherd.runs", "run-1", scope="ground") == {"run_ref": "run-1"}
    assert store.list_selected(FakeVcsCore(), "shepherd.runs", scope="ground") == ({"run_ref": "run-1"},)
