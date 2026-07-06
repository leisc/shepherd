"""Reusable helpers for addressable JSON records in substrate revision trees."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from vcs_core._substrate_driver import KeyedJsonPut, KeyedJsonTreeDraft

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True)
class ShardingSpec:
    """Prefix-shard configuration for keyed JSON records."""

    prefix_chars: int = 2

    def __post_init__(self) -> None:
        if self.prefix_chars < 1:
            raise ValueError("prefix_chars must be positive")


@dataclass(frozen=True)
class KeyedJsonRecord:
    """One logical keyed JSON record."""

    key: str
    payload: dict[str, object]


@dataclass(frozen=True)
class KeyedJsonTreeStore:
    """Pathing and draft construction for keyed JSON-tree revisions."""

    record_root: str
    sharding: ShardingSpec = ShardingSpec()
    content_root: str = "data"

    def __post_init__(self) -> None:
        _validate_root(self.record_root, field_name="record_root")
        _validate_root(self.content_root, field_name="content_root")

    def shard_for_key(self, key: str) -> str:
        normalized = _validate_key(key)
        return normalized[: self.sharding.prefix_chars].ljust(self.sharding.prefix_chars, "_")

    def path_for_key(self, key: str) -> str:
        normalized = _validate_key(key)
        shard = self.shard_for_key(normalized)
        return f"{self.record_root}/{shard}/{normalized}.json"

    def entry_path_for_key(self, key: str) -> str:
        return f"{self.content_root}/{self.path_for_key(key)}"

    def prefix_path(self) -> str:
        return f"{self.content_root}/{self.record_root}"

    def draft(
        self,
        *,
        manifest: dict[str, object],
        base_head: str | None,
        puts: tuple[KeyedJsonRecord, ...],
        deletes: tuple[str, ...] = (),
    ) -> KeyedJsonTreeDraft:
        seen_keys: set[str] = set()
        keyed_puts: list[KeyedJsonPut] = []
        for record in puts:
            key = _validate_key(record.key)
            if key in seen_keys:
                raise ValueError(f"duplicate keyed JSON record: {key!r}")
            if not isinstance(record.payload, dict):
                raise TypeError("keyed JSON record payload must be an object")
            keyed_puts.append(KeyedJsonPut(key=key, path=self.path_for_key(key), payload=dict(record.payload)))
            seen_keys.add(key)
        delete_paths = tuple(self.path_for_key(_validate_key(key)) for key in deletes)
        if set(delete_paths) & {put.path for put in keyed_puts}:
            raise ValueError("keyed JSON draft cannot put and delete the same key")
        return KeyedJsonTreeDraft(
            manifest=dict(manifest),
            base_head=base_head,
            puts=tuple(keyed_puts),
            deletes=delete_paths,
            content_root=self.content_root,
        )

    def read_selected(self, mg: Any, binding: str, key: str, *, scope: Any = None) -> dict[str, object] | None:
        reader = getattr(mg, "read_selected_binding_json_entry", None)
        if not callable(reader):
            return None
        return cast("dict[str, object] | None", reader(binding, self.entry_path_for_key(key), scope=scope))

    def list_selected(self, mg: Any, binding: str, *, scope: Any = None) -> tuple[dict[str, object], ...]:
        reader = getattr(mg, "read_selected_binding_json_entries", None)
        if not callable(reader):
            return ()
        return tuple(value for _path, value in reader(binding, self.prefix_path(), scope=scope))


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError("keyed JSON key must be a non-empty string")
    if _KEY_RE.fullmatch(key) is None:
        raise ValueError(f"keyed JSON key contains unsupported characters: {key!r}")
    return key


def _validate_root(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError(f"{field_name} must be a relative path")
    if any(part in {"", ".", "..", "meta", "workspace"} for part in value.split("/")):
        raise ValueError(f"{field_name} contains a reserved or invalid path segment")
