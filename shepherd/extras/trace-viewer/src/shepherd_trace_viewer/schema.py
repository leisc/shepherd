"""JSON-Schema artifact for ``shepherd.trace-view.v3``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shepherd_trace_viewer.model import EDGE_KINDS, NODE_ROLES, SCHEMA_VERSION, SOURCE_KINDS

SCHEMA_PATH = Path(__file__).with_name("schema.v3.json")


def build_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for the current trace-viewer contract."""
    nullable_str = {"type": ["string", "null"]}
    nullable_num = {"type": ["number", "null"]}
    nullable_int = {"type": ["integer", "null"]}
    obj = {"type": "object"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "title": "Shepherd trace-view v3",
        "type": "object",
        "required": ["schema_version", "source", "run", "lanes", "nodes", "edges", "resources"],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "source": {
                "type": "object",
                "required": [
                    "trace_runtime",
                    "trace_owner_id",
                    "frontier_id",
                    "source_kind",
                    "visibility_profile",
                    "mode_filter",
                ],
                "additionalProperties": False,
                "properties": {
                    "trace_runtime": {"type": "string"},
                    "trace_owner_id": {"type": "string"},
                    "frontier_id": {"type": "string"},
                    "source_kind": {"enum": list(SOURCE_KINDS)},
                    "visibility_profile": nullable_str,
                    "mode_filter": nullable_str,
                    "identity_domain": nullable_str,
                    "schema": nullable_str,
                    "kind": nullable_str,
                },
            },
            "run": {
                "type": "object",
                "required": ["id", "terminal_status", "transition", "summary", "detail"],
                "additionalProperties": False,
                "properties": {
                    "id": nullable_str,
                    "terminal_status": nullable_str,
                    "transition": nullable_str,
                    "summary": obj,
                    "detail": obj,
                },
            },
            "lanes": {"type": "array", "items": _lane_schema()},
            "nodes": {"type": "array", "items": _node_schema(nullable_str, nullable_num, nullable_int, obj)},
            "edges": {"type": "array", "items": _edge_schema()},
            "resources": {"type": "array", "items": _resource_schema(obj)},
        },
    }


def _lane_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["id", "label", "node_ids"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "node_ids": {"type": "array", "items": {"type": "string"}},
        },
    }


def _node_schema(nullable_str: dict, nullable_num: dict, nullable_int: dict, obj: dict) -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "id",
            "kind",
            "family",
            "role",
            "lane_ids",
            "sequence",
            "timestamp",
            "label",
            "identity_domain",
            "record_digest",
            "body",
            "payload",
        ],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "kind": {"type": "string"},
            "family": {"type": "string"},
            "role": {"enum": list(NODE_ROLES)},
            "lane_ids": {"type": "array", "items": {"type": "string"}},
            "sequence": nullable_int,
            "timestamp": nullable_num,
            "label": {"type": "string"},
            "identity_domain": nullable_str,
            "record_digest": nullable_str,
            "body": obj,
            "payload": obj,
        },
    }


def _edge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["id", "kind", "source", "target", "label"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "kind": {"enum": list(EDGE_KINDS)},
            "source": {"type": "string"},
            "target": {"type": "string"},
            "label": {"type": "string"},
        },
    }


def _resource_schema(obj: dict) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["id", "kind", "label", "detail"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "kind": {"type": "string"},
            "label": {"type": "string"},
            "detail": obj,
        },
    }


def write_schema() -> None:
    """Write the committed JSON Schema artifact."""
    SCHEMA_PATH.write_text(json.dumps(build_json_schema(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_schema()
    import sys

    sys.stdout.write(f"wrote {SCHEMA_PATH}\n")
