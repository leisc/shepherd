"""Shared JSON Schema utilities for task and step output schemas (W2 port).

Ported verbatim from the legacy ``shepherd_core._shared.schema`` (authoring
re-pin W2 — the dialect carries no legacy imports); the single source of truth
for type-to-schema conversion via Pydantic's TypeAdapter.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import PydanticSchemaGenerationError, TypeAdapter


class SchemaGenerationError(Exception):
    """Error generating JSON schema for task/step outputs (legacy observable)."""

    def __init__(self, message: str, conflicting_key: str | None = None, field_name: str | None = None):
        self.conflicting_key = conflicting_key
        self.field_name = field_name
        super().__init__(message)


# =============================================================================
# Constants
# =============================================================================

# Key used for single-value outputs in step schemas.
# Used by _return_type_to_output_schema() and _parse_single_output().
SINGLE_OUTPUT_KEY = "result"


# =============================================================================
# JSON Schema Generation
# =============================================================================


def type_to_json_schema(type_annotation: Any) -> dict[str, Any]:
    """Convert Python type annotation to JSON Schema using Pydantic's TypeAdapter.

    This is the single source of truth for type-to-schema conversion.
    Handles all types Pydantic supports: primitives, generics, unions,
    Literal, Enum, datetime, UUID, Pydantic models, etc.

    Note: TypeAdapter.json_schema() returns a new dict each call, so callers
    may safely mutate the result (e.g., popping $defs for hoisting).

    Args:
        type_annotation: Any Python type annotation

    Returns:
        JSON Schema dict representing the type. Falls back to {"type": "string"}
        with a warning for unsupported types.
    """
    # Handle None/NoneType explicitly (TypeAdapter doesn't like bare None)
    if type_annotation is None or type_annotation is type(None):
        return {"type": "null"}

    # Handle Any - no constraints (empty schema allows anything)
    if type_annotation is Any:
        return {}

    try:
        ta = TypeAdapter(type_annotation)
        schema = ta.json_schema()

        # Strip Pydantic-added title (noise for LLM providers)
        schema.pop("title", None)

        return schema
    except (PydanticSchemaGenerationError, TypeError) as e:
        # PydanticSchemaGenerationError: Pydantic can't generate schema
        # TypeError: Exotic types that fail during introspection
        warnings.warn(
            f"Cannot generate JSON schema for {type_annotation!r}: {e}. "
            f"Falling back to string type. Consider using a supported type.",
            stacklevel=2,
        )
        return {"type": "string"}
    except Exception as e:  # noqa: BLE001
        # Unexpected failure — still fall back, but include exception class for debugging
        warnings.warn(
            f"Unexpected error generating JSON schema for {type_annotation!r} "
            f"({type(e).__name__}: {e}). Falling back to string type.",
            stacklevel=2,
        )
        return {"type": "string"}


# Deprecated alias for backward compatibility
# (internal function, but kept for safety during transition)
python_type_to_json_schema = type_to_json_schema


# =============================================================================
# Schema Helper Functions
# =============================================================================


def merge_schema_defs(
    schema: dict[str, Any],
    all_defs: dict[str, Any],
    *,
    field_name: str | None = None,
    context: str = "output fields",
) -> None:
    """Extract and merge $defs from schema into all_defs.

    Mutates both arguments:
    - schema: $defs key removed if present
    - all_defs: merged with definitions from this schema's $defs

    Args:
        schema: Schema dict that may contain $defs
        all_defs: Accumulator dict for all $defs
        field_name: Optional field name for error attribution
        context: Description of context for error message (e.g., "output fields", "tuple return type")

    Raises:
        SchemaGenerationError: If same $def name has different structure
    """
    if "$defs" not in schema:
        return

    for key, value in schema.pop("$defs").items():
        if key in all_defs and all_defs[key] != value:
            raise SchemaGenerationError(
                f"Conflicting $defs for '{key}' in {context}. "
                f"Two types define nested classes with the same name "
                f"but different structures. Consider renaming one of the "
                f"nested classes to be unique.",
                conflicting_key=key,
                field_name=field_name,
            )
        all_defs[key] = value


def wrap_as_json_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
    all_defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap properties dict into JSON schema format for LLM providers.

    Args:
        properties: Dict of property name -> schema
        required: List of required property names (defaults to all properties)
        all_defs: Optional $defs to include at top level

    Returns:
        {"type": "json_schema", "schema": {"type": "object", ...}}
    """
    result_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties.keys()),
        "additionalProperties": False,
    }
    if all_defs:
        result_schema["$defs"] = all_defs
    return {"type": "json_schema", "schema": result_schema}
