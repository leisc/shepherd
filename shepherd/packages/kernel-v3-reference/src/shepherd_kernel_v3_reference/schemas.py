from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Schema(Protocol):
    """Runtime schema validator.

    Returns `None` on success, or a human-readable error message on failure.
    """

    def validate(self, value: Any) -> str | None: ...


class ValidationError(ValueError):
    """Raised when a value does not match the required schema."""


@dataclass(frozen=True)
class AnySchema:
    def validate(self, value: Any) -> str | None:
        return None


@dataclass(frozen=True)
class TypeSchema:
    expected: type

    def validate(self, value: Any) -> str | None:
        if not isinstance(value, self.expected):
            return f"expected {self.expected.__name__}, got {type(value).__name__}"
        return None


@dataclass(frozen=True)
class TaggedRecordSchema:
    """Tagged record: a mapping with a fixed `kind` field.

    The freed `LiteralSchema` name is bound to the `-lite` profile's
    single-int-literal schema below.
    """

    tag: str

    def validate(self, value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return f"expected mapping tagged {self.tag!r}, got {type(value).__name__}"
        actual = value.get("kind")
        if actual != self.tag:
            return f"expected kind={self.tag!r}, got kind={actual!r}"
        return None


@dataclass(frozen=True)
class IntSchema:
    """Integer schema admitted by `core-reference-v0-lite`.

    Maps to Lean `Nat` for differential testing. Rejects booleans (Python's
    `bool` is a subclass of `int`; the kernel treats them as a separate domain).
    """

    def validate(self, value: Any) -> str | None:
        if isinstance(value, bool):
            return f"expected int, got bool ({value!r})"
        if not isinstance(value, int):
            return f"expected int, got {type(value).__name__}"
        return None


@dataclass(frozen=True)
class NullSchema:
    """Null schema admitted by `core-reference-v0-lite` (the only non-int value)."""

    def validate(self, value: Any) -> str | None:
        if value is not None:
            return f"expected null, got {type(value).__name__}"
        return None


@dataclass(frozen=True)
class LiteralSchema:
    """Single-int-literal schema admitted by `core-reference-v0-lite`.

    Accepts exactly the integer `value`; rejects anything else (including
    booleans, per `IntSchema`'s discipline).
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError(f"LiteralSchema(value=...) requires int, got {type(self.value).__name__}")

    def validate(self, value: Any) -> str | None:
        if isinstance(value, bool):
            return f"expected literal {self.value!r}, got bool ({value!r})"
        if not isinstance(value, int) or value != self.value:
            return f"expected literal {self.value!r}, got {value!r}"
        return None


@dataclass(frozen=True)
class RecordSchema:
    """Mapping with required field schemas."""

    fields: Mapping[str, Schema]

    def validate(self, value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return f"expected mapping, got {type(value).__name__}"
        for name, sub in self.fields.items():
            if name not in value:
                return f"missing field {name!r}"
            err = sub.validate(value[name])
            if err is not None:
                return f"in field {name!r}: {err}"
        return None


def check(schema: Schema, value: Any, *, context: str = "") -> None:
    """Raise `ValidationError` if `value` does not match `schema`."""
    err = schema.validate(value)
    if err is not None:
        prefix = f"{context}: " if context else ""
        raise ValidationError(f"{prefix}{err}")


def schema_fingerprint(schema: Schema) -> object:
    """Return a deterministic semantic fingerprint for a schema.

    The auditable kernel path cannot use Python object identity or `repr(...)`
    for schemas. Built-in schemas have structural fingerprints. Custom schemas
    must provide a zero-argument `fingerprint()` method returning a
    JSON-compatible value.
    """

    if isinstance(schema, AnySchema):
        return {"schema": "any"}
    if isinstance(schema, TypeSchema):
        return {
            "schema": "type",
            "module": schema.expected.__module__,
            "qualname": schema.expected.__qualname__,
        }
    if isinstance(schema, TaggedRecordSchema):
        return {"schema": "tagged-record", "tag": schema.tag}
    if isinstance(schema, IntSchema):
        return {"schema": "int"}
    if isinstance(schema, NullSchema):
        return {"schema": "null"}
    if isinstance(schema, LiteralSchema):
        return {"schema": "literal", "value": schema.value}
    if isinstance(schema, RecordSchema):
        return {
            "schema": "record",
            "fields": tuple((name, schema_fingerprint(subschema)) for name, subschema in sorted(schema.fields.items())),
        }

    custom = getattr(schema, "fingerprint", None)
    if callable(custom):
        return {"schema": "custom", "fingerprint": custom()}

    raise TypeError(
        f"schema of type {type(schema).__name__} has no stable fingerprint; "
        "use a built-in schema or provide fingerprint()"
    )
