"""Value-level coercion helpers for command contract normalization."""

from __future__ import annotations

import json
import math
from typing import Any, Literal, TypeAlias

from vcs_core._errors import VcsCoreError
from vcs_core._typed_json import decode_typed_json, encode_typed_json

CommandValueSource: TypeAlias = Literal["cli", "native", "typed-json"]


class CommandValueError(VcsCoreError, ValueError):
    """Raised when one command value cannot be coerced or rendered faithfully."""


def parse_command_type(type_name: object) -> tuple[str, bool]:
    """Return ``(base_type, nullable)`` for a command type declaration."""
    if not isinstance(type_name, str) or not type_name:
        raise CommandValueError("invalid command type declaration; expected a non-empty string.")
    nullable = type_name.endswith("?")
    declared_base_type = type_name[:-1] if nullable else type_name
    if not declared_base_type or "?" in declared_base_type:
        raise CommandValueError(
            f"invalid command type declaration {type_name!r}; expected a non-empty base type with at most one "
            "trailing '?'."
        )
    if declared_base_type == "dict":
        raise CommandValueError(
            f"unsupported command type declaration {type_name!r}; use 'object' for opaque JSON-like payloads."
        )
    return declared_base_type, nullable


def base_type(type_name: str) -> str:
    return parse_command_type(type_name)[0]


def is_nullable(type_name: str) -> bool:
    return parse_command_type(type_name)[1]


def coerce_ingress_value(
    *,
    owner_label: str,
    context: str,
    expected_type: str,
    value: Any,
    source: CommandValueSource = "native",
) -> Any:
    """Coerce one ingress value according to a declared ingress type."""
    try:
        expected_base_type, nullable = parse_command_type(expected_type)
    except CommandValueError as exc:
        raise CommandValueError(f"{owner_label} {context} has {exc}") from exc
    expected_type = expected_base_type
    if source == "cli" and expected_type in {"object", "list"} and isinstance(value, str):
        value = _decode_cli_json(owner_label, context, expected_type, value)
    if value is None and nullable:
        return None
    if value is None:
        raise CommandValueError(f"{owner_label} {context} expected {expected_base_type}, got NoneType.")
    if expected_type == "str":
        if isinstance(value, str):
            return value
    elif expected_type == "bytes":
        if isinstance(value, bytes):
            return value
        if source == "typed-json":
            try:
                decoded_value = decode_typed_json(value)
            except (TypeError, ValueError) as exc:
                raise CommandValueError(f"{owner_label} {context} has invalid typed JSON bytes payload: {exc}") from exc
            if isinstance(decoded_value, bytes):
                return decoded_value
    elif expected_type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                try:
                    return int(value)
                except ValueError:
                    pass
    elif expected_type == "float":
        coerced_float = _coerce_float(value)
        if coerced_float is not None:
            return coerced_float
    elif expected_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
    elif expected_type == "object":
        return value
    elif expected_type == "list":
        if isinstance(value, list):
            return value
    else:
        return value

    actual_type = type(value).__name__
    raise CommandValueError(f"{owner_label} {context} expected {expected_type}, got {actual_type}.")


def coerce_command_value(
    *,
    substrate_name: str,
    context: str,
    expected_type: str,
    value: Any,
    source: CommandValueSource = "native",
) -> Any:
    """Coerce one command value according to a declared command type."""
    return coerce_ingress_value(
        owner_label=f"Substrate '{substrate_name}'",
        context=context,
        expected_type=expected_type,
        value=value,
        source=source,
    )


def coerce_ingress_repeated_value(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: str,
    expected_type: str,
    value: Any,
    source: CommandValueSource,
) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        values = (value,)
    else:
        values = tuple(value)
    return [
        coerce_ingress_value(
            owner_label=owner_label,
            context=f"{ingress_label} parameter '{param_name}'",
            expected_type=expected_type,
            value=item,
            source=source,
        )
        for item in values
    ]


def coerce_ingress_param_value(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: str,
    expected_type: str,
    repeated: bool,
    value: Any,
    source: CommandValueSource,
) -> Any:
    if repeated:
        return coerce_ingress_repeated_value(
            owner_label=owner_label,
            ingress_label=ingress_label,
            param_name=param_name,
            expected_type=expected_type,
            value=value,
            source=source,
        )
    return coerce_ingress_value(
        owner_label=owner_label,
        context=f"{ingress_label} parameter '{param_name}'",
        expected_type=expected_type,
        value=value,
        source=source,
    )


def coerce_repeated_value(
    *,
    substrate_name: str,
    command_name: str,
    param_name: str,
    expected_type: str,
    value: Any,
    source: CommandValueSource,
) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        values = (value,)
    else:
        values = tuple(value)
    return [
        coerce_command_value(
            substrate_name=substrate_name,
            context=f"command '{command_name}' parameter '{param_name}'",
            expected_type=expected_type,
            value=item,
            source=source,
        )
        for item in values
    ]


def coerce_param_value(
    *,
    substrate_name: str,
    command_name: str,
    param_name: str,
    expected_type: str,
    repeated: bool,
    value: Any,
    source: CommandValueSource,
) -> Any:
    if repeated:
        return coerce_ingress_repeated_value(
            owner_label=f"Substrate '{substrate_name}'",
            ingress_label=f"command '{command_name}'",
            param_name=param_name,
            expected_type=expected_type,
            value=value,
            source=source,
        )
    return coerce_ingress_value(
        owner_label=f"Substrate '{substrate_name}'",
        context=f"command '{command_name}' parameter '{param_name}'",
        expected_type=expected_type,
        value=value,
        source=source,
    )


def coerce_ingress_choice_value(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: str,
    expected_type: str,
    value: Any,
) -> Any:
    return coerce_ingress_value(
        owner_label=owner_label,
        context=f"{ingress_label} parameter '{param_name}' choice",
        expected_type=expected_type,
        value=value,
    )


def coerce_choice_value(
    *,
    substrate_name: str,
    command_name: str,
    param_name: str,
    expected_type: str,
    value: Any,
) -> Any:
    return coerce_ingress_choice_value(
        owner_label=f"Substrate '{substrate_name}'",
        ingress_label=f"command '{command_name}'",
        param_name=param_name,
        expected_type=expected_type,
        value=value,
    )


def validate_ingress_choice(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: str,
    value: Any,
    choices: tuple[Any, ...],
    repeated: bool,
) -> None:
    if not choices:
        return
    if repeated:
        values = value if isinstance(value, list) else [value]
        bad = [item for item in values if item not in choices]
        if not bad:
            return
        rendered = ", ".join(repr(item) for item in bad)
        raise CommandValueError(
            f"{owner_label} {ingress_label} parameter '{param_name}' got unsupported repeated value(s): {rendered}."
        )
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise CommandValueError(f"{owner_label} {ingress_label} parameter '{param_name}' must be one of: {allowed}.")


def validate_choice(
    *,
    substrate_name: str,
    command_name: str,
    param_name: str,
    value: Any,
    choices: tuple[Any, ...],
    repeated: bool,
) -> None:
    validate_ingress_choice(
        owner_label=f"Substrate '{substrate_name}'",
        ingress_label=f"command '{command_name}'",
        param_name=param_name,
        value=value,
        choices=choices,
        repeated=repeated,
    )


def typed_json_projection(value: object) -> tuple[object | None, bool]:
    try:
        return encode_typed_json(value), True
    except (TypeError, ValueError):
        return None, False


def json_schema_projection(value: object) -> tuple[object | None, bool]:
    if isinstance(value, bytes):
        return None, False
    try:
        encoded = json.dumps(value, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return None, False
    if decoded != value:
        return None, False
    return decoded, True


def cli_literal(value: object, *, expected_type: str | None = None) -> str | None:
    if isinstance(value, bytes):
        return None
    expected_base_type = base_type(expected_type) if expected_type is not None else None
    if expected_base_type in {"dict", "object", "list"}:
        projected, renderable = json_schema_projection(value)
        if not renderable:
            return None
        return json.dumps(projected, allow_nan=False, sort_keys=True)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return repr(value)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        try:
            candidate = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def _decode_cli_json(owner_label: str, context: str, expected_type: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value, parse_constant=_reject_non_finite_json_constant)
    except ValueError as exc:
        raise CommandValueError(f"{owner_label} {context} expected valid JSON for {expected_type}.") from exc


def _reject_non_finite_json_constant(value: str) -> object:
    raise ValueError(f"unsupported non-finite JSON constant {value!r}")
