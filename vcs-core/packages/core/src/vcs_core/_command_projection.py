"""Backend-specific projections over compiled command contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from vcs_core._command_contract import (
    CommandContract,
    CommandContractError,
    CommandParamContract,
    RequiredOneOfContract,
    compile_command_contract,
)
from vcs_core._command_values import cli_literal, json_schema_projection, typed_json_projection
from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vcs_core.spi import DriverSchema


ProjectionBackend = Literal["cli", "tool"]
RESERVED_CLI_OPTION_NAMES = frozenset({"scope", "json", "help"})
RESERVED_CLI_CLICK_NAMES = frozenset({"scope_name", "as_json"})
_CLI_PARAM_NAME_RE = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*\Z")


class CommandProjectionError(VcsCoreError, ValueError):
    """Raised when a command cannot be lowered to a generated projection."""


ProjectionKind = Literal[
    "exact-visible-xor",
    "restricted-visible-xor",
    "default-satisfied-xor",
    "unprojectable-xor",
]


@dataclass(frozen=True)
class ProjectionBackendCapabilities:
    backend: ProjectionBackend
    supported_param_types: frozenset[str]
    supports_repeated_bool: bool
    enforces_cli_identity: bool


BACKEND_CAPABILITIES: dict[ProjectionBackend, ProjectionBackendCapabilities] = {
    "cli": ProjectionBackendCapabilities(
        backend="cli",
        supported_param_types=frozenset({"str", "int", "float", "bool", "bytes", "object", "list"}),
        supports_repeated_bool=False,
        enforces_cli_identity=True,
    ),
    "tool": ProjectionBackendCapabilities(
        backend="tool",
        supported_param_types=frozenset({"str", "int", "float", "bool", "bytes", "object", "list"}),
        supports_repeated_bool=True,
        enforces_cli_identity=False,
    ),
}


@dataclass(frozen=True)
class ProjectedValue:
    dispatch_value: object
    typed_json_value: object | None
    typed_json_transportable: bool
    json_schema_value: object | None
    json_schema_renderable: bool
    cli_literal: str | None


@dataclass(frozen=True)
class ProjectedParamFidelity:
    preserves_type: bool
    preserves_nullability: bool
    preserves_repeated_cardinality: bool
    renders_default: bool
    renders_choices: bool
    spelling: Literal["native", "cli-option"]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectedParamContract:
    param: CommandParamContract
    option_name: str
    rendered_default: ProjectedValue | None
    rendered_choices: tuple[ProjectedValue, ...]
    fidelity: ProjectedParamFidelity


@dataclass(frozen=True)
class ProjectedOneOfContract:
    semantic_group: RequiredOneOfContract
    visible_members: tuple[str, ...]
    hidden_members: tuple[str, ...]
    projection_kind: ProjectionKind


@dataclass(frozen=True)
class ProjectedCommandContract:
    contract: CommandContract
    backend: ProjectionBackend
    params: tuple[ProjectedParamContract, ...]
    one_of: tuple[ProjectedOneOfContract, ...]
    projectable: bool
    hidden_reasons: Mapping[str, tuple[str, ...]]
    command_reasons: tuple[str, ...]


def project_command_contract(
    contract: CommandContract,
    *,
    backend: ProjectionBackend,
) -> ProjectedCommandContract:
    """Project a semantic command contract to one backend surface."""
    capabilities = _backend_capabilities(backend)
    command_reasons: list[str] = []
    hidden_reasons: dict[str, tuple[str, ...]] = {}
    visible_params: list[ProjectedParamContract] = []

    command_surface_enabled = True
    if not contract.projectable:
        command_surface_enabled = False
        command_reasons.append("command-not-projectable")
    if contract.selectable:
        command_surface_enabled = False
        command_reasons.append("selectable-driver")

    for param in contract.params.values():
        reasons = _hidden_param_reasons(
            param,
            capabilities=capabilities,
            command_surface_enabled=command_surface_enabled,
        )
        if reasons:
            hidden_reasons[param.name] = reasons
            continue
        rendered_default = (
            _project_value(param.normalized_default, expected_type=param.base_type) if param.has_default else None
        )
        rendered_choices = tuple(_project_value(choice, expected_type=param.base_type) for choice in param.choices)
        visible_params.append(
            ProjectedParamContract(
                param=param,
                option_name=_cli_option_name(param.name),
                rendered_default=rendered_default,
                rendered_choices=rendered_choices,
                fidelity=_project_param_fidelity(
                    param,
                    capabilities=capabilities,
                    rendered_default=rendered_default,
                    rendered_choices=rendered_choices,
                ),
            )
        )

    if capabilities.enforces_cli_identity:
        _validate_cli_projection_identity(visible_params)

    visible_names = {param.param.name for param in visible_params}
    one_of: list[ProjectedOneOfContract] = []
    for group in contract.required_one_of:
        visible_members = tuple(member for member in group.members if member in visible_names)
        hidden_members = tuple(member for member in group.members if member not in visible_names)
        projection_kind: ProjectionKind
        if group.defaulted_member is not None:
            projection_kind = "default-satisfied-xor"
        elif visible_members and not hidden_members:
            projection_kind = "exact-visible-xor"
        elif visible_members:
            projection_kind = "restricted-visible-xor"
        else:
            projection_kind = "unprojectable-xor"
        one_of.append(
            ProjectedOneOfContract(
                semantic_group=group,
                visible_members=visible_members,
                hidden_members=hidden_members,
                projection_kind=projection_kind,
            )
        )
        if command_surface_enabled and projection_kind == "unprojectable-xor":
            command_reasons.append(f"required-one-of-hidden:{','.join(group.members)}")

    for param in contract.params.values():
        if command_surface_enabled and param.user_required and param.name in hidden_reasons:
            command_reasons.append(f"required-param-hidden:{param.name}")

    return ProjectedCommandContract(
        contract=contract,
        backend=backend,
        params=tuple(visible_params),
        one_of=tuple(one_of),
        projectable=command_surface_enabled and not command_reasons,
        hidden_reasons=MappingProxyType(hidden_reasons),
        command_reasons=tuple(command_reasons),
    )


def projectable_command_names(schema: DriverSchema) -> tuple[str, ...]:
    names: list[str] = []
    for command_name in sorted(schema.commands):
        try:
            contract = compile_command_contract(schema, command_name)
            projection = project_command_contract(contract, backend="cli")
        except (CommandContractError, CommandProjectionError):
            continue
        if projection.projectable:
            names.append(command_name)
    return tuple(names)


def project_cli_command(binding_name: str, schema: DriverSchema, command_name: str) -> ProjectedCommandContract:
    """Project one command to the generated CLI backend."""
    if command_name not in schema.commands:
        available = ", ".join(sorted(schema.commands)) or "(none)"
        raise CommandProjectionError(f"unknown {binding_name} command '{command_name}'. Available: {available}")
    try:
        contract = compile_command_contract(schema, command_name, binding_name=binding_name)
        projected = project_command_contract(contract, backend="cli")
    except CommandContractError as exc:
        raise CommandProjectionError(str(exc)) from exc
    if not projected.projectable:
        reasons = ", ".join(projected.command_reasons) or "not projectable"
        raise CommandProjectionError(reasons)
    return projected


def project_tool_command(binding_name: str, schema: DriverSchema, command_name: str) -> ProjectedCommandContract:
    """Project one command to the Anthropic tool-schema backend."""
    if command_name not in schema.commands:
        available = ", ".join(sorted(schema.commands)) or "(none)"
        raise CommandProjectionError(f"unknown {binding_name} command '{command_name}'. Available: {available}")
    try:
        contract = compile_command_contract(schema, command_name, binding_name=binding_name)
        projected = project_command_contract(contract, backend="tool")
    except CommandContractError as exc:
        raise CommandProjectionError(str(exc)) from exc
    if not projected.projectable:
        reasons = ", ".join(projected.command_reasons) or "not projectable"
        raise CommandProjectionError(reasons)
    return projected


def anthropic_tool_schema_for_command(binding_name: str, schema: DriverSchema, command_name: str) -> dict[str, object]:
    """Emit an Anthropic-compatible tool-use schema without going through CLI projection."""
    return _anthropic_tool_schema_from_projected(project_tool_command(binding_name, schema, command_name))


def _anthropic_tool_schema_from_projected(projected: ProjectedCommandContract) -> dict[str, object]:
    if projected.backend != "tool":
        raise CommandProjectionError("Anthropic tool schema rendering requires a tool command projection.")
    if not projected.projectable:
        reasons = ", ".join(projected.command_reasons) or "not projectable"
        raise CommandProjectionError(reasons)

    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {param.param.name: _tool_json_schema_for_param(param) for param in projected.params},
        "additionalProperties": False,
    }

    required = _tool_required_params(projected)
    if required:
        input_schema["required"] = list(required)
    constraints = _tool_group_constraints(projected)
    if len(constraints) == 1 and "oneOf" in constraints[0]:
        input_schema["oneOf"] = constraints[0]["oneOf"]
    elif constraints:
        input_schema["allOf"] = constraints
    binding_name = projected.contract.binding_name or projected.contract.driver_id
    return {
        "name": _tool_name(binding_name, projected.contract.command_name),
        "description": projected.contract.description,
        "input_schema": input_schema,
    }


def validate_projected_surface_input(projection: ProjectedCommandContract, raw_params: Mapping[str, object]) -> None:
    """Validate only constraints the projected CLI surface can faithfully enforce."""
    if projection.backend != "cli":
        raise CommandProjectionError("Projected surface input validation requires a CLI command projection.")

    param_by_name = {param.param.name: param for param in projection.params}
    for param in projection.params:
        if param.param.user_required and param.param.name not in raw_params:
            raise CommandProjectionError(f"Missing option '{_option_label(param.param.name)}'.")

    for group in projection.one_of:
        supplied = [param_name for param_name in group.visible_members if param_name in raw_params]
        if group.projection_kind == "default-satisfied-xor":
            if len(supplied) > 1:
                _raise_visible_xor_error(projection, group.visible_members)
            continue
        if group.projection_kind == "exact-visible-xor":
            if len(supplied) == 1:
                continue
            _raise_visible_xor_error(projection, group.visible_members)
        if group.projection_kind == "restricted-visible-xor":
            if len(supplied) == 1:
                continue
            if len(group.visible_members) == 1 and not supplied:
                raise CommandProjectionError(f"Missing option '{_option_label(group.visible_members[0])}'.")
            _raise_visible_xor_error(projection, group.visible_members)
        if group.projection_kind == "unprojectable-xor" and group.visible_members:
            _raise_visible_xor_error(projection, group.visible_members)

    unknown = sorted(set(raw_params) - set(param_by_name))
    if unknown:
        binding_name = projection.contract.binding_name or projection.contract.driver_id
        raise CommandProjectionError(
            f"Binding '{binding_name}' command '{projection.contract.command_name}' got unknown option(s): "
            f"{', '.join(_option_label(name) for name in unknown)}."
        )


def _choice_labels(projected_choices: tuple[ProjectedValue, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for choice in projected_choices:
        value = choice.dispatch_value
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return ()
        labels.append(str(value))
    return tuple(labels)


def _project_param_fidelity(
    param: CommandParamContract,
    *,
    capabilities: ProjectionBackendCapabilities,
    rendered_default: ProjectedValue | None,
    rendered_choices: tuple[ProjectedValue, ...],
) -> ProjectedParamFidelity:
    notes: list[str] = []
    preserves_nullability = True
    if capabilities.backend == "cli" and param.nullable and param.base_type not in {"object", "list"}:
        preserves_nullability = False
        notes.append("explicit-null-not-representable")

    renders_default = _renders_default(capabilities.backend, rendered_default)
    if param.has_default and not renders_default:
        notes.append("default-not-rendered")

    renders_choices = _renders_choices(capabilities.backend, param, rendered_choices)
    if param.choices and not renders_choices:
        notes.append("choices-not-rendered")

    return ProjectedParamFidelity(
        preserves_type=True,
        preserves_nullability=preserves_nullability,
        preserves_repeated_cardinality=True,
        renders_default=renders_default,
        renders_choices=renders_choices,
        spelling="cli-option" if capabilities.backend == "cli" else "native",
        notes=tuple(notes),
    )


def _renders_default(backend: ProjectionBackend, rendered_default: ProjectedValue | None) -> bool:
    if rendered_default is None:
        return False
    if backend == "cli":
        return rendered_default.cli_literal is not None
    return rendered_default.json_schema_renderable


def _renders_choices(
    backend: ProjectionBackend,
    param: CommandParamContract,
    rendered_choices: tuple[ProjectedValue, ...],
) -> bool:
    if not rendered_choices:
        return False
    if backend == "cli":
        return param.base_type == "str" and bool(_choice_labels(rendered_choices))
    return all(choice.json_schema_renderable for choice in rendered_choices)


def _hidden_param_reasons(
    param: CommandParamContract,
    *,
    capabilities: ProjectionBackendCapabilities,
    command_surface_enabled: bool,
) -> tuple[str, ...]:
    if not command_surface_enabled:
        return ("command-not-projectable",)
    if not param.projectable:
        return ("param-not-projectable",)
    if param.base_type not in capabilities.supported_param_types:
        raise CommandProjectionError(
            f"Driver command parameter '{param.name}' is projectable but has unsupported type '{param.type}'."
        )
    if capabilities.enforces_cli_identity:
        _validate_cli_param_name(param.name)
        if param.repeated and param.base_type == "bool" and not capabilities.supports_repeated_bool:
            raise CommandProjectionError(
                f"Driver command parameter '{param.name}' is projectable but repeated bool is not faithfully "
                "representable by the generated CLI."
            )
        option_name = _cli_option_name(param.name)
        if option_name in RESERVED_CLI_OPTION_NAMES:
            raise CommandProjectionError(
                f"Driver command parameter '{param.name}' collides with reserved CLI option '--{option_name}'."
            )
        click_name = _cli_click_param_name(param.name)
        if click_name in RESERVED_CLI_CLICK_NAMES:
            raise CommandProjectionError(
                f"Driver command parameter '{param.name}' collides with reserved CLI parameter name '{click_name}'."
            )
    return ()


def _validate_cli_projection_identity(params: list[ProjectedParamContract]) -> None:
    _reject_duplicate_cli_options(params)
    _reject_duplicate_projection_values(
        params,
        label="projected CLI parameter",
        render=lambda value: value,
        key=lambda param: _cli_click_param_name(param.param.name),
    )
    _reject_reserved_negative_option_namespace(params)


def _validate_cli_param_name(param_name: str) -> None:
    if _CLI_PARAM_NAME_RE.fullmatch(param_name) is None:
        raise CommandProjectionError(
            f"Driver command parameter '{param_name}' is projectable but cannot be represented as a generated CLI "
            "option; use ASCII alphanumeric words separated by '_' or '-'."
        )


def _reject_duplicate_cli_options(params: list[ProjectedParamContract]) -> None:
    seen: dict[str, list[str]] = {}
    for param in params:
        for option_name in _cli_emitted_option_names(param):
            seen.setdefault(option_name, []).append(param.param.name)
    for option_name, names in seen.items():
        if len(names) > 1:
            raise CommandProjectionError(
                f"projected CLI option '--{option_name}' collides for parameters: {', '.join(names)}."
            )


def _reject_duplicate_projection_values(
    params: list[ProjectedParamContract],
    *,
    label: str,
    render: Callable[[str], str],
    key: Callable[[ProjectedParamContract], str],
) -> None:
    seen: dict[str, list[str]] = {}
    for param in params:
        seen.setdefault(key(param), []).append(param.param.name)
    for value, names in seen.items():
        if len(names) > 1:
            raise CommandProjectionError(f"{label} '{render(value)}' collides for parameters: {', '.join(names)}.")


def _reject_reserved_negative_option_namespace(params: list[ProjectedParamContract]) -> None:
    for param in params:
        if param.option_name.startswith("no-"):
            raise CommandProjectionError(
                f"projected CLI option '--{param.option_name}' for parameter '{param.param.name}' uses the reserved "
                "'--no-*' negative-option namespace."
            )


def _cli_emitted_option_names(param: ProjectedParamContract) -> tuple[str, ...]:
    option_name = param.option_name
    if param.param.base_type == "bool":
        return (option_name, f"no-{option_name}")
    return (option_name,)


def _project_value(value: object, *, expected_type: str) -> ProjectedValue:
    typed_json_value, typed_json_transportable = typed_json_projection(value)
    json_schema_value, json_schema_renderable = json_schema_projection(value)
    return ProjectedValue(
        dispatch_value=value,
        typed_json_value=typed_json_value,
        typed_json_transportable=typed_json_transportable,
        json_schema_value=json_schema_value,
        json_schema_renderable=json_schema_renderable,
        cli_literal=cli_literal(value, expected_type=expected_type),
    )


def _tool_json_schema_for_param(projected: ProjectedParamContract) -> dict[str, object]:
    param = projected.param
    if param.repeated:
        schema: dict[str, object] = {
            "type": "array",
            "items": _tool_json_schema_for_type(param.base_type, nullable=param.nullable),
        }
    else:
        schema = _tool_json_schema_for_type(param.base_type, nullable=param.nullable)
    if param.description:
        schema["description"] = param.description
    if projected.rendered_choices and all(choice.json_schema_renderable for choice in projected.rendered_choices):
        enum_values = [choice.json_schema_value for choice in projected.rendered_choices]
        if param.repeated:
            items_schema = dict(schema["items"]) if isinstance(schema["items"], dict) else {}
            items_schema["enum"] = enum_values
            schema["items"] = items_schema
        else:
            schema["enum"] = enum_values
    if projected.rendered_default is not None and projected.rendered_default.json_schema_renderable:
        schema["default"] = projected.rendered_default.json_schema_value
    return schema


def _tool_json_schema_for_type(type_name: str, *, nullable: bool) -> dict[str, object]:
    if type_name in {"str", "bytes"}:
        json_types = ["string"]
    elif type_name == "int":
        json_types = ["integer"]
    elif type_name == "float":
        json_types = ["number"]
    elif type_name == "bool":
        json_types = ["boolean"]
    elif type_name == "list":
        json_types = ["array"]
    elif type_name == "object":
        json_types = ["object", "array", "string", "number", "boolean"]
    else:
        json_types = ["object"]
    if nullable:
        json_types = [*json_types, "null"]
    if len(json_types) == 1:
        return {"type": json_types[0]}
    return {"type": json_types}


def _tool_required_params(projected: ProjectedCommandContract) -> tuple[str, ...]:
    grouped_required: set[str] = set()
    required: list[str] = []
    for group in projected.one_of:
        if group.projection_kind in {"exact-visible-xor", "restricted-visible-xor"} and len(group.visible_members) == 1:
            grouped_required.add(group.visible_members[0])
            required.append(group.visible_members[0])
        elif group.projection_kind in {"exact-visible-xor", "restricted-visible-xor"}:
            grouped_required.update(group.visible_members)

    for param in projected.params:
        if param.param.user_required and param.param.name not in grouped_required:
            required.append(param.param.name)
    return tuple(dict.fromkeys(required))


def _tool_group_constraints(projected: ProjectedCommandContract) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = []
    for group in projected.one_of:
        if group.projection_kind in {"exact-visible-xor", "restricted-visible-xor"}:
            if len(group.visible_members) > 1:
                constraints.append({"oneOf": [{"required": [name]} for name in group.visible_members]})
            continue
        if group.projection_kind == "default-satisfied-xor" and len(group.visible_members) > 1:
            constraints.extend(
                {"not": {"required": [left, right]}} for left, right in combinations(group.visible_members, 2)
            )
    return constraints


def _raise_visible_xor_error(projection: ProjectedCommandContract, visible_members: tuple[str, ...]) -> None:
    labels = [_option_label(param_name) for param_name in visible_members]
    binding_name = projection.contract.binding_name or projection.contract.driver_id
    message = (
        f"Binding '{binding_name}' command '{projection.contract.command_name}' requires exactly one of: "
        f"{', '.join(labels)}."
    )
    raise CommandProjectionError(message)


def _backend_capabilities(backend: ProjectionBackend) -> ProjectionBackendCapabilities:
    return BACKEND_CAPABILITIES[backend]


def _option_label(param_name: str) -> str:
    return f"--{_cli_option_name(param_name)}"


def _tool_name(binding_name: str, command_name: str) -> str:
    return "vcs_core__" + "__".join(_tool_part(part) for part in (binding_name, command_name))


def _tool_part(value: str) -> str:
    if not value:
        return "unnamed"
    parts: list[str] = []
    for char in value:
        if char.isascii() and char.isalnum():
            parts.append(char)
        elif char == "_":
            parts.append("_u_")
        else:
            parts.append(f"_x{ord(char):x}_")
    return "".join(parts) or "unnamed"


def _cli_option_name(param_name: str) -> str:
    return param_name.replace("_", "-")


def _cli_click_param_name(param_name: str) -> str:
    return param_name.replace("-", "_")
