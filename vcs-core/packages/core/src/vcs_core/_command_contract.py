"""Compiled command contracts and canonical invocation normalization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

from vcs_core._errors import InvalidRepositoryStateError, VcsCoreError
from vcs_core._ingress_params import (
    IngressParamContract,
    IngressParamError,
    compile_ingress_params,
    normalize_ingress_params,
)
from vcs_core._substrate_driver import CommandSpec, DriverSchema, validate_driver_identity

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vcs_core._command_values import CommandValueSource


class CommandContractError(VcsCoreError, ValueError):
    """Raised when a command schema or invocation violates the semantic contract."""


CommandParamContract: TypeAlias = IngressParamContract


@dataclass(frozen=True)
class RequiredOneOfContract:
    members: tuple[str, ...]
    defaulted_member: str | None


@dataclass(frozen=True)
class CommandContract:
    binding_name: str | None
    driver_id: str
    driver_version: str
    command_name: str
    description: str
    examples: tuple[str, ...]
    params: Mapping[str, CommandParamContract]
    required_one_of: tuple[RequiredOneOfContract, ...]
    projectable: bool
    selectable: bool


@dataclass(frozen=True)
class NormalizedCommandInvocation:
    params: dict[str, object]
    supplied: frozenset[str]
    defaulted: frozenset[str]
    selected_one_of: Mapping[tuple[str, ...], str]


def compile_command_contract(
    schema: DriverSchema,
    command_name: str,
    *,
    binding_name: str | None = None,
) -> CommandContract:
    """Compile one declared command into a backend-neutral semantic contract."""
    if not isinstance(schema, DriverSchema):
        raise CommandContractError(f"Driver schema must be DriverSchema, got {type(schema).__name__}.")
    try:
        validate_driver_identity(driver_id=schema.driver_id, driver_version=schema.driver_version)
    except InvalidRepositoryStateError as exc:
        raise CommandContractError(str(exc)) from exc
    if command_name not in schema.commands:
        raise CommandContractError(f"Driver '{schema.driver_id}' has no command named '{command_name}'.")
    return _compile_command_spec(
        driver_id=schema.driver_id,
        driver_version=schema.driver_version,
        command_name=command_name,
        command_spec=schema.commands[command_name],
        binding_name=binding_name,
        selectable=schema.capabilities.selectable,
        strict_schema=True,
    )


def compile_all_command_contracts(schema: DriverSchema) -> tuple[CommandContract, ...]:
    if not isinstance(schema, DriverSchema):
        raise CommandContractError(f"Driver schema must be DriverSchema, got {type(schema).__name__}.")
    return tuple(compile_command_contract(schema, command_name) for command_name in schema.commands)


def normalize_command_params(
    contract: CommandContract,
    raw_params: Mapping[str, object],
    *,
    source: CommandValueSource = "native",
) -> NormalizedCommandInvocation:
    """Normalize source input into dispatch params plus presence metadata."""
    xor_members = {name for group in contract.required_one_of for name in group.members}
    try:
        normalized = normalize_ingress_params(
            contract.params,
            raw_params,
            source=source,
            owner_label=f"Substrate '{_substrate_label(contract)}'",
            ingress_label=f"command '{contract.command_name}'",
            skip_defaults=frozenset(xor_members),
        )
    except IngressParamError as exc:
        raise CommandContractError(str(exc)) from exc

    supplied = normalized.supplied
    params = normalized.params
    defaulted = set(normalized.defaulted)

    selected_one_of: dict[tuple[str, ...], str] = {}
    for group in contract.required_one_of:
        explicitly_supplied = [name for name in group.members if name in supplied]
        if not explicitly_supplied and group.defaulted_member is not None:
            params[group.defaulted_member] = _copy_invocation_default(
                contract,
                contract.params[group.defaulted_member],
            )
            defaulted.add(group.defaulted_member)

        present = [name for name in group.members if name in params]
        if len(present) == 1:
            selected_one_of[group.members] = present[0]
            continue
        if not present:
            raise CommandContractError(
                f"Substrate '{_substrate_label(contract)}' command '{contract.command_name}' "
                f"requires exactly one of: {', '.join(group.members)}."
            )
        raise CommandContractError(
            f"Substrate '{_substrate_label(contract)}' command '{contract.command_name}' "
            f"accepts only one of: {', '.join(group.members)}."
        )

    return NormalizedCommandInvocation(
        params=params,
        supplied=supplied,
        defaulted=frozenset(defaulted),
        selected_one_of=MappingProxyType(selected_one_of),
    )


def _compile_command_spec(
    *,
    driver_id: str,
    driver_version: str,
    command_name: object,
    command_spec: object,
    binding_name: str | None,
    selectable: bool,
    strict_schema: bool,
) -> CommandContract:
    if not isinstance(command_name, str) or not command_name:
        raise CommandContractError(f"Driver '{driver_id}' has a command with an invalid name.")
    if strict_schema and not isinstance(command_spec, CommandSpec):
        raise CommandContractError(f"Driver '{driver_id}' command '{command_name}' must be a CommandSpec.")
    if not isinstance(command_spec, CommandSpec):
        raise CommandContractError(f"Driver '{driver_id}' command '{command_name}' must be a CommandSpec.")
    if not isinstance(command_spec.description, str) or not command_spec.description:
        raise CommandContractError(f"Driver '{driver_id}' command '{command_name}' must have a description.")
    examples = _normalize_command_examples(
        driver_id=driver_id,
        command_name=command_name,
        value=command_spec.examples,
    )
    _validate_bool(driver_id, command_name, "projectable", command_spec.projectable)
    if not isinstance(command_spec.params, dict) and not hasattr(command_spec.params, "items"):
        raise CommandContractError(f"Driver '{driver_id}' command '{command_name}' params must be a mapping.")

    try:
        params = compile_ingress_params(
            owner_label=f"Driver '{driver_id}'",
            ingress_label=f"command '{command_name}'",
            specs=command_spec.params,
            strict_schema=strict_schema,
        )
    except IngressParamError as exc:
        raise CommandContractError(str(exc)) from exc
    required_one_of = _compile_required_one_of(
        driver_id=driver_id,
        command_name=command_name,
        command_spec=command_spec,
        params=params,
    )
    return CommandContract(
        binding_name=binding_name,
        driver_id=driver_id,
        driver_version=driver_version,
        command_name=command_name,
        description=command_spec.description,
        examples=examples,
        params=params,
        required_one_of=required_one_of,
        projectable=command_spec.projectable,
        selectable=selectable,
    )


def _normalize_command_examples(
    *,
    driver_id: str,
    command_name: str,
    value: object,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise CommandContractError(
            f"Driver '{driver_id}' command '{command_name}' examples must be a tuple of strings."
        )
    for item in value:
        if not isinstance(item, str) or not item:
            raise CommandContractError(
                f"Driver '{driver_id}' command '{command_name}' examples entries must be non-empty strings."
            )
    return value


def _compile_required_one_of(
    *,
    driver_id: str,
    command_name: str,
    command_spec: CommandSpec,
    params: Mapping[str, CommandParamContract],
) -> tuple[RequiredOneOfContract, ...]:
    if not isinstance(command_spec.required_one_of, tuple):
        raise CommandContractError(
            f"Driver '{driver_id}' command '{command_name}' required_one_of must be a tuple of tuples."
        )
    compiled: list[RequiredOneOfContract] = []
    members_by_group: dict[str, tuple[str, ...]] = {}
    for group in command_spec.required_one_of:
        if not isinstance(group, tuple) or not group:
            raise CommandContractError(
                f"Driver '{driver_id}' command '{command_name}' required_one_of groups must be non-empty tuples."
            )
        group_members = tuple(group)
        seen_members: set[str] = set()
        for param_name in group_members:
            if not isinstance(param_name, str) or not param_name:
                raise CommandContractError(
                    f"Driver '{driver_id}' command '{command_name}' required_one_of members must be parameter names."
                )
            if param_name in seen_members:
                raise CommandContractError(
                    f"Driver '{driver_id}' command '{command_name}' required_one_of group contains duplicate "
                    f"member '{param_name}'."
                )
            seen_members.add(param_name)
            if param_name not in params:
                raise CommandContractError(
                    f"Driver '{driver_id}' command '{command_name}' required_one_of references unknown parameter "
                    f"'{param_name}'."
                )
            if params[param_name].user_required:
                raise CommandContractError(
                    f"Driver '{driver_id}' command '{command_name}' required_one_of member '{param_name}' must not "
                    "also be individually required."
                )
            previous_group = members_by_group.get(param_name)
            if previous_group is not None:
                raise CommandContractError(
                    f"Driver '{driver_id}' command '{command_name}' required_one_of groups must not overlap; "
                    f"member '{param_name}' appears in both ({', '.join(previous_group)}) and "
                    f"({', '.join(group_members)})."
                )
        for param_name in group_members:
            members_by_group[param_name] = group_members
        defaulted = [
            param_name
            for param_name in group_members
            if params[param_name].has_default and params[param_name].normalized_default is not None
        ]
        if len(defaulted) > 1:
            raise CommandContractError(
                f"Driver '{driver_id}' command '{command_name}' required_one_of group has multiple defaulted "
                f"members: {', '.join(defaulted)}; must not have multiple non-None defaults: {', '.join(defaulted)}."
            )
        compiled.append(
            RequiredOneOfContract(members=group_members, defaulted_member=defaulted[0] if defaulted else None)
        )
    return tuple(compiled)


def _copy_invocation_default(contract: CommandContract, param: CommandParamContract) -> object:
    try:
        return deepcopy(param.normalized_default)
    except Exception as exc:
        raise CommandContractError(
            f"Substrate '{_substrate_label(contract)}' command '{contract.command_name}' parameter "
            f"'{param.name}' default cannot be copied: {exc}"
        ) from exc


def _validate_bool(driver_id: str, command_name: str, label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise CommandContractError(f"Driver '{driver_id}' command '{command_name}' {label} must be a bool.")


def _substrate_label(contract: CommandContract) -> str:
    return contract.binding_name or contract.driver_id
