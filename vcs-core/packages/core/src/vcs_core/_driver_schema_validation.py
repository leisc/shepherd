"""Validation and projection classification for driver introspection schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vcs_core._command_contract import (
    CommandContractError,
    compile_all_command_contracts,
    compile_command_contract,
)
from vcs_core._command_projection import CommandProjectionError, project_command_contract
from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._ingress_params import IngressParamError, compile_ingress_params
from vcs_core._substrate_driver import (
    DriverSchema,
    MergeSpec,
    RevisionStorageProfile,
    ScanSpec,
    validate_driver_identity,
)


class DriverSchemaValidationError(ValueError):
    """Raised when a driver schema or projected command surface is invalid."""


@dataclass(frozen=True)
class HiddenParamProjectability:
    """Why one schema parameter is omitted from generated CLI/tool-like projections."""

    param_name: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CommandProjectability:
    """Projection classification for one command in a driver schema."""

    driver_id: str
    command_name: str
    projectable: bool
    projectable_params: tuple[str, ...]
    hidden_params: tuple[HiddenParamProjectability, ...]
    command_reasons: tuple[str, ...] = ()
    required_one_of: tuple[tuple[str, ...], ...] = ()


def validate_driver_schema(schema: DriverSchema) -> None:
    """Validate schema-validity only; CLI projectability is checked separately."""
    if not isinstance(schema, DriverSchema):
        raise DriverSchemaValidationError(f"Driver schema must be DriverSchema, got {type(schema).__name__}.")
    try:
        validate_driver_identity(driver_id=schema.driver_id, driver_version=schema.driver_version)
    except InvalidRepositoryStateError as exc:
        raise DriverSchemaValidationError(str(exc)) from exc

    _validate_mapping(schema.driver_id, "commands", schema.commands)
    _validate_mapping(schema.driver_id, "scans", schema.scans)
    _validate_mapping(schema.driver_id, "merges", schema.merges)
    _validate_storage_profile(schema.driver_id, schema.storage_profile)

    try:
        compile_all_command_contracts(schema)
    except CommandContractError as exc:
        raise DriverSchemaValidationError(str(exc)) from exc
    _validate_named_param_specs(schema.driver_id, "scan", schema.scans, ScanSpec)
    _validate_named_param_specs(schema.driver_id, "merge", schema.merges, MergeSpec)


def validate_projectable_command(schema: DriverSchema, command_name: str) -> CommandProjectability:
    """Validate and classify one command for generated CLI/tool-like projections."""
    try:
        contract = compile_command_contract(schema, command_name)
        projection = project_command_contract(contract, backend="cli")
    except (CommandContractError, CommandProjectionError) as exc:
        raise DriverSchemaValidationError(str(exc)) from exc

    hidden_params = tuple(
        HiddenParamProjectability(param_name=param.name, reasons=projection.hidden_reasons[param.name])
        for param in contract.params.values()
        if param.name in projection.hidden_reasons
    )
    return CommandProjectability(
        driver_id=contract.driver_id,
        command_name=contract.command_name,
        projectable=projection.projectable,
        projectable_params=tuple(param.param.name for param in projection.params),
        hidden_params=hidden_params,
        command_reasons=projection.command_reasons,
        required_one_of=tuple(group.members for group in contract.required_one_of),
    )


def is_projectable_command(schema: DriverSchema, command_name: str) -> bool:
    """Return whether one schema command can be emitted into generated projections."""
    return validate_projectable_command(schema, command_name).projectable


def _validate_mapping(driver_id: str, label: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise DriverSchemaValidationError(f"Driver '{driver_id}' {label} must be a mapping.")


def _validate_storage_profile(driver_id: str, profile: object) -> None:
    if not isinstance(profile, RevisionStorageProfile):
        raise DriverSchemaValidationError(f"Driver '{driver_id}' storage_profile must be a RevisionStorageProfile.")
    if (
        profile.shape == "json-snapshot"
        and profile.growth_bound == "unbounded"
        and not profile.allow_totalized_snapshot
    ):
        raise DriverSchemaValidationError(
            f"Driver '{driver_id}' declares unbounded json-snapshot storage; use an addressable shape "
            "or set allow_totalized_snapshot with a reviewed reason."
        )
    if profile.allow_totalized_snapshot and not profile.notes:
        raise DriverSchemaValidationError(
            f"Driver '{driver_id}' allow_totalized_snapshot requires notes explaining the waiver."
        )
    if profile.authority_role == "accelerator":
        if profile.read_safety is None or profile.crash_lag is None:
            raise DriverSchemaValidationError(
                f"Driver '{driver_id}' accelerator storage requires read_safety and crash_lag."
            )
        if (profile.read_safety, profile.crash_lag) not in {
            ("superset", "index-leads"),
            ("exact", "atomic"),
            ("exact", "authority-leads"),
        }:
            raise DriverSchemaValidationError(
                f"Driver '{driver_id}' accelerator storage has incoherent read_safety/crash_lag."
            )
    elif profile.read_safety is not None or profile.crash_lag is not None:
        raise DriverSchemaValidationError(
            f"Driver '{driver_id}' read_safety/crash_lag are only valid for accelerator storage."
        )
    if profile.shape == "keyed-json-tree" and profile.growth_bound != "unbounded":
        raise DriverSchemaValidationError(
            f"Driver '{driver_id}' keyed-json-tree storage must declare growth_bound='unbounded'."
        )


def _validate_named_param_specs(
    driver_id: str,
    ingress_kind: str,
    specs: Mapping[str, object],
    expected_type: type[ScanSpec | MergeSpec],
) -> None:
    for name, spec in specs.items():
        if not isinstance(name, str) or not name:
            raise DriverSchemaValidationError(f"Driver '{driver_id}' has a {ingress_kind} with an invalid name.")
        if not isinstance(spec, expected_type):
            raise DriverSchemaValidationError(
                f"Driver '{driver_id}' {ingress_kind} '{name}' must be a {expected_type.__name__}."
            )
        if not isinstance(spec.params, Mapping):
            raise DriverSchemaValidationError(f"Driver '{driver_id}' {ingress_kind} '{name}' params must be a mapping.")
        try:
            compile_ingress_params(
                owner_label=f"Driver '{driver_id}'",
                ingress_label=f"{ingress_kind} '{name}'",
                specs=spec.params,
                strict_schema=True,
            )
        except IngressParamError as exc:
            raise DriverSchemaValidationError(str(exc)) from exc
