"""Schema-oriented helpers shared by CLI command handlers."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from vcs_core._typed_json import encode_typed_json

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vcs_core._binding_contracts import ResolvedDriverBinding
    from vcs_core.spi import DriverSchema


def build_exec_ipc_params(
    substrate_name: str,
    command: str,
    params: Mapping[str, Any],
    *,
    scope_name: str | None,
    execution_options: object | None = None,
) -> dict[str, Any]:
    from vcs_core._command_contract import CommandContractError, normalize_command_params
    from vcs_core._command_envelope import (
        CommandEnvelopeError,
        CommandExecutionOptions,
        command_execution_options_to_mapping,
    )

    resolved = resolve_exec_binding(substrate_name)
    command_contract = resolved.command_contracts.get(command)
    if command_contract is None:
        available = ", ".join(sorted(resolved.command_contracts)) or "(none)"
        click.echo(f"Error: unknown {substrate_name} command '{command}'. Available: {available}")
        sys.exit(1)

    try:
        if execution_options is None:
            execution_options = CommandExecutionOptions()
        if not isinstance(execution_options, CommandExecutionOptions):
            raise CommandEnvelopeError(
                f"execution_options must be CommandExecutionOptions, got {type(execution_options).__name__}."
            )
        cli_params = expand_cli_bytes_params(command_contract, params)
        invocation = normalize_command_params(command_contract, cli_params, source="cli")
        supplied_params = {name: invocation.params[name] for name in invocation.supplied}
        encoded_params = cast("dict[str, object]", encode_typed_json(supplied_params))
        encoded_options = command_execution_options_to_mapping(execution_options)
    except (CommandContractError, CommandEnvelopeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        click.echo(f"Error: {exc}")
        sys.exit(1)

    payload: dict[str, object] = {
        "binding": substrate_name,
        "command": command,
        "params": encoded_params,
        "options": encoded_options,
    }
    if scope_name is not None:
        payload["scope"] = scope_name
    return payload


def expand_cli_bytes_params(command_contract: object, params: Mapping[str, Any]) -> dict[str, Any]:
    """Expand explicit CLI byte references before semantic contract normalization."""
    contract_params = getattr(command_contract, "params", {})
    prepared = dict(params)
    for name, value in params.items():
        param = contract_params.get(name) if hasattr(contract_params, "get") else None
        if getattr(param, "base_type", None) != "bytes":
            continue
        if getattr(param, "repeated", False):
            if isinstance(value, list):
                prepared[name] = [_expand_cli_bytes_value(name, item) for item in value]
            elif isinstance(value, tuple):
                prepared[name] = tuple(_expand_cli_bytes_value(name, item) for item in value)
            else:
                prepared[name] = _expand_cli_bytes_value(name, value)
            continue
        prepared[name] = _expand_cli_bytes_value(name, value)
    return prepared


def _expand_cli_bytes_value(param_name: str, value: object) -> object:
    if isinstance(value, bytes) or not isinstance(value, str):
        return value
    if value == "-":
        return sys.stdin.buffer.read()
    if value.startswith("@"):
        try:
            return Path(value[1:]).read_bytes()
        except OSError as exc:
            raise ValueError(f"parameter '{param_name}' could not read bytes from {value!r}: {exc}") from exc
    return value


def resolve_exec_schema(name: str) -> DriverSchema:
    return resolve_exec_binding(name).schema


def resolve_exec_binding(name: str) -> ResolvedDriverBinding:
    resolved = load_exec_binding_for_cli(name, workspace=".")
    if resolved is None:
        click.echo(f"Error: unknown binding '{name}'.")
        sys.exit(1)
    return resolved


def load_binding_surface_records_for_cli(*, workspace: str) -> tuple[object, ...]:
    from vcs_core._binding_surface import BindingSurface
    from vcs_core.config import load_config
    from vcs_core.discovery import SubstrateResolutionError, collect_binding_specs

    config = load_config(workspace)
    try:
        specs = collect_binding_specs(config, Path(workspace))
        return BindingSurface(specs=specs).records()
    except (SubstrateResolutionError, TypeError, ValueError, OSError) as exc:
        click.echo(f"Error: unable to load binding inventory: {exc}")
        sys.exit(1)


def load_exec_schema_for_cli(name: str, *, workspace: str) -> DriverSchema | None:
    resolved = load_exec_binding_for_cli(name, workspace=workspace)
    return None if resolved is None else resolved.schema


def load_exec_binding_for_cli(name: str, *, workspace: str) -> ResolvedDriverBinding | None:
    from vcs_core._binding_contracts import BindingContractResolver
    from vcs_core._binding_surface import BindingSurface
    from vcs_core.config import load_config
    from vcs_core.discovery import SubstrateResolutionError, collect_binding_specs, resolve_bindings
    from vcs_core.store import Store

    config = load_config(workspace)
    try:
        specs = collect_binding_specs(config, Path(workspace))
        metadata_surface = BindingSurface(specs=specs)
        try:
            metadata_surface.get(name)
        except ValueError:
            return None

        with tempfile.TemporaryDirectory(prefix="vcs-core-show-") as tmpdir:
            repo_path = os.path.join(tmpdir, ".vcscore")  # noqa: PTH118
            os.makedirs(repo_path, exist_ok=True)  # noqa: PTH103
            store = Store(repo_path)
            bindings = resolve_bindings(config, Path(workspace), store)
            return BindingContractResolver(specs=specs, live_bindings=bindings).resolve_driver(name)
    except (SubstrateResolutionError, TypeError, ValueError, OSError) as exc:
        click.echo(f"Error: unable to load binding schema for '{name}': {exc}")
        sys.exit(1)


def load_driver_substrate_schema_for_cli(name: str, *, workspace: str) -> DriverSchema | None:
    from vcs_core._binding_contracts import BindingContractResolver
    from vcs_core.discovery import discover_plugin_registrations, instantiate_substrate_class
    from vcs_core.spi import SubstrateDriver
    from vcs_core.store import Store
    from vcs_core.types import BoundSubstrate

    registrations = discover_plugin_registrations(strict=True)
    registration = registrations.get(name)
    if registration is None:
        return None
    if registration.implementation_kind != "driver":
        return None  # type: ignore[unreachable]  # future-proof: implementation_kind is Literal["driver"] today

    try:
        module = importlib.import_module(registration.module_name)
        cls = getattr(module, registration.class_name)
        with tempfile.TemporaryDirectory(prefix="vcs-core-show-") as tmpdir:
            repo_path = os.path.join(tmpdir, ".vcscore")  # noqa: PTH118
            os.makedirs(repo_path, exist_ok=True)  # noqa: PTH103
            store = Store(repo_path)
            instance = instantiate_substrate_class(
                cls,
                source=registration.source,
                implementation_kind=registration.implementation_kind,
                workspace=Path(workspace),
                store=store,
                config={},
            )
            if not isinstance(instance, SubstrateDriver):
                raise TypeError(f"Substrate type '{name}' is marked driver but does not implement SubstrateDriver.")
            driver_id = instance.driver_id
            if driver_id != name:
                raise ValueError(f"resolved driver instance must report driver_id='{name}', got {driver_id!r}.")
            binding = BoundSubstrate(binding_name=name, substrate_type=name, instance=instance)
            return BindingContractResolver(live_bindings=[binding]).schema(name)
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        click.echo(f"Error: unable to load driver schema for '{name}': {exc}")
        sys.exit(1)


def format_schema_type(type_name: str) -> str:
    if type_name.endswith("?"):
        return f"{type_name[:-1]} (optional)"
    return type_name
