"""Binding and substrate-related CLI command groups."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from vcs_core._cli_schema import format_schema_type, load_driver_substrate_schema_for_cli

if TYPE_CHECKING:
    from collections.abc import Mapping


def _render_command_schema(commands: Mapping[str, Any], *, detailed: bool) -> None:
    if not commands:
        return

    click.echo()
    click.echo("  Commands:")
    for command_name, command_spec in sorted(commands.items()):
        click.echo(f"    {command_name}: {command_spec.description}")
        if not detailed:
            continue
        if command_spec.params:
            click.echo("      Params:")
            for param_name, param_spec in sorted(command_spec.params.items()):
                required = "required" if param_spec.required else "optional"
                line = f"        {param_name}: {format_schema_type(param_spec.type)} ({required})"
                if param_spec.description:
                    line += f" -- {param_spec.description}"
                click.echo(line)
        if command_spec.examples:
            click.echo("      Examples:")
            for example in command_spec.examples:
                click.echo(f"        {example}")


@click.group()
def binding() -> None:
    """Manage configured and implicit bindings."""


@binding.command("list")
def binding_list() -> None:
    """Show active bindings for this repository."""
    workspace = os.path.abspath(".")
    repo_path = os.path.join(workspace, ".vcscore")  # noqa: PTH118
    if not os.path.exists(repo_path):  # noqa: PTH110
        click.echo("Error: not a vcs-core repository. Run `vcs-core init` first.")
        sys.exit(1)

    from vcs_core.config import load_config
    from vcs_core.discovery import discover_manifests, resolve_bindings
    from vcs_core.store import Store

    config = load_config(workspace)
    bindings = resolve_bindings(config, Path(workspace), Store(repo_path))
    manifests = discover_manifests(strict=False)

    click.echo(f"  {'Binding':<20s} {'Type':<20s} {'Source'}")
    click.echo(f"  {'─' * 56}")
    for bound in bindings:
        manifest = manifests.get(bound.substrate_type)
        source = "configured" if bound.binding_name in config.bindings else "implicit"
        suffix = " [planned]" if manifest is not None and manifest.status == "planned" else ""
        click.echo(f"  {bound.binding_name:<20s} {bound.substrate_type:<20s} {source}{suffix}")


@binding.command("add")
@click.argument("name")
@click.option("--type", "substrate_type", required=True, help="Substrate type for this binding")
@click.option("--repo", is_flag=True, help="Write to local config instead of project config")
def binding_add(name: str, substrate_type: str, repo: bool) -> None:
    """Add a binding to configuration."""
    import tomli_w

    from vcs_core.discovery import discover_manifests

    manifests = discover_manifests(strict=False)
    if substrate_type not in manifests:
        click.echo(f"Error: unknown substrate type '{substrate_type}'.")
        sys.exit(1)

    config_path = Path(".vcscore") / "config.toml" if repo else Path("vcscore.toml")

    existing: dict[str, object] = {}
    if config_path.exists():
        import tomllib

        with config_path.open("rb") as f:
            existing = tomllib.load(f)

    bindings = existing.get("bindings")
    if not isinstance(bindings, dict):
        bindings = {}
        existing["bindings"] = bindings
    if name in bindings:
        click.echo(f"Error: binding '{name}' already exists in {config_path}.")
        sys.exit(1)
    bindings[name] = {"type": substrate_type}

    config_path.write_bytes(tomli_w.dumps(existing).encode())
    click.echo(f"Added [bindings.{name}] to {config_path}.")


@binding.command("remove")
@click.argument("name")
def binding_remove(name: str) -> None:
    """Remove a binding from configuration."""
    for config_path in [Path("vcscore.toml"), Path(".vcscore") / "config.toml"]:
        if config_path.exists():
            import tomli_w
            import tomllib

            with config_path.open("rb") as f:
                data = tomllib.load(f)
            if "bindings" in data and name in data["bindings"]:
                del data["bindings"][name]
                config_path.write_bytes(tomli_w.dumps(data).encode())
                click.echo(f"Removed [bindings.{name}] from {config_path}.")
                return

    click.echo(f"Binding '{name}' not found in any config file.")


@binding.command("show")
@click.argument("name")
def binding_show(name: str) -> None:
    """Show binding config, resolved type, and schema."""
    workspace = os.path.abspath(".")
    repo_path = os.path.join(workspace, ".vcscore")  # noqa: PTH118
    if not os.path.exists(repo_path):  # noqa: PTH110
        click.echo("Error: not a vcs-core repository. Run `vcs-core init` first.")
        sys.exit(1)

    from vcs_core._binding_contracts import BindingContractResolver
    from vcs_core.config import load_config
    from vcs_core.discovery import SubstrateResolutionError, discover_manifests, resolve_bindings
    from vcs_core.spi import SubstrateDriver
    from vcs_core.store import Store

    config = load_config(workspace)
    try:
        resolved_bindings = resolve_bindings(config, Path(workspace), Store(repo_path))
    except SubstrateResolutionError as exc:
        click.echo(f"Error: unable to resolve bindings: {exc}")
        sys.exit(1)

    resolved = {bound.binding_name: bound for bound in resolved_bindings}
    bound = resolved.get(name)
    if bound is None:
        click.echo(f"Error: unknown binding '{name}'.")
        sys.exit(1)

    manifest = discover_manifests(strict=False).get(bound.substrate_type)
    source = "configured" if name in config.bindings else "implicit"
    click.echo(f"{name} -- {bound.substrate_type}")
    click.echo()
    click.echo(f"  Source:            {source}")
    if manifest is not None:
        click.echo(f"  Description:       {manifest.description}")
        click.echo(f"  Tier:              {manifest.tier}")
        click.echo(f"  Depends on:        {', '.join(manifest.depends_on) or '(none)'}")

    if bound.config:
        click.echo()
        click.echo("  Config:")
        for key, value in bound.config.items():
            click.echo(f"    {key} = {value!r}")

    if isinstance(bound.instance, SubstrateDriver):
        schema = BindingContractResolver(live_bindings=[bound]).schema(name)
        _render_command_schema(schema.commands, detailed=False)
        return

    click.echo(f"Error: resolved binding '{name}' does not implement SubstrateDriver.")
    sys.exit(1)


@binding.command("check")
@click.argument("name", required=False)
def binding_check(name: str | None) -> None:
    """Validate binding config and prerequisites."""
    workspace = os.path.abspath(".")
    repo_path = os.path.join(workspace, ".vcscore")  # noqa: PTH118
    if not os.path.exists(repo_path):  # noqa: PTH110
        click.echo("Error: not a vcs-core repository. Run `vcs-core init` first.")
        sys.exit(1)

    from vcs_core.config import load_config
    from vcs_core.discovery import check_substrate, resolve_bindings
    from vcs_core.store import Store

    config = load_config(workspace)
    resolved = resolve_bindings(config, Path(workspace), Store(repo_path))
    names = [name] if name else sorted(bound.binding_name for bound in resolved)
    for binding_name in names:
        click.echo(f"Checking binding '{binding_name}'...")
        results = check_substrate(binding_name, config, Path())
        for check_name, result in results.items():
            status_icon = "OK" if "FAIL" not in result and "MISSING" not in result else "FAIL"
            click.echo(f"  {check_name:20s} {status_icon}: {result}")


@click.group()
def substrate() -> None:
    """Inspect substrate types."""


@substrate.group("trace")
def substrate_trace() -> None:
    """Inspect the trace substrate (read-only)."""


@substrate_trace.command("read")
@click.argument("rev")
def substrate_trace_read(rev: str) -> None:
    """Print one durable trace revision payload as JSON (B4b slice 3 W1).

    REV is the revision head returned by the append route (`mg exec trace
    append` / the dialect's `append_run_trace`). Inspection-only — the
    substrate CLI group never composes (runtime-call-api.md §2).
    """
    workspace = os.path.abspath(".")
    repo_path = os.path.join(workspace, ".vcscore")  # noqa: PTH118
    if not os.path.exists(repo_path):  # noqa: PTH110
        click.echo("Error: not a vcs-core repository. Run `vcs-core init` first.")
        sys.exit(1)

    import json

    from vcs_core._world_storage_installation import open_or_init_default_world_storage

    manager = open_or_init_default_world_storage(repo_path)
    try:
        payload = manager.store("store_trace").read_revision_payload(rev)
    except (KeyError, ValueError) as exc:
        click.echo(f"Error: cannot read trace revision {rev!r}: {exc}")
        sys.exit(1)
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@substrate_trace.command("head")
def substrate_trace_head() -> None:
    """Print the trace binding's currently selected revision head (ground world).

    The companion to ``trace read``: emits the REV that ``read`` (and the
    dialect's ``shepherd-dialect run trace``) take, resolved exactly as
    ``VcsCore.read_trace_revision(head=None)`` resolves it. Inspection-only.
    """
    workspace = os.path.abspath(".")
    if not os.path.exists(os.path.join(workspace, ".vcscore")):  # noqa: PTH110, PTH118
        click.echo("Error: not a vcs-core repository. Run `vcs-core init` first.")
        sys.exit(1)

    from vcs_core.runtime_api import VcsCore

    mg = VcsCore(workspace)
    mg.activate()
    try:
        world = mg.world_oid()
        heads = mg._world_storage().read_world(world).snapshot.heads if world else ()
        selected = next((h for h in heads if h.binding == "trace"), None)
    finally:
        mg.deactivate()
    if selected is None:
        click.echo("Error: no selected trace head yet (no trace revision appended in this world).")
        sys.exit(1)
    click.echo(selected.head)


@substrate.command("list")
@click.option("--available", is_flag=True, help="Show installed substrate implementations")
def substrate_list(available: bool) -> None:
    """Show substrate types."""
    from vcs_core.discovery import discover_manifests, discover_plugin_registrations

    if available:
        all_subs = discover_plugin_registrations(strict=False)
        manifests = discover_manifests(strict=False)
        click.echo(f"  {'Substrate':<20s} {'Tier':<15s} {'Source'}")
        click.echo(f"  {'─' * 50}")
        for name in sorted(all_subs):
            manifest = manifests.get(name)
            tier = manifest.tier if manifest else "explicit"
            source = all_subs[name].source
            click.echo(f"  {name:<20s} {tier:<15s} {source}")
    else:
        for name, manifest in discover_manifests(strict=False).items():
            tag = " [planned]" if manifest.status == "planned" else ""
            click.echo(f"  {name:<20s} {manifest.tier:<15s} {manifest.description}{tag}")


@substrate.command("show")
@click.argument("name")
def substrate_show(name: str) -> None:
    """Show substrate type schema and manifest."""
    from vcs_core.discovery import discover_manifests, discover_plugin_registrations

    manifest = discover_manifests(strict=False).get(name)
    registration = discover_plugin_registrations(strict=False).get(name)
    if manifest is None or registration is None:
        click.echo(f"Error: unknown substrate '{name}'.")
        sys.exit(1)
    click.echo(f"{name} -- {manifest.description}")
    click.echo()
    click.echo(f"  Tier:              {manifest.tier}")
    click.echo(f"  Requires daemon:   {'yes' if manifest.requires_daemon else 'no'}")
    click.echo(f"  Depends on:        {', '.join(manifest.depends_on) or '(none)'}")

    if registration.implementation_kind != "driver":
        click.echo(f"Error: substrate '{name}' is not driver-kind.")  # type: ignore[unreachable]  # future-proof: implementation_kind is Literal["driver"] today
        return

    schema = load_driver_substrate_schema_for_cli(name, workspace=".")
    if schema is not None:
        _render_command_schema(schema.commands, detailed=True)


@substrate.command("check")
@click.argument("name", required=False)
def substrate_check(name: str | None) -> None:
    """Validate substrate-type installation and prerequisites."""
    from vcs_core.config import load_config
    from vcs_core.discovery import check_substrate, discover_manifests

    config = load_config(".")

    names = [name] if name else sorted(discover_manifests(strict=False))
    for substrate_name in names:
        click.echo(f"Checking substrate '{substrate_name}'...")
        results = check_substrate(substrate_name, config, Path())
        for check_name, result in results.items():
            status_icon = "OK" if "FAIL" not in result and "MISSING" not in result else "FAIL"
            click.echo(f"  {check_name:20s} {status_icon}: {result}")
