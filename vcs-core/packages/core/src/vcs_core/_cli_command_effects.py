"""Helpers for CLI command execution and effect recording flows."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import click

from vcs_core import _cli_delegation, _cli_schema
from vcs_core._admission.identifiers import ParseError, parse_optional_scope_name
from vcs_core._cli_errors import exit_app_error
from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vcs_core._app import AppError, VcsCoreApp


class CommandValueRenderError(VcsCoreError, RuntimeError):
    """Raised when a command result cannot be rendered safely in the CLI."""


def _exit_app_error(exc: AppError) -> None:
    exit_app_error(exc)


def run_exec(
    *,
    binding_name: str,
    command: str,
    raw_params: tuple[str, ...],
    scope_name: str | None,
    non_reversible_run: bool = False,
    as_json: bool = False,
) -> None:
    """Run the `exec` CLI flow with session-aware delegation."""
    from vcs_core._command_envelope import CommandExecutionOptions

    params = _parse_assignments(raw_params, "parameter")
    run_exec_prepared(
        binding_name=binding_name,
        command=command,
        params=params,
        scope_name=scope_name,
        execution_options=CommandExecutionOptions(non_reversible_run=non_reversible_run),
        as_json=as_json,
    )


def run_exec_prepared(
    *,
    binding_name: str,
    command: str,
    params: Mapping[str, Any],
    scope_name: str | None,
    execution_options: object | None = None,
    as_json: bool = False,
) -> None:
    """Run an already-parsed `exec` flow with session-aware delegation."""
    from vcs_core._command_envelope import CommandEnvelopeError, CommandExecutionOptions

    if execution_options is None:
        execution_options = CommandExecutionOptions()
    if not isinstance(execution_options, CommandExecutionOptions):
        raise TypeError(f"execution_options must be CommandExecutionOptions, got {type(execution_options).__name__}.")

    try:
        scope_name = parse_optional_scope_name(scope_name)
    except ParseError as exc:
        click.echo(f"Error: cannot exec: {exc}")
        sys.exit(2)

    def _render_exec_result(result: dict[str, object]) -> None:
        if as_json:
            _emit_json(result)
            return
        try:
            emit_command_result(_result_oids(result), result.get("value"))
        except CommandValueRenderError as exc:
            click.echo(f"Error: {exc}")
            sys.exit(1)

    def _fallback() -> None:
        from vcs_core._app import AppError, AppOpenMode, VcsCoreApp

        try:
            with VcsCoreApp.open_existing(".", mode=AppOpenMode.CONTROL) as app:
                resolved = app.mg.binding_contracts.resolve_driver(binding_name)
                command_contract = resolved.command_contracts.get(command)
                if command_contract is None:
                    available = ", ".join(sorted(resolved.command_contracts)) or "(none)"
                    click.echo(f"Error: unknown {binding_name} command '{command}'. Available: {available}")
                    sys.exit(1)

                resolved_scope_name = _stateless_scope_name(app, scope_name, command_name="exec")
                cli_params = _cli_schema.expand_cli_bytes_params(command_contract, params)
                outcome = app.execute(
                    binding_name=binding_name,
                    command=command,
                    scope_name=resolved_scope_name,
                    params=cli_params,
                    command_source="cli",
                    execution_options=execution_options,
                )
                if as_json:
                    from vcs_core.types import normalize_recorded_command_outcome

                    _emit_json(normalize_recorded_command_outcome(outcome))
                    return
                emit_command_result(list(outcome.oids), outcome.value)
        except AppError as exc:
            _exit_app_error(exc)
        except (
            CommandValueRenderError,
            CommandEnvelopeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            click.echo(f"Error: {exc}")
            sys.exit(1)

    _cli_delegation.with_session_result(
        "exec",
        _cli_schema.build_exec_ipc_params(
            binding_name,
            command,
            params,
            scope_name=scope_name,
            execution_options=execution_options,
        ),
        on_result=_render_exec_result,
        on_fallback=_fallback,
    )


def emit_command_result(oids: list[str], value: object | None = None) -> None:
    from vcs_core.types import DRIVER_INGRESS_RESULT_VALUE_SCHEMA, normalize_command_value

    click.echo(f"Recorded {len(oids)} effect(s)")
    for oid in oids:
        click.echo(f"  {oid[:8]}")
    if value is None:
        return
    try:
        normalized = normalize_command_value(value)
    except TypeError as exc:
        msg = f"command returned a value that cannot be rendered in the CLI: {exc}"
        raise CommandValueRenderError(msg) from exc
    click.echo("Value:")
    if isinstance(normalized, dict) and normalized.get("schema") == DRIVER_INGRESS_RESULT_VALUE_SCHEMA:
        _emit_driver_ingress_summary(normalized)
        return
    rendered = json.dumps(normalized, indent=2, sort_keys=True)
    click.echo(rendered)


def _emit_json(payload: object) -> None:
    click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _emit_driver_ingress_summary(payload: dict[str, object]) -> None:
    summary = payload.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    click.echo(
        "DriverIngressResult: "
        f"{summary.get('observation_count', 0)} observation(s), "
        f"{summary.get('transition_count', 0)} transition(s), "
        f"{summary.get('diagnostic_count', 0)} diagnostic(s)"
    )
    for transition in _payload_items(payload, "transitions"):
        click.echo(f"  transition {transition.get('transition_id')} op={transition.get('semantic_op')}")
    for observation in _payload_items(payload, "observations"):
        click.echo(f"  observation {observation.get('observation_id')} kind={observation.get('evidence_kind')}")
    for diagnostic in _payload_items(payload, "diagnostics"):
        click.echo(f"  diagnostic {diagnostic.get('code', 'diagnostic')} subject={diagnostic.get('subject')}")


def _payload_items(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _result_oids(result: dict[str, object]) -> list[str]:
    raw_oids = result.get("oids", [])
    if isinstance(raw_oids, list) and all(isinstance(oid, str) for oid in raw_oids):
        return raw_oids
    msg = "command result contained a non-string oid list"
    raise CommandValueRenderError(msg)


def _parse_assignments(values: tuple[str, ...], label: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            click.echo(f"Error: {label} must be key=value, got '{value}'")
            sys.exit(1)
        key, _, raw = value.partition("=")
        assignments[key] = raw
    return assignments


def _stateless_scope_name(app: VcsCoreApp, requested_scope: str | None, *, command_name: str) -> str:
    """Resolve stateless exec/record defaults without hidden current-scope state."""
    if requested_scope is not None:
        return requested_scope
    live_scopes = tuple(entry.name for entry in app.scope_index.entries if entry.name != "ground")
    if not live_scopes:
        return "ground"
    sample = ", ".join(live_scopes[:5])
    remainder = len(live_scopes) - min(len(live_scopes), 5)
    suffix = f", and {remainder} more" if remainder > 0 else ""
    click.echo(
        f"Error: cannot {command_name}: live scope(s) exist ({sample}{suffix}); "
        "pass `--scope ground` or `--scope <name>` explicitly."
    )
    sys.exit(1)
