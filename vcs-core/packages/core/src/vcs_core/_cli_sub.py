"""Generated substrate command projection for ``vcs-core sub``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from vcs_core import _cli_command_effects, _cli_schema
from vcs_core._command_projection import (
    CommandProjectionError,
    ProjectedCommandContract,
    ProjectedParamContract,
    project_cli_command,
    projectable_command_names,
    validate_projected_surface_input,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vcs_core._binding_surface import BindingSurfaceRecord
    from vcs_core.spi import DriverSchema


class _BytesInput(click.ParamType):
    name = "bytes"

    def convert(self, value: object, param: click.Parameter | None, ctx: click.Context | None) -> object:
        if not isinstance(value, str):
            return value
        if value == "-":
            return sys.stdin.buffer.read()
        if value.startswith("@"):
            try:
                return Path(value[1:]).read_bytes()
            except OSError as exc:
                self.fail(f"could not read {value!r}: {exc}", param, ctx)
        return value


class _DeferredContractInput(click.ParamType):
    def __init__(self, name: str) -> None:
        self.name = name

    def convert(self, value: object, param: click.Parameter | None, ctx: click.Context | None) -> object:
        del param, ctx
        return value


class SubstrateProjectionRoot(click.Group):
    """Lazy root group: ``vcs-core sub <binding> <command>``."""

    def __init__(self) -> None:
        super().__init__(
            name="sub",
            help="Execute projectable binding commands.",
            no_args_is_help=True,
        )

    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        return sorted(
            record.binding_name for record in _load_binding_records() if record.implementation_kind == "driver"
        )

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        del ctx
        record = _find_binding_record(cmd_name)
        if record is None:
            return None
        if record.implementation_kind != "driver":
            return _DiagnosticCommand(  # type: ignore[unreachable]  # future-proof: implementation_kind is Literal["driver"] today
                name=cmd_name,
                message=(
                    f"Binding '{cmd_name}' is not a driver binding, so it is not projected under "
                    "`vcs-core sub`. Use `vcs-core exec` for raw binding command invocation."
                ),
            )
        return _BindingProjectionGroup(cmd_name)


class _BindingProjectionGroup(click.Group):
    def __init__(self, binding_name: str) -> None:
        self.binding_name = binding_name
        super().__init__(
            name=binding_name,
            help=f"Execute projectable commands on binding '{binding_name}'.",
            no_args_is_help=True,
        )

    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        schema = _load_schema(self.binding_name)
        return list(projectable_command_names(schema))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        del ctx
        schema = _load_schema(self.binding_name)
        command_spec = schema.commands.get(cmd_name)
        if command_spec is None:
            return None
        try:
            command_projection = project_cli_command(self.binding_name, schema, cmd_name)
        except CommandProjectionError as exc:
            return _DiagnosticCommand(
                name=cmd_name,
                message=(
                    f"Binding '{self.binding_name}' command '{cmd_name}' cannot be projected "
                    f"({exc}). "
                    "Use `vcs-core exec` for raw binding command invocation."
                ),
            )
        return _build_projected_command(command_projection)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands = self.list_commands(ctx)
        if commands:
            super().format_commands(ctx, formatter)
            return
        formatter.write_paragraph()
        formatter.write_text(
            "No commands on this binding can be projected. Use `vcs-core exec` for raw binding command invocation."
        )


class _DiagnosticCommand(click.Command):
    def __init__(self, *, name: str, message: str) -> None:
        self._message = message
        super().__init__(name=name, help=message, callback=self._callback)

    def _callback(self) -> None:
        raise click.ClickException(self._message)


def _build_projected_command(projection: ProjectedCommandContract) -> click.Command:
    params: list[click.Parameter] = [
        click.Option(
            ["--scope", "scope_name"],
            default=None,
            help="Scope to operate on (default: ground; current session scope when a session is active).",
        ),
        click.Option(["--json", "as_json"], is_flag=True, help="Render machine-readable JSON output."),
    ]
    param_by_click_name: dict[str, str] = {}
    for param in projection.params:
        option = _build_option(param)
        params.append(option)
        if option.name is None:
            raise AssertionError(f"Click did not derive an option name for {param.param.name!r}")
        param_by_click_name[option.name] = param.param.name

    def _callback(**kwargs: Any) -> None:
        scope_name = kwargs.pop("scope_name")
        as_json = kwargs.pop("as_json")
        raw_params = _collect_raw_params(kwargs, param_by_click_name, projection.params)
        try:
            validate_projected_surface_input(projection, raw_params)
        except CommandProjectionError as exc:
            raise click.UsageError(str(exc)) from exc
        binding_name = projection.contract.binding_name or projection.contract.driver_id
        _cli_command_effects.run_exec_prepared(
            binding_name=binding_name,
            command=projection.contract.command_name,
            params=raw_params,
            scope_name=scope_name,
            as_json=as_json,
        )

    return click.Command(
        name=projection.contract.command_name,
        callback=_callback,
        params=params,
        help=projection.contract.description,
        epilog="\n".join(projection.contract.examples),
    )


def _build_option(param: ProjectedParamContract) -> click.Option:
    command_param = param.param
    click_name = _click_param_name(command_param.name)
    help_text = command_param.description
    default_label = (
        param.rendered_default.cli_literal
        if param.rendered_default is not None and param.rendered_default.cli_literal is not None
        else None
    )
    if default_label is not None:
        help_text = f"{help_text} Default: {default_label}.".strip()
    param_decls = [f"--{param.option_name}", click_name]
    option_kwargs: dict[str, Any] = {
        "required": False,
        "help": help_text,
        "multiple": command_param.repeated,
    }
    base_type = command_param.base_type
    if base_type == "bool":
        param_decls = [f"--{param.option_name}/--no-{param.option_name}", click_name]
        option_kwargs["default"] = None
        option_kwargs.pop("multiple")
    else:
        option_kwargs["type"] = _click_type(param)
    return click.Option(param_decls, **option_kwargs)


def _click_type(param: ProjectedParamContract) -> click.ParamType | type[str]:
    choice_labels = _choice_labels(param.rendered_choices)
    if param.param.base_type == "str" and choice_labels:
        return click.Choice(choice_labels)
    base_type = param.param.base_type
    if base_type == "int":
        return _DeferredContractInput("integer")
    if base_type == "float":
        return _DeferredContractInput("float")
    if base_type == "bytes":
        return _BytesInput()
    return str


def _collect_raw_params(
    kwargs: Mapping[str, object],
    param_by_click_name: Mapping[str, str],
    params: tuple[ProjectedParamContract, ...],
) -> dict[str, object]:
    param_by_name = {param.param.name: param for param in params}
    raw_params: dict[str, object] = {}
    for click_name, param_name in param_by_click_name.items():
        value = kwargs.get(click_name)
        if value is None or value == ():
            continue
        if param_by_name[param_name].param.repeated and isinstance(value, tuple):
            raw_params[param_name] = list(value)
        else:
            raw_params[param_name] = value
    return raw_params


def _choice_labels(projected_choices: tuple[object, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for choice in projected_choices:
        value = getattr(choice, "dispatch_value", None)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return ()
        labels.append(str(value))
    return tuple(labels)


def _click_param_name(param_name: str) -> str:
    return param_name.replace("-", "_")


def _load_binding_records() -> tuple[BindingSurfaceRecord, ...]:
    return _cli_schema.load_binding_surface_records_for_cli(workspace=".")  # type: ignore[return-value]


def _find_binding_record(name: str) -> BindingSurfaceRecord | None:
    for record in _load_binding_records():
        if record.binding_name == name:
            return record
    return None


def _load_schema(binding_name: str) -> DriverSchema:
    return _cli_schema.resolve_exec_schema(binding_name)


sub = SubstrateProjectionRoot()
