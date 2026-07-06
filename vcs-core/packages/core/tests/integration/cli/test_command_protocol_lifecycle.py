# under-test: vcs_core._command_projection
"""End-to-end command protocol lifecycle regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from click.testing import CliRunner
from vcs_core._command_contract import compile_command_contract
from vcs_core._command_projection import (
    anthropic_tool_schema_for_command,
    project_cli_command,
    projectable_command_names,
)
from vcs_core._driver_schema_validation import validate_driver_schema
from vcs_core.cli import main
from vcs_core.runtime_api import CommandExecutionOptions, substrate_client
from vcs_core.spi import (
    BaseSubstrateDriver,
    CapabilitySet,
    CommandRequest,
    Diagnostic,
    DriverContext,
    DriverIngressResult,
    DriverSchema,
    IngressRequest,
    ParamSpec,
    SubstrateStoreIdentity,
    command,
)


@dataclass(frozen=True)
class _LifecycleDriver(BaseSubstrateDriver):
    driver_id: str = "test.lifecycle"
    driver_version: str = "v1"
    binding: str = "lifecycle"
    role: str = "test.Lifecycle"

    @property
    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False)

    def prepare(self, context: DriverContext, request: IngressRequest) -> DriverIngressResult:
        return self.dispatch_decorated_command(context, request)

    @command("configure", required_one_of=(("name", "alias"),))
    def configure(
        self,
        context: DriverContext,
        *,
        name: str | None = None,
        alias: str | None = None,
        mode: Annotated[str, {"choices": ("safe", "fast"), "description": "Execution mode."}] = "safe",
        tags: Annotated[
            object,
            ParamSpec(type="str", required=False, description="Tag.", repeated=True),
        ] = (),
    ) -> DriverIngressResult:
        return DriverIngressResult(
            diagnostics=(
                Diagnostic(
                    code="configured",
                    message="configured",
                    detail={
                        "operation_id": context.operation_id,
                        "name": name,
                        "alias": alias,
                        "mode": mode,
                        "tags": list(tags) if isinstance(tags, list) else list(tags or ()),
                    },
                ),
            ),
        )

    @command("route")
    def route(self, *, binding: str, command: str) -> DriverIngressResult:
        return DriverIngressResult(
            diagnostics=(
                Diagnostic(
                    code="route",
                    message="route",
                    detail={"binding": binding, "command": command},
                ),
            ),
        )


class _Record:
    binding_name = "lifecycle"
    implementation_kind = "driver"


class _BindingSurface:
    def __init__(self, schema: DriverSchema) -> None:
        self._schema = schema

    def schema(self, name: str) -> DriverSchema:
        assert name == "lifecycle"
        return self._schema

    def resolve_driver(self, name: str) -> object:
        schema = self.schema(name)
        return _ResolvedBinding(schema=schema, binding_name=name)


class _ResolvedBinding:
    def __init__(self, *, schema: DriverSchema, binding_name: str) -> None:
        self.schema = schema
        self.command_contracts = {
            name: compile_command_contract(schema, name, binding_name=binding_name) for name in schema.commands
        }


class _FakeVcsCore:
    def __init__(self, schema: DriverSchema) -> None:
        self.binding_contracts = _BindingSurface(schema)
        self.ground = object()
        self.calls: list[tuple[str, str, object, CommandExecutionOptions | None, dict[str, Any]]] = []

    def exec(
        self,
        binding_name: str,
        command: str,
        *,
        scope: object,
        execution_options: CommandExecutionOptions | None = None,
        **params: Any,
    ) -> object:
        self.calls.append((binding_name, command, scope, execution_options, params))
        return object()


def _context() -> DriverContext:
    return DriverContext(
        operation_id="op-lifecycle",
        binding="lifecycle",
        role="test.Lifecycle",
        store_identity=SubstrateStoreIdentity(
            store_id="store_lifecycle",
            kind="test.lifecycle",
            resource_id="lifecycle:test",
        ),
    )


def test_decorated_command_lifecycle_projects_to_cli_tool_proxy_and_dispatch(monkeypatch) -> None:
    driver = _LifecycleDriver()
    schema = driver.describe()

    validate_driver_schema(schema)
    assert projectable_command_names(schema) == ("configure", "route")
    tags_spec = schema.commands["configure"].params["tags"]
    assert tags_spec.has_default is True
    assert tags_spec.default == ()

    projection = project_cli_command("lifecycle", schema, "configure")
    assert tuple(param.param.name for param in projection.params) == ("name", "alias", "mode", "tags")
    assert tuple(group.visible_members for group in projection.one_of) == (("name", "alias"),)
    tool_schema = anthropic_tool_schema_for_command("lifecycle", schema, "configure")
    input_schema = tool_schema["input_schema"]
    assert isinstance(input_schema, dict)
    assert input_schema["oneOf"] == [{"required": ["name"]}, {"required": ["alias"]}]
    assert input_schema["properties"]["tags"]["type"] == "array"

    cli_calls: list[dict[str, Any]] = []
    monkeypatch.setattr("vcs_core._cli_sub._load_binding_records", lambda: (_Record(),))
    monkeypatch.setattr("vcs_core._cli_sub._load_schema", lambda name: schema)

    def fake_run_exec_prepared(**kwargs: Any) -> None:
        cli_calls.append(kwargs)

    monkeypatch.setattr("vcs_core._cli_command_effects.run_exec_prepared", fake_run_exec_prepared)
    result = CliRunner().invoke(
        main,
        [
            "sub",
            "lifecycle",
            "configure",
            "--name",
            "alpha",
            "--mode",
            "fast",
            "--tags",
            "one",
            "--tags",
            "two",
        ],
    )

    assert result.exit_code == 0, result.output
    assert cli_calls == [
        {
            "binding_name": "lifecycle",
            "command": "configure",
            "params": {"name": "alpha", "mode": "fast", "tags": ["one", "two"]},
            "scope_name": None,
            "as_json": False,
        }
    ]

    mg = _FakeVcsCore(schema)
    substrate_client(mg, "lifecycle").configure(alias="beta", mode="safe", tags=["one", "two"])
    assert mg.calls == [
        (
            "lifecycle",
            "configure",
            mg.ground,
            CommandExecutionOptions(),
            {"alias": "beta", "mode": "safe", "tags": ["one", "two"]},
        )
    ]

    dispatched = driver.prepare(
        _context(),
        CommandRequest(command="configure", params={"name": "gamma", "tags": ("x", "y")}),
    )
    assert dispatched.diagnostics[0].detail == {
        "operation_id": "op-lifecycle",
        "name": "gamma",
        "alias": None,
        "mode": "safe",
        "tags": ["x", "y"],
    }

    defaulted_dispatch = driver.prepare(
        _context(),
        CommandRequest(command="configure", params={"name": "delta"}),
    )
    assert defaulted_dispatch.diagnostics[0].detail["tags"] == []


def test_generated_cli_preserves_params_named_like_exec_envelope_fields(monkeypatch) -> None:
    schema = _LifecycleDriver().describe()
    cli_calls: list[dict[str, Any]] = []
    monkeypatch.setattr("vcs_core._cli_sub._load_binding_records", lambda: (_Record(),))
    monkeypatch.setattr("vcs_core._cli_sub._load_schema", lambda name: schema)
    monkeypatch.setattr("vcs_core._cli_command_effects.run_exec_prepared", lambda **kwargs: cli_calls.append(kwargs))

    result = CliRunner().invoke(
        main,
        [
            "sub",
            "lifecycle",
            "route",
            "--binding",
            "user-binding",
            "--command",
            "user-command",
        ],
    )

    assert result.exit_code == 0, result.output
    assert cli_calls == [
        {
            "binding_name": "lifecycle",
            "command": "route",
            "params": {"binding": "user-binding", "command": "user-command"},
            "scope_name": None,
            "as_json": False,
        }
    ]
