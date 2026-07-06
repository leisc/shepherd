"""Session IPC and stateless rendering CLI integration tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner
from vcs_core._command_contract import compile_command_contract
from vcs_core.cli import main
from vcs_core.spi import CapabilitySet, CommandRequest, CommandSpec, DriverSchema, ParamSpec
from vcs_core.types import RecordedCommandOutcome, ScopeInfo

from ...support.cli import init_repo as _init


def _full_summary(**overrides: object) -> dict[str, object]:
    summary = {
        "operation_id": "op-default",
        "label": "Default Op",
        "kind": "test.default",
        "status": "ok",
        "visibility": "visible",
        "world_name": "default",
        "world_ref": "refs/vcscore/scopes/default",
        "world_id": "world_default",
        "carrier_ref": "refs/vcscore/scopes/default",
        "anchor_oid": None,
        "effect_count": 0,
        "parent_operation_id": None,
        "final_phase": None,
        "archived_via": None,
    }
    summary.update(overrides)
    return summary


def _value_driver_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="value",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={"inspect": CommandSpec(description="Inspect", params={})},
    )


class _StaticBindingSurface:
    def schema(self, name: str) -> DriverSchema:
        assert name == "value"
        return _value_driver_schema()

    def resolve_driver(self, name: str) -> object:
        return _resolved_binding(self.schema(name), binding_name=name)


def _resolved_binding(schema: DriverSchema, *, binding_name: str) -> object:
    return SimpleNamespace(
        schema=schema,
        command_contracts={
            command_name: compile_command_contract(schema, command_name, binding_name=binding_name)
            for command_name in schema.commands
        },
    )


def test_exec_uses_session_ipc_when_available(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["exec", "marker", "mark", "-p", "label=session-driven"])

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "marker",
                "command": "mark",
                "params": {"label": "session-driven"},
                "options": {"non_reversible_run": False},
            },
        )
    ]


def test_exec_non_reversible_run_uses_structural_session_option(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["exec", "marker", "mark", "--non-reversible-run", "-p", "label=session-driven"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "marker",
                "command": "mark",
                "params": {"label": "session-driven"},
                "options": {"non_reversible_run": True},
            },
        )
    ]


def test_exec_ipc_params_load_configured_driver_schema_via_binding_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import sys
    import types

    from vcs_core import discovery
    from vcs_core._cli_schema import build_exec_ipc_params
    from vcs_core.manifest import SubstrateManifest

    from ...support.drivers import PlainCommandDriver

    driver_module = types.ModuleType("_test_cli_binding_surface_driver")

    class CliDriver(PlainCommandDriver):
        driver_id = "test.cli_binding_surface_driver"

    monkeypatch.setitem(sys.modules, "_test_cli_binding_surface_driver", driver_module)
    driver_module.CliDriver = CliDriver  # type: ignore[attr-defined]
    real_discover = discovery.discover_plugin_registrations

    def patched_discover(*, strict: bool = True):
        available = dict(real_discover(strict=strict))
        available["test.cli_binding_surface_driver"] = discovery.DiscoveredSubstrate(
            name="test.cli_binding_surface_driver",
            module_name="_test_cli_binding_surface_driver",
            class_name="CliDriver",
            source="plugin",
            manifest=SubstrateManifest(name="test.cli_binding_surface_driver"),
            implementation_kind="driver",
        )
        return available

    monkeypatch.setattr(discovery, "discover_plugin_registrations", patched_discover)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vcscore.toml").write_text(
        '[bindings.runtime]\ntype = "test.cli_binding_surface_driver"\n',
        encoding="utf-8",
    )

    payload = build_exec_ipc_params(
        "runtime",
        "echo",
        {"message": "hello"},
        scope_name="ground",
    )

    assert payload == {
        "binding": "runtime",
        "command": "echo",
        "params": {"message": "hello"},
        "options": {"non_reversible_run": False},
        "scope": "ground",
    }


def test_exec_ipc_params_nests_user_params_that_share_routing_names(monkeypatch) -> None:
    from vcs_core._cli_schema import build_exec_ipc_params

    schema = DriverSchema(
        driver_id="router",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "route": CommandSpec(
                description="Route a request.",
                params={
                    "binding": ParamSpec(type="str"),
                    "command": ParamSpec(type="str"),
                },
            )
        },
    )
    monkeypatch.setattr(
        "vcs_core._cli_schema.resolve_exec_binding", lambda name: _resolved_binding(schema, binding_name=name)
    )

    payload = build_exec_ipc_params(
        "runtime",
        "route",
        {"binding": "user-binding", "command": "user-command"},
        scope_name=None,
    )

    assert payload == {
        "binding": "runtime",
        "command": "route",
        "params": {
            "binding": "user-binding",
            "command": "user-command",
        },
        "options": {"non_reversible_run": False},
    }


def test_exec_object_param_sends_json_object_over_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["exec", "marker", "mark", "-p", "label=session-driven", "-p", 'metadata={"phase":"start"}'],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "marker",
                "command": "mark",
                "params": {
                    "label": "session-driven",
                    "metadata": {"phase": "start"},
                },
                "options": {"non_reversible_run": False},
            },
        )
    ]


def test_exec_filesystem_write_rejects_plain_text_bytes_over_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["exec", "filesystem", "write", "-p", "path=from-exec.txt", "-p", "content=active"],
    )

    assert result.exit_code == 1
    assert "expected bytes, got str" in result.output
    assert calls == []


def test_exec_filesystem_write_sends_explicit_file_bytes_over_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)
    payload = tmp_path / "content.bin"
    payload.write_text("active")

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["exec", "filesystem", "write", "-p", "path=from-exec.txt", "-p", f"content=@{payload}"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "filesystem",
                "command": "write",
                "params": {
                    "path": "from-exec.txt",
                    "content": {"__type__": "bytes", "encoding": "base64", "data": "YWN0aXZl"},
                },
                "options": {"non_reversible_run": False},
            },
        )
    ]


def test_exec_reports_non_finite_cli_json_without_traceback(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["exec", "marker", "mark", "-p", "label=bad", "-p", 'metadata={"bad":NaN}'],
    )

    assert result.exit_code != 0
    assert "expected valid JSON for object" in result.output
    assert "Traceback" not in result.output
    assert calls == []


def test_exec_renders_command_value_from_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: {
            "ok": True,
            "result": {
                "oids": ["1234567890abcdef"],
                "value": {"answer": 42},
            },
        },
    )

    result = runner.invoke(main, ["exec", "marker", "mark", "-p", "label=session-driven"])

    assert result.exit_code == 0, result.output
    assert "Recorded 1 effect" in result.output
    assert "Value:" in result.output
    assert '"answer": 42' in result.output


def test_exec_json_renders_session_ipc_machine_payload(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo
    from vcs_core.types import DRIVER_INGRESS_RESULT_VALUE_SCHEMA

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    payload = {
        "oids": [],
        "value": {
            "schema": DRIVER_INGRESS_RESULT_VALUE_SCHEMA,
            "summary": {
                "observation_count": 0,
                "transition_count": 1,
                "retention_hint_count": 0,
                "selection_requirement_count": 0,
                "diagnostic_count": 0,
            },
            "observations": [],
            "transitions": [{"transition_id": "tr-1", "semantic_op": "append"}],
            "retention_hints": [],
            "selection_requirements": [],
            "diagnostics": [],
        },
    }

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: {"ok": True, "result": payload},
    )

    result = runner.invoke(main, ["exec", "marker", "mark", "-p", "label=session-driven", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload


def test_exec_renders_command_value_in_stateless_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    scope = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")

    class ValueSubstrate:
        name = "value"
        commands = {"inspect": CommandSpec(description="Inspect", params={})}

    class FakeVcsCore:
        def __init__(self) -> None:
            self.substrates = [ValueSubstrate()]
            self.binding_surface = _StaticBindingSurface()
            self.binding_contracts = _StaticBindingSurface()
            self.ground = scope

        def resolve_binding(self, name: str):
            assert name == "value"
            return SimpleNamespace(instance=self.substrates[0])

        def exec(self, substrate_name: str, command: str, *, scope: ScopeInfo, **params: Any) -> RecordedCommandOutcome:
            del substrate_name, command, scope, params
            return RecordedCommandOutcome(oids=("1234567890abcdef",), value={"answer": 42})

    class FakeApp:
        def __init__(self) -> None:
            self.mg = FakeVcsCore()

        def execute(
            self,
            *,
            binding_name: str,
            command: str,
            scope_name: str,
            params: dict[str, object],
            execution_options: object | None = None,
            command_source: str = "native",
        ) -> RecordedCommandOutcome:
            del execution_options
            assert command_source == "cli"
            scope_arg = self.resolve_scope(scope_name)
            return self.mg.exec(binding_name, command, scope=scope_arg, **params)

        def resolve_scope(self, name: str) -> ScopeInfo:
            assert name == "ground"
            return scope

    @contextmanager
    def fake_app_context(workspace: str = ".", *, mode=None):  # type: ignore[no-untyped-def]
        del workspace, mode
        yield FakeApp()

    monkeypatch.setattr(
        "vcs_core._cli_schema.resolve_exec_binding",
        lambda name: _resolved_binding(_value_driver_schema(), binding_name=name),
    )
    monkeypatch.setattr("vcs_core._cli_ipc.try_session_ipc", lambda method, params=None: None)
    monkeypatch.setattr("vcs_core._app.VcsCoreApp.open_existing", fake_app_context)

    result = runner.invoke(main, ["exec", "value", "inspect", "--scope", "ground"])

    assert result.exit_code == 0, result.output
    assert "Recorded 1 effect" in result.output
    assert "Value:" in result.output
    assert '"answer": 42' in result.output


def test_exec_renders_driver_ingress_result_in_stateless_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from vcs_core.runtime_api import DriverIngressResult
    from vcs_core.spi import Diagnostic, TransitionDraft
    from vcs_core.types import DRIVER_INGRESS_RESULT_VALUE_SCHEMA

    runner = CliRunner()
    scope = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")

    class ValueSubstrate:
        name = "value"
        commands = {"inspect": CommandSpec(description="Inspect", params={})}

    class FakeVcsCore:
        def __init__(self) -> None:
            self.substrates = [ValueSubstrate()]
            self.binding_surface = _StaticBindingSurface()
            self.binding_contracts = _StaticBindingSurface()
            self.ground = scope

        def resolve_binding(self, name: str):
            assert name == "value"
            return SimpleNamespace(instance=self.substrates[0])

        def exec(self, substrate_name: str, command: str, *, scope: ScopeInfo, **params: Any) -> RecordedCommandOutcome:
            del substrate_name, command, scope, params
            return RecordedCommandOutcome(
                value=DriverIngressResult(
                    transitions=(
                        TransitionDraft(
                            transition_id="tr-1",
                            semantic_op="append",
                            payload={"schema": "test/transition/v1"},
                            observation_ids=(),
                        ),
                    ),
                    diagnostics=(Diagnostic(code="note", message="hello", subject="tr-1"),),
                )
            )

    class FakeApp:
        def __init__(self) -> None:
            self.mg = FakeVcsCore()

        def execute(
            self,
            *,
            binding_name: str,
            command: str,
            scope_name: str,
            params: dict[str, object],
            execution_options: object | None = None,
            command_source: str = "native",
        ) -> RecordedCommandOutcome:
            del execution_options
            assert command_source == "cli"
            scope_arg = self.resolve_scope(scope_name)
            return self.mg.exec(binding_name, command, scope=scope_arg, **params)

        def resolve_scope(self, name: str) -> ScopeInfo:
            assert name == "ground"
            return scope

    @contextmanager
    def fake_app_context(workspace: str = ".", *, mode=None):  # type: ignore[no-untyped-def]
        del workspace, mode
        yield FakeApp()

    monkeypatch.setattr(
        "vcs_core._cli_schema.resolve_exec_binding",
        lambda name: _resolved_binding(_value_driver_schema(), binding_name=name),
    )
    monkeypatch.setattr("vcs_core._cli_ipc.try_session_ipc", lambda method, params=None: None)
    monkeypatch.setattr("vcs_core._app.VcsCoreApp.open_existing", fake_app_context)

    result = runner.invoke(main, ["exec", "value", "inspect", "--scope", "ground"])
    json_result = runner.invoke(main, ["exec", "value", "inspect", "--scope", "ground", "--json"])

    assert result.exit_code == 0, result.output
    assert "DriverIngressResult: 0 observation(s), 1 transition(s), 1 diagnostic(s)" in result.output
    assert "transition tr-1 op=append" in result.output
    assert '"transitions"' not in result.output
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload["oids"] == []
    assert payload["value"]["schema"] == DRIVER_INGRESS_RESULT_VALUE_SCHEMA
    assert payload["value"]["transitions"][0]["semantic_op"] == "append"


def test_exec_reports_unrenderable_command_value_in_stateless_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    scope = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")

    class ValueSubstrate:
        name = "value"
        commands = {"inspect": CommandSpec(description="Inspect", params={})}

    class FakeVcsCore:
        def __init__(self) -> None:
            self.substrates = [ValueSubstrate()]
            self.binding_surface = _StaticBindingSurface()
            self.binding_contracts = _StaticBindingSurface()
            self.ground = scope

        def resolve_binding(self, name: str):
            assert name == "value"
            return SimpleNamespace(instance=self.substrates[0])

        def exec(self, substrate_name: str, command: str, *, scope: ScopeInfo, **params: Any) -> RecordedCommandOutcome:
            del substrate_name, command, scope, params
            return RecordedCommandOutcome(oids=("1234567890abcdef",), value={1: "bad-key"})

    class FakeApp:
        def __init__(self) -> None:
            self.mg = FakeVcsCore()

        def execute(
            self,
            *,
            binding_name: str,
            command: str,
            scope_name: str,
            params: dict[str, object],
            execution_options: object | None = None,
            command_source: str = "native",
        ) -> RecordedCommandOutcome:
            del execution_options
            assert command_source == "cli"
            scope_arg = self.resolve_scope(scope_name)
            return self.mg.exec(binding_name, command, scope=scope_arg, **params)

        def resolve_scope(self, name: str) -> ScopeInfo:
            assert name == "ground"
            return scope

    @contextmanager
    def fake_app_context(workspace: str = ".", *, mode=None):  # type: ignore[no-untyped-def]
        del workspace, mode
        yield FakeApp()

    monkeypatch.setattr(
        "vcs_core._cli_schema.resolve_exec_binding",
        lambda name: _resolved_binding(_value_driver_schema(), binding_name=name),
    )
    monkeypatch.setattr("vcs_core._cli_ipc.try_session_ipc", lambda method, params=None: None)
    monkeypatch.setattr("vcs_core._app.VcsCoreApp.open_existing", fake_app_context)

    result = runner.invoke(main, ["exec", "value", "inspect", "--scope", "ground"])

    assert result.exit_code != 0
    assert "cannot be rendered in the CLI" in result.output
    assert "dict keys must be strings" in result.output


def test_exec_uses_current_session_scope_after_switch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        if method == "switch":
            return {"ok": True, "result": {"current_scope": "experiment", "mount_path": str(tmp_path / "overlay")}}
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    switch_result = runner.invoke(main, ["switch", "experiment"])
    exec_result = runner.invoke(main, ["exec", "marker", "mark", "-p", "label=session-driven"])

    assert switch_result.exit_code == 0, switch_result.output
    assert exec_result.exit_code == 0, exec_result.output
    assert calls == [
        ("switch", {"name": "experiment"}),
        (
            "exec",
            {
                "binding": "marker",
                "command": "mark",
                "params": {"label": "session-driven"},
                "options": {"non_reversible_run": False},
            },
        ),
    ]


def test_switch_missing_scope_renders_app_error_once(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: {"ok": False, "error": "Error: no tracked scope 'missing'."},
    )

    result = runner.invoke(main, ["switch", "missing"])

    assert result.exit_code != 0
    assert result.output == "Error: no tracked scope 'missing'.\n"


def test_delegated_scope_lifecycle_error_is_not_double_prefixed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: {"ok": False, "error": "Error: no tracked scope 'missing'."},
    )

    result = runner.invoke(main, ["merge", "missing"])

    assert result.exit_code != 0
    assert result.output == "Error: no tracked scope 'missing'.\n"


def test_operations_uses_session_ipc_with_current_scope_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        if method == "switch":
            return {"ok": True, "result": {"current_scope": "experiment", "mount_path": str(tmp_path / "overlay")}}
        return {
            "ok": True,
            "result": {
                "requested_mode": "visible",
                "scope": None,
                "visible": [
                    _full_summary(
                        operation_id="op_visible",
                        label="Session Op",
                        kind="test.session",
                        world_name="experiment",
                        world_ref="refs/vcscore/scopes/experiment",
                        world_id="world_experiment",
                        carrier_ref="refs/vcscore/scopes/experiment",
                        effect_count=1,
                    )
                ],
                "open": [],
                "archived": [],
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    switch_result = runner.invoke(main, ["switch", "experiment"])
    operations_result = runner.invoke(main, ["operations"])

    assert switch_result.exit_code == 0, switch_result.output
    assert operations_result.exit_code == 0, operations_result.output
    assert "op_visible  [visible/ok]" in operations_result.output
    assert "world:world_experiment (experiment)" in operations_result.output
    assert "scope_name" not in operations_result.output
    assert "scope_ref" not in operations_result.output
    assert calls == [
        ("switch", {"name": "experiment"}),
        ("operations", {"mode": "visible", "max_count": 20}),
    ]


def test_operations_archived_uses_repo_wide_session_ipc_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        if method == "switch":
            return {"ok": True, "result": {"current_scope": "experiment", "mount_path": str(tmp_path / "overlay")}}
        return {
            "ok": True,
            "result": {
                "requested_mode": "archived",
                "scope": None,
                "visible": [],
                "open": [],
                "archived": [
                    _full_summary(
                        operation_id="archived-op",
                        label="Archived Op",
                        kind="test.archived",
                        status="error",
                        visibility="archived",
                        world_name="experiment",
                        world_ref="refs/vcscore/scopes/experiment",
                        world_id="world_experiment",
                        carrier_ref="refs/vcscore/archive/ops/archived-op",
                        archived_via="operation_ref",
                    )
                ],
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    switch_result = runner.invoke(main, ["switch", "experiment"])
    operations_result = runner.invoke(main, ["operations", "--archived"])

    assert switch_result.exit_code == 0, switch_result.output
    assert operations_result.exit_code == 0, operations_result.output
    assert "archived via: archived operation ref" in operations_result.output
    assert calls == [
        ("switch", {"name": "experiment"}),
        ("operations", {"mode": "archived", "max_count": 20}),
    ]


def test_operation_show_uses_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {
            "ok": True,
            "result": {
                "requested_selector": "show-op",
                "scope": None,
                "summary": {
                    **_full_summary(
                        operation_id="op_show",
                        label="Show Op",
                        kind="test.session",
                        status="error",
                        visibility="archived",
                        world_name="ground",
                        world_ref="refs/vcscore/ground",
                        world_id="world_ground",
                        carrier_ref="refs/vcscore/archive/ops/op_show",
                        effect_count=1,
                        archived_via="operation_ref",
                    )
                },
                "commits": [
                    {
                        "oid": "deadbeefcafebabe",
                        "message": "Marker",
                        "timestamp": 0.0,
                        "metadata": {"type": "Marker"},
                        "parent_oids": [],
                    }
                ],
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["operation", "show", "show-op"])

    assert result.exit_code == 0, result.output
    assert "Operation:    op_show" in result.output
    assert "World:        ground" in result.output
    assert "World ID:     world_ground" in result.output
    assert "Archived via: archived operation ref" in result.output
    assert "Carrier:      refs/vcscore/archive/ops/op_show" in result.output
    assert "scope_name" not in result.output
    assert "scope_ref" not in result.output
    assert calls == [("operation_history", {"selector": "show-op"})]


def test_recovery_uses_session_ipc(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {
            "ok": True,
            "result": {
                "orphaned_scope_refs": ["refs/vcscore/scopes/experiment"],
                "open_operations": [],
                "archived_recovery_operations": [],
                "orphaned_operations": [],
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["recovery"])

    assert result.exit_code == 0, result.output
    assert "Orphaned scopes:" in result.output
    assert "experiment" in result.output
    assert "Archived recovery operations:" in result.output
    assert calls == [("recovery", {"max_count": 20})]


def test_operations_session_ipc_uses_fixed_envelope_for_all_mode(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )
    calls: list[tuple[str, dict | None]] = []

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)

    def fake_send_request(socket_path: str, method: str, params: dict | None = None) -> dict:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {
            "ok": True,
            "result": {
                "requested_mode": "all",
                "scope": "experiment",
                "visible": [
                    _full_summary(
                        operation_id="op_visible",
                        label="Visible Op",
                        kind="test.visible",
                        world_name="experiment",
                        world_ref="refs/vcscore/scopes/experiment",
                        world_id="world_experiment",
                        carrier_ref="refs/vcscore/scopes/experiment",
                        effect_count=1,
                    )
                ],
                "open": [
                    _full_summary(
                        operation_id="op_open",
                        label="Open Op",
                        kind="test.open",
                        status="open",
                        visibility="staged",
                        world_name="experiment",
                        world_ref="refs/vcscore/scopes/experiment",
                        world_id="world_experiment",
                        carrier_ref="refs/vcscore/operations/open/op_open",
                    )
                ],
                "archived": [],
            },
        }

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["operations", "--all", "--scope", "experiment"])

    assert result.exit_code == 0, result.output
    assert "Visible operations:" in result.output
    assert "Open operations:" in result.output
    assert "Archived operations:" in result.output
    assert "op_visible  [visible/ok]" in result.output
    assert "op_open  [staged/open]" in result.output
    assert calls == [("operations", {"mode": "all", "max_count": 20, "scope": "experiment"})]


def test_exec_errors_when_session_daemon_is_unreachable(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import time

    from vcs_core.testing import SessionInfo

    runner = CliRunner()
    _init(runner, tmp_path)

    info = SessionInfo(
        pid=12345,
        socket_path="/tmp/fake.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", lambda repo_path: True)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", lambda repo_path: info)
    monkeypatch.setattr(
        "vcs_core._ipc.send_request",
        lambda socket_path, method, params=None: (_ for _ in ()).throw(ConnectionError("boom")),
    )

    result = runner.invoke(main, ["exec", "marker", "mark", "-p", "label=session-driven"])

    assert result.exit_code != 0
    assert "unreachable" in result.output
