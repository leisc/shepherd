"""Generated ``vcs-core sub`` CLI projection tests."""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from vcs_core._command_contract import compile_command_contract
from vcs_core.cli import main
from vcs_core.spi import CapabilitySet, CommandRequest, CommandSpec, DriverSchema, ParamSpec
from vcs_core.testing import SessionInfo

from ...support.cli import init_repo as _init


def _runtime_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "register": CommandSpec(description="Register a task.", projectable=False),
            "run": CommandSpec(
                description="Run a task.",
                params={
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                    "task_id": ParamSpec(type="str", required=False),
                    "args": ParamSpec(type="object", required=False),
                    "may": ParamSpec(type="str", required=False),
                    "limit": ParamSpec(
                        type="int",
                        required=False,
                        description="Maximum retry count.",
                        has_default=True,
                        default=7,
                    ),
                    "provider": ParamSpec(type="ExecutionProvider", required=False, projectable=False),
                },
                required_one_of=(("task_body", "task_id"),),
            ),
        },
    )


def _filesystem_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="filesystem",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "write": CommandSpec(
                description="Write a file.",
                params={
                    "path": ParamSpec(type="str"),
                    "content": ParamSpec(type="bytes"),
                },
            )
        },
    )


def _filesystem_default_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="filesystem",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "write": CommandSpec(
                description="Write a file.",
                params={
                    "content": ParamSpec(
                        type="bytes",
                        required=False,
                        description="File content bytes.",
                        has_default=True,
                        default=b"hello",
                        choices=(b"hello",),
                    ),
                },
            )
        },
    )


def _mode_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="mode",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "set": CommandSpec(
                description="Set mode.",
                params={
                    "mode": ParamSpec(type="str", required=False, choices=("safe", "fast")),
                },
            )
        },
    )


def _numeric_choice_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="numbers",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "set": CommandSpec(
                description="Set numeric value.",
                params={
                    "limit": ParamSpec(
                        type="int",
                        required=False,
                        description="Limit.",
                        choices=(1,),
                    ),
                },
            )
        },
    )


def _json_default_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="json-defaults",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={
                    "payload": ParamSpec(
                        type="object",
                        required=False,
                        description="Payload.",
                        has_default=True,
                        default={"marker": "cli"},
                    ),
                    "items": ParamSpec(
                        type="list",
                        required=False,
                        description="Items.",
                        has_default=True,
                        default=["one", "two"],
                    ),
                },
            )
        },
    )


def _json_payload_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="json-payloads",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={"payload": ParamSpec(type="object", required=False, description="Payload.")},
            )
        },
    )


def _choice_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="chooser",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "choose": CommandSpec(
                description="Choose one input.",
                params={
                    "left": ParamSpec(type="str", required=False),
                    "right": ParamSpec(type="str", required=False),
                },
                required_one_of=(("left", "right"),),
            )
        },
    )


def _option_collision_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="collision",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={
                    "foo_bar": ParamSpec(type="str", required=False),
                    "foo-bar": ParamSpec(type="str", required=False),
                },
            )
        },
    )


def _bool_option_alias_collision_schema() -> DriverSchema:
    return DriverSchema(
        driver_id="collision",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={
                    "foo": ParamSpec(type="bool", required=False),
                    "no_foo": ParamSpec(type="str", required=False),
                },
            )
        },
    )


def _hidden_default_xor_schema() -> DriverSchema:
    def default_task_body() -> None:
        return None

    return DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run a task.",
                params={
                    "task_body": ParamSpec(
                        type="callable",
                        required=False,
                        projectable=False,
                        has_default=True,
                        default=default_task_body,
                    ),
                    "task_id": ParamSpec(type="str", required=False),
                },
                required_one_of=(("task_body", "task_id"),),
            ),
        },
    )


class _Record:
    def __init__(self, binding_name: str, implementation_kind: str = "driver") -> None:
        self.binding_name = binding_name
        self.implementation_kind = implementation_kind


def _patch_projection_inventory(
    monkeypatch: pytest.MonkeyPatch,
    schemas: dict[str, DriverSchema],
    *,
    non_driver: tuple[str, ...] = (),
) -> None:
    records = tuple(_Record(name) for name in schemas) + tuple(_Record(name, "non-driver") for name in non_driver)

    def _resolved_binding(name: str) -> object:
        schema = schemas[name]
        return SimpleNamespace(
            schema=schema,
            command_contracts={
                command_name: compile_command_contract(schema, command_name, binding_name=name)
                for command_name in schema.commands
            },
        )

    monkeypatch.setattr("vcs_core._cli_sub._load_binding_records", lambda: records)
    monkeypatch.setattr("vcs_core._cli_sub._load_schema", lambda name: schemas[name])
    monkeypatch.setattr("vcs_core._cli_schema.resolve_exec_schema", lambda name: schemas[name])
    monkeypatch.setattr("vcs_core._cli_schema.resolve_exec_binding", _resolved_binding)


def _session_info(tmp_path: Path) -> SessionInfo:
    return SessionInfo(
        pid=12345,
        socket_path="/tmp/fake-sub.sock",
        mount_path=str(tmp_path),
        workspace=str(tmp_path),
        started_at=time.time(),
    )


def _runtime_run_ipc_params(*, limit: int | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "task_id": "pkg.task:run",
        "args": {"marker": "cli"},
        "may": "write",
    }
    if limit is not None:
        params["limit"] = limit
    return {
        "binding": "runtime",
        "command": "run",
        "params": params,
        "options": {"non_reversible_run": False},
    }


def _patch_live_session(monkeypatch: pytest.MonkeyPatch, *, repo_path: str, info: SessionInfo) -> None:
    def fake_is_session_alive(candidate_repo_path: str) -> bool:
        assert candidate_repo_path == repo_path
        return True

    def fake_read_session_info(candidate_repo_path: str) -> SessionInfo:
        assert candidate_repo_path == repo_path
        return info

    monkeypatch.setattr("vcs_core._ipc.is_session_alive", fake_is_session_alive)
    monkeypatch.setattr("vcs_core._ipc.read_session_info", fake_read_session_info)


def test_sub_root_help_lists_driver_bindings_only(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()}, non_driver=("raw",))

    result = runner.invoke(main, ["sub", "--help"])

    assert result.exit_code == 0, result.output
    assert "runtime" in result.output
    assert "raw" not in result.output


def test_sub_runtime_run_help_projects_only_cli_safe_options(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})

    result = runner.invoke(main, ["sub", "runtime", "run", "--help"], terminal_width=100)

    assert result.exit_code == 0, result.output
    assert result.output == textwrap.dedent(
        """\
        Usage: main sub runtime run [OPTIONS]

          Run a task.

        Options:
          --scope TEXT     Scope to operate on (default: ground; current session scope when a session is
                           active).
          --json           Render machine-readable JSON output.
          --task-id TEXT
          --args TEXT
          --may TEXT
          --limit INTEGER  Maximum retry count. Default: 7.
          --help           Show this message and exit.
        """
    )
    assert "--task-body" not in result.output
    assert "--provider" not in result.output


def test_sub_help_omits_non_cli_renderable_default(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"filesystem": _filesystem_default_schema()})

    result = runner.invoke(main, ["sub", "filesystem", "write", "--help"], terminal_width=100)

    assert result.exit_code == 0, result.output
    assert "--content BYTES  File content bytes." in result.output
    assert "Default:" not in result.output
    assert "b'hello'" not in result.output


def test_sub_string_choices_still_use_click_prevalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"mode": _mode_schema()})

    result = runner.invoke(main, ["sub", "mode", "set", "--mode", "danger"])

    assert result.exit_code == 2
    assert "Invalid value for '--mode'" in result.output
    assert "'danger' is not one of 'safe', 'fast'" in result.output


def test_sub_numeric_choices_defer_to_contract_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"numbers": _numeric_choice_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    leading_zero = runner.invoke(main, ["sub", "numbers", "set", "--limit", "01"])
    hexadecimal = runner.invoke(main, ["sub", "numbers", "set", "--limit", "0x1"])

    assert leading_zero.exit_code == 0, leading_zero.output
    assert hexadecimal.exit_code == 0, hexadecimal.output
    assert calls == [
        (
            "exec",
            {
                "binding": "numbers",
                "command": "set",
                "params": {"limit": 1},
                "options": {"non_reversible_run": False},
            },
        ),
        (
            "exec",
            {
                "binding": "numbers",
                "command": "set",
                "params": {"limit": 1},
                "options": {"non_reversible_run": False},
            },
        ),
    ]


def test_sub_numeric_choice_rejections_come_from_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"numbers": _numeric_choice_schema()})

    result = runner.invoke(main, ["sub", "numbers", "set", "--limit", "2"])

    assert result.exit_code == 1
    assert "must be one of: 1" in result.output
    assert "Invalid value for '--limit'" not in result.output


def test_sub_json_defaults_render_as_cli_json_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"json-defaults": _json_default_schema()})

    result = runner.invoke(main, ["sub", "json-defaults", "configure", "--help"], terminal_width=120)

    assert result.exit_code == 0, result.output
    assert 'Payload. Default: {"marker": "cli"}.' in result.output
    assert 'Items. Default: ["one", "two"].' in result.output
    assert "Default: {'marker': 'cli'}" not in result.output


def test_sub_and_raw_exec_reject_cli_json_null_for_nonnullable_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"payloads": _json_payload_schema()})

    sub_result = runner.invoke(main, ["sub", "payloads", "configure", "--payload", "null"])
    raw_result = runner.invoke(main, ["exec", "payloads", "configure", "-p", "payload=null"])

    for result in (sub_result, raw_result):
        assert result.exit_code != 0
        assert "expected object, got NoneType" in result.output


def test_sub_runtime_run_uses_session_exec_ipc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["sub", "runtime", "run", "--task-id", "pkg.task:run", "--args", '{"marker":"cli"}', "--may", "write"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("exec", _runtime_run_ipc_params())]


def test_sub_runtime_run_explicit_option_overrides_default_before_session_ipc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        [
            "sub",
            "runtime",
            "run",
            "--task-id",
            "pkg.task:run",
            "--args",
            '{"marker":"cli"}',
            "--may",
            "write",
            "--limit",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("exec", _runtime_run_ipc_params(limit=9))]


def test_sub_hidden_defaulted_xor_accepts_omission_under_live_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _hidden_default_xor_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(main, ["sub", "runtime", "run"])

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "runtime",
                "command": "run",
                "params": {},
                "options": {"non_reversible_run": False},
            },
        )
    ]


def test_sub_runtime_run_matches_raw_exec_under_live_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    session_payload = {
        "oids": ["1234567890abcdef"],
        "value": {
            "answer": 42,
            "source": "session",
        },
    }
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": session_payload}

    def fail_if_control_opened(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("sub/raw exec should use session IPC while a live session exists")

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)
    monkeypatch.setattr("vcs_core._app.VcsCoreApp.open_existing", fail_if_control_opened)

    sub_result = runner.invoke(
        main,
        [
            "sub",
            "runtime",
            "run",
            "--json",
            "--task-id",
            "pkg.task:run",
            "--args",
            '{"marker":"cli"}',
            "--may",
            "write",
        ],
    )
    raw_result = runner.invoke(
        main,
        [
            "exec",
            "runtime",
            "run",
            "--json",
            "-p",
            "task_id=pkg.task:run",
            "-p",
            'args={"marker":"cli"}',
            "-p",
            "may=write",
        ],
    )

    assert sub_result.exit_code == 0, sub_result.output
    assert raw_result.exit_code == 0, raw_result.output
    assert json.loads(sub_result.output) == session_payload
    assert json.loads(raw_result.output) == session_payload
    assert calls == [
        ("exec", _runtime_run_ipc_params()),
        ("exec", _runtime_run_ipc_params()),
    ]


def test_sub_live_session_does_not_open_control_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vcs_core._lock import acquire_session_lock, release_session_lock

    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})
    repo_path = str(tmp_path / ".vcscore")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert socket_path == info.socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    def fail_if_control_opened(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("sub should use session IPC while a live session exists")

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)
    monkeypatch.setattr("vcs_core._app.VcsCoreApp.open_existing", fail_if_control_opened)

    acquire_session_lock(repo_path, "foreign-session")
    try:
        result = runner.invoke(
            main,
            ["sub", "runtime", "run", "--task-id", "pkg.task:run", "--args", '{"marker":"cli"}', "--may", "write"],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / ".vcscore" / "session.lock").exists()
    finally:
        release_session_lock(repo_path, "foreign-session")

    assert calls == [("exec", _runtime_run_ipc_params())]


def test_sub_without_live_session_fails_closed_on_foreign_session_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vcs_core._lock import acquire_session_lock, release_session_lock

    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})
    repo_path = str(tmp_path / ".vcscore")

    acquire_session_lock(repo_path, "foreign-session")
    try:
        result = runner.invoke(
            main,
            ["sub", "runtime", "run", "--task-id", "pkg.task:run", "--args", '{"marker":"cli"}', "--may", "write"],
        )

        assert result.exit_code != 0
        assert "Repository locked by session" in result.output
        assert "Traceback" not in result.output
        assert (tmp_path / ".vcscore" / "session.lock").exists()
    finally:
        release_session_lock(repo_path, "foreign-session")


def test_sub_runtime_run_enforces_visible_required_one_of(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})

    result = runner.invoke(main, ["sub", "runtime", "run"])

    assert result.exit_code == 2
    assert "Missing option '--task-id'" in result.output
    assert "task_body" not in result.output


def test_sub_visible_required_one_of_rejects_multiple_options(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"chooser": _choice_schema()})

    result = runner.invoke(main, ["sub", "chooser", "choose", "--left", "a", "--right", "b"])

    assert result.exit_code == 2
    assert "requires exactly one of: --left, --right" in result.output


def test_sub_visible_required_one_of_rejects_missing_options(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"chooser": _choice_schema()})

    result = runner.invoke(main, ["sub", "chooser", "choose"])

    assert result.exit_code == 2
    assert "requires exactly one of: --left, --right" in result.output


def test_sub_nonprojectable_command_points_to_raw_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()})

    result = runner.invoke(main, ["sub", "runtime", "register"])

    assert result.exit_code == 1
    assert "cannot be projected" in result.output
    assert "vcs-core exec" in result.output


def test_sub_projected_option_collision_points_to_raw_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"collision": _option_collision_schema()})

    result = runner.invoke(main, ["sub", "collision", "run"])

    assert result.exit_code == 1
    assert "cannot be projected" in result.output
    assert "projected CLI option '--foo-bar' collides" in result.output
    assert "vcs-core exec" in result.output


def test_sub_bool_negative_alias_collision_points_to_raw_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"collision": _bool_option_alias_collision_schema()})

    result = runner.invoke(main, ["sub", "collision", "run"])

    assert result.exit_code == 1
    assert "cannot be projected" in result.output
    assert "projected CLI option '--no-foo' collides" in result.output
    assert "vcs-core exec" in result.output


def test_sub_non_driver_binding_points_to_raw_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    _patch_projection_inventory(monkeypatch, {"runtime": _runtime_schema()}, non_driver=("raw",))

    result = runner.invoke(main, ["sub", "raw"])

    assert result.exit_code == 1
    assert "not a driver binding" in result.output
    assert "vcs-core exec" in result.output


def test_sub_bytes_option_expands_file_before_session_ipc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_projection_inventory(monkeypatch, {"filesystem": _filesystem_schema()})
    repo_path = str(tmp_path / ".vcscore")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"hello")

    info = _session_info(tmp_path)
    calls: list[tuple[str, dict[str, Any] | None]] = []
    _patch_live_session(monkeypatch, repo_path=repo_path, info=info)

    def fake_send_request(socket_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        del socket_path
        calls.append((method, params))
        return {"ok": True, "result": {"oids": ["1234567890abcdef"]}}

    monkeypatch.setattr("vcs_core._ipc.send_request", fake_send_request)

    result = runner.invoke(
        main,
        ["sub", "filesystem", "write", "--path", "out.txt", "--content", f"@{payload}"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "exec",
            {
                "binding": "filesystem",
                "command": "write",
                "params": {
                    "path": "out.txt",
                    "content": {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="},
                },
                "options": {"non_reversible_run": False},
            },
        )
    ]
