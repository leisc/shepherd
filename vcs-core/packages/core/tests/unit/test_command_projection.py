# under-test: vcs_core._command_projection
"""Tests for backend-neutral command projection descriptors."""

from __future__ import annotations

import json
import math

import pytest
from vcs_core._command_contract import compile_command_contract, normalize_command_params
from vcs_core._command_projection import (
    CommandProjectionError,
    ProjectedParamContract,
    anthropic_tool_schema_for_command,
    project_cli_command,
    project_tool_command,
)
from vcs_core.spi import CapabilitySet, CommandRequest, CommandSpec, DriverSchema, ParamSpec


def _assert_json_schema_accepts(schema: dict[str, object], value: object) -> None:
    assert _json_schema_accepts(schema, value), f"expected schema to accept {value!r}: {schema!r}"


def _assert_json_schema_rejects(schema: dict[str, object], value: object) -> None:
    assert not _json_schema_accepts(schema, value), f"expected schema to reject {value!r}: {schema!r}"


def _json_schema_accepts(schema: dict[str, object], value: object) -> bool:
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and not all(
        isinstance(item, dict) and _json_schema_accepts(item, value) for item in all_of
    ):
        return False

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        matches = [item for item in one_of if isinstance(item, dict) and _json_schema_accepts(item, value)]
        if len(matches) != 1:
            return False

    not_schema = schema.get("not")
    if isinstance(not_schema, dict) and _json_schema_accepts(not_schema, value):
        return False

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False

    raw_type = schema.get("type")
    if raw_type is not None:
        allowed_types = tuple(raw_type) if isinstance(raw_type, list) else (raw_type,)
        if not any(_json_type_accepts(type_name, value) for type_name in allowed_types if isinstance(type_name, str)):
            return False

    required = schema.get("required")
    if isinstance(required, list):
        if not isinstance(value, dict):
            return False
        if any(not isinstance(name, str) or name not in value for name in required):
            return False

    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(value, dict):
        additional = schema.get("additionalProperties", True)
        if additional is False and any(key not in properties for key in value):
            return False
        for key, nested_schema in properties.items():
            if (
                key in value
                and isinstance(key, str)
                and isinstance(nested_schema, dict)
                and not _json_schema_accepts(nested_schema, value[key])
            ):
                return False

    items = schema.get("items")
    if isinstance(items, dict) and isinstance(value, list):
        return all(_json_schema_accepts(items, item) for item in value)

    return True


def _json_type_accepts(type_name: str, value: object) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _schema(*, selectable: bool = False) -> DriverSchema:
    return DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=selectable),
        commands={
            "run": CommandSpec(
                description="Run a task.",
                params={
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                    "task_id": ParamSpec(type="str", required=False, description="Task identity."),
                    "args": ParamSpec(type="object", required=False, description="Argument payload."),
                    "may": ParamSpec(type="str", required=False, choices=("read", "write")),
                },
                required_one_of=(("task_body", "task_id"),),
            )
        },
    )


def _default_label(param: ProjectedParamContract) -> str | None:
    rendered = param.rendered_default
    if rendered is None:
        return None
    return rendered.cli_literal


def _choice_labels(param: ProjectedParamContract) -> tuple[str, ...]:
    labels: list[str] = []
    for choice in param.rendered_choices:
        value = choice.dispatch_value
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return ()
        labels.append(str(value))
    return tuple(labels)


def test_project_command_returns_backend_neutral_descriptor() -> None:
    projection = project_cli_command("runtime", _schema(), "run")

    assert projection.contract.binding_name == "runtime"
    assert projection.contract.command_name == "run"
    assert tuple(param.param.name for param in projection.params) == ("task_id", "args", "may")
    assert tuple(param.option_name for param in projection.params) == ("task-id", "args", "may")
    assert tuple(group.visible_members for group in projection.one_of) == (("task_id",),)
    may = projection.params[2]
    assert may.param.choices == ("read", "write")
    assert tuple(choice.cli_literal for choice in may.rendered_choices) == ("'read'", "'write'")


def test_anthropic_tool_schema_uses_same_projected_params() -> None:
    tool = anthropic_tool_schema_for_command("runtime", _schema(), "run")

    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    properties = input_schema["properties"]
    assert isinstance(properties, dict)
    assert tuple(properties) == ("task_id", "args", "may")
    assert tool["name"] == "vcs_core__runtime__run"
    assert properties["task_id"]["type"] == "string"
    assert properties["args"]["type"] == ["object", "array", "string", "number", "boolean"]
    assert properties["may"]["enum"] == ["read", "write"]
    assert input_schema["required"] == ["task_id"]
    assert "oneOf" not in input_schema


def test_anthropic_tool_schema_uses_one_of_for_visible_xor_groups() -> None:
    schema = DriverSchema(
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

    tool = anthropic_tool_schema_for_command("chooser", schema, "choose")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    assert input_schema["oneOf"] == [{"required": ["left"]}, {"required": ["right"]}]


def test_hidden_defaulted_xor_branch_does_not_require_visible_tool_branch() -> None:
    def default_task_body() -> None:
        return None

    schema = DriverSchema(
        driver_id="chooser",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "choose": CommandSpec(
                description="Choose one input.",
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
            )
        },
    )

    projection = project_cli_command("chooser", schema, "choose")
    assert projection.one_of[0].projection_kind == "default-satisfied-xor"

    input_schema = anthropic_tool_schema_for_command("chooser", schema, "choose")["input_schema"]
    assert isinstance(input_schema, dict)
    assert "required" not in input_schema
    assert "oneOf" not in input_schema


def test_visible_defaulted_xor_branch_does_not_emit_one_of_requirement() -> None:
    schema = DriverSchema(
        driver_id="chooser",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "choose": CommandSpec(
                description="Choose one input.",
                params={
                    "left": ParamSpec(type="str", required=False, has_default=True, default="fallback"),
                    "right": ParamSpec(type="str", required=False),
                },
                required_one_of=(("left", "right"),),
            )
        },
    )

    tool = anthropic_tool_schema_for_command("chooser", schema, "choose")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    assert input_schema["properties"]["left"]["default"] == "fallback"
    assert "required" not in input_schema
    assert "oneOf" not in input_schema
    assert input_schema["allOf"] == [{"not": {"required": ["left", "right"]}}]
    _assert_json_schema_accepts(input_schema, {})
    _assert_json_schema_accepts(input_schema, {"left": "explicit"})
    _assert_json_schema_accepts(input_schema, {"right": "explicit"})
    _assert_json_schema_rejects(input_schema, {"left": "explicit", "right": "explicit"})


def test_anthropic_tool_schema_omits_non_json_safe_default_and_enum_annotations() -> None:
    schema = DriverSchema(
        driver_id="filesystem",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "write": CommandSpec(
                description="Write bytes.",
                params={
                    "content": ParamSpec(
                        type="bytes",
                        required=False,
                        has_default=True,
                        default=b"hello",
                        choices=(b"hello",),
                    )
                },
            )
        },
    )

    projection = project_cli_command("filesystem", schema, "write")
    content = projection.params[0]
    assert content.param.has_default is True
    assert content.param.normalized_default == b"hello"
    assert _default_label(content) is None
    assert content.param.choices == (b"hello",)
    assert _choice_labels(content) == ()

    tool = anthropic_tool_schema_for_command("filesystem", schema, "write")
    json.dumps(tool)
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    properties = input_schema["properties"]
    assert isinstance(properties, dict)
    content_schema = properties["content"]
    assert isinstance(content_schema, dict)
    assert "default" not in content_schema
    assert "enum" not in content_schema


def test_cli_projection_renders_json_shaped_defaults_as_json_literals() -> None:
    schema = DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={
                    "payload": ParamSpec(
                        type="object",
                        required=False,
                        has_default=True,
                        default={"marker": "cli"},
                    ),
                    "object_string": ParamSpec(
                        type="object",
                        required=False,
                        has_default=True,
                        default="literal",
                    ),
                    "items": ParamSpec(
                        type="list",
                        required=False,
                        has_default=True,
                        default=["one", "two"],
                    ),
                    "name": ParamSpec(type="str", required=False, has_default=True, default="literal"),
                },
            )
        },
    )

    projection = project_cli_command("runtime", schema, "configure")
    defaults = {param.param.name: _default_label(param) for param in projection.params}

    assert defaults == {
        "payload": '{"marker": "cli"}',
        "object_string": '"literal"',
        "items": '["one", "two"]',
        "name": "'literal'",
    }


def test_retired_dict_params_do_not_project_as_json_objects() -> None:
    schema = DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={
                    "payload": ParamSpec(
                        type="dict",
                        required=False,
                        has_default=True,
                        default={"marker": "cli"},
                    ),
                },
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="unsupported command type declaration 'dict'"):
        project_cli_command("runtime", schema, "configure")


def test_tool_schema_omits_non_standard_json_default() -> None:
    schema = DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={
                    "payload": ParamSpec(
                        type="object",
                        required=False,
                        has_default=True,
                        default={"value": math.nan},
                    ),
                },
            )
        },
    )

    projection = project_cli_command("runtime", schema, "configure")
    assert _default_label(projection.params[0]) is None

    tool = anthropic_tool_schema_for_command("runtime", schema, "configure")
    json.dumps(tool, allow_nan=False)
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    properties = input_schema["properties"]
    assert isinstance(properties, dict)
    payload_schema = properties["payload"]
    assert isinstance(payload_schema, dict)
    assert "default" not in payload_schema


def test_projection_omits_json_serializable_but_unfaithful_defaults_and_enums() -> None:
    schema = DriverSchema(
        driver_id="runtime",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure payload.",
                params={
                    "payload": ParamSpec(
                        type="object",
                        required=False,
                        has_default=True,
                        default={1: "one"},
                    ),
                    "items": ParamSpec(
                        type="list",
                        required=False,
                        has_default=True,
                        default=[{1: "one"}],
                    ),
                    "shape": ParamSpec(
                        type="object",
                        required=False,
                        choices=({1: "one"},),
                    ),
                },
            )
        },
    )

    projection = project_cli_command("runtime", schema, "configure")
    params = {param.param.name: param for param in projection.params}
    assert params["payload"].param.normalized_default == {1: "one"}
    assert _default_label(params["payload"]) is None
    assert params["items"].param.normalized_default == [{1: "one"}]
    assert _default_label(params["items"]) is None

    tool = anthropic_tool_schema_for_command("runtime", schema, "configure")
    json.dumps(tool, allow_nan=False)
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    properties = input_schema["properties"]
    assert isinstance(properties, dict)
    assert "default" not in properties["payload"]
    assert "default" not in properties["items"]
    assert "enum" not in properties["shape"]


def test_project_command_rejects_selectable_driver_commands() -> None:
    with pytest.raises(CommandProjectionError, match="selectable-driver"):
        project_cli_command("trace", _schema(selectable=True), "run")


def test_project_command_rejects_projected_cli_option_collisions() -> None:
    schema = DriverSchema(
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

    with pytest.raises(CommandProjectionError, match="projected CLI option '--foo-bar' collides"):
        project_cli_command("collision", schema, "run")


@pytest.mark.parametrize(
    "param_name",
    [
        "bad name",
        "-dash",
        "_private",
        "---",
        "name.with.dot",
        "trailing-",
        "double__gap",
        "double--gap",
    ],
)
def test_cli_projection_rejects_cli_unsafe_parameter_names(param_name: str) -> None:
    schema = DriverSchema(
        driver_id="unsafe",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={param_name: ParamSpec(type="str", required=False)},
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="cannot be represented as a generated CLI option"):
        project_cli_command("unsafe", schema, "run")

    projected = project_tool_command("unsafe", schema, "run")
    assert tuple(param.param.name for param in projected.params) == (param_name,)


def test_cli_projection_reserves_negative_option_namespace() -> None:
    schema = DriverSchema(
        driver_id="unsafe",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={"no_flag": ParamSpec(type="str", required=False)},
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="'--no-\\*' negative-option namespace"):
        project_cli_command("unsafe", schema, "run")

    projected = project_tool_command("unsafe", schema, "run")
    assert tuple(param.param.name for param in projected.params) == ("no_flag",)


@pytest.mark.parametrize(("bool_param", "colliding_param"), [("foo", "no_foo"), ("foo_bar", "no_foo_bar")])
def test_project_command_rejects_bool_negative_option_alias_collisions(
    bool_param: str,
    colliding_param: str,
) -> None:
    schema = DriverSchema(
        driver_id="collision",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={
                    bool_param: ParamSpec(type="bool", required=False),
                    colliding_param: ParamSpec(type="str", required=False),
                },
            )
        },
    )

    option = colliding_param.replace("_", "-")
    with pytest.raises(CommandProjectionError, match=rf"projected CLI option '--{option}' collides"):
        project_cli_command("collision", schema, "run")

    projected = project_tool_command("collision", schema, "run")
    assert tuple(param.param.name for param in projected.params) == (bool_param, colliding_param)


def test_cli_projection_rejects_repeated_bool_while_tool_projection_renders_array() -> None:
    schema = DriverSchema(
        driver_id="flags",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "set": CommandSpec(
                description="Set flags.",
                params={"flag": ParamSpec(type="bool", required=False, repeated=True)},
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="repeated bool"):
        project_cli_command("flags", schema, "set")

    tool = anthropic_tool_schema_for_command("flags", schema, "set")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    flag_schema = input_schema["properties"]["flag"]
    assert isinstance(flag_schema, dict)
    assert flag_schema == {"type": "array", "items": {"type": "boolean"}}


def test_project_command_rejects_reserved_click_parameter_name_collisions() -> None:
    schema = DriverSchema(
        driver_id="collision",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={
                    "scope_name": ParamSpec(type="str", required=False),
                },
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="reserved CLI parameter name 'scope_name'"):
        project_cli_command("collision", schema, "run")


def test_tool_projection_does_not_apply_cli_reserved_parameter_names() -> None:
    schema = DriverSchema(
        driver_id="tool",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "run": CommandSpec(
                description="Run.",
                params={
                    "scope": ParamSpec(type="str", required=False),
                },
            )
        },
    )

    with pytest.raises(CommandProjectionError, match="reserved CLI option"):
        project_cli_command("tool", schema, "run")

    projected = project_tool_command("tool", schema, "run")
    assert tuple(param.param.name for param in projected.params) == ("scope",)

    tool = anthropic_tool_schema_for_command("tool", schema, "run")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    assert tuple(input_schema["properties"]) == ("scope",)


def test_tool_projection_is_not_blocked_by_cli_option_name_collisions() -> None:
    schema = DriverSchema(
        driver_id="tool",
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

    with pytest.raises(CommandProjectionError, match="projected CLI option '--foo-bar' collides"):
        project_cli_command("tool", schema, "run")

    tool = anthropic_tool_schema_for_command("tool", schema, "run")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    assert tuple(input_schema["properties"]) == ("foo_bar", "foo-bar")


def test_tool_schema_represents_nullable_params() -> None:
    schema = DriverSchema(
        driver_id="nullable",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={"name": ParamSpec(type="str?", required=False)},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("nullable", schema, "configure")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    name_schema = input_schema["properties"]["name"]
    assert isinstance(name_schema, dict)
    assert name_schema["type"] == ["string", "null"]


def test_projection_fidelity_labels_cli_nullable_scalar_loss_without_hiding_param() -> None:
    schema = DriverSchema(
        driver_id="nullable",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={
                    "name": ParamSpec(type="str?", required=False),
                    "payload": ParamSpec(type="object?", required=False),
                },
            )
        },
    )

    cli_projection = project_cli_command("nullable", schema, "configure")
    cli_params = {param.param.name: param for param in cli_projection.params}

    assert tuple(cli_params) == ("name", "payload")
    assert cli_params["name"].fidelity.preserves_nullability is False
    assert cli_params["name"].fidelity.spelling == "cli-option"
    assert cli_params["name"].fidelity.notes == ("explicit-null-not-representable",)
    assert cli_params["payload"].fidelity.preserves_nullability is True

    tool_projection = project_tool_command("nullable", schema, "configure")
    tool_params = {param.param.name: param for param in tool_projection.params}
    assert tool_params["name"].fidelity.preserves_nullability is True
    assert tool_params["name"].fidelity.spelling == "native"
    assert tool_params["name"].fidelity.notes == ()


def test_projection_fidelity_tracks_rendered_defaults_and_choices_by_backend() -> None:
    schema = DriverSchema(
        driver_id="rendering",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={
                    "mode": ParamSpec(type="str", required=False, choices=("fast", "safe")),
                    "count": ParamSpec(type="int", required=False, choices=(1, 2)),
                    "blob": ParamSpec(type="bytes", required=False, has_default=True, default=b"raw"),
                },
            )
        },
    )

    cli_params = {param.param.name: param for param in project_cli_command("rendering", schema, "configure").params}
    assert cli_params["mode"].fidelity.renders_choices is True
    assert cli_params["count"].fidelity.renders_choices is False
    assert cli_params["count"].fidelity.notes == ("choices-not-rendered",)
    assert cli_params["blob"].fidelity.renders_default is False
    assert cli_params["blob"].fidelity.notes == ("default-not-rendered",)

    tool_params = {param.param.name: param for param in project_tool_command("rendering", schema, "configure").params}
    assert tool_params["mode"].fidelity.renders_choices is True
    assert tool_params["count"].fidelity.renders_choices is True
    assert tool_params["blob"].fidelity.renders_default is False
    assert tool_params["blob"].fidelity.notes == ("default-not-rendered",)


def test_tool_schema_represents_object_as_any_non_null_json_value() -> None:
    schema = DriverSchema(
        driver_id="payloads",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={"payload": ParamSpec(type="object", required=False)},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("payloads", schema, "configure")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    payload_schema = input_schema["properties"]["payload"]
    assert isinstance(payload_schema, dict)
    assert payload_schema["type"] == ["object", "array", "string", "number", "boolean"]

    for value in ({"marker": "tool"}, ["one", "two"], "literal", 3, 3.5, True):
        _assert_json_schema_accepts(input_schema, {"payload": value})
        contract = compile_command_contract(schema, "configure", binding_name="payloads")
        assert normalize_command_params(contract, {"payload": value}).params == {"payload": value}
    _assert_json_schema_rejects(input_schema, {"payload": None})


def test_tool_schema_represents_nullable_object_as_any_json_value() -> None:
    schema = DriverSchema(
        driver_id="payloads",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={"payload": ParamSpec(type="object?", required=False)},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("payloads", schema, "configure")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    payload_schema = input_schema["properties"]["payload"]
    assert isinstance(payload_schema, dict)
    assert payload_schema["type"] == ["object", "array", "string", "number", "boolean", "null"]
    _assert_json_schema_accepts(input_schema, {"payload": None})


def test_tool_schema_list_enum_examples_normalize_through_contract() -> None:
    schema = DriverSchema(
        driver_id="lists",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "configure": CommandSpec(
                description="Configure.",
                params={"items": ParamSpec(type="list", required=False, choices=([1, 2],))},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("lists", schema, "configure")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    items_schema = input_schema["properties"]["items"]
    assert isinstance(items_schema, dict)
    assert items_schema["enum"] == [[1, 2]]
    _assert_json_schema_accepts(input_schema, {"items": [1, 2]})

    contract = compile_command_contract(schema, "configure", binding_name="lists")
    assert normalize_command_params(contract, {"items": [1, 2]}).params == {"items": [1, 2]}


def test_tool_schema_places_repeated_choices_on_items() -> None:
    schema = DriverSchema(
        driver_id="tags",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "set": CommandSpec(
                description="Set tags.",
                params={"tag": ParamSpec(type="str", required=False, repeated=True, choices=("red", "blue"))},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("tags", schema, "set")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    tag_schema = input_schema["properties"]["tag"]
    assert isinstance(tag_schema, dict)
    assert tag_schema["type"] == "array"
    assert "enum" not in tag_schema
    assert tag_schema["items"] == {"type": "string", "enum": ["red", "blue"]}


def test_tool_schema_represents_repeated_nullable_items() -> None:
    schema = DriverSchema(
        driver_id="tags",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "set": CommandSpec(
                description="Set tags.",
                params={"tag": ParamSpec(type="str?", required=False, repeated=True)},
            )
        },
    )

    tool = anthropic_tool_schema_for_command("tags", schema, "set")
    input_schema = tool["input_schema"]
    assert isinstance(input_schema, dict)
    tag_schema = input_schema["properties"]["tag"]
    assert isinstance(tag_schema, dict)
    assert tag_schema["items"] == {"type": ["string", "null"]}


def test_tool_names_preserve_distinct_native_command_names() -> None:
    schema = DriverSchema(
        driver_id="collision",
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            "do-thing": CommandSpec(description="Do the dashed thing."),
            "do_thing": CommandSpec(description="Do the underscored thing."),
        },
    )

    names = [
        anthropic_tool_schema_for_command("binding-name", schema, command_name)["name"]
        for command_name in sorted(schema.commands)
    ]

    assert names == [
        "vcs_core__binding_x2d_name__do_x2d_thing",
        "vcs_core__binding_x2d_name__do_u_thing",
    ]
