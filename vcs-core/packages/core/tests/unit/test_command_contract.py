# under-test: vcs_core._command_contract
from __future__ import annotations

import math
from typing import Any

import pytest
from vcs_core._command_contract import (
    CommandContract,
    CommandContractError,
    compile_command_contract,
    normalize_command_params,
)
from vcs_core._command_values import coerce_command_value
from vcs_core.spi import CapabilitySet, CommandRequest, CommandSpec, DriverSchema, ParamSpec


def _compile_contract(
    *,
    substrate_name: str,
    command_name: str,
    spec_params: dict[str, Any],
    required_one_of: tuple[tuple[str, ...], ...] = (),
    examples: Any = (),
    description: str | None = None,
) -> CommandContract:
    schema = DriverSchema(
        driver_id=substrate_name,
        driver_version="test",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False),
        commands={
            command_name: CommandSpec(
                description=description or f"{command_name}.",
                params=spec_params,
                examples=examples,
                required_one_of=required_one_of,
            )
        },
    )
    return compile_command_contract(schema, command_name, binding_name=substrate_name)


def _normalize_command_mapping(
    *,
    substrate_name: str,
    command_name: str,
    spec_params: dict[str, Any],
    raw_params: dict[str, Any],
    required_one_of: tuple[tuple[str, ...], ...] = (),
    source: str = "native",
) -> dict[str, object]:
    contract = _compile_contract(
        substrate_name=substrate_name,
        command_name=command_name,
        spec_params=spec_params,
        required_one_of=required_one_of,
    )
    return normalize_command_params(contract, raw_params, source=source).params


def test_cli_object_json_is_opaque_and_does_not_decode_nested_typed_json() -> None:
    value = coerce_command_value(
        substrate_name="runtime",
        context="command 'run' parameter 'args'",
        expected_type="object",
        value='{"nested":{"__type__":"bytes","encoding":"base64","data":"aGVsbG8="}}',
        source="cli",
    )

    assert value == {"nested": {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="}}


def test_cli_list_json_is_opaque_and_does_not_decode_nested_typed_json() -> None:
    value = coerce_command_value(
        substrate_name="runtime",
        context="command 'run' parameter 'items'",
        expected_type="list",
        value='[{"__type__":"bytes","encoding":"base64","data":"aGVsbG8="}]',
        source="cli",
    )

    assert value == [{"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="}]


def test_native_values_pass_through_without_cli_string_json_parsing() -> None:
    assert (
        coerce_command_value(
            substrate_name="runtime",
            context="command 'run' parameter 'args'",
            expected_type="object",
            value='{"already":"a string"}',
        )
        == '{"already":"a string"}'
    )
    assert coerce_command_value(
        substrate_name="runtime",
        context="command 'run' parameter 'items'",
        expected_type="list",
        value=[{"native": True}],
    ) == [{"native": True}]
    assert (
        coerce_command_value(
            substrate_name="runtime",
            context="command 'run' parameter 'enabled'",
            expected_type="bool",
            value=True,
        )
        is True
    )


def test_nullable_declared_params_accept_explicit_none_values() -> None:
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="run",
        spec_params={
            "task_id": ParamSpec(type="str?", required=False),
            "limit": ParamSpec(type="int?", required=False),
            "payload": ParamSpec(type="object?", required=False),
            "task_ref": ParamSpec(type="TaskRef?", required=False, projectable=False),
        },
        raw_params={"task_id": None, "limit": None, "payload": None, "task_ref": None},
    ) == {"task_id": None, "limit": None, "payload": None, "task_ref": None}
    with pytest.raises(ValueError, match="expected str, got NoneType"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="run",
            spec_params={"task_id": ParamSpec(type="str", required=False)},
            raw_params={"task_id": None},
        )


def test_cli_json_null_respects_declared_nullability() -> None:
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params={"payload": ParamSpec(type="object?", required=False)},
        raw_params={"payload": "null"},
        source="cli",
    ) == {"payload": None}

    with pytest.raises(ValueError, match="expected object, got NoneType"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={"payload": ParamSpec(type="object", required=False)},
            raw_params={"payload": "null"},
            source="cli",
        )


@pytest.mark.parametrize(
    "type_name",
    [
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "list",
        "object",
        "TaskRef",
    ],
)
def test_nonnullable_declared_params_reject_explicit_none_values(type_name: str) -> None:
    with pytest.raises(ValueError, match=rf"expected {type_name}, got NoneType"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="run",
            spec_params={"value": ParamSpec(type=type_name, required=False, projectable=False)},
            raw_params={"value": None},
        )


def test_malformed_nullable_type_declarations_do_not_disable_coercion() -> None:
    with pytest.raises(ValueError, match="invalid command type declaration 'str\\?\\?'"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={"name": ParamSpec(type="str??", required=True)},
            raw_params={"name": {"not": "a string"}},
        )


def test_command_contract_rejects_retired_dict_type_declarations() -> None:
    with pytest.raises(ValueError, match="unsupported command type declaration 'dict'"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={"payload": ParamSpec(type="dict", required=False)},
            raw_params={"payload": {"legacy": True}},
        )

    with pytest.raises(ValueError, match="unsupported command type declaration 'dict\\?'"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={"payload": ParamSpec(type="dict?", required=False)},
            raw_params={"payload": None},
        )


@pytest.mark.parametrize(
    ("examples", "match"),
    [
        (["vcs-core exec runtime run"], "examples must be a tuple"),
        ("not-a-sequence", "examples must be a tuple"),
        (("ok", ""), "examples entries must be non-empty strings"),
        (("ok", object()), "examples entries must be non-empty strings"),
    ],
)
def test_command_contract_rejects_malformed_examples(examples: object, match: str) -> None:
    with pytest.raises(CommandContractError, match=match):
        _compile_contract(
            substrate_name="runtime",
            command_name="run",
            spec_params={},
            examples=examples,
        )


def test_missing_required_param_uses_legacy_compatible_single_name_message() -> None:
    with pytest.raises(ValueError, match="missing required parameter 'content'"):
        _normalize_command_mapping(
            substrate_name="filesystem",
            command_name="write",
            spec_params={
                "path": ParamSpec(type="str", required=True),
                "content": ParamSpec(type="bytes", required=True),
            },
            raw_params={"path": "hello.txt"},
            source="cli",
        )


def test_missing_required_params_use_plural_message() -> None:
    with pytest.raises(ValueError, match="missing required parameters: content, path"):
        _normalize_command_mapping(
            substrate_name="filesystem",
            command_name="write",
            spec_params={
                "path": ParamSpec(type="str", required=True),
                "content": ParamSpec(type="bytes", required=True),
            },
            raw_params={},
            source="cli",
        )


def test_required_one_of_requires_exactly_one_present_param() -> None:
    spec_params = {
        "task_body": ParamSpec(type="callable", required=False, projectable=False),
        "task_id": ParamSpec(type="str", required=False),
    }
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="run",
        spec_params=spec_params,
        raw_params={"task_id": "pkg.task:run"},
        required_one_of=(("task_body", "task_id"),),
    ) == {"task_id": "pkg.task:run"}

    with pytest.raises(ValueError, match="requires exactly one of: task_body, task_id"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="run",
            spec_params=spec_params,
            raw_params={},
            required_one_of=(("task_body", "task_id"),),
        )

    with pytest.raises(ValueError, match="accepts only one of: task_body, task_id"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="run",
            spec_params=spec_params,
            raw_params={"task_body": lambda: None, "task_id": "pkg.task:run"},
            required_one_of=(("task_body", "task_id"),),
        )


def test_defaults_are_applied_before_command_param_coercion() -> None:
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="run",
        spec_params={
            "task_id": ParamSpec(type="str", required=False),
            "limit": ParamSpec(type="int", required=False, has_default=True, default="7"),
        },
        raw_params={"task_id": "pkg.task:run"},
        source="cli",
    ) == {"task_id": "pkg.task:run", "limit": 7}

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="run",
        spec_params={
            "task_id": ParamSpec(type="str", required=False),
            "limit": ParamSpec(type="int", required=False, has_default=True, default=7),
        },
        raw_params={"task_id": "pkg.task:run", "limit": "9"},
        source="cli",
    ) == {"task_id": "pkg.task:run", "limit": 9}


def test_required_one_of_uses_single_non_none_default_only_without_explicit_branch() -> None:
    spec_params = {
        "name": ParamSpec(type="str", required=False),
        "alias": ParamSpec(type="str", required=False, has_default=True, default="fallback"),
        "mode": ParamSpec(type="str", required=False, has_default=True, default="safe"),
    }

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={},
        required_one_of=(("name", "alias"),),
    ) == {"alias": "fallback", "mode": "safe"}

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={"name": "explicit"},
        required_one_of=(("name", "alias"),),
    ) == {"name": "explicit", "mode": "safe"}


def test_repeated_defaults_normalize_scalar_and_sequence_values() -> None:
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "tag": ParamSpec(type="str", required=False, repeated=True, has_default=True, default="one"),
        },
        raw_params={},
    ) == {"tag": ["one"]}

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "tag": ParamSpec(type="str", required=False, repeated=True, has_default=True, default=("one", "two")),
        },
        raw_params={},
    ) == {"tag": ["one", "two"]}


def test_repeated_default_choices_are_checked_per_item() -> None:
    with pytest.raises(ValueError, match="unsupported repeated value\\(s\\): 'bad'"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={
                "tag": ParamSpec(
                    type="str",
                    required=False,
                    repeated=True,
                    has_default=True,
                    default=("good", "bad"),
                    choices=("good",),
                ),
            },
            raw_params={},
        )


def test_non_repeated_list_choices_validate_whole_values() -> None:
    spec_params = {
        "items": ParamSpec(
            type="list",
            required=False,
            has_default=True,
            default=[1, 2],
            choices=([1, 2],),
        )
    }

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={},
    ) == {"items": [1, 2]}
    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={"items": [1, 2]},
    ) == {"items": [1, 2]}

    with pytest.raises(ValueError, match="must be one of: \\[1, 2\\]"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params=spec_params,
            raw_params={"items": [1, 3]},
        )


def test_non_repeated_object_choices_validate_whole_values() -> None:
    spec_params = {"payload": ParamSpec(type="object", required=False, choices=([1, 2],))}

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={"payload": [1, 2]},
    ) == {"payload": [1, 2]}

    with pytest.raises(ValueError, match="must be one of: \\[1, 2\\]"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params=spec_params,
            raw_params={"payload": [1, 3]},
        )


def test_default_injection_returns_fresh_values_and_does_not_alias_schema_defaults() -> None:
    payload_default = {"items": []}
    contract = _compile_contract(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "payload": ParamSpec(type="object", required=False, has_default=True, default=payload_default),
            "tag": ParamSpec(type="str", required=False, repeated=True, has_default=True, default=("one",)),
        },
    )

    first = normalize_command_params(contract, {}).params
    first["payload"]["items"].append("mutated")
    first["tag"].append("mutated")
    second = normalize_command_params(contract, {}).params

    assert second == {"payload": {"items": []}, "tag": ["one"]}
    assert payload_default == {"items": []}
    assert contract.params["payload"].normalized_default == {"items": []}
    assert first["payload"] is not second["payload"]
    assert first["tag"] is not second["tag"]


def test_choice_compilation_does_not_alias_mutable_schema_choices() -> None:
    choice = [1, 2]
    contract = _compile_contract(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "items": ParamSpec(type="list", required=False, choices=(choice,)),
        },
    )

    choice.append(3)
    param = contract.params["items"]

    assert param.declared_choices == ([1, 2],)
    assert param.choices == ([1, 2],)
    assert normalize_command_params(contract, {"items": [1, 2]}).params == {"items": [1, 2]}
    with pytest.raises(CommandContractError, match="must be one of: \\[1, 2\\]"):
        normalize_command_params(contract, {"items": [1, 2, 3]})


def test_defaulted_xor_branch_returns_fresh_values() -> None:
    contract = _compile_contract(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "payload": ParamSpec(type="object", required=False, has_default=True, default={"items": []}),
            "name": ParamSpec(type="str", required=False),
        },
        required_one_of=(("payload", "name"),),
    )

    first = normalize_command_params(contract, {}).params
    first["payload"]["items"].append("mutated")
    second = normalize_command_params(contract, {}).params

    assert second == {"payload": {"items": []}}
    assert first["payload"] is not second["payload"]


def test_required_one_of_ignores_none_defaults_as_absent() -> None:
    spec_params = {
        "name": ParamSpec(type="str?", required=False, has_default=True, default=None),
        "alias": ParamSpec(type="str?", required=False, has_default=True, default=None),
    }

    with pytest.raises(ValueError, match="requires exactly one of: name, alias"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params=spec_params,
            raw_params={},
            required_one_of=(("name", "alias"),),
        )


def test_required_one_of_counts_explicit_nullable_none_as_supplied_branch() -> None:
    spec_params = {
        "name": ParamSpec(type="str?", required=False),
        "alias": ParamSpec(type="str?", required=False),
    }

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params=spec_params,
        raw_params={"name": None},
        required_one_of=(("name", "alias"),),
    ) == {"name": None}


def test_required_one_of_rejects_multiple_default_branches() -> None:
    spec_params = {
        "name": ParamSpec(type="str", required=False, has_default=True, default="primary"),
        "alias": ParamSpec(type="str", required=False, has_default=True, default="fallback"),
    }

    with pytest.raises(ValueError, match="multiple defaulted members: name, alias"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params=spec_params,
            raw_params={},
            required_one_of=(("name", "alias"),),
        )


def test_required_one_of_rejects_duplicate_members() -> None:
    with pytest.raises(ValueError, match="duplicate member 'task_id'"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="run",
            spec_params={"task_id": ParamSpec(type="str", required=False)},
            raw_params={"task_id": "pkg.task:run"},
            required_one_of=(("task_id", "task_id"),),
        )


def test_required_one_of_rejects_overlapping_groups_before_default_injection() -> None:
    with pytest.raises(ValueError, match="required_one_of groups must not overlap"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={
                "name": ParamSpec(type="str", required=False),
                "alias": ParamSpec(type="str", required=False, has_default=True, default="fallback"),
                "fallback": ParamSpec(type="str", required=False),
            },
            raw_params={"name": "explicit"},
            required_one_of=(("name", "alias"), ("alias", "fallback")),
        )


def test_cli_json_rejects_invalid_object_or_list_payloads() -> None:
    with pytest.raises(ValueError, match="expected valid JSON"):
        coerce_command_value(
            substrate_name="runtime",
            context="command 'run' parameter 'args'",
            expected_type="object",
            value="{bad json",
            source="cli",
        )

    with pytest.raises(ValueError, match="expected list, got dict"):
        coerce_command_value(
            substrate_name="runtime",
            context="command 'run' parameter 'items'",
            expected_type="list",
            value='{"not":"a list"}',
            source="cli",
        )


@pytest.mark.parametrize(
    ("type_name", "value"),
    [
        ("object", "NaN"),
        ("object", "Infinity"),
        ("object", "-Infinity"),
        ("list", "[NaN]"),
    ],
)
def test_cli_json_rejects_non_finite_constants(type_name: str, value: str) -> None:
    with pytest.raises(ValueError, match="expected valid JSON"):
        _normalize_command_mapping(
            substrate_name="runtime",
            command_name="configure",
            spec_params={"payload": ParamSpec(type=type_name, required=False)},
            raw_params={"payload": value},
            source="cli",
        )


def test_float_coercion_accepts_only_finite_values() -> None:
    assert (
        coerce_command_value(
            substrate_name="metrics",
            context="command 'set' parameter 'threshold'",
            expected_type="float",
            value="1.25",
            source="cli",
        )
        == 1.25
    )
    assert (
        coerce_command_value(
            substrate_name="metrics",
            context="command 'set' parameter 'threshold'",
            expected_type="float",
            value=2,
        )
        == 2.0
    )

    for value in ("nan", "inf", "-inf", math.inf):
        with pytest.raises(ValueError, match="expected float"):
            coerce_command_value(
                substrate_name="metrics",
                context="command 'set' parameter 'threshold'",
                expected_type="float",
                value=value,
                source="cli",
            )


def test_bytes_declared_param_decodes_typed_json() -> None:
    assert (
        coerce_command_value(
            substrate_name="filesystem",
            context="command 'write' parameter 'content'",
            expected_type="bytes",
            value={"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="},
            source="typed-json",
        )
        == b"hello"
    )
    assert _normalize_command_mapping(
        substrate_name="filesystem",
        command_name="write",
        spec_params={"content": ParamSpec(type="bytes")},
        raw_params={"content": {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="}},
        source="typed-json",
    ) == {"content": b"hello"}


def test_bytes_declared_param_rejects_implicit_string_encoding() -> None:
    with pytest.raises(ValueError, match="expected bytes, got str"):
        coerce_command_value(
            substrate_name="filesystem",
            context="command 'write' parameter 'content'",
            expected_type="bytes",
            value="hello",
            source="cli",
        )

    with pytest.raises(ValueError, match="expected bytes, got str"):
        coerce_command_value(
            substrate_name="filesystem",
            context="command 'write' parameter 'content'",
            expected_type="bytes",
            value="hello",
        )


def test_typed_json_bytes_envelope_requires_typed_json_source() -> None:
    with pytest.raises(ValueError, match="expected bytes, got dict"):
        _normalize_command_mapping(
            substrate_name="filesystem",
            command_name="write",
            spec_params={"content": ParamSpec(type="bytes")},
            raw_params={"content": {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="}},
        )


def test_typed_json_source_decodes_only_declared_bytes_values() -> None:
    typed_bytes = {"__type__": "bytes", "encoding": "base64", "data": "aGVsbG8="}

    assert _normalize_command_mapping(
        substrate_name="runtime",
        command_name="configure",
        spec_params={
            "content": ParamSpec(type="bytes"),
            "payload": ParamSpec(type="object", required=False),
            "items": ParamSpec(type="list", required=False),
        },
        raw_params={
            "content": typed_bytes,
            "payload": {"nested": typed_bytes},
            "items": [typed_bytes],
        },
        source="typed-json",
    ) == {
        "content": b"hello",
        "payload": {"nested": typed_bytes},
        "items": [typed_bytes],
    }


def test_malformed_typed_json_bytes_errors_use_command_error_surface() -> None:
    with pytest.raises(ValueError, match="invalid typed JSON bytes payload"):
        _normalize_command_mapping(
            substrate_name="filesystem",
            command_name="write",
            spec_params={"content": ParamSpec(type="bytes")},
            raw_params={"content": {"__type__": "bytes", "encoding": "base64", "data": "not base64!"}},
            source="typed-json",
        )
