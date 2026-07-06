# under-test: vcs_core._driver_schema_validation
"""Tests for driver schema validation and projectability classification."""

from __future__ import annotations

import pytest
from vcs_core._command_projection import projectable_command_names
from vcs_core._driver_schema_validation import (
    DriverSchemaValidationError,
    is_projectable_command,
    validate_driver_schema,
    validate_projectable_command,
)
from vcs_core._execution_capability import NON_REVERSIBLE_RUN_FLAG
from vcs_core.spi import CapabilitySet, CommandRequest, CommandSpec, DriverSchema, MergeSpec, ParamSpec, ScanSpec


def _schema(
    *,
    commands: dict[str, CommandSpec],
    scans: dict[str, ScanSpec] | None = None,
    merges: dict[str, MergeSpec] | None = None,
    selectable: bool = False,
) -> DriverSchema:
    return DriverSchema(
        driver_id="test.driver",
        driver_version="v1",
        capabilities=CapabilitySet(accepts=frozenset({CommandRequest}), selectable=selectable),
        commands=commands,
        scans=scans or {},
        merges=merges or {},
    )


def test_validate_driver_schema_accepts_canonical_projection_metadata() -> None:
    command = CommandSpec(
        description="Run",
        params={
            "task_body": ParamSpec(type="callable", required=False, projectable=False),
            "task_id": ParamSpec(
                type="str",
                required=False,
                has_default=True,
                default="pkg.module:task",
                choices=("pkg.module:task", "pkg.other:task"),
            ),
        },
        required_one_of=(("task_body", "task_id"),),
    )

    validate_driver_schema(_schema(commands={"run": command}))

    assert command.required_one_of == (("task_body", "task_id"),)
    assert command.params["task_id"].choices == ("pkg.module:task", "pkg.other:task")


def test_validate_driver_schema_rejects_list_choices_metadata() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_id": ParamSpec(
                        type="str",
                        required=False,
                        choices=["pkg.module:task", "pkg.other:task"],  # type: ignore[arg-type]
                    ),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="parameter 'task_id' choices must be a tuple"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_list_required_one_of_metadata() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                    "task_id": ParamSpec(type="str", required=False),
                },
                required_one_of=[["task_body", "task_id"]],  # type: ignore[arg-type]
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="required_one_of must be a tuple of tuples"):
        validate_driver_schema(schema)


def test_validate_driver_schema_accepts_param_named_like_execution_option() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={NON_REVERSIBLE_RUN_FLAG: ParamSpec(type="bool", required=False)},
            )
        }
    )

    validate_driver_schema(schema)


def test_validate_projectable_command_hides_optional_python_params() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                    "task_id": ParamSpec(type="str", required=False),
                    "provider": ParamSpec(type="ExecutionProvider", required=False, projectable=False),
                },
                required_one_of=(("task_body", "task_id"),),
            )
        }
    )

    result = validate_projectable_command(schema, "run")

    assert result.projectable is True
    assert result.projectable_params == ("task_id",)
    assert {param.param_name for param in result.hidden_params} == {"task_body", "provider"}
    assert is_projectable_command(schema, "run") is True


def test_validate_projectable_command_downgrades_required_nonprojectable_param() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"callback": ParamSpec(type="callable", required=True, projectable=False)},
            )
        }
    )

    result = validate_projectable_command(schema, "run")

    assert result.projectable is False
    assert result.command_reasons == ("required-param-hidden:callback",)


def test_validate_projectable_command_downgrades_all_hidden_required_one_of_group() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                    "task_ref": ParamSpec(type="TaskRef", required=False, projectable=False),
                },
                required_one_of=(("task_body", "task_ref"),),
            )
        }
    )

    result = validate_projectable_command(schema, "run")

    assert result.projectable is False
    assert result.command_reasons == ("required-one-of-hidden:task_body,task_ref",)


def test_validate_projectable_command_rejects_projectable_python_type() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"provider": ParamSpec(type="ExecutionProvider", required=False)},
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="unsupported type 'ExecutionProvider'"):
        validate_projectable_command(schema, "run")


def test_validate_projectable_command_rejects_reserved_option_names() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"scope": ParamSpec(type="str", required=False)},
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="reserved CLI option"):
        validate_projectable_command(schema, "run")


def test_validate_projectable_command_rejects_projected_option_name_collisions() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "foo_bar": ParamSpec(type="str", required=False),
                    "foo-bar": ParamSpec(type="str", required=False),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="projected CLI option '--foo-bar' collides"):
        validate_projectable_command(schema, "run")


def test_validate_driver_schema_rejects_bad_default_through_coercer() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "limit": ParamSpec(type="int", required=False, has_default=True, default="abc"),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="parameter 'limit' default expected int"):
        validate_driver_schema(schema)


def test_validate_driver_schema_accepts_nullable_none_defaults() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_id": ParamSpec(type="str?", required=False, has_default=True, default=None),
                    "limit": ParamSpec(type="int?", required=False, has_default=True, default=None),
                    "payload": ParamSpec(type="object?", required=False, has_default=True, default=None),
                    "task_ref": ParamSpec(
                        type="TaskRef?",
                        required=False,
                        has_default=True,
                        default=None,
                        projectable=False,
                    ),
                },
            )
        }
    )

    validate_driver_schema(schema)


@pytest.mark.parametrize("type_name", [pytest.param("?"), pytest.param("str??"), pytest.param("str?extra")])
def test_validate_driver_schema_rejects_malformed_type_declarations(type_name: str) -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"name": ParamSpec(type=type_name, required=False)},
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="invalid command type declaration"):
        validate_driver_schema(schema)


def test_validate_driver_schema_accepts_custom_nonprojectable_type_names() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "provider": ParamSpec(type="ExecutionProvider", required=False, projectable=False),
                    "task_ref": ParamSpec(
                        type="TaskRef?",
                        required=False,
                        projectable=False,
                        has_default=True,
                        default=None,
                    ),
                },
            )
        }
    )

    validate_driver_schema(schema)


@pytest.mark.parametrize("type_name", ["str", "object", "TaskRef"])
def test_validate_driver_schema_rejects_nonnullable_none_defaults(type_name: str) -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_id": ParamSpec(
                        type=type_name,
                        required=False,
                        has_default=True,
                        default=None,
                        projectable=False,
                    ),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match=rf"parameter 'task_id' default expected {type_name}"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_bad_choice_through_coercer() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "mode": ParamSpec(type="str", required=False, choices=("fast", 3)),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="parameter 'mode' choice expected str"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_default_outside_choices() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "mode": ParamSpec(type="str", required=False, has_default=True, default="fast", choices=("safe",)),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="parameter 'mode' must be one of: 'safe'"):
        validate_driver_schema(schema)


def test_strict_schema_validation_does_not_hide_invalid_commands_as_unprojectable() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "mode": ParamSpec(type="str", required=False, has_default=True, default="fast", choices=("safe",)),
                },
            )
        }
    )

    assert projectable_command_names(schema) == ()
    with pytest.raises(DriverSchemaValidationError, match="parameter 'mode' must be one of: 'safe'"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_invalid_scan_param_declarations() -> None:
    schema = _schema(
        commands={},
        scans={
            "workspace-scan": ScanSpec(
                description="Scan",
                params={"payload": ParamSpec(type="str", required=False, choices=(object(),))},
            )
        },
    )

    with pytest.raises(DriverSchemaValidationError, match="scan 'workspace-scan' parameter 'payload' choice"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_invalid_merge_param_declarations() -> None:
    schema = _schema(
        commands={},
        merges={
            "workspace-overlay-merge": MergeSpec(
                description="Merge",
                params={"other_head": ParamSpec(type="str", required=False, has_default=True, default=None)},
            )
        },
    )

    with pytest.raises(DriverSchemaValidationError, match="merge 'workspace-overlay-merge' parameter 'other_head'"):
        validate_driver_schema(schema)


def test_scan_and_merge_specs_do_not_accept_command_required_one_of() -> None:
    scan = ScanSpec(description="Scan", params={"payload": ParamSpec(type="object")})
    merge = MergeSpec(description="Merge", params={"payload": ParamSpec(type="object")})

    assert not hasattr(scan, "required_one_of")
    assert not hasattr(merge, "required_one_of")
    validate_driver_schema(_schema(commands={}, scans={"workspace-scan": scan}, merges={"overlay": merge}))


def test_validate_driver_schema_rejects_uncopyable_command_defaults() -> None:
    class UncopyableDefault:
        def __deepcopy__(self, memo: object) -> object:
            del memo
            raise TypeError("not copyable")

    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "payload": ParamSpec(
                        type="object",
                        required=False,
                        has_default=True,
                        default=UncopyableDefault(),
                    ),
                },
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="default cannot be copied"):
        validate_driver_schema(schema)


def test_defaulted_required_param_is_not_user_required_for_projection() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "mode": ParamSpec(type="str", required=True, has_default=True, default="safe"),
                },
            )
        }
    )

    validate_driver_schema(schema)
    result = validate_projectable_command(schema, "run")

    assert result.projectable is True
    assert result.command_reasons == ()


def test_validate_projectable_command_classifies_selectable_driver_commands_nonprojectable() -> None:
    schema = _schema(
        selectable=True,
        commands={
            "append": CommandSpec(
                description="Append",
                params={"payload": ParamSpec(type="object")},
            )
        },
    )

    result = validate_projectable_command(schema, "append")

    assert result.projectable is False
    assert result.command_reasons == ("selectable-driver",)


def test_validate_driver_schema_rejects_required_one_of_unknown_member() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"task_id": ParamSpec(type="str", required=False)},
                required_one_of=(("task_id", "task_body"),),
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="unknown parameter 'task_body'"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_required_one_of_individually_required_member() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "task_id": ParamSpec(type="str", required=True),
                    "task_body": ParamSpec(type="callable", required=False, projectable=False),
                },
                required_one_of=(("task_id", "task_body"),),
            )
        }
    )

    with pytest.raises(
        DriverSchemaValidationError,
        match="required_one_of member 'task_id' must not also be individually required",
    ):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_required_one_of_duplicate_members() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={"task_id": ParamSpec(type="str", required=False)},
                required_one_of=(("task_id", "task_id"),),
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="duplicate member 'task_id'"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_overlapping_required_one_of_groups() -> None:
    schema = _schema(
        commands={
            "run": CommandSpec(
                description="Run",
                params={
                    "left": ParamSpec(type="str", required=False),
                    "middle": ParamSpec(type="str", required=False),
                    "right": ParamSpec(type="str", required=False),
                },
                required_one_of=(("left", "middle"), ("middle", "right")),
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="required_one_of groups must not overlap"):
        validate_driver_schema(schema)


def test_validate_driver_schema_rejects_required_one_of_multiple_non_none_defaults() -> None:
    schema = _schema(
        commands={
            "configure": CommandSpec(
                description="Configure",
                params={
                    "name": ParamSpec(type="str", required=False, has_default=True, default="primary"),
                    "alias": ParamSpec(type="str", required=False, has_default=True, default="fallback"),
                },
                required_one_of=(("name", "alias"),),
            )
        }
    )

    with pytest.raises(DriverSchemaValidationError, match="multiple non-None defaults: name, alias"):
        validate_driver_schema(schema)
