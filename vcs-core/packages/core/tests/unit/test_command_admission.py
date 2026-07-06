# under-test: vcs_core._command_admission
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from vcs_core._command_admission import CommandAdmissionError, admit_command_invocation
from vcs_core.types import ScopeInfo

_SCOPE = ScopeInfo(
    name="task",
    ref="refs/heads/task",
    instance_id="task-1",
    creation_oid="root",
)


@dataclass
class _ValidatingSubstrate:
    error: Exception | None = None
    seen_params: list[Mapping[str, Any]] = field(default_factory=list)

    def validate_command_invocation(
        self,
        command: str,
        scope: ScopeInfo,
        *,
        params: Mapping[str, Any],
    ) -> None:
        del command, scope
        self.seen_params.append(params)
        if self.error is not None:
            raise self.error


class _MutatingSubstrate:
    def validate_command_invocation(
        self,
        command: str,
        scope: ScopeInfo,
        *,
        params: Mapping[str, Any],
    ) -> None:
        del command, scope
        params["limit"] = 9  # type: ignore[index]


class _MutatingNestedSubstrate:
    def validate_command_invocation(
        self,
        command: str,
        scope: ScopeInfo,
        *,
        params: Mapping[str, Any],
    ) -> None:
        del command, scope
        params["payload"]["approved"] = False  # type: ignore[index]


def test_admit_command_invocation_is_noop_without_provider_hook() -> None:
    admit_command_invocation(object(), "inspect", _SCOPE, params={"limit": 7})


def test_admit_command_invocation_passes_immutable_param_snapshot() -> None:
    substrate = _ValidatingSubstrate()
    params = {"limit": 7}

    admit_command_invocation(substrate, "inspect", _SCOPE, params=params)
    params["limit"] = 9

    assert dict(substrate.seen_params[0]) == {"limit": 7}


def test_admit_command_invocation_rejects_provider_mutation_of_top_level_params() -> None:
    with pytest.raises(TypeError):
        admit_command_invocation(_MutatingSubstrate(), "inspect", _SCOPE, params={"limit": 7})


def test_admit_command_invocation_rejects_provider_mutation_of_nested_params() -> None:
    params = {"payload": {"approved": True}}

    with pytest.raises(TypeError):
        admit_command_invocation(_MutatingNestedSubstrate(), "inspect", _SCOPE, params=params)

    assert params == {"payload": {"approved": True}}


def test_admit_command_invocation_preserves_named_error() -> None:
    substrate = _ValidatingSubstrate(error=CommandAdmissionError("unsafe invocation"))

    with pytest.raises(CommandAdmissionError, match="unsafe invocation"):
        admit_command_invocation(substrate, "inspect", _SCOPE, params={})
    with pytest.raises(ValueError, match="unsafe invocation"):
        admit_command_invocation(substrate, "inspect", _SCOPE, params={})


def test_admit_command_invocation_wraps_plain_value_error() -> None:
    substrate = _ValidatingSubstrate(error=ValueError("unsafe invocation"))

    with pytest.raises(CommandAdmissionError, match="unsafe invocation"):
        admit_command_invocation(substrate, "inspect", _SCOPE, params={})


def test_admit_command_invocation_propagates_unrelated_bug() -> None:
    substrate = _ValidatingSubstrate(error=KeyError("missing-key"))

    with pytest.raises(KeyError, match="missing-key"):
        admit_command_invocation(substrate, "inspect", _SCOPE, params={})
