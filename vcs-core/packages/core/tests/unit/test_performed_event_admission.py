# under-test: vcs_core._performed_event_admission
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from vcs_core._performed_event_admission import (
    PerformedEventAdmissionError,
    admit_performed_event,
    compile_performed_event_contracts,
)
from vcs_core._substrate_runtime import PerformedEventSpec
from vcs_core.spi import ParamSpec
from vcs_core.types import EffectRecord


@dataclass
class _Provider:
    specs: object
    name: str = "provider"

    def __post_init__(self) -> None:
        self.performed_calls = 0

    def performed_event_specs(self) -> object:
        return self.specs

    def performed_effects(self, event: str, scope: object, *, params: object) -> tuple[EffectRecord, ...]:
        del event, scope, params
        self.performed_calls += 1
        return ()


class _MutatingValidatorProvider(_Provider):
    def validate_performed_event(self, event: str, scope: object, *, params: object) -> None:
        del event, scope
        params["payload"]["approved"] = False  # type: ignore[index]


def test_compile_performed_event_contract_rejects_non_mapping_params() -> None:
    provider = _Provider({"inspect": PerformedEventSpec(params=["not", "a", "mapping"])})  # type: ignore[arg-type]

    with pytest.raises(PerformedEventAdmissionError, match="params must be a mapping"):
        compile_performed_event_contracts(provider)  # type: ignore[arg-type]


def test_compile_performed_event_contract_rejects_bare_string_effect_types() -> None:
    provider = _Provider({"inspect": PerformedEventSpec(effect_types="Marker")})  # type: ignore[arg-type]

    with pytest.raises(PerformedEventAdmissionError, match="effect_types must be a tuple"):
        compile_performed_event_contracts(provider)  # type: ignore[arg-type]


def test_compile_performed_event_contract_rejects_non_bool_allow_unknown_params() -> None:
    provider = _Provider({"inspect": PerformedEventSpec(allow_unknown_params="yes")})  # type: ignore[arg-type]

    with pytest.raises(PerformedEventAdmissionError, match="allow_unknown_params must be a bool"):
        compile_performed_event_contracts(provider)  # type: ignore[arg-type]


def test_compile_performed_event_contract_rejects_invalid_event_names() -> None:
    provider = _Provider({"": PerformedEventSpec()})

    with pytest.raises(PerformedEventAdmissionError, match="invalid name"):
        compile_performed_event_contracts(provider)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (PerformedEventSpec(description=123), "description must be a string"),  # type: ignore[arg-type]
        (PerformedEventSpec(examples="not-a-sequence"), "examples must be a tuple"),  # type: ignore[arg-type]
        (PerformedEventSpec(examples=["ok"]), "examples must be a tuple"),  # type: ignore[arg-type]
        (PerformedEventSpec(examples=("ok", 123)), "examples entries must be non-empty strings"),  # type: ignore[arg-type]
        (PerformedEventSpec(effect_types=["Marker"]), "effect_types must be a tuple"),  # type: ignore[arg-type]
        (PerformedEventSpec(effect_types=frozenset({"Marker"})), "effect_types must be a tuple"),  # type: ignore[arg-type]
        (PerformedEventSpec(effect_types=("Marker", "")), "effect_types entries must be non-empty strings"),
        ({"inspect": object()}, "must be a PerformedEventSpec"),
    ],
)
def test_compile_performed_event_contract_rejects_malformed_metadata(spec: Any, message: str) -> None:
    specs = spec if isinstance(spec, dict) else {"inspect": spec}
    provider = _Provider(specs)

    with pytest.raises(PerformedEventAdmissionError, match=message):
        compile_performed_event_contracts(provider)  # type: ignore[arg-type]


def test_admit_performed_event_honors_explicit_allow_unknown_params() -> None:
    marker = object()
    provider = _Provider({"inspect": PerformedEventSpec(allow_unknown_params=True, effect_types=("Marker",))})

    normalized = admit_performed_event(provider, "inspect", object(), params={"extra": marker})  # type: ignore[arg-type]

    assert normalized.params["extra"] is marker
    assert normalized.supplied == frozenset({"extra"})
    assert normalized.effect_types == frozenset({"Marker"})


def test_admit_performed_event_applies_defaults_through_contract() -> None:
    provider = _Provider(
        {
            "inspect": PerformedEventSpec(
                params={"label": ParamSpec(type="str", required=False, has_default=True, default="defaulted")}
            )
        }
    )

    normalized = admit_performed_event(provider, "inspect", object(), params={})  # type: ignore[arg-type]

    assert normalized.params == {"label": "defaulted"}
    assert normalized.defaulted == frozenset({"label"})


def test_admit_performed_event_rejects_validator_mutation_of_nested_params() -> None:
    provider = _MutatingValidatorProvider(
        {"inspect": PerformedEventSpec(params={"payload": ParamSpec(type="object", required=True)})}
    )
    params = {"payload": {"approved": True}}

    with pytest.raises(TypeError):
        admit_performed_event(provider, "inspect", object(), params=params)  # type: ignore[arg-type]

    assert params == {"payload": {"approved": True}}
