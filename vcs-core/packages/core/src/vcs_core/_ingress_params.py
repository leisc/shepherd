"""Shared parameter contracts for framework-owned ingress families."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from vcs_core._command_values import (
    CommandValueError,
    CommandValueSource,
    coerce_ingress_choice_value,
    coerce_ingress_param_value,
    coerce_ingress_value,
    parse_command_type,
    validate_ingress_choice,
)
from vcs_core._errors import VcsCoreError
from vcs_core._substrate_driver import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


IngressValueSource = CommandValueSource


class IngressParamError(VcsCoreError, ValueError):
    """Raised when an ingress parameter declaration or payload is invalid."""


@dataclass(frozen=True)
class IngressParamContract:
    name: str
    type: str
    base_type: str
    nullable: bool
    required: bool
    user_required: bool
    required_after_defaults: bool
    description: str
    has_default: bool
    declared_default: object | None
    normalized_default: object | None
    choices: tuple[object, ...]
    declared_choices: tuple[object, ...]
    repeated: bool
    projectable: bool


@dataclass(frozen=True)
class NormalizedIngressParams:
    params: dict[str, object]
    supplied: frozenset[str]
    defaulted: frozenset[str]


def compile_ingress_params(
    *,
    owner_label: str,
    ingress_label: str,
    specs: Mapping[str, object],
    strict_schema: bool = True,
) -> Mapping[str, IngressParamContract]:
    params: dict[str, IngressParamContract] = {}
    for param_name, spec in specs.items():
        params[param_name] = _compile_param_contract(
            owner_label=owner_label,
            ingress_label=ingress_label,
            param_name=param_name,
            spec=spec,
            strict_schema=strict_schema,
        )
    return MappingProxyType(params)


def normalize_ingress_params(
    contracts: Mapping[str, IngressParamContract],
    raw_params: Mapping[str, object],
    *,
    source: IngressValueSource = "native",
    owner_label: str,
    ingress_label: str,
    allow_unknown: bool = False,
    skip_defaults: frozenset[str] = frozenset(),
) -> NormalizedIngressParams:
    unknown = sorted(set(raw_params) - set(contracts))
    if unknown and not allow_unknown:
        raise IngressParamError(f"{owner_label} {ingress_label} got unknown parameter(s): {', '.join(unknown)}.")

    supplied = frozenset(raw_params)
    params: dict[str, object] = {}
    for name, value in raw_params.items():
        if name not in contracts:
            params[name] = value
            continue
        param = contracts[name]
        try:
            coerced_value = coerce_ingress_param_value(
                owner_label=owner_label,
                ingress_label=ingress_label,
                param_name=name,
                expected_type=param.type,
                repeated=param.repeated,
                value=value,
                source=source,
            )
            validate_ingress_choice(
                owner_label=owner_label,
                ingress_label=ingress_label,
                param_name=name,
                value=coerced_value,
                choices=param.choices,
                repeated=param.repeated,
            )
        except CommandValueError as exc:
            raise IngressParamError(str(exc)) from exc
        params[name] = coerced_value

    defaulted: set[str] = set()
    for name, param in contracts.items():
        if name in params or name in skip_defaults or not param.has_default:
            continue
        params[name] = _copy_invocation_default(owner_label, ingress_label, param)
        defaulted.add(name)

    missing = sorted(name for name, param in contracts.items() if param.user_required and name not in params)
    if missing:
        if len(missing) == 1:
            raise IngressParamError(f"{owner_label} {ingress_label} is missing required parameter '{missing[0]}'.")
        raise IngressParamError(f"{owner_label} {ingress_label} is missing required parameters: {', '.join(missing)}.")

    return NormalizedIngressParams(params=params, supplied=supplied, defaulted=frozenset(defaulted))


def _compile_param_contract(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: object,
    spec: object,
    strict_schema: bool,
) -> IngressParamContract:
    if not isinstance(param_name, str) or not param_name:
        raise IngressParamError(f"{owner_label} {ingress_label} has a parameter with an invalid name.")
    if strict_schema and not isinstance(spec, ParamSpec):
        raise IngressParamError(f"{owner_label} {ingress_label} parameter '{param_name}' must be a ParamSpec.")
    type_name = getattr(spec, "type", None)
    if not isinstance(type_name, str) or not type_name:
        raise IngressParamError(f"{owner_label} {ingress_label} parameter '{param_name}' must declare a type.")
    try:
        param_base_type, nullable = parse_command_type(type_name)
    except CommandValueError as exc:
        raise IngressParamError(f"{owner_label} {ingress_label} parameter '{param_name}' has {exc}") from exc
    description = getattr(spec, "description", "")
    if not isinstance(description, str):
        raise IngressParamError(f"{owner_label} {ingress_label} parameter '{param_name}' description must be a string.")
    required = getattr(spec, "required", True)
    has_default = getattr(spec, "has_default", False)
    repeated = getattr(spec, "repeated", False)
    projectable = getattr(spec, "projectable", True)
    _validate_bool(owner_label, ingress_label, f"parameter '{param_name}' required", required)
    _validate_bool(owner_label, ingress_label, f"parameter '{param_name}' has_default", has_default)
    _validate_bool(owner_label, ingress_label, f"parameter '{param_name}' repeated", repeated)
    _validate_bool(owner_label, ingress_label, f"parameter '{param_name}' projectable", projectable)

    raw_choices = getattr(spec, "choices", ())
    if strict_schema and not isinstance(raw_choices, tuple):
        raise IngressParamError(f"{owner_label} {ingress_label} parameter '{param_name}' choices must be a tuple.")
    raw_declared_choices = tuple(raw_choices) if isinstance(raw_choices, (tuple, list)) else (raw_choices,)
    declared_choices = tuple(
        _copy_declared_value(owner_label, ingress_label, param_name, "choice", choice)
        for choice in raw_declared_choices
    )
    try:
        choices = tuple(
            _copy_declared_value(
                owner_label,
                ingress_label,
                param_name,
                "normalized choice",
                coerce_ingress_choice_value(
                    owner_label=owner_label,
                    ingress_label=ingress_label,
                    param_name=param_name,
                    expected_type=type_name,
                    value=choice,
                ),
            )
            for choice in declared_choices
        )
    except CommandValueError as exc:
        raise IngressParamError(str(exc)) from exc

    raw_default = getattr(spec, "default", None)
    declared_default = (
        _copy_declared_value(owner_label, ingress_label, param_name, "default", raw_default) if has_default else None
    )
    normalized_default = None
    if has_default:
        try:
            normalized_default = _coerce_declared_default(
                owner_label=owner_label,
                ingress_label=ingress_label,
                param_name=param_name,
                type_name=type_name,
                repeated=repeated,
                value=raw_default,
            )
            normalized_default = _copy_declared_value(
                owner_label,
                ingress_label,
                param_name,
                "normalized default",
                normalized_default,
            )
            validate_ingress_choice(
                owner_label=owner_label,
                ingress_label=ingress_label,
                param_name=param_name,
                value=normalized_default,
                choices=choices,
                repeated=repeated,
            )
        except CommandValueError as exc:
            raise IngressParamError(str(exc)) from exc

    user_required = required and not has_default
    return IngressParamContract(
        name=param_name,
        type=type_name,
        base_type=param_base_type,
        nullable=nullable,
        required=required,
        user_required=user_required,
        required_after_defaults=user_required,
        description=description,
        has_default=has_default,
        declared_default=declared_default,
        normalized_default=normalized_default,
        choices=choices,
        declared_choices=declared_choices,
        repeated=repeated,
        projectable=projectable,
    )


def _coerce_declared_default(
    *,
    owner_label: str,
    ingress_label: str,
    param_name: str,
    type_name: str,
    repeated: bool,
    value: object,
) -> object:
    if repeated:
        if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
            values = (value,)
        else:
            values = tuple(value)
        return [
            coerce_ingress_value(
                owner_label=owner_label,
                context=f"{ingress_label} parameter '{param_name}' default",
                expected_type=type_name,
                value=item,
            )
            for item in values
        ]
    return coerce_ingress_value(
        owner_label=owner_label,
        context=f"{ingress_label} parameter '{param_name}' default",
        expected_type=type_name,
        value=value,
    )


def _copy_declared_value(
    owner_label: str,
    ingress_label: str,
    param_name: str,
    label: str,
    value: object,
) -> object:
    try:
        return deepcopy(value)
    except Exception as exc:
        raise IngressParamError(
            f"{owner_label} {ingress_label} parameter '{param_name}' {label} cannot be copied: {exc}"
        ) from exc


def _copy_invocation_default(owner_label: str, ingress_label: str, param: IngressParamContract) -> object:
    try:
        return deepcopy(param.normalized_default)
    except Exception as exc:
        raise IngressParamError(
            f"{owner_label} {ingress_label} parameter '{param.name}' default cannot be copied: {exc}"
        ) from exc


def _validate_bool(owner_label: str, ingress_label: str, label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise IngressParamError(f"{owner_label} {ingress_label} {label} must be a bool.")
