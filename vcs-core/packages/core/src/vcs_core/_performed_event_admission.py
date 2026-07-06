"""Internal framework-owned admission for already-performed events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from vcs_core._errors import VcsCoreError
from vcs_core._immutable_payload import immutable_payload_view
from vcs_core._ingress_params import (
    IngressParamContract,
    IngressParamError,
    compile_ingress_params,
    normalize_ingress_params,
)
from vcs_core._substrate_runtime import PerformedEventProvider, PerformedEventSpec

if TYPE_CHECKING:
    from vcs_core.types import ScopeInfo


class PerformedEventAdmissionError(VcsCoreError, ValueError):
    """Named validation error for framework-routed performed-event admission."""


@dataclass(frozen=True)
class NormalizedPerformedEvent:
    params: Mapping[str, object]
    supplied: frozenset[str]
    defaulted: frozenset[str]
    effect_types: frozenset[str]


@dataclass(frozen=True)
class PerformedEventContract:
    provider_name: str
    event: str
    description: str
    params: Mapping[str, IngressParamContract]
    examples: tuple[str, ...]
    effect_types: frozenset[str]
    allow_unknown_params: bool


def compile_performed_event_contracts(provider: PerformedEventProvider) -> Mapping[str, PerformedEventContract]:
    """Compile and validate a provider's declared performed-event ingress contracts."""
    provider_name = _provider_name(provider)
    specs = provider.performed_event_specs()
    if not isinstance(specs, Mapping):
        raise PerformedEventAdmissionError(f"Provider '{provider_name}' performed_event_specs() must return a mapping.")

    contracts: dict[str, PerformedEventContract] = {}
    for event, spec in specs.items():
        if not isinstance(event, str) or not event:
            raise PerformedEventAdmissionError(
                f"Provider '{provider_name}' has a performed event with an invalid name."
            )
        contracts[event] = _compile_performed_event_contract(
            provider_name=provider_name,
            event=event,
            spec=spec,
        )
    return MappingProxyType(contracts)


def admit_performed_event(
    provider: PerformedEventProvider,
    event: str,
    scope: ScopeInfo,
    *,
    params: Mapping[str, Any],
) -> NormalizedPerformedEvent:
    provider_name = _provider_name(provider)
    contracts = compile_performed_event_contracts(provider)
    contract = contracts.get(event)
    if contract is None:
        available = ", ".join(sorted(contracts)) or "(none)"
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' has no performed event named {event!r}; available: {available}."
        )

    owner_label = f"Provider '{provider_name}'"
    ingress_label = f"event '{event}'"
    try:
        normalized = normalize_ingress_params(
            contract.params,
            params,
            owner_label=owner_label,
            ingress_label=ingress_label,
            allow_unknown=contract.allow_unknown_params,
        )
    except IngressParamError as exc:
        raise PerformedEventAdmissionError(str(exc)) from exc

    normalized_params = MappingProxyType(dict(normalized.params))
    validator = getattr(provider, "validate_performed_event", None)
    if validator is not None:
        try:
            validator(event, scope, params=immutable_payload_view(normalized.params))
        except PerformedEventAdmissionError:
            raise
        except ValueError as exc:
            raise PerformedEventAdmissionError(
                f"Provider '{provider_name}' performed event {event!r} failed validation: {exc}"
            ) from exc

    return NormalizedPerformedEvent(
        params=normalized_params,
        supplied=normalized.supplied,
        defaulted=normalized.defaulted,
        effect_types=contract.effect_types,
    )


def _provider_name(provider: PerformedEventProvider) -> str:
    provider_name = provider.name
    if not isinstance(provider_name, str) or not provider_name:
        raise PerformedEventAdmissionError("Performed-event provider name must be a non-empty string.")
    return provider_name


def _compile_performed_event_contract(
    *,
    provider_name: str,
    event: str,
    spec: object,
) -> PerformedEventContract:
    if type(spec) is not PerformedEventSpec:
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' performed event {event!r} must be a PerformedEventSpec."
        )
    if not isinstance(spec.description, str):
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' performed event {event!r} description must be a string."
        )
    if not isinstance(spec.params, Mapping):
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' performed event {event!r} params must be a mapping."
        )
    if type(spec.allow_unknown_params) is not bool:
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' performed event {event!r} allow_unknown_params must be a bool."
        )

    owner_label = f"Provider '{provider_name}'"
    ingress_label = f"event '{event}'"
    try:
        params = compile_ingress_params(
            owner_label=owner_label,
            ingress_label=ingress_label,
            specs=spec.params,
            strict_schema=True,
        )
    except IngressParamError as exc:
        raise PerformedEventAdmissionError(str(exc)) from exc

    return PerformedEventContract(
        provider_name=provider_name,
        event=event,
        description=spec.description,
        params=params,
        examples=_normalize_string_sequence(
            provider_name=provider_name,
            event=event,
            field_name="examples",
            value=spec.examples,
            allow_frozenset=False,
        ),
        effect_types=frozenset(
            _normalize_string_sequence(
                provider_name=provider_name,
                event=event,
                field_name="effect_types",
                value=spec.effect_types,
                allow_frozenset=True,
            )
        ),
        allow_unknown_params=spec.allow_unknown_params,
    )


def _normalize_string_sequence(
    *,
    provider_name: str,
    event: str,
    field_name: str,
    value: object,
    allow_frozenset: bool,
) -> tuple[str, ...]:
    del allow_frozenset
    if not isinstance(value, tuple):
        expected = "tuple"
        raise PerformedEventAdmissionError(
            f"Provider '{provider_name}' performed event {event!r} {field_name} must be a {expected} of strings."
        )
    raw_items = cast("tuple[object, ...]", value)
    normalized: list[str] = []
    for item in raw_items:
        if not isinstance(item, str) or not item:
            raise PerformedEventAdmissionError(
                f"Provider '{provider_name}' performed event {event!r} {field_name} entries must be non-empty strings."
            )
        normalized.append(item)
    return tuple(normalized)
