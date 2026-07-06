"""Validated runtime binding contracts for driver-capable paths."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from vcs_core._binding_surface import BindingSurface, BindingSurfaceRecord
from vcs_core._command_contract import CommandContract, compile_command_contract
from vcs_core._driver_schema_validation import DriverSchemaValidationError, validate_driver_schema
from vcs_core._errors import VcsCoreError
from vcs_core._ingress_params import IngressParamContract, IngressParamError, compile_ingress_params
from vcs_core.spi import DriverSchema, SubstrateDriver

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vcs_core.discovery import BindingSpec
    from vcs_core.types import BoundSubstrate


class BindingContractError(VcsCoreError, ValueError):
    """Raised when a live binding cannot form a valid driver contract."""


@dataclass(frozen=True)
class BindingIdentity:
    binding_name: str
    substrate_type: str
    implementation_kind: Literal["driver"]
    source: str


@dataclass(frozen=True)
class ResolvedDriverBinding:
    identity: BindingIdentity
    record: BindingSurfaceRecord
    bound: BoundSubstrate
    driver: SubstrateDriver
    schema: DriverSchema
    command_contracts: Mapping[str, CommandContract]
    scan_contracts: Mapping[str, NamedIngressContract]
    merge_contracts: Mapping[str, NamedIngressContract]


@dataclass(frozen=True)
class NamedIngressContract:
    name: str
    params: Mapping[str, IngressParamContract]


class BindingContractResolver:
    """Resolve live bindings into validated driver contracts."""

    def __init__(
        self,
        *,
        specs: Iterable[BindingSpec] = (),
        live_bindings: Iterable[BoundSubstrate] = (),
    ) -> None:
        self._live_by_name = {binding.binding_name: binding for binding in live_bindings}
        self._surface = BindingSurface(specs=specs, live_bindings=self._live_by_name.values())
        self._resolved_cache: dict[str, ResolvedDriverBinding] = {}

    def records(self) -> tuple[BindingSurfaceRecord, ...]:
        return self._surface.records()

    def names(self) -> tuple[str, ...]:
        return self._surface.names()

    def get(self, name: str) -> BindingSurfaceRecord:
        return self._surface.get(name)

    def schema(self, name: str) -> DriverSchema:
        return self.resolve_driver(name).schema

    def resolve_driver(self, binding_name: str) -> ResolvedDriverBinding:
        cached = self._resolved_cache.get(binding_name)
        if cached is not None:
            return cached

        record = self._surface.get(binding_name)
        bound = self._live_by_name.get(binding_name)
        if bound is None:
            raise BindingContractError(
                f"Binding '{binding_name}' is not live; driver resolution requires a live binding."
            )
        if record.implementation_kind != "driver":
            raise BindingContractError(
                f"Binding '{binding_name}' is marked {record.implementation_kind!r}; driver resolution is driver-only."
            )
        driver = bound.instance
        if not isinstance(driver, SubstrateDriver):
            raise BindingContractError(f"Binding '{binding_name}' does not implement SubstrateDriver.")

        schema = driver.describe()
        try:
            validate_driver_schema(schema)
        except DriverSchemaValidationError as exc:
            raise BindingContractError(
                f"Binding '{binding_name}' ({bound.substrate_type}) has an invalid driver schema: {exc}"
            ) from exc
        driver_id = driver.driver_id
        if driver_id != bound.substrate_type:
            raise BindingContractError(
                f"Binding '{binding_name}' ({bound.substrate_type}) driver_id must match substrate type; "
                f"got {driver_id!r}."
            )
        if schema.driver_id != bound.substrate_type:
            raise BindingContractError(
                f"Binding '{binding_name}' ({bound.substrate_type}) schema driver_id must match substrate type; "
                f"got {schema.driver_id!r}."
            )
        driver_version = driver.driver_version
        if schema.driver_version != driver_version:
            raise BindingContractError(
                f"Binding '{binding_name}' ({bound.substrate_type}) schema driver_version must match live driver "
                f"driver_version; schema has {schema.driver_version!r}, driver has {driver_version!r}."
            )
        driver_capabilities = driver.capabilities
        if schema.capabilities != driver_capabilities:
            raise BindingContractError(
                f"Binding '{binding_name}' ({bound.substrate_type}) schema capabilities must match live driver "
                f"capabilities; schema has {schema.capabilities!r}, driver has {driver_capabilities!r}."
            )

        contracts = {
            contract.command_name: contract
            for contract in compile_all_command_contracts_for_binding(schema, binding_name=binding_name)
        }
        scan_contracts = {
            contract.name: contract
            for contract in compile_named_ingress_contracts_for_binding(
                schema,
                binding_name=binding_name,
                ingress_kind="scan",
            )
        }
        merge_contracts = {
            contract.name: contract
            for contract in compile_named_ingress_contracts_for_binding(
                schema,
                binding_name=binding_name,
                ingress_kind="merge",
            )
        }
        resolved = ResolvedDriverBinding(
            identity=BindingIdentity(
                binding_name=binding_name,
                substrate_type=bound.substrate_type,
                implementation_kind="driver",
                source=record.binding_source,
            ),
            record=record,
            bound=bound,
            driver=driver,
            schema=schema,
            command_contracts=MappingProxyType(contracts),
            scan_contracts=MappingProxyType(scan_contracts),
            merge_contracts=MappingProxyType(merge_contracts),
        )
        self._resolved_cache[binding_name] = resolved
        return resolved


def compile_all_command_contracts_for_binding(
    schema: DriverSchema,
    *,
    binding_name: str,
) -> tuple[CommandContract, ...]:
    return tuple(
        compile_command_contract(schema, command_name, binding_name=binding_name) for command_name in schema.commands
    )


def compile_named_ingress_contracts_for_binding(
    schema: DriverSchema,
    *,
    binding_name: str,
    ingress_kind: Literal["scan", "merge"],
) -> tuple[NamedIngressContract, ...]:
    specs = schema.scans if ingress_kind == "scan" else schema.merges
    contracts: list[NamedIngressContract] = []
    for name, spec in specs.items():
        try:
            params = compile_ingress_params(
                owner_label=f"Driver '{schema.driver_id}'",
                ingress_label=f"{ingress_kind} '{name}'",
                specs=spec.params,
                strict_schema=True,
            )
        except IngressParamError as exc:
            raise BindingContractError(
                f"Binding '{binding_name}' ({schema.driver_id}) has an invalid {ingress_kind} contract: {exc}"
            ) from exc
        contracts.append(NamedIngressContract(name=name, params=params))
    return tuple(contracts)
