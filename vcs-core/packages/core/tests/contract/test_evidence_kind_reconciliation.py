# under-test: vcs_core._substrate_driver
"""Evidence-kind reconciliation positive contract test (SPI v0.1 SP3.6).

The Q4 invariant — "every observation ``evidence_kind`` emitted by a
driver must appear in some ``describe().capture_adapters[*].evidence_kinds``
set, and follow the ``<mechanism>:<kind>`` convention" — is enforced at
validation time when an observation is lowered. This test catches the
*declarative* half before any observation ever ships.

Since SP-substrate-authoring (2026-06-12) the reconciliation check lives in
the exportable conformance kit (``vcs_core.spi.testing``, the ``evidence_kinds``
case); this file consumes it so built-ins and out-of-tree drivers share one
source of truth (``decisions.md`` ``substrate-conformance-kit``). The
inventory guard below documents which built-ins carry default adapters.

Both this and Phase A.2's ``test_capabilities_runtime_contract.py`` consume
the shared ``DRIVERS_UNDER_TEST`` fixture; adding a built-in there opts it
into both contract checks.
"""

from __future__ import annotations

import pytest
from vcs_core._substrate_driver import DriverContext, SubstrateDriver
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.spi.testing import conformance_cases

from tests.contract.conftest import DRIVERS_UNDER_TEST


def _evidence_kinds_case(driver: SubstrateDriver):
    """The kit's ``evidence_kinds`` conformance case for the driver."""
    cases = {case.id: case for case in conformance_cases(driver)}
    return cases["evidence_kinds"]


@pytest.mark.parametrize("driver_cls", DRIVERS_UNDER_TEST, ids=lambda cls: cls.__name__)
def test_driver_default_adapter_evidence_kinds_reconcile_with_describe(driver_cls: type) -> None:
    """For every driver-default CaptureAdapter, the adapter's evidence_kinds
    property must match the corresponding describe() row, and each kind must
    follow the mechanism-prefixed convention.

    Delegates to the conformance kit's reconciliation check (the single source
    of truth the kit also applies to out-of-tree drivers). Drivers with no
    default adapters pass trivially (nothing to reconcile).
    """
    _evidence_kinds_case(driver_cls()).run()


def test_drivers_with_no_default_adapters_pass_trivially() -> None:
    """Drivers that return empty capture_adapters(context) have nothing to
    reconcile. This guard confirms the empty-adapter case is intentional, not
    a drift indicator, and documents which built-in carries a default adapter.
    """
    drivers_with_no_default_adapters: list[str] = []
    drivers_with_default_adapters: list[str] = []
    for driver_cls in DRIVERS_UNDER_TEST:
        driver = driver_cls()  # type: ignore[call-arg]
        context = DriverContext(
            operation_id="op_inventory_check",
            binding=driver.binding,  # type: ignore[attr-defined]
            role=driver.role,  # type: ignore[attr-defined]
            store_identity=SubstrateStoreIdentity(
                store_id=driver.store_id,  # type: ignore[attr-defined]
                kind=f"test.{driver.binding}",  # type: ignore[attr-defined]
                resource_id=f"{driver.binding}:test",  # type: ignore[attr-defined]
            ),
        )
        adapters = driver.capture_adapters(context)
        bucket = drivers_with_default_adapters if adapters else drivers_with_no_default_adapters
        bucket.append(driver_cls.__name__)
    # Workspace is the only driver expected to have a default adapter
    # (OverlayCaptureAdapter) in v0.1. Session/trace/world-ref are command-only
    # state substrates without driver-default adapters. Cross-cutting adapters
    # (PythonRuntimeCaptureAdapter) live on the CaptureAdapterRegistry.
    expected_with_adapters = {"WorkspaceSubstrateDriver"}
    expected_without_adapters = {
        "SessionStateSubstrateDriver",
        "TaskTraceSubstrateDriver",
        "WorldRefSubstrateDriver",
    }
    assert set(drivers_with_default_adapters) == expected_with_adapters, (
        f"drivers with default capture adapters {set(drivers_with_default_adapters)} "
        f"drifted from expected {expected_with_adapters}"
    )
    assert set(drivers_with_no_default_adapters) == expected_without_adapters, (
        f"drivers without default capture adapters {set(drivers_with_no_default_adapters)} "
        f"drifted from expected {expected_without_adapters}"
    )
