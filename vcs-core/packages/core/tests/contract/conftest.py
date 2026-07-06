"""Shared fixtures for SPI v0.1 contract tests.

``DRIVERS_UNDER_TEST`` is the canonical inventory of built-in
``SubstrateDriver`` implementations that contract tests must cover.
Adding a new built-in driver requires adding it here and confirming
the contract tests still pass. Both ``test_capabilities_runtime_contract.py``
(Phase A.2) and ``test_evidence_kind_reconciliation.py`` (SP3.6 / Phase D
companion) consume this tuple so the inventory has a single source of
truth.

The inventory-drift guard
(``test_drivers_under_test_covers_known_built_in_drivers``) lives next
to its primary consumer in ``test_capabilities_runtime_contract.py`` and
checks against the expected built-in set; the same check is implicitly
honored here because both test files import this tuple. If a new driver
is added, the guard test fails until ``DRIVERS_UNDER_TEST`` here is
updated AND the new driver name is added to the guard's expected set.

The *checks* those two files run now live in the exportable conformance kit
``vcs_core.spi.testing`` (``decisions.md`` ``substrate-conformance-kit``) — the
single source of truth shared with out-of-tree drivers. The contract files
here pair the kit's checks with this built-in inventory; the kit's own unit
tests (``tests/unit/test_spi_testing_kit.py``) prove it passes/fails correctly.
"""

from __future__ import annotations

from vcs_core._world_substrate_adapters import (
    SessionStateSubstrateDriver,
    WorkspaceSubstrateDriver,
    WorldRefSubstrateDriver,
)
from vcs_core.runtime_substrate import TaskTraceSubstrateDriver
from vcs_core.spi import SubstrateDriver

DRIVERS_UNDER_TEST: tuple[type[SubstrateDriver], ...] = (
    WorkspaceSubstrateDriver,
    SessionStateSubstrateDriver,
    TaskTraceSubstrateDriver,
    WorldRefSubstrateDriver,
)
