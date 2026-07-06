# under-test: vcs_core._operation_journal_controller
"""Focused tests for the extracted OperationJournalController (V2.1).

The journal state machine is exercised end-to-end through the WSM shims by
test_world_operation_journal.py; this file pins the two properties the extraction
exists to create — the controller is a real collaborator, and it holds no
back-reference to the manager — plus a direct round-trip that does not route
through a shim.
"""

from __future__ import annotations

import pytest
from vcs_core._operation_journal_controller import OperationJournalController
from vcs_core.spi import SubstrateStoreIdentity
from vcs_core.testing import SubstrateStoreSpec, WorldStorageManager


def _workspace_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="fs:repo-main")


@pytest.fixture
def manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(SubstrateStoreSpec(identity=_workspace_identity(), locator="substrates/workspace.git"),),
    )


def test_manager_owns_a_journal_controller(manager: WorldStorageManager) -> None:
    assert isinstance(manager._journal, OperationJournalController)


def test_controller_holds_no_direct_back_reference_to_the_manager(manager: WorldStorageManager) -> None:
    """The extraction's load-bearing property: no attribute of the controller *is* the
    manager — the ``owner: VcsCore`` anti-pattern the plan retires. (One injected callable
    remains transitionally manager-bound; that residual is pinned separately below.)"""
    controller = manager._journal
    for name, value in vars(controller).items():
        assert value is not manager, f"{name} is a direct back-reference to the manager"


def test_admission_validator_is_coordinator_bound_not_manager_bound(manager: WorldStorageManager) -> None:
    # The prepared-admission validator is the coordinator's method, cleanly off the manager.
    admission = manager._journal._validate_prepared_operation_admission
    assert admission.__self__ is manager._transition_coordinator
    assert admission.__self__ is not manager


def test_publication_validator_is_bound_to_the_pubret_controller_after_v2_2(manager: WorldStorageManager) -> None:
    """V2.2c flipped the V2.1 tripwire: the journal controller's publication-plan validator
    now binds to the PublicationRetentionController, not the manager — publication logic left
    WSM. If this ever rebinds to the manager, the residual coupling has silently returned."""
    publication = manager._journal._validate_publication_plan
    assert publication.__self__ is manager._pubret
    assert publication.__self__ is not manager


def test_controller_open_and_read_round_trip_without_a_shim(manager: WorldStorageManager) -> None:
    controller = manager._journal
    controller.open_operation_journal(
        operation_id="op-controller-direct",
        operation_kind="world_merge",
        target_ref="refs/vcscore/authority/main",
        input_world_oid=None,
    )
    history = controller.read_operation_journal("op-controller-direct")
    assert history.tip.payload["status"] == "opened"
    # The open-journal accelerator index reflects the new open ref.
    summaries = controller.list_operation_journals(family="open")
    assert any(summary.operation_id == "op-controller-direct" for summary in summaries)


def test_manager_shim_delegates_to_the_controller(manager: WorldStorageManager) -> None:
    """`manager.open_operation_journal` (shim) routes through the controller a monkeypatch on
    the manager attribute still intercepts (standing rule 9)."""
    manager.open_operation_journal(
        operation_id="op-shim-parity",
        operation_kind="world_merge",
        target_ref="refs/vcscore/authority/main",
        input_world_oid=None,
    )
    assert manager.read_operation_journal("op-shim-parity").tip.payload["status"] == "opened"
