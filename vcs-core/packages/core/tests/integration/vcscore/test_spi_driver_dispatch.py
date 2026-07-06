"""PD3a: the ``mg exec`` → SPI ``prepare`` dispatch bridge.

An SPI driver bound into a ``VcsCore`` coordinator is dispatchable through the
same ``execute_recorded`` route built-in substrates use; the bridge routes to
the driver's ``prepare(ctx, CommandRequest)`` through the validated SPI entry
(capability acceptance + ingress validation) inside a recorded operation
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from vcs_core import VcsCore, build_builtin_substrate_context
from vcs_core.runtime_api import CommandRequest, DriverContext, DriverIngressResult
from vcs_core.spi import BaseSubstrateDriver, CapabilitySet, CommandSpec, DriverSchema, ParamSpec, TransitionDraft
from vcs_core.store import Store
from vcs_core.substrates import DeclarativeFilesystemSubstrate, MarkerSubstrate


@dataclass(frozen=True)
class _ProbeDriver(BaseSubstrateDriver):
    """Journal-only SPI driver with one command — the run-driver fingerprint."""

    store_id: str = "store_probe"
    binding: str = "probe"
    role: str = "test.ProbeDriver"
    driver_id: str = "test.probe_driver"
    driver_version: str = "v0.1"

    @property
    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            accepts=frozenset({CommandRequest}),
            selectable=False,
            materializable=False,
            journal_only=True,
        )

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            commands={
                "ping": CommandSpec(
                    description="Echo the message back through prepare.",
                    params={
                        "message": ParamSpec(type="str", required=True),
                        "extra": ParamSpec(type="str", required=False),
                    },
                ),
            },
        )

    def prepare(self, context: DriverContext, request: Any) -> DriverIngressResult:
        assert isinstance(request, CommandRequest)
        return DriverIngressResult(
            transitions=(
                TransitionDraft(
                    transition_id="primary",
                    semantic_op="execute",
                    payload={
                        "schema": "test/probe-ping/v0",
                        "reached": "prepare",
                        "command": request.command,
                        "message": request.params.get("message"),
                        "operation_id": context.operation_id,
                        "binding": context.binding,
                    },
                    observation_ids=(),
                    base_heads=context.base_heads,
                    materialization_class="noop",
                ),
            ),
        )


@pytest.fixture
def mg_with_driver(tmp_path: Path) -> VcsCore:
    root = tmp_path / "ws"
    root.mkdir()
    store = Store(str(root / ".vcscore"))
    ctx = build_builtin_substrate_context(store, workspace=root, config={})
    mg = VcsCore(
        str(root),
        substrates=[MarkerSubstrate(ctx), DeclarativeFilesystemSubstrate(ctx), _ProbeDriver()],
        store=store,
    )
    mg.activate()
    yield mg
    mg.deactivate()


def test_spi_driver_binds_and_dispatches_to_prepare(mg_with_driver: VcsCore) -> None:
    mg = mg_with_driver
    outcome = mg.execute_recorded("probe", "ping", scope=mg.ground, message="hello")
    result = outcome.value
    assert isinstance(result, DriverIngressResult)
    payload = result.transitions[0].payload
    assert payload["reached"] == "prepare"
    assert payload["message"] == "hello"
    assert payload["binding"] == "probe"
    assert payload["operation_id"]  # a real journaled operation id, not empty


def test_spi_driver_dispatch_is_a_recorded_operation(mg_with_driver: VcsCore) -> None:
    mg = mg_with_driver
    outcome = mg.execute_recorded("probe", "ping", scope=mg.ground, message="journaled")
    operation_id = outcome.value.transitions[0].payload["operation_id"]
    assert isinstance(operation_id, str)
    assert operation_id
    # The operation closed cleanly: nothing in-flight remains.
    assert mg._pipeline.current_operation() is None


def test_unknown_driver_command_rejected(mg_with_driver: VcsCore) -> None:
    with pytest.raises(ValueError, match="Unknown probe command"):
        mg_with_driver.execute_recorded("probe", "zap", scope=mg_with_driver.ground)


def test_unknown_param_rejected_naming_accepted(mg_with_driver: VcsCore) -> None:
    with pytest.raises(ValueError, match="mesage"):
        mg_with_driver.execute_recorded("probe", "ping", scope=mg_with_driver.ground, mesage="typo")


def test_missing_required_param_rejected(mg_with_driver: VcsCore) -> None:
    with pytest.raises(ValueError, match="message"):
        mg_with_driver.execute_recorded("probe", "ping", scope=mg_with_driver.ground)


def test_performed_keyword_is_not_a_command_dispatch_mode(mg_with_driver: VcsCore) -> None:
    with pytest.raises(ValueError, match="unknown parameter\\(s\\): performed"):
        mg_with_driver.execute_recorded("probe", "ping", scope=mg_with_driver.ground, performed=True, message="x")


def test_builtin_lifecycle_substrates_unaffected_beside_a_bound_driver(mg_with_driver: VcsCore) -> None:
    mg = mg_with_driver
    filesystem = next(s for s in mg.lifecycle_substrates if getattr(s, "name", None) == "filesystem")
    filesystem.record_changes([("note.txt", b"built-in path still works\n")], scope=mg.ground)
