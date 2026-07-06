# under-test: vcs_core._world_transition_coordinator
"""Tests for ActiveSurface profiles used by Shepherd integration."""

from __future__ import annotations

import pytest
from vcs_core._substrate_evidence_kinds import EvidenceKind
from vcs_core._world_substrate_adapters import WorkspaceSubstrateDriver
from vcs_core._world_transition_coordinator import _check_active_surface_post_dispatch
from vcs_core.spi import DriverIngressResult, ObservationDraft, SurfacePolicyError, TransitionDraft
from vcs_core.surface_profiles import (
    FILESYSTEM_WRITE_EVIDENCE_KINDS,
    WORKSPACE_MUTATION_SEMANTIC_OPS,
    ensure_session_capture_admitted,
    permissive_active_surface,
    read_only_filesystem_surface,
)


def test_read_only_filesystem_surface_is_deny_shaped() -> None:
    surface = read_only_filesystem_surface()

    assert surface.allow_request_types is None
    assert surface.allow_evidence_kinds is None
    assert surface.allow_semantic_ops is None
    assert surface.deny_request_types == frozenset()
    assert surface.deny_evidence_kinds == frozenset(FILESYSTEM_WRITE_EVIDENCE_KINDS)
    assert surface.deny_semantic_ops == frozenset(WORKSPACE_MUTATION_SEMANTIC_OPS)


def test_read_only_filesystem_surface_denies_python_runtime_write_observations() -> None:
    driver = WorkspaceSubstrateDriver()
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="write-1",
                evidence_kind=EvidenceKind.PYTHON_RUNTIME_WRITE,
                stable_observation={"path": "out.txt"},
            ),
        )
    )

    with pytest.raises(SurfacePolicyError, match="python-runtime:write"):
        _check_active_surface_post_dispatch(driver, read_only_filesystem_surface(), result)


def test_read_only_filesystem_surface_denies_workspace_mutation_transitions() -> None:
    driver = WorkspaceSubstrateDriver()
    result = DriverIngressResult(
        transitions=(
            TransitionDraft(
                transition_id="t-1",
                semantic_op="workspace-capture-reduction",
                payload={"schema": "test"},
                observation_ids=(),
            ),
        )
    )

    with pytest.raises(SurfacePolicyError, match="workspace-capture-reduction"):
        _check_active_surface_post_dispatch(driver, read_only_filesystem_surface(), result)


def test_permissive_active_surface_allows_write_observation_and_transition() -> None:
    driver = WorkspaceSubstrateDriver()
    result = DriverIngressResult(
        observations=(
            ObservationDraft(
                observation_id="write-1",
                evidence_kind=EvidenceKind.PYTHON_RUNTIME_WRITE,
                stable_observation={"path": "out.txt"},
            ),
        ),
        transitions=(
            TransitionDraft(
                transition_id="t-1",
                semantic_op="workspace-capture-reduction",
                payload={"schema": "test"},
                observation_ids=("write-1",),
            ),
        ),
    )

    _check_active_surface_post_dispatch(driver, permissive_active_surface(), result)


def test_session_capture_admission_refuses_under_read_only_surface() -> None:
    with pytest.raises(SurfacePolicyError, match="overlay:write-observed") as excinfo:
        ensure_session_capture_admitted(read_only_filesystem_surface())
    assert excinfo.value.operation == "session exec --capture"
    assert excinfo.value.driver_id == "shepherd.workspace_ref"


def test_session_capture_admission_allows_under_permissive_surface() -> None:
    # Permissive and None both admit (no exception).
    ensure_session_capture_admitted(permissive_active_surface())
    ensure_session_capture_admitted(None)


def test_session_capture_admission_uses_supplied_operation_label() -> None:
    with pytest.raises(SurfacePolicyError) as excinfo:
        ensure_session_capture_admitted(read_only_filesystem_surface(), operation="agent.run")
    assert excinfo.value.operation == "agent.run"
