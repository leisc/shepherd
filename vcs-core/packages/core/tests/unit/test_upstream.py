# under-test: vcs_core._upstream
from __future__ import annotations

import pytest
from vcs_core._upstream import PendingSelector, PreflightResult, UpstreamBaseAvailability, UpstreamBasisState


def test_pending_selector_rejects_missing_stable_target() -> None:
    with pytest.raises(ValueError, match="Exactly one stable selector"):
        PendingSelector(target_id="sqlite:main")


def test_pending_selector_rejects_unit_id_and_frontier_together() -> None:
    with pytest.raises(ValueError, match="Exactly one stable selector"):
        PendingSelector(target_id="sqlite:main", unit_id="unit-1", frontier="abc123")


def test_pending_selector_accepts_frontier_with_scope_context() -> None:
    selector = PendingSelector(target_id="sqlite:main", frontier="abc123", scope_context="task")

    assert selector.frontier == "abc123"
    assert selector.unit_id is None


def test_upstream_basis_state_uses_explicit_exact_match_contract() -> None:
    state = UpstreamBasisState(
        substrate="sqlite",
        target_id="sqlite:main",
        basis_token="basis-1",
        last_observed_token="basis-1",
        local_frontier="frontier-1",
    )

    assert state.comparison == "exact"


def test_upstream_base_availability_requires_consistent_source() -> None:
    with pytest.raises(ValueError, match="source='none'"):
        UpstreamBaseAvailability(
            substrate="sqlite",
            target_id="sqlite:main",
            basis_token="basis-1",
            base_available=True,
            source="none",
        )


def test_preflight_result_base_unavailable_requires_unavailable_base() -> None:
    with pytest.raises(ValueError, match="status='base-unavailable'"):
        PreflightResult(
            status="base-unavailable",
            reason="missing base",
            observed_token="missing",
            base_availability=UpstreamBaseAvailability(
                substrate="sqlite",
                target_id="sqlite:main",
                basis_token="basis-1",
                base_available=True,
                source="live-upstream",
            ),
        )
