# under-test: vcs_core._claims
from __future__ import annotations

import pytest
from vcs_core._claims import ClaimConflictError, ClaimRegistry


def test_register_is_idempotent_for_same_claim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = ClaimRegistry()
    path = tmp_path / "state.db"

    first = registry.register(
        substrate="sqlite",
        target_id="sqlite:main",
        path=path,
        policy="exclusive",
    )
    second = registry.register(
        substrate="sqlite",
        target_id="sqlite:main",
        path=path,
        policy="exclusive",
    )

    assert second is first


def test_register_rejects_conflicting_claimants_for_same_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = ClaimRegistry()
    path = tmp_path / "state.db"

    registry.register(
        substrate="sqlite",
        target_id="sqlite:main",
        path=path,
        policy="exclusive",
    )

    with pytest.raises(ClaimConflictError, match="already claimed by sqlite:sqlite:main"):
        registry.register(
            substrate="filesystem",
            target_id="workspace",
            path=path,
            policy="observe",
        )


def test_register_rejects_conflicts_after_path_normalization(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = ClaimRegistry()
    path = tmp_path / "runtime" / "shadow.db"
    path.parent.mkdir()

    registry.register(
        substrate="sqlite",
        target_id="sqlite:shadow",
        path=path,
        policy="authoritative_suppress_fs",
    )

    alias_path = tmp_path / "runtime" / "." / ".." / "runtime" / "shadow.db"
    with pytest.raises(ClaimConflictError, match=r"shadow\.db"):
        registry.register(
            substrate="filesystem",
            target_id="workspace",
            path=alias_path,
            policy="observe",
        )
