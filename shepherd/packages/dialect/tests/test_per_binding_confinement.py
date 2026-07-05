"""v0.2 Lane B — per-binding grant → ConfinementSpec lowering (pure) + disjoint-root soundness.

These exercise the dialect-side lowering in isolation (no workspace-control facade, no
``workspace.py``): per-binding grants lower to a deny-closed multi-root ``ConfinementSpec``, and
overlapping/nested bound roots fail closed at lowering time (the §4 soundness precondition — a
nested root is sub-root semantics, i.e. Tier-3). One macOS-gated seam test feeds the lowered spec
through the generalized Seatbelt jail (Lane A) to prove the dialect→jail seam is deny-closed.
"""

from __future__ import annotations

import os
import sys

import pytest

from shepherd_dialect.confinement import (
    BindingRootGrant,
    OverlappingBoundRootsError,
    lower_grants_to_confinement,
    validate_disjoint_roots,
)

_macos = pytest.mark.skipif(sys.platform != "darwin", reason="native Seatbelt jail is macOS-only")


def _real(path) -> str:
    return os.path.realpath(str(path))


# --- disjoint-root validation (the soundness load-bearer) -----------------------------------


def test_disjoint_sibling_roots_are_accepted(tmp_path) -> None:
    backend = tmp_path / "backend"
    docs = tmp_path / "docs"
    canonical = validate_disjoint_roots([str(backend), str(docs)])
    assert set(canonical) == {_real(backend), _real(docs)}


def test_identical_roots_fail_closed(tmp_path) -> None:
    backend = tmp_path / "backend"
    with pytest.raises(OverlappingBoundRootsError):
        validate_disjoint_roots([str(backend), str(backend)])


def test_nested_root_fails_closed(tmp_path) -> None:
    """A ReadOnly subtree nested inside a ReadWrite root is exactly the excluded Tier-3 case."""
    backend = tmp_path / "backend"
    vendor = backend / "vendor"
    with pytest.raises(OverlappingBoundRootsError):
        validate_disjoint_roots([str(backend), str(vendor)])
    # order-independent: parent-after-child fails closed too.
    with pytest.raises(OverlappingBoundRootsError):
        validate_disjoint_roots([str(vendor), str(backend)])


# --- per-binding lowering -------------------------------------------------------------------


def test_readwrite_grants_become_writable_roots(tmp_path) -> None:
    backend = tmp_path / "backend"
    docs = tmp_path / "docs"
    spec = lower_grants_to_confinement(
        [
            BindingRootGrant(binding="backend", root=str(backend), writable=True),
            BindingRootGrant(binding="docs", root=str(docs), writable=False),
        ]
    )
    # only the ReadWrite root is writable; the ReadOnly root contributes nothing.
    assert spec.writable_roots == (_real(backend),)
    # v0.2 makes no network claim: the spec denies egress by default.
    assert spec.network.mode.value == "deny_all"


def test_all_readonly_grants_lower_to_no_writable_root(tmp_path) -> None:
    docs = tmp_path / "docs"
    spec = lower_grants_to_confinement([BindingRootGrant(binding="docs", root=str(docs), writable=False)])
    assert spec.writable_roots == ()  # the ReadOnly / deny-closed floor


def test_multiple_readwrite_roots_union(tmp_path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    spec = lower_grants_to_confinement(
        [
            BindingRootGrant(binding="a", root=str(a), writable=True),
            BindingRootGrant(binding="b", root=str(b), writable=True),
        ]
    )
    assert set(spec.writable_roots) == {_real(a), _real(b)}


def test_lowering_rejects_overlapping_bound_roots(tmp_path) -> None:
    backend = tmp_path / "backend"
    vendor = backend / "vendor"
    with pytest.raises(OverlappingBoundRootsError):
        lower_grants_to_confinement(
            [
                BindingRootGrant(binding="backend", root=str(backend), writable=True),
                BindingRootGrant(binding="vendor", root=str(vendor), writable=False),
            ]
        )


# --- dialect → jail seam (macOS) ------------------------------------------------------------


@_macos
def test_lowered_spec_is_deny_closed_at_the_seatbelt_jail(tmp_path) -> None:
    """End-to-end at the dialect→jail seam: per-binding grants lower to a spec the native
    Seatbelt jail enforces deny-closed — writes land under the ReadWrite root and are refused
    at the syscall under the ReadOnly root and at unbound paths."""
    from vcs_core._execution_capability import _resolve_spec
    from vcs_core._seatbelt_containment import SeatbeltContainmentBackend

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    spec = lower_grants_to_confinement(
        [
            BindingRootGrant(binding="backend", root=str(backend_dir), writable=True),
            BindingRootGrant(binding="docs", root=str(docs_dir), writable=False),
        ]
    )
    writable_roots, allow_network = _resolve_spec(spec)
    assert not allow_network  # deny-all lowered through

    jail = SeatbeltContainmentBackend()
    profile = jail.profile_for(writable_roots, allow_network=allow_network)
    jail.probe(profile, tmp_path, writable_roots=writable_roots)  # fail-closed pre-flight

    ok = backend_dir / "candidate.txt"
    jail.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(ok)])
    assert ok.exists()  # write beneath the ReadWrite root lands

    denied = docs_dir / "nope.txt"
    jail.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(denied)])
    assert not denied.exists()  # write beneath the ReadOnly root refused at the syscall

    unbound = tmp_path / "stray.txt"
    jail.launch(profile, tmp_path, ["/usr/bin/touch", "--", str(unbound)])
    assert not unbound.exists()  # unbound in-workspace write refused at the syscall
