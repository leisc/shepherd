"""Internal containment-axis backend contract — the jail that makes `may=` real.

The two-axis model (containment-and-carriers.md §2): a Device = Containment x
Carrier. This module is the **Containment** axis: the syscall-deny jail that
enforces an effect surface (`may=`) at the OS boundary, distinct from the
reversibility-axis `CarrierBackend` (_substrate_runtime.py). The family spans
Seatbelt (macOS), Landlock+seccomp (Linux), AppContainer, container, and VM
tiers; each member lives in its own `_*_containment.py` (cf. `_seatbelt_containment.py`,
`_landlock_containment.py`).

Lifted move-not-build from the skeleton staging (skeleton/src/skeleton/containment.py),
grounded in spikes/sandbox-jail (macOS Seatbelt x clonefile 8/8, Linux Landlock x
fuse 8/8, and the may=->lowering seam 15/15). Fail-closed is mandatory (§6): probe
the jail for liveness AND `may=` conformance before the body runs; never proceed
unconfined.

Internal runtime surface — NOT part of the frozen consumer SPI (SPI_VERSION=0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Sequence


class SandboxError(Exception):
    """A jailed run failed for a non-policy reason (launch/runner error)."""


class SandboxDenied(SandboxError):  # noqa: N818
    """An out-of-`may=` operation was refused at the syscall (EPERM/EACCES).

    The native syscall-deny tier's signal — stronger than the carrier's optimistic
    check-at-commit: the write never reached disk. Mapped by the runtime to a
    discarded, `Failed` run.
    """


class JailNotEstablished(SandboxError):  # noqa: N818
    """Fail-closed: confinement could not be established, so the run refuses."""


@runtime_checkable
class ContainmentBackend(Protocol):
    """Enforce `may=` — the syscall-deny jail.

    Family (containment-and-carriers.md §2): Seatbelt, Landlock+seccomp, AppContainer,
    container, VM. A backend lowers a set of writable roots (+ the network axis) to an
    OS-specific profile, probes it fail-closed, then launches a command confined by it.

    The writable-root set is the per-binding surface: zero roots = ReadOnly
    (no managed writable root at all); one root == realpath(WORKDIR) = the whole-workspace
    Permissive floor; a proper subset of the workspace = per-binding grants
    (``ReadWrite backend/`` alongside ``ReadOnly docs/``). The profile is **deny-closed**:
    a write to any path outside every writable root is refused at the syscall. Roots must
    be disjoint/non-nested (enforced dialect-side at ``ws.bind``); nested roots would be
    sub-root semantics, which this whole-root surface deliberately excludes.
    """

    name: str
    enforcement_tier: str

    def available(self) -> tuple[bool, str]: ...

    def profile_for(self, writable_roots: Sequence[str], *, allow_network: bool) -> str: ...

    def probe(self, profile: str, working_root: Any, *, writable_roots: Sequence[str]) -> None: ...

    def launch(
        self, profile: str, working_root: Any, command: list[str], *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]: ...
