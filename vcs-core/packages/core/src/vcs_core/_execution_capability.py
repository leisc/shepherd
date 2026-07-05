"""The execution-mechanism capability surface (PD1).

Execution is an *opt-in capability*, not a field on the universal driver
context: a driver that runs opaque code declares it by implementing
``ExecutionBoundDriver`` (detected via ``isinstance`` — the
``RuntimeBoundSubstrate`` house style), and the coordinator hands it a per-run
``ExecutionCapability`` *through the dispatch call*. A pure-data driver is
structurally incapable of receiving execution authority. The frozen
``DriverContext`` is untouched (additive growth — no ``SPI_VERSION`` bump);
the capability surface is separately versioned (``EXECUTION_CAPABILITY_VERSION``).

``launch_confined`` is the only real-execution verb; there is deliberately no
unconfined-real verb. ``ConfinementSpec`` is the generic (``may=``-agnostic)
confinement policy; its v0 lowering is exact-match onto today's two profile
names (byte-identical by construction — ``profile_for`` IS the live lowering)
and refuses anything not representable, fail-closed. ``NetworkPolicy`` carries
the ``broker_grants`` seam (``allowed_hosts``) the egress broker will consume;
the jail is host-blind and never reads it.

Shapes spike-validated: ``spikes/260608-exec-spi-verbs`` (8/8),
``spikes/260609-confinement-spec`` (19/19),
``spikes/260609-dialect-jailed-run`` (6/6).
Architecture: ``docs/engineering/convergence/execution-boundary.md`` §2.
"""

from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path

    from vcs_core._runtime_types import ExecutionContext
    from vcs_core._substrate_driver import DriverContext, DriverIngressResult, IngressRequest

#: The separately-versioned execution capability (PD4 pins the negotiation rule).
EXECUTION_CAPABILITY_VERSION = "v0"

#: The loud, greppable opt-out from reversible-by-default. This is the
#: structural execution-option key; it is not mixed into driver params, so a
#: same-named driver parameter remains ordinary command payload.
NON_REVERSIBLE_RUN_FLAG = "non_reversible_run"


class UnsupportedConfinementSpecError(ValueError):
    """The spec has no faithful lowering on this host's enforcement surface.

    Fail-closed: a spec that cannot be enforced exactly as declared refuses to
    run real rather than running under a weaker (or silently different)
    profile.
    """


class ExecutionAuthorityRequired(RuntimeError):  # noqa: N818 — refusal-state name, per JailNotEstablished
    """An execution command was dispatched without execution authority.

    The negotiation rule (PD4, ``decisions.md`` ``spi-additive-no-bump``): an
    opted-in driver dispatched by a coordinator that does not recognize the
    execution capability (version skew; plain ``prepare``) **refuses to run
    real** — never a silent in-process fallback. Drivers raise this from
    ``prepare`` for any command in their ``execution_commands``, before
    touching params.
    """


class NetMode(enum.Enum):
    DENY_ALL = "deny_all"  # no egress
    ALLOW_ALL = "allow_all"  # unrestricted egress
    BROKER = "broker"  # host-filtered: jail pins egress to loopback; broker allowlists hosts


@dataclass(frozen=True)
class NetworkPolicy:
    """The network axis of a confinement spec — coarse today, host-granular later.

    ``allowed_hosts`` is the host predicate (``broker_grants`` in
    egress-broker.md). The jail is host-blind (Seatbelt/Landlock structurally
    cannot filter by host) and ignores it; the broker consumes it. Adding the
    broker changes neither this type nor ``ConfinementSpec``.
    """

    mode: NetMode
    allowed_hosts: tuple[str, ...] = ()
    broker_port: int | None = None

    @classmethod
    def deny_all(cls) -> NetworkPolicy:
        return cls(NetMode.DENY_ALL)

    @classmethod
    def allow_all(cls) -> NetworkPolicy:
        return cls(NetMode.ALLOW_ALL)

    @classmethod
    def via_broker(cls, hosts: tuple[str, ...], *, port: int | None = None) -> NetworkPolicy:
        return cls(NetMode.BROKER, tuple(hosts), port)


@dataclass(frozen=True)
class ConfinementSpec:
    """Generic, ``may=``-vocabulary-agnostic confinement policy.

    The dialect lowers its ``may=`` ``Match`` to this; substrates stay
    ``may=``-blind. v0 representable lowerings (exact match, byte-identical to
    the live 2-name profiles):

    - ``read_only()`` — no writable roots, no egress → the ``ReadOnly`` profile.
    - ``permissive_for(working_path)`` — the run's working path writable, all
      egress → the ``Permissive`` profile (the documented coarse network hole;
      host-filtering arrives with the broker).

    Anything else refuses fail-closed (``UnsupportedConfinementSpecError``).
    """

    writable_roots: tuple[str, ...] = ()
    network: NetworkPolicy = field(default_factory=NetworkPolicy.deny_all)

    @classmethod
    def read_only(cls) -> ConfinementSpec:
        return cls()

    @classmethod
    def permissive_for(cls, working_path: Path | str) -> ConfinementSpec:
        return cls(writable_roots=(str(working_path),), network=NetworkPolicy.allow_all())


def _resolve_spec(spec: ConfinementSpec) -> tuple[tuple[str, ...], bool]:
    """Resolve a ConfinementSpec to the jail's inputs: canonical writable roots + a network flag.

    The writable-root set lowers directly (zero roots = ReadOnly; one root == WORKDIR =
    Permissive; a proper subset = per-binding grants), so multi-root specs are now
    faithfully representable — the backend compiles a deny-closed per-root profile.

    Only the network axis can still refuse: the syscall jail is host-blind, so BROKER
    (host-filtered) egress has no faithful jail lowering and fails closed here — the egress
    broker, not the jail, filters hosts. DENY_ALL / ALLOW_ALL map to the coarse jail profiles.
    """
    mode = spec.network.mode
    if mode is NetMode.DENY_ALL:
        allow_network = False
    elif mode is NetMode.ALLOW_ALL:
        allow_network = True
    else:
        raise UnsupportedConfinementSpecError(
            f"ConfinementSpec(network={mode.value!r}) has no jail lowering: host-filtered egress is the "
            "egress broker's job, not the syscall jail's (the jail is host-blind). "
            "Representable network axes: NetMode.DENY_ALL, NetMode.ALLOW_ALL."
        )
    writable_roots = tuple(os.path.realpath(root) for root in spec.writable_roots)
    return writable_roots, allow_network


@dataclass(frozen=True)
class ExecutionCapability:
    """The per-run execution handle, passed through the dispatch call.

    Constructed by the coordinator only after the run's scope exists — for a
    reversible run, only after the isolated fork has *succeeded* (the
    capability carries the proof; a non-isolated capability for a reversible
    run is never handed out). Carries the existing identity value
    (``vcs_core._runtime_types.ExecutionContext``) as ``.identity``; the
    carrier/jail backends stay private behind the verbs.

    ``isolation`` is read-only observability (``"isolated"`` | ``"ground"``) —
    load-bearing for the loud-opt-out mode where ground is a legitimate value,
    not an assert target for the reversible mode where the invariant is
    structural.
    """

    identity: ExecutionContext
    working_path: Path
    isolation: str
    _containment: Any | None = None

    def launch_confined(self, command: list[str], spec: ConfinementSpec) -> subprocess.CompletedProcess:
        """The only real-execution verb. Fail-closed: no jail, no real run (D-4)."""
        from vcs_core._containment import JailNotEstablished

        if self._containment is None:
            raise JailNotEstablished(
                "launch_confined: no jail-capable containment on this host; real execution "
                "refuses rather than running unconfined (real-implies-jail-capable-host)."
            )
        writable_roots, allow_network = _resolve_spec(spec)
        profile = self._containment.profile_for(writable_roots, allow_network=allow_network)
        self._containment.probe(profile, self.working_path, writable_roots=writable_roots)  # fail-closed first
        return self._containment.launch(profile, self.working_path, command)


@runtime_checkable
class ExecutionBoundDriver(Protocol):
    """Opt-in capability: a driver that runs opaque code.

    The coordinator detects the opt-in via ``isinstance`` and passes a per-run
    ``ExecutionCapability`` through ``prepare_bound`` (per-call, never stored —
    the driver stays frozen/stateless/concurrency-safe). A driver that does not
    define ``prepare_bound`` structurally cannot receive execution authority.

    ``execution_commands`` names the commands that actually carry execution —
    only those are dispatched through the reversible wrap with a capability.
    Everything else (``list``, ``get``, ``register``, …) routes through plain
    ``prepare`` with **no** execution authority and **no** scope fork: least
    authority per command, and a ``list`` never clones a workspace. A driver
    implementing ``prepare_bound`` without ``execution_commands`` does not
    satisfy this protocol and dispatches entirely through ``prepare``.
    """

    @property
    def execution_commands(self) -> frozenset[str]: ...

    def prepare_bound(
        self,
        context: DriverContext,
        request: IngressRequest,
        execution: ExecutionCapability,
    ) -> DriverIngressResult: ...


def verify_execution_negotiation(driver: ExecutionBoundDriver) -> None:
    """Conformance check for the negotiation rule — fail-closed under skew.

    Pins, for an opted-in driver:

    - every declared execution command exists in the driver's schema;
    - plain ``prepare`` of an execution command raises
      ``ExecutionAuthorityRequired`` **before any other failure** (probed with
      empty params — a driver that validates params first, or worse runs the
      body in-process, fails conformance).

    Raises ``AssertionError`` describing the violation; passes silently.
    """
    from vcs_core._substrate_driver import CommandRequest, DriverContext
    from vcs_core._world_types import SubstrateStoreIdentity

    schema = driver.describe()
    undeclared = sorted(set(driver.execution_commands) - set(schema.commands))
    if undeclared:
        raise AssertionError(
            f"{type(driver).__name__} declares execution command(s) {undeclared!r} absent from its schema."
        )
    probe_context = DriverContext(
        operation_id="negotiation-probe",
        binding=getattr(driver, "binding", "probe"),
        role=getattr(driver, "role", "probe"),
        store_identity=SubstrateStoreIdentity(
            store_id=getattr(driver, "store_id", "store_probe"),
            kind="runtime.journal_only",
            resource_id="negotiation-probe",
        ),
        base_heads=(),
    )
    for command in sorted(driver.execution_commands):
        try:
            driver.prepare(probe_context, CommandRequest(command=command, params={}))
        except ExecutionAuthorityRequired:
            continue
        except Exception as exc:
            raise AssertionError(
                f"{type(driver).__name__}.prepare({command!r}) must refuse with ExecutionAuthorityRequired "
                f"before any other failure (fail-closed under version skew); got {type(exc).__name__}: {exc}"
            ) from exc
        raise AssertionError(
            f"{type(driver).__name__}.prepare({command!r}) ran without execution authority — "
            "the silent in-process fallback the negotiation rule forbids."
        )


def detect_containment_backend() -> Any | None:
    """The host's jail backend, or ``None`` on a jail-less host (advisory tier only)."""
    if sys.platform == "darwin":
        from vcs_core._seatbelt_containment import SeatbeltContainmentBackend

        backend = SeatbeltContainmentBackend()
        return backend if backend.available()[0] else None
    if sys.platform.startswith("linux"):
        from vcs_core._landlock_containment import LandlockContainmentBackend

        backend = LandlockContainmentBackend()
        return backend if backend.available()[0] else None
    return None
