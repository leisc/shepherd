"""The egress broker — userspace mediation for irreversible network egress (MVP).

The fine network cell of the Containment axis (`docs/engineering/convergence/
egress-broker.md`). A syscall jail (Seatbelt/Landlock) is host-blind — it can
pin egress to a loopback port but cannot allowlist *which host*. Network egress
is irreversible (no undo, no compensation), so per the check-time law it must
be checked *before* it happens. The broker is that check: a loopback-bound
HTTP-CONNECT proxy that evaluates the value-level host predicate the jail
could not, allows or denies per connection, and records every verdict.

Composition (not built here — the launch-path wiring is a signposted
follow-up): the jail confines the body to `loopback_only:broker_port`, so the
broker is the *only* route out — it is unbypassable by construction, not by the
client's cooperation.

This MVP productizes `spikes/260609-egress-broker-mvp`'s throwaway proxy into a
tested unit:

- ``EgressPolicy`` — the decision function, with the **pinned evaluation
  order** (egress-broker.md "The evaluation-order contract"): within a grant,
  negation scopes the positives (deny-wins); across grants, union; absent any
  admitting grant, default-deny. ``escalate`` is enumerated but deny-loud in
  the MVP (no human-approval path until increment 3).
- ``EgressBroker`` — the loopback CONNECT proxy that consumes a policy, forwards
  allowed connections, refuses denied ones (HTTP 403), and appends an
  ``EgressDecision`` record per connection (the irreversible-effect evidence).

MVP scope: HTTP ``CONNECT`` only (covers ~all HTTPS clients, hands the broker
the hostname pre-DNS without TLS interception). SOCKS5, transparent redirect,
and the DNS stub are increment 2; escalate→supervisor is increment 3.
"""

from __future__ import annotations

import contextlib
import enum
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


class Verdict(enum.Enum):
    """A per-connection broker decision."""

    ALLOW = "allow"
    DENY = "deny"
    #: Enumerated for the irreversible-effect human-approval gate; the MVP
    #: treats it as deny-loud (egress-broker.md increment 3 builds the path).
    ESCALATE = "escalate"


@dataclass(frozen=True)
class HostAtom:
    """One value-level host predicate the syscall jail could not evaluate.

    ``op`` is one of: ``exact`` (``host``), ``in`` (set membership),
    ``contains`` (substring), ``matches`` (full-match regex). The lowering
    compiler emits these from the ``may=`` ``Match``'s ``broker_grants``.
    """

    op: str
    value: object

    def matches(self, host: str) -> bool:
        if self.op == "exact":
            return host == self.value
        if self.op == "in":
            return isinstance(self.value, (list, tuple, set, frozenset)) and host in self.value
        if self.op == "contains":
            return isinstance(self.value, str) and self.value in host
        if self.op == "matches":
            return isinstance(self.value, str) and re.fullmatch(self.value, host) is not None
        msg = f"unknown host-atom op {self.op!r} (lowering error — never a silent allow)"
        raise ValueError(msg)


@dataclass(frozen=True)
class HostGrant:
    """One DNF disjunct: admits a host iff a positive matches and no negative does.

    The grant-local negation is the pinned ¬-scope (egress-broker.md): deny
    lives only here, carving an exclusion out of *this* grant's positives —
    never a cross-grant veto. ``verdict`` lets a grant classify its admitted
    hosts as ``ALLOW`` (the common case) or ``ESCALATE`` (deny-loud in the MVP).
    """

    positive: tuple[HostAtom, ...]
    negative: tuple[HostAtom, ...] = ()
    verdict: Verdict = Verdict.ALLOW

    def admits(self, host: str) -> bool:
        if any(atom.matches(host) for atom in self.negative):
            return False
        return any(atom.matches(host) for atom in self.positive)


@dataclass(frozen=True)
class EgressDecision:
    """The evidence record for one mediated connection (egress-broker.md observability)."""

    host: str
    port: int
    verdict: Verdict
    #: The broker:{kind} EvidenceKind convention; deny-loud escalate is distinct.
    kind: str


@dataclass(frozen=True)
class EgressPolicy:
    """The compiled host predicate — ``broker_grants`` plus the pinned order.

    ``decide(host, port)`` applies the evaluation-order contract:
    union-across-grants of (positive ∧ ¬negative), default-deny, escalate→
    deny-loud. Pure and deterministic — unit-testable with no sockets.
    """

    grants: tuple[HostGrant, ...] = ()

    def decide(self, host: str, port: int) -> EgressDecision:
        admitting = [grant for grant in self.grants if grant.admits(host)]
        if not admitting:
            return EgressDecision(host, port, Verdict.DENY, "broker:connect-denied")
        # An admitting ALLOW grant wins outright (union). Otherwise the only
        # admitting grants are escalate — deny-loud in the MVP.
        if any(grant.verdict is Verdict.ALLOW for grant in admitting):
            return EgressDecision(host, port, Verdict.ALLOW, "broker:connect-allowed")
        return EgressDecision(host, port, Verdict.ESCALATE, "broker:connect-escalate-unbuilt")


_CONNECT_RE = re.compile(rb"^CONNECT\s+(?P<host>[^:\s]+):(?P<port>\d+)\s+HTTP", re.IGNORECASE)


@dataclass
class EgressBroker:
    """A loopback-bound HTTP-CONNECT proxy enforcing an ``EgressPolicy``.

    Forwards allowed connections (200 + bidirectional splice); refuses denied/
    escalate ones (403); records every verdict. Start it, read ``.port``, point
    the jail's ``loopback_only`` at it, stop it on scope close. The launch-path
    provisioning (the ``jail.loopback_only_port == broker.bound_port`` seam) is
    the run-driver increment — not wired here.
    """

    policy: EgressPolicy
    #: Sink for decisions (default: an in-memory list, also the test surface).
    on_decision: Callable[[EgressDecision], None] | None = None
    decisions: list[EgressDecision] = field(default_factory=list)
    _listen: socket.socket | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _port: int = 0
    _stopping: bool = field(default=False, repr=False)

    @property
    def port(self) -> int:
        if not self._port:
            msg = "broker not started"
            raise RuntimeError(msg)
        return self._port

    def start(self) -> int:
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind(("127.0.0.1", 0))
        listen.listen(16)
        self._listen = listen
        self._port = listen.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self._port

    def stop(self) -> None:
        self._stopping = True
        if self._listen is not None:
            with contextlib.suppress(OSError):
                self._listen.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _record(self, decision: EgressDecision) -> None:
        self.decisions.append(decision)
        if self.on_decision is not None:
            self.on_decision(decision)

    def _serve(self) -> None:
        assert self._listen is not None
        while not self._stopping:
            try:
                conn, _ = self._listen.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(3)
        try:
            request = self._read_head(conn)
            match = _CONNECT_RE.match(request)
            if match is None:
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            host = match.group("host").decode("ascii", "replace")
            port = int(match.group("port"))
            decision = self.policy.decide(host, port)
            self._record(decision)
            if decision.verdict is Verdict.ALLOW:
                self._forward(conn, host, port)
            else:
                # Deny and escalate-unbuilt both refuse the connection (403);
                # the distinct EvidenceKind keeps escalate auditable.
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    @staticmethod
    def _read_head(conn: socket.socket) -> bytes:
        buf = b""
        while b"\r\n\r\n" not in buf and len(buf) < 8192:
            chunk = conn.recv(256)
            if not chunk:
                break
            buf += chunk
        return buf

    def _forward(self, conn: socket.socket, host: str, port: int) -> None:
        try:
            upstream = socket.create_connection((host, port), timeout=3)
        except OSError:
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        self._splice(conn, upstream)

    @staticmethod
    def _splice(a: socket.socket, b: socket.socket) -> None:
        def pump(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass

        threads = [
            threading.Thread(target=pump, args=(a, b), daemon=True),
            threading.Thread(target=pump, args=(b, a), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4)
        for sock in (a, b):
            with contextlib.suppress(OSError):
                sock.close()


def policy_from_grants(grants: tuple[HostGrant, ...]) -> EgressPolicy:
    """Build an ``EgressPolicy`` from lowered ``broker_grants`` (the dialect seam)."""
    return EgressPolicy(grants=grants)


def allow_hosts(*hosts: str) -> EgressPolicy:
    """The common case: an exact-host allowlist (one ALLOW grant per host)."""
    return EgressPolicy(grants=tuple(HostGrant(positive=(HostAtom("exact", host),)) for host in hosts))
