"""The egress broker MVP — the decision function's pinned evaluation order + the proxy.

The decision tests are pure and cross-platform: they pin the
evaluation-order contract (egress-broker.md) — within-grant negation scopes
the positives, across-grant union, default-deny, escalate→deny-loud. The
proxy tests use only loopback (no internet); a Seatbelt composition test
(macOS-gated) proves the jail makes the broker non-optional.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from vcs_core._egress_broker import (
    EgressBroker,
    EgressPolicy,
    HostAtom,
    HostGrant,
    Verdict,
    allow_hosts,
)

# --- the decision function: the pinned evaluation order --------------------


def test_default_deny_empty_policy() -> None:
    assert EgressPolicy().decide("any.host", 443).verdict is Verdict.DENY


def test_exact_allowlist_admits_only_listed() -> None:
    policy = allow_hosts("model.api", "tools.api")
    assert policy.decide("model.api", 443).verdict is Verdict.ALLOW
    assert policy.decide("evil.exfil", 443).verdict is Verdict.DENY


def test_negation_scopes_within_a_grant_deny_wins() -> None:
    """A grant admitting *.internal EXCEPT secret.internal: the negative wins
    within the grant (the pinned ¬-scope)."""
    grant = HostGrant(
        positive=(HostAtom("contains", ".internal"),),
        negative=(HostAtom("exact", "secret.internal"),),
    )
    policy = EgressPolicy(grants=(grant,))
    assert policy.decide("api.internal", 443).verdict is Verdict.ALLOW
    assert policy.decide("secret.internal", 443).verdict is Verdict.DENY


def test_union_across_grants_one_admitting_suffices() -> None:
    policy = EgressPolicy(
        grants=(
            HostGrant(positive=(HostAtom("exact", "a.host"),)),
            HostGrant(positive=(HostAtom("matches", r"b\..*"),)),
        )
    )
    assert policy.decide("a.host", 443).verdict is Verdict.ALLOW
    assert policy.decide("b.anything", 443).verdict is Verdict.ALLOW
    assert policy.decide("c.host", 443).verdict is Verdict.DENY


def test_negation_is_grant_local_not_a_cross_grant_veto() -> None:
    """A ¬ in grant 1 cannot veto grant 2's allow — deny is grant-scoped."""
    policy = EgressPolicy(
        grants=(
            HostGrant(
                positive=(HostAtom("contains", ".host"),),
                negative=(HostAtom("exact", "shared.host"),),
            ),
            HostGrant(positive=(HostAtom("exact", "shared.host"),)),  # grant 2 admits it
        )
    )
    assert policy.decide("shared.host", 443).verdict is Verdict.ALLOW


def test_escalate_is_deny_loud_in_the_mvp() -> None:
    policy = EgressPolicy(grants=(HostGrant(positive=(HostAtom("exact", "approve.me"),), verdict=Verdict.ESCALATE),))
    decision = policy.decide("approve.me", 443)
    assert decision.verdict is Verdict.ESCALATE
    assert decision.kind == "broker:connect-escalate-unbuilt"  # auditable, never a silent allow


def test_allow_grant_wins_over_escalate_grant_when_both_admit() -> None:
    policy = EgressPolicy(
        grants=(
            HostGrant(positive=(HostAtom("exact", "h"),), verdict=Verdict.ESCALATE),
            HostGrant(positive=(HostAtom("exact", "h"),), verdict=Verdict.ALLOW),
        )
    )
    assert policy.decide("h", 443).verdict is Verdict.ALLOW


def test_unknown_atom_op_is_a_lowering_error_not_a_silent_allow() -> None:
    policy = EgressPolicy(grants=(HostGrant(positive=(HostAtom("regexp_typo", "x"),)),))
    with pytest.raises(ValueError, match="unknown host-atom op"):
        policy.decide("x", 443)


# --- the proxy: loopback, no internet --------------------------------------


def _banner_server(banner: bytes) -> tuple[socket.socket, int]:
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind(("127.0.0.1", 0))
    listen.listen(8)
    port = listen.getsockname()[1]

    def serve() -> None:
        while True:
            try:
                conn, _ = listen.accept()
            except OSError:
                return
            try:
                conn.sendall(banner)
            finally:
                conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return listen, port


def _connect_through(broker_port: int, host: str, port: int) -> bytes:
    sock = socket.create_connection(("127.0.0.1", broker_port), timeout=3)
    sock.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    sock.settimeout(0.5)
    buf = b""
    try:
        # Read the response head + any spliced banner. The broker keeps the
        # tunnel open after the short banner (the other splice direction has no
        # EOF), so treat a read-idle as end-of-data rather than waiting for EOF.
        while True:
            try:
                chunk = sock.recv(256)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b" 200 " in head:
            return b"STATUS200:" + rest
        return head.split(b"\r\n", 1)[0]
    finally:
        sock.close()


@pytest.mark.loopback
def test_proxy_forwards_allowed_and_refuses_denied_and_records() -> None:
    upstream, upstream_port = _banner_server(b"MODEL_API_OK")
    # The broker resolves the logical host to the loopback upstream (the broker
    # IS the resolver in the CONNECT model); the policy allows only "model.api".
    policy = EgressPolicy(grants=(HostGrant(positive=(HostAtom("exact", "127.0.0.1"),)),))
    try:
        with EgressBroker(policy=policy) as broker:
            allowed = _connect_through(broker.port, "127.0.0.1", upstream_port)
            assert allowed.startswith(b"STATUS200:MODEL_API_OK")
            denied = _connect_through(broker.port, "evil.exfil", upstream_port)
            assert b"403" in denied
        verdicts = [(d.host, d.verdict) for d in broker.decisions]
        assert ("127.0.0.1", Verdict.ALLOW) in verdicts
        assert ("evil.exfil", Verdict.DENY) in verdicts
    finally:
        upstream.close()


# --- the jail makes the broker non-optional (macOS Seatbelt) ---------------

_DARWIN_JAIL = sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()

_DIRECT_CLIENT = (
    "import socket,sys\n"
    "try:\n"
    "    s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=3)\n"
    "    print('REACH:'+s.recv(64).decode('replace').strip()); s.close()\n"
    "except Exception as e:\n"
    "    print('BLOCKED:'+type(e).__name__)\n"
)


@pytest.mark.skipif(not _DARWIN_JAIL, reason="needs macOS sandbox-exec")
@pytest.mark.loopback
def test_loopback_only_jail_blocks_direct_egress_making_broker_non_optional(tmp_path: Path) -> None:
    """The composition guarantee: under a loopback-only:broker_port jail, a
    DIRECT connect to a non-broker port is refused at the syscall — so the
    broker is the only route out (unbypassable, not by cooperation)."""
    secret, secret_port = _banner_server(b"SECRET_DATA_LEAK")
    try:
        with EgressBroker(policy=allow_hosts("127.0.0.1")) as broker:
            client = tmp_path / "client.py"
            client.write_text(_DIRECT_CLIENT)
            profile = (
                "(version 1)\n(allow default)\n(deny network-outbound)\n"
                f'(allow network-outbound (remote ip "localhost:{broker.port}"))\n'
            )
            proc = subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", profile, sys.executable, str(client), "127.0.0.1", str(secret_port)],
                capture_output=True,
                text=True,
                check=False,
            )
            out = (proc.stdout or proc.stderr).strip()
            assert out.startswith("BLOCKED:"), f"direct egress to the secret should be jailed: {out!r}"
    finally:
        secret.close()
