"""Interfaces reserved for later tranches — DEFINED, not implemented.

The lease validation cut exercises ``DeltaIndex`` (B) + the shared contract ONLY.
It does not exercise the cut-frontier (A), the DAG ``reach_delta``, or the
monotone-vs-retraction precondition. Those are defined here so retention and
journal customers can adopt the package later without re-litigating its shape, but
they ship unimplemented: a concrete A/Dag implementation is the next (harder)
tranche.

The substrate is a **cut frontier over a DAG** (git's "have"-set); a linear
``MonotonicStream`` is the degenerate case (a DAG whose frontier is a single tip).
See ``260621-1730-incremental-frontier-primitive.md`` rev2.
"""

from __future__ import annotations

from typing import Any, NewType, Protocol

NodeId = NewType("NodeId", str)  # world_oid | journal-entry oid | lease ref | trace fact id
Ordinal = NewType("Ordinal", int)  # monotonic position within a LINEAR stream
Cut = frozenset[NodeId]  # an antichain boundary over NodeId; {tip} for a linear stream
Watermark = Any  # Cut | Ordinal — the boundary of a certified prefix

_RESERVED = "reserved for a later tranche; the lease cut implements DeltaIndex (B) only"


class Dag(Protocol):
    """The general substrate: a cut frontier over a DAG. NOT implemented in the lease cut."""

    def reach_delta(self, head: NodeId, *, behind: Cut) -> tuple[NodeId, ...]:
        r"""Return ``reach(head) \ behind(cut)``.

        MUST traverse only the boundary-bounded delta — never enumerate the whole
        node/ref namespace.
        """
        ...

    def dominates(self, cut: Cut, node: NodeId) -> bool: ...


class MonotonicStream(Protocol):
    """Linear convenience wrapper (journals / leases / trace owners). Reserved."""

    def append(self, stream_id: str, entries: Any) -> Any: ...
    def head(self, stream_id: str) -> Ordinal: ...
    def read_delta(self, stream_id: str, *, since: Ordinal) -> tuple[Any, ...]: ...


class Question(Protocol):
    """A monotone fold supplied per A call site. Reserved."""

    def fold(self, base_result: Any, delta: Any) -> Any: ...
    def recompute(self, full: Any) -> Any: ...
    def digest(self, result: Any) -> str: ...


class Frontier(Protocol):
    """Primitive A — an inductive certificate over a monotone fold. Reserved."""


class FrontierStore(Protocol):
    """Advance/verify inductive frontiers. Reserved — see ``_RESERVED``."""

    def advance(self, base: Frontier | None, question: Question, dag: Dag) -> Any: ...
    def verify(self, frontier: Frontier) -> bool: ...
