"""Trace container with kernel/surface partition and canonical JSON serde.

CONTRACTS E4 pins:

- three frozen fields: ``run_ref``, ``kernel``, ``surface``
- ``filter(sub_tag)`` returns a narrowed ``Trace``
- ``cite(ref)`` raises ``LookupError`` on miss
- ``to_json()`` returns a ``dict``; ``from_json(payload)`` is the inverse
- round-trip equality: ``Trace.from_json(t.to_json()) == t``
- kernel half delegates to ``shepherd_kernel_v3_reference.trace.serde``
- surface half uses ``dataclasses.asdict(...)`` + the registry in
  ``surface.py`` for class lookup

Pinned by `docs/design/proposed/260505-plans/CONTRACTS.md` E4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from shepherd_kernel_v3_reference.trace.serde import (
    trace_from_json,
    trace_to_json,
)

from shepherd_runtime.identities import run_ref_from_json, run_ref_to_json
from shepherd_runtime.trace.kernel import KernelRecord  # noqa: TC001
from shepherd_runtime.trace.surface import (
    SURFACE_REGISTRY,
    SurfaceBase,
    SurfaceRecord,
)
from shepherd_runtime.trace.types import Ref, RunRef, SubTag

__all__ = ["SCHEMA_VERSION", "Trace"]


SCHEMA_VERSION = 1
"""Version tag for the JSON envelope. Bumped on incompatible shape changes."""


@dataclass(frozen=True)
class Trace:
    """Append-only run trace partitioned into kernel and surface halves.

    Kernel records are proof-oriented and normative when produced by the
    reference path. Runtime-normalized profiles may also use the kernel
    record dataclasses as structured evidence without claiming proof-backed
    source lowering. Surface records are projection: filterable by ``SubTag``
    for human readers and OTel spans. The two halves share the ``run_ref`` but
    live in separate tuples so proof envelopes can inspect the kernel half
    independently.
    """

    run_ref: RunRef
    kernel: tuple[KernelRecord, ...] = ()
    surface: tuple[SurfaceRecord, ...] = ()

    def filter(self, sub_tag: SubTag, /) -> Trace:
        """Return a Trace narrowed to surface records with this sub_tag.

        Kernel records are preserved unchanged; this is a *projection*
        on the surface half, not a new Trace identity.
        """
        return Trace(
            run_ref=self.run_ref,
            kernel=self.kernel,
            surface=tuple(s for s in self.surface if s.sub_tag is sub_tag),
        )

    def cite(self, ref: Ref, /) -> KernelRecord | SurfaceRecord:
        """Look up a record by ``ref`` across both halves.

        Raises:
            LookupError: if no record matches.
        """
        for record in self.kernel:
            if getattr(record, "ref", None) == ref:
                return record
        for record in self.surface:
            if record.ref == ref:
                return record
        raise LookupError(f"no record with ref={ref!r} in this trace")

    def to_json(self) -> dict[str, Any]:
        """Serialize to canonical-JSON-compatible dict (CONTRACTS E4).

        The kernel half delegates to
        ``shepherd_kernel_v3_reference.trace.serde.trace_to_json``. The
        surface half uses ``dataclasses.asdict(...)`` plus a ``type``
        discriminator (the dataclass class name).
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "run_ref": _run_ref_to_json(self.run_ref),
            "kernel": list(trace_to_json(self.kernel)),
            "surface": [_surface_to_json(s) for s in self.surface],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any], /) -> Trace:
        """Reconstruct a Trace from the JSON envelope produced by ``to_json``.

        Round-trip equality: ``Trace.from_json(t.to_json()) == t`` for
        every well-formed ``t``.
        """
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Trace.from_json schema_version={version!r} not supported (expected {SCHEMA_VERSION})")
        return cls(
            run_ref=_run_ref_from_json(payload["run_ref"]),
            kernel=tuple(trace_from_json(payload["kernel"])),
            surface=tuple(_surface_from_json(s) for s in payload["surface"]),
        )


def _run_ref_to_json(run_ref: RunRef) -> dict[str, Any]:
    # Trace JSON is a compatibility boundary, not raw dataclass serde. RunRef
    # owns nested identity hydration so richer links round-trip safely.
    return dict(run_ref_to_json(run_ref))


def _run_ref_from_json(data: dict[str, Any]) -> RunRef:
    return run_ref_from_json(data)


def _surface_to_json(record: SurfaceBase) -> dict[str, Any]:
    body = asdict(record)
    body["sub_tag"] = record.sub_tag.value
    body["citing"] = list(record.citing)
    if record.run_ref is not None:
        body["run_ref"] = _run_ref_to_json(record.run_ref)
    return {"type": type(record).__name__, "fields": body}


def _surface_from_json(data: dict[str, Any]) -> SurfaceRecord:
    discriminator = data["type"]
    cls = SURFACE_REGISTRY.get(discriminator)
    if cls is None:
        raise LookupError(
            f"surface record type {discriminator!r} is not registered; "
            f"register concrete surface classes in "
            f"shepherd_runtime.trace.surface.SURFACE_REGISTRY"
        )
    body = dict(data["fields"])
    body["sub_tag"] = SubTag(body["sub_tag"])
    body["citing"] = tuple(body.get("citing", ()))
    raw_run_ref = body.get("run_ref")
    if isinstance(raw_run_ref, dict):
        body["run_ref"] = _run_ref_from_json(raw_run_ref)
    # Drop fields the dataclass doesn't carry (defensive on schema drift).
    field_names = {f.name for f in fields(cls)}
    body = {k: v for k, v in body.items() if k in field_names}
    return cls(**body)
