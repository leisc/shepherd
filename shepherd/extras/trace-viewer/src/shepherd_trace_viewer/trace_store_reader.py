"""TraceStore source adapter for local Trace Viewer reads."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from shepherd_trace_viewer.projection import TraceSliceSelection, _occurrences, project_trace_slice_to_view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shepherd_trace_viewer.model import TraceView

TraceStoreSelector = Literal["cut", "owner", "causal_root"]
Visibility = Literal["shape_only", "payload", "full_internal"]
ModeFilter = Literal["declarations_only", "captures_only", "both"]


class TraceStoreReadError(RuntimeError):
    """Raised when a TraceStore source cannot be read."""


def _caused_by_fact_ids(fact: object) -> tuple[str, ...]:
    """Causal-parent fact ids for a visible fact (empty when shape-only/absent)."""
    envelope = getattr(fact, "envelope", None)
    return tuple(getattr(envelope, "caused_by_fact_ids", ()) or ())


def _read_context(actor: str, visibility: Visibility, trusted_internal: bool) -> Any:
    """Build the kernel ReadContext for a viewer read (raises if shepherd2 absent)."""
    try:
        from shepherd2 import ReadContext
    except ImportError as exc:
        raise TraceStoreReadError("shepherd2 is required for --trace-store") from exc
    presented_witness_refs = ("trusted:internal",) if trusted_internal else ()
    return ReadContext(
        actor_ref=actor,
        presented_witness_refs=presented_witness_refs,
        visibility_profile=visibility,
    )


def read_trace_store_view(
    store_path: str | Path,
    *,
    selector: TraceStoreSelector,
    selector_value: str,
    through: int | None = None,
    visibility: Visibility = "payload",
    mode_filter: ModeFilter = "both",
    actor: str = "trace-viewer",
    trusted_internal: bool = False,
) -> TraceView:
    """Read a selected TraceStore slice from SQLite and project it for the viewer."""
    try:
        from shepherd2 import SQLiteTraceStore
    except ImportError as exc:
        raise TraceStoreReadError("shepherd2 is required for --trace-store") from exc

    path = Path(store_path)
    if not path.exists():
        raise TraceStoreReadError(f"trace store does not exist: {path}")
    read_context = _read_context(actor, visibility, trusted_internal)
    try:
        with SQLiteTraceStore(path) as store:
            if selector == "cut":
                trace_slice = store.resolve_cut(read_context, selector_value, mode_filter=mode_filter)
            elif selector == "owner":
                if through is None:
                    raise TraceStoreReadError("--owner requires --through")
                trace_slice = store.read_owner_prefix(read_context, selector_value, through, mode_filter=mode_filter)
            elif selector == "causal_root":
                trace_slice = store.read_causal_closure(
                    read_context,
                    (selector_value,),
                    mode_filter=mode_filter,
                    closure_policy="include_external_anchors",
                )
            else:
                raise TraceStoreReadError(f"unknown TraceStore selector: {selector}")
    except Exception as exc:
        if isinstance(exc, TraceStoreReadError):
            raise
        raise TraceStoreReadError(f"cannot read trace store slice: {exc}") from exc

    return project_trace_slice_to_view(
        trace_slice,
        selection=TraceSliceSelection(selector=selector, selector_value=selector_value, store_path=str(path)),
    )


# Owner-prefix cutoff meaning "read this owner path in full" (every ordinal).
_THROUGH_ALL = 2**62


def read_trace_store_session_view(
    store_path: str | Path,
    *,
    owners: Sequence[str],
    visibility: Visibility = "payload",
    mode_filter: ModeFilter = "both",
    actor: str = "trace-viewer",
    trusted_internal: bool = False,
) -> TraceView:
    """Project a whole session (every supplied owner as a lane) as one viewer view.

    A forked or pipelined session has several runs, each on its own owner path
    (its execution id). A single ``cut``/``owner``/``causal_root`` read only sees
    one owner, and a causal closure over the terminals only reaches each run's
    causal ancestors (started -> completed), dropping the surrounding lifecycle
    records. This instead reads every supplied owner's *full* prefix and unions
    them into one slice, so each run renders as a complete lane -- its whole
    record lifecycle (created, started, output, completed, frontier) -- side by
    side. ``owners`` are owner-path ids (run execution ids). This is the notebook
    fork-board / pipeline entry point.
    """
    owners = tuple(dict.fromkeys(owners))
    if not owners:
        raise TraceStoreReadError("read_trace_store_session_view requires at least one owner id")
    try:
        from shepherd2 import SQLiteTraceStore
    except ImportError as exc:
        raise TraceStoreReadError("shepherd2 is required for --trace-store") from exc

    path = Path(store_path)
    if not path.exists():
        raise TraceStoreReadError(f"trace store does not exist: {path}")
    read_context = _read_context(actor, visibility, trusted_internal)
    try:
        with SQLiteTraceStore(path) as store:
            slices = [
                store.read_owner_prefix(read_context, owner, _THROUGH_ALL, mode_filter=mode_filter) for owner in owners
            ]
    except Exception as exc:
        if isinstance(exc, TraceStoreReadError):
            raise
        raise TraceStoreReadError(f"cannot read trace store session: {exc}") from exc

    merged = _merge_slices(slices, visibility=visibility, mode_filter=mode_filter)
    return project_trace_slice_to_view(
        merged,
        selection=TraceSliceSelection(selector="session", selector_value=f"{len(owners)} owners", store_path=str(path)),
    )


def _merge_slices(slices: Sequence[Any], *, visibility: Visibility, mode_filter: ModeFilter) -> Any:
    """Union per-owner TraceSlices into one multi-lane slice for projection.

    Each owner read is independent, so the unions are disjoint by construction
    (distinct owner paths and record ids); the dedup guards only protect against
    a record legitimately shared across owner paths.
    """
    from shepherd2 import TraceSlice

    facts_by_id: dict = {}
    contexts_by_id: dict = {}
    owner_paths: dict = {}
    visible_witnesses_by_id: dict = {}
    causal_edges: list = []
    external_anchors: list = []
    context_anchors: list = []
    witness_anchors: list = []
    occurrences: list = []
    seen_edge: set = set()
    seen_ext: set = set()
    seen_ctx_anchor: set = set()
    seen_wit_anchor: set = set()
    seen_occ: set = set()

    for sl in slices:
        facts_by_id.update(sl.facts_by_id)
        contexts_by_id.update(sl.contexts_by_id)
        owner_paths.update(sl.owner_paths)
        visible_witnesses_by_id.update(sl.visible_witnesses_by_id)
        for edge in sl.causal_edges:
            if edge not in seen_edge:
                seen_edge.add(edge)
                causal_edges.append(edge)
        for anchor in sl.external_anchors:
            if anchor.ref not in seen_ext:
                seen_ext.add(anchor.ref)
                external_anchors.append(anchor)
        for anchor in sl.context_anchors:
            if anchor.context_id not in seen_ctx_anchor:
                seen_ctx_anchor.add(anchor.context_id)
                context_anchors.append(anchor)
        for anchor in sl.witness_anchors:
            if anchor.witness_ref not in seen_wit_anchor:
                seen_wit_anchor.add(anchor.witness_ref)
                witness_anchors.append(anchor)
        for occ in _occurrences(sl):
            key = (occ.trace_owner_id, occ.owner_ordinal, occ.record_id)
            if key not in seen_occ:
                seen_occ.add(key)
                occurrences.append(occ)

    # Cross-owner causal edges (e.g. an origin run -> its forked children) are
    # invisible to each single-owner read -- the parent fact is outside the owner,
    # so it surfaces as an external anchor instead of an edge. Recompute edges over
    # the unioned fact set so those links render as branch edges, and demote the
    # now-internal external anchors.
    for child_id, fact in facts_by_id.items():
        for parent_id in _caused_by_fact_ids(fact):
            if parent_id in facts_by_id:
                edge = (parent_id, child_id)
                if edge not in seen_edge:
                    seen_edge.add(edge)
                    causal_edges.append(edge)
    external_anchors = [anchor for anchor in external_anchors if anchor.ref not in facts_by_id]

    kwargs = {
        "frontier": None,
        "visibility_profile": visibility,
        "mode_filter": mode_filter,
        "facts_by_id": facts_by_id,
        "contexts_by_id": contexts_by_id,
        "owner_paths": owner_paths,
        "causal_edges": tuple(causal_edges),
        "external_anchors": tuple(external_anchors),
        "context_anchors": tuple(context_anchors),
        "visible_witnesses_by_id": visible_witnesses_by_id,
        "witness_anchors": tuple(witness_anchors),
    }
    if "occurrences" in getattr(TraceSlice, "__dataclass_fields__", {}):
        kwargs["occurrences"] = tuple(occurrences)
    return TraceSlice(**kwargs)
