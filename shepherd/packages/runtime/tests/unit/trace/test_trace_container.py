"""Contract-import + round-trip test for E4 ``Trace`` container.

Satisfies CONTRACTS Maintenance Rule 3 for E4: every consumer can
import ``Trace`` from ``shepherd_runtime.trace`` and exercise the
filter / cite / to_json / from_json surface against real kernel
records.

Pinned by `docs/design/proposed/260505-plans/CONTRACTS.md` E4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_args, get_type_hints

import pytest
from shepherd_kernel_v3_reference import (
    AnySchema,
    Handle,
    HandlerEnv,
    Let,
    Lit,
    Perform,
    Resume,
    Return,
    StaticHandlerInstall,
    Var,
)
from shepherd_kernel_v3_reference.kernel import elaborate
from shepherd_kernel_v3_reference.trace import run_trace, validate_core_trace
from shepherd_runtime.identities import VcsCoreExecutionLink
from shepherd_runtime.nucleus.types import Run
from shepherd_runtime.trace import (
    SCHEMA_VERSION,
    SURFACE_REGISTRY,
    EffectDeclaration,
    KernelRecord,
    RunRef,
    RuntimeSurfaceEvent,
    SubTag,
    SurfaceBase,
    Trace,
)

# ---------------------------------------------------------------------------
# Helpers — build a real kernel trace and a couple of surface records
# ---------------------------------------------------------------------------


def _build_kernel_trace() -> tuple[KernelRecord, ...]:
    program = Handle(
        Let("y", Perform("ask", Lit(None)), Return(Var("y"))),
        HandlerEnv(
            (
                StaticHandlerInstall(
                    effect_kind="ask",
                    handler_id="local.handler.v1",
                    handled_result_schema=AnySchema(),
                    payload_name="_payload",
                    body=Let("r", Resume(Lit(42)), Return(Var("r"))),
                ),
            )
        ),
    )
    result = run_trace(elaborate(program))
    validate_core_trace(result.trace)
    return result.trace


# Concrete surface record for round-trip testing. Concrete subclasses
# in production land via Plan 02 audit-by-emitter; this is a local
# fixture that registers itself into SURFACE_REGISTRY only for these
# tests.


@dataclass(frozen=True, kw_only=True)
class _RunStartedFixture(SurfaceBase):
    sub_tag: SubTag = SubTag.run
    task_id: str = ""
    inputs: dict = field(default_factory=dict)


SURFACE_REGISTRY.setdefault("_RunStartedFixture", _RunStartedFixture)


def _vcscore_execution_link() -> VcsCoreExecutionLink:
    return VcsCoreExecutionLink(
        workspace_root="/repo",
        vcscore_repo="/repo/.vcscore",
        parent_ref="refs/vcscore/ground",
        child_scope_ref="refs/vcscore/scopes/run-child",
        input_world_oid="world-in",
        output_world_oid="world-out",
        trace_revision_ref="trace-revision",
        carrier_ref="carrier-revision",
        input_head="head-in",
        output_head="head-out",
        terminal_status="merged",
    )


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def test_trace_imports_from_runtime_trace() -> None:
    """E4: ``Trace`` is importable from ``shepherd_runtime.trace``."""
    from shepherd_runtime.trace import Trace

    assert Trace.__name__ == "Trace"


def test_public_trace_annotations_resolve_at_runtime() -> None:
    """Public dataclass annotations remain usable by runtime schema tools."""
    trace_hints = get_type_hints(Trace)
    surface_hints = get_type_hints(SurfaceBase)
    runtime_event_hints = get_type_hints(RuntimeSurfaceEvent)
    run_hints = get_type_hints(Run)

    assert trace_hints["run_ref"] is RunRef
    kernel_record_union = get_args(trace_hints["kernel"])[0]
    assert EffectDeclaration in get_args(kernel_record_union)
    assert surface_hints["sub_tag"] is SubTag
    assert runtime_event_hints["sub_tag"] is SubTag
    assert run_hints["trace"] == Trace | None


def test_trace_default_is_empty_kernel_and_surface() -> None:
    """E4: kernel and surface default to empty tuples."""
    trace = Trace(run_ref=RunRef(id="run-empty"))
    assert trace.kernel == ()
    assert trace.surface == ()


# ---------------------------------------------------------------------------
# filter / cite / round-trip
# ---------------------------------------------------------------------------


def test_filter_returns_narrowed_trace() -> None:
    run_ref = RunRef(id="run-1")
    s_run = _RunStartedFixture(ref="surf_run", timestamp_us=0, run_ref=run_ref, task_id="t")
    trace = Trace(run_ref=run_ref, kernel=(), surface=(s_run,))

    runs = trace.filter(SubTag.run)
    assert isinstance(runs, Trace)
    assert runs.surface == (s_run,)

    control = trace.filter(SubTag.control)
    assert control.surface == ()


def test_cite_locates_kernel_records() -> None:
    kernel = _build_kernel_trace()
    decls = [r for r in kernel if isinstance(r, EffectDeclaration)]
    assert decls, "expected at least one EffectDeclaration"
    trace = Trace(run_ref=RunRef(id="run-1"), kernel=kernel)

    found = trace.cite(decls[0].ref)
    assert found is decls[0]


def test_cite_locates_surface_records() -> None:
    run_ref = RunRef(id="run-1")
    record = _RunStartedFixture(ref="surf_1", timestamp_us=10, run_ref=run_ref, task_id="t")
    trace = Trace(run_ref=run_ref, surface=(record,))
    assert trace.cite("surf_1") is record


def test_cite_raises_lookup_error_on_miss() -> None:
    trace = Trace(run_ref=RunRef(id="run-1"))
    with pytest.raises(LookupError):
        trace.cite("nonexistent")


def test_to_json_returns_dict_with_schema_version() -> None:
    trace = Trace(run_ref=RunRef(id="run-1"))
    payload = trace.to_json()
    assert isinstance(payload, dict)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_ref"] == {"id": "run-1"}
    assert payload["kernel"] == []
    assert payload["surface"] == []


def test_round_trip_empty_trace() -> None:
    trace = Trace(run_ref=RunRef(id="run-1"))
    again = Trace.from_json(trace.to_json())
    assert again == trace


def test_round_trip_kernel_records() -> None:
    kernel = _build_kernel_trace()
    trace = Trace(run_ref=RunRef(id="run-1"), kernel=kernel)
    again = Trace.from_json(trace.to_json())
    assert again == trace


def test_round_trip_with_surface_record() -> None:
    run_ref = RunRef(id="run-1")
    s = _RunStartedFixture(
        ref="surf_1",
        timestamp_us=42,
        run_ref=run_ref,
        branch_scope_ref="branch:root",
        citing=("decl:x",),
        task_id="summarize",
        inputs={"article": "hello"},
    )
    trace = Trace(run_ref=run_ref, kernel=(), surface=(s,))
    again = Trace.from_json(trace.to_json())
    assert again == trace


def test_round_trip_preserves_vcscore_run_ref_link() -> None:
    run_ref = RunRef(id="run-vcscore", vcscore=_vcscore_execution_link())
    trace = Trace(run_ref=run_ref)

    payload = trace.to_json()
    again = Trace.from_json(payload)

    assert "schema" not in payload["run_ref"]
    assert payload["run_ref"]["vcscore"]["output_world_oid"] == "world-out"
    assert again == trace
    assert again.run_ref.vcscore == run_ref.vcscore


def test_round_trip_preserves_surface_record_vcscore_run_ref_link() -> None:
    run_ref = RunRef(id="run-surface-vcscore", vcscore=_vcscore_execution_link())
    surface = _RunStartedFixture(ref="surf_vcscore", timestamp_us=5, run_ref=run_ref, task_id="task")
    trace = Trace(run_ref=RunRef(id="run-parent"), surface=(surface,))

    again = Trace.from_json(trace.to_json())

    assert again.surface == (surface,)
    assert again.surface[0].run_ref == run_ref


def test_from_json_accepts_schema_shaped_run_ref_payload() -> None:
    run_ref = RunRef(id="run-schema", vcscore=_vcscore_execution_link())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_ref": run_ref.to_payload(),
        "kernel": [],
        "surface": [],
    }

    trace = Trace.from_json(payload)

    assert trace.run_ref == run_ref


def test_round_trip_preserves_surface_record_order() -> None:
    run_ref = RunRef(id="run-order")
    first = _RunStartedFixture(ref="surf_1", timestamp_us=1, run_ref=run_ref, task_id="first")
    second = _RunStartedFixture(ref="surf_2", timestamp_us=2, run_ref=run_ref, task_id="second")
    trace = Trace(run_ref=run_ref, surface=(first, second))

    again = Trace.from_json(trace.to_json())

    assert [record.ref for record in again.surface] == ["surf_1", "surf_2"]
    assert again == trace


def test_round_trip_full_trace() -> None:
    kernel = _build_kernel_trace()
    decl_refs = [r.ref for r in kernel if isinstance(r, EffectDeclaration)]
    assert decl_refs

    run_ref = RunRef(id="run-full")
    trace = Trace(
        run_ref=run_ref,
        kernel=kernel,
        surface=(
            _RunStartedFixture(
                ref="surf_1",
                timestamp_us=0,
                run_ref=run_ref,
                citing=(decl_refs[0],),
                task_id="summarize",
            ),
        ),
    )
    again = Trace.from_json(trace.to_json())
    assert again == trace


def test_from_json_rejects_unknown_schema_version() -> None:
    bad = {"schema_version": 999, "run_ref": {"id": "x"}, "kernel": [], "surface": []}
    with pytest.raises(ValueError, match="schema_version"):
        Trace.from_json(bad)


def test_from_json_rejects_missing_envelope_keys() -> None:
    bad = {"schema_version": SCHEMA_VERSION, "run_ref": {"id": "x"}, "kernel": []}
    with pytest.raises(KeyError, match="surface"):
        Trace.from_json(bad)


def test_from_json_rejects_malformed_run_ref() -> None:
    bad = {"schema_version": SCHEMA_VERSION, "run_ref": {}, "kernel": [], "surface": []}
    with pytest.raises(TypeError, match="id"):
        Trace.from_json(bad)


def test_from_json_rejects_surface_entry_without_type() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_ref": {"id": "x"},
        "kernel": [],
        "surface": [{"fields": {"ref": "s", "sub_tag": "run", "timestamp_us": 0}}],
    }
    with pytest.raises(KeyError, match="type"):
        Trace.from_json(payload)


def test_from_json_rejects_unregistered_surface_type() -> None:
    """If a surface record class isn't in SURFACE_REGISTRY, raise LookupError."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_ref": {"id": "x"},
        "kernel": [],
        "surface": [
            {
                "type": "_NotRegistered",
                "fields": {
                    "ref": "s",
                    "sub_tag": "run",
                    "timestamp_us": 0,
                    "citing": [],
                },
            }
        ],
    }
    with pytest.raises(LookupError, match="not registered"):
        Trace.from_json(payload)


def test_from_json_rejects_invalid_surface_sub_tag() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_ref": {"id": "x"},
        "kernel": [],
        "surface": [
            {
                "type": "_RunStartedFixture",
                "fields": {
                    "ref": "s",
                    "sub_tag": "not-a-subtag",
                    "timestamp_us": 0,
                    "citing": [],
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="not-a-subtag"):
        Trace.from_json(payload)
