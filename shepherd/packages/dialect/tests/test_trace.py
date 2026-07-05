"""W1: the hybrid trace builder + the D1 harvest import boundary (B4b slice 1)."""

from __future__ import annotations

import json
import subprocess
import sys

from shepherd2.kernel.canonical import canonical_digest

from shepherd_dialect.trace import (
    SHEPHERD_KERNEL_DOMAIN,
    VCSCORE_DOMAIN,
    build_run_trace_revision,
    task_invocation_record,
)


def _revision(**overrides):
    kwargs = {
        "run_ref": "run_abc",
        "trace_owner_id": "task:demo:run_abc",
        "frontier_id": "frontier:1",
        "task_id": "pkg.mod:demo_task",
        "args": {"target": "README.md"},
        "may_profile": "ReadOnly",
        "terminal_status": "merged",
        "input_world_oid": "w-in",
        "output_world_oid": "w-out",
    }
    kwargs.update(overrides)
    return build_run_trace_revision(**kwargs)


def test_hybrid_payload_shape_floor_and_structure() -> None:
    rev = _revision()
    # The 5-field floor's semantic fields + the hybrid structure.
    for field in ("trace_runtime", "trace_owner_id", "frontier_id", "run_ref", "identity_domain"):
        assert rev[field]
    kinds = [e["kind"] for e in rev["events"]]
    assert kinds == ["task.invocation", "substrate.transition", "run.lifecycle"]
    assert rev["identity_domain"] == VCSCORE_DOMAIN
    assert rev["owner_paths"]["task:demo:run_abc"] == [e["id"] for e in rev["events"]]
    assert rev["causal_edges"] == [
        ["task-invocation", "workspace-transition"],
        ["workspace-transition", "run-lifecycle"],
    ]


def test_fourth_row_record_is_shepherd_domain_and_recomputes() -> None:
    rev = _revision()
    invocation = rev["events"][0]
    assert invocation["identity_domain"] == SHEPHERD_KERNEL_DOMAIN
    # Body digest (D2's pinned shape): recomputes byte-exactly after a JSON round-trip.
    round_tripped = json.loads(json.dumps(invocation["body"]))
    assert canonical_digest(round_tripped) == invocation["record_digest"]


def test_cross_run_identity_same_task_same_digest() -> None:
    one = task_invocation_record(task_id="pkg:fn", args={"a": 1}, may_profile="Permissive")
    two = task_invocation_record(task_id="pkg:fn", args={"a": 1}, may_profile="Permissive")
    other_args = task_invocation_record(task_id="pkg:fn", args={"a": 2}, may_profile="Permissive")
    other_may = task_invocation_record(task_id="pkg:fn", args={"a": 1}, may_profile="ReadOnly")
    assert one["record_digest"] == two["record_digest"]
    assert one["record_digest"] != other_args["record_digest"]
    # The effect surface is part of the cross-run identity, deliberately (D2).
    assert one["record_digest"] != other_may["record_digest"]


def test_defaulted_and_declared_permissive_are_the_same_cross_run_fact() -> None:
    """Provenance is housekeeping, not identity: a defaulted-Permissive run and a
    declared-Permissive run carry the SAME fourth-row digest (same semantic
    surface), while the lifecycle event keeps them distinguishable/countable.
    """
    declared = _revision(may_profile="Permissive", may_source="declared")
    defaulted = _revision(may_profile="Permissive", may_source="defaulted")
    assert declared["events"][0]["record_digest"] == defaulted["events"][0]["record_digest"]
    assert declared["events"][-1]["may_source"] == "declared"
    assert defaulted["events"][-1]["may_source"] == "defaulted"
    # The countable marker: the defaulted population is one filter away.
    revisions = [declared, defaulted]
    counted = [r for r in revisions if r["events"][-1]["may_source"] == "defaulted"]
    assert counted == [defaulted]


def test_pointer_entries_carry_no_record_digest() -> None:
    rev = _revision()
    transition = rev["events"][1]
    assert transition["binding"] == "workspace"
    assert "record_digest" not in transition
    assert transition["head_from"] == "w-in"
    assert transition["head_to"] == "w-out"


def test_failure_path_payload_discarded_with_no_output_world() -> None:
    rev = _revision(terminal_status="discarded", output_world_oid=None)
    lifecycle = rev["events"][-1]
    assert lifecycle["transition"] == "failed"
    assert lifecycle["terminal_status"] == "discarded"
    assert rev["events"][1]["head_to"] is None


def test_retained_path_payload_is_not_failed() -> None:
    rev = _revision(terminal_status="retained", output_world_oid="w-out")
    lifecycle = rev["events"][-1]
    assert lifecycle["transition"] == "retained"
    assert lifecycle["terminal_status"] == "retained"
    assert rev["events"][1]["head_to"] == "w-out"


def test_extra_events_get_ids_and_ride_the_owner_path() -> None:
    rev = _revision(extra_events=({"kind": "model.call", "prompt": "p"},))
    kinds = [e["kind"] for e in rev["events"]]
    assert kinds == ["task.invocation", "substrate.transition", "model.call", "run.lifecycle"]
    assert rev["events"][2]["id"] == "e1"


def test_harvest_import_boundary_kernel_ring_only() -> None:
    """D1: importing the dialect's trace component loads shepherd2's kernel ring ONLY.

    Fresh interpreter with -P (safe path): the repo-root `shepherd2/` DIRECTORY
    otherwise shadows the installed package as a namespace package when cwd is
    on sys.path — the assert on __file__ guards against testing the shadow.
    """
    code = (
        "import sys\n"
        "import shepherd_dialect.trace\n"
        "import shepherd2\n"
        "assert shepherd2.__file__ is not None, 'namespace-shadow: not the installed package'\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('shepherd2'))\n"
        "forbidden = [m for m in loaded if m.startswith("
        "('shepherd2.runtime', 'shepherd2.vnext', 'shepherd2.trace_store'))]\n"
        "assert not forbidden, f'forbidden rings loaded: {forbidden}'\n"
        "assert any(m.startswith('shepherd2.kernel') for m in loaded), loaded\n"
        "print('boundary OK')\n"
    )
    proc = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, check=True)
    assert "boundary OK" in proc.stdout
