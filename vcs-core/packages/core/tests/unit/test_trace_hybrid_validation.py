# under-test: vcs_core._world_substrate_adapters
"""B4b slice 2 (W1/W2): driver-side hybrid trace validation + the evidence-class pin.

The 5-field floor stays a floor (payloads without `events` are untouched); hybrid
payloads must obey the pointer/record discipline; unknown kinds are admitted iff
namespaced — the explicit extension point (trace-substrate-hybrid.md, slice-2 status).
"""

from __future__ import annotations

import pytest
from vcs_core._world_substrate_adapters import (
    TaskTraceSubstrateDriver,
    _trace_revision_payload,
)

FLOOR = {"trace_runtime": "rt", "trace_owner_id": "owner", "frontier_id": "f"}


def hybrid(**overrides):
    payload = {
        **FLOOR,
        "identity_domain": "vcscore.canonical.v2",
        "events": [
            {
                "id": "task-invocation",
                "kind": "task.invocation",
                "identity_domain": "shepherd.kernel.canonical.v2",
                "record_digest": "sha256:abc",
                "body": {"task_id": "t"},
            },
            {"id": "workspace-transition", "kind": "substrate.transition", "head_to": "oid"},
            {"id": "run-lifecycle", "kind": "run.lifecycle", "terminal_status": "merged"},
        ],
        "causal_edges": [["task-invocation", "workspace-transition"]],
        "owner_paths": {"owner": ["task-invocation", "run-lifecycle"]},
    }
    payload.update(overrides)
    return payload


def test_floor_payload_passes_untouched():
    assert _trace_revision_payload(dict(FLOOR))["trace_owner_id"] == "owner"


def test_hybrid_payload_passes():
    assert len(_trace_revision_payload(hybrid())["events"]) == 3


def test_unknown_namespaced_kind_is_the_extension_point():
    extra = hybrid()
    extra["events"].append({"id": "w1", "kind": "weld.observed"})
    _trace_revision_payload(extra)


def test_effect_type_record_kind_admitted():
    # The skeleton records capture-lane effects with effect-type kinds (FileCreate);
    # existing producers are the validator's calibration.
    extra = hybrid()
    extra["events"].append({"id": "fc1", "kind": "FileCreate", "path": "README.md"})
    _trace_revision_payload(extra)


def test_unnamespaced_kind_refused():
    bad = hybrid()
    bad["events"][2] = {"id": "x", "kind": "lifecycle"}
    with pytest.raises(ValueError, match="namespaced"):
        _trace_revision_payload(bad)


def test_pointer_with_digest_refused():
    bad = hybrid()
    bad["events"][1]["record_digest"] = "sha256:nope"
    with pytest.raises(ValueError, match="no record_digest"):
        _trace_revision_payload(bad)


def test_digested_record_without_domain_refused():
    bad = hybrid()
    del bad["events"][0]["identity_domain"]
    with pytest.raises(ValueError, match="identity_domain"):
        _trace_revision_payload(bad)


def test_missing_header_domain_refused():
    bad = hybrid()
    del bad["identity_domain"]
    with pytest.raises(ValueError, match="identity_domain"):
        _trace_revision_payload(bad)


def test_duplicate_event_ids_refused():
    bad = hybrid()
    bad["events"][2]["id"] = "task-invocation"
    with pytest.raises(ValueError, match="duplicate"):
        _trace_revision_payload(bad)


def test_causal_edge_to_unknown_id_refused():
    bad = hybrid(causal_edges=[["task-invocation", "ghost"]])
    with pytest.raises(ValueError, match="causal edge"):
        _trace_revision_payload(bad)


def test_owner_path_unknown_id_refused():
    bad = hybrid(owner_paths={"owner": ["ghost"]})
    with pytest.raises(ValueError, match="owner path"):
        _trace_revision_payload(bad)


def test_trace_substrate_is_evidence_class():
    assert TaskTraceSubstrateDriver().lifecycle_class == "evidence"
