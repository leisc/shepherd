# under-test: vcs_core._python_runtime_capture_adapter
"""Unit tests for ``PythonRuntimeCaptureAdapter``.

Validates the SPI v0.1 ``CaptureAdapter`` Protocol against the
python-runtime adapter (T2b): parse contract, evidence-kind coverage,
mechanism declaration, deterministic observation ids, single-sink and
multi-sink fan-out under the Q5b discipline, and malformed-event
diagnostic emission.

The adapter is registry-owned per SPI v0.1 §Q2 Discovery boundary
criterion (the patch manager owns its lifetime, not a single substrate
driver). T2c wires it into ``CaptureAdapterRegistry`` and rewires
``_vcscore_runtime.py`` to route Python-tier events through the
adapter → coordinator persistence → ``ReduceRequest`` flow.
"""

from __future__ import annotations

from typing import Any

from vcs_core._python_runtime_capture_adapter import (
    PYTHON_RUNTIME_ADAPTER_ID,
    PYTHON_RUNTIME_ADAPTER_VERSION,
    PYTHON_RUNTIME_EFFECT_TYPE,
    PYTHON_RUNTIME_MECHANISM,
    PythonRuntimeCaptureAdapter,
)
from vcs_core._substrate_evidence_kinds import (
    PYTHON_RUNTIME_EVIDENCE_KINDS,
    EvidenceKind,
    Mechanism,
)
from vcs_core.runtime_api import DriverContext
from vcs_core.spi import (
    CaptureAdapter,
    Diagnostic,
    FanOutSink,
    ObservationDraft,
    ParseResult,
    SubstrateStoreIdentity,
    TupleSink,
)


def _ctx() -> DriverContext:
    return DriverContext(
        operation_id="op-test",
        binding="workspace",
        role="shepherd.WorkspaceRef",
        store_identity=SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="ws:test"),
    )


def _write_event(
    *,
    path: str = "src/app.py",
    content_digest: str = "sha256:abcd",
    mode: int = 0o100644,
    global_seq: int = 1,
    command_operation_id: str = "op-cmd",
) -> dict[str, Any]:
    return {
        "type": PYTHON_RUNTIME_EFFECT_TYPE,
        "op": "write",
        "path": path,
        "content_digest": content_digest,
        "mode": mode,
        "command_operation_id": command_operation_id,
        "binding_name": "workspace",
        "global_seq": global_seq,
    }


def _delete_event(
    *,
    path: str = "src/old.py",
    global_seq: int = 2,
    command_operation_id: str = "op-cmd",
) -> dict[str, Any]:
    return {
        "type": PYTHON_RUNTIME_EFFECT_TYPE,
        "op": "delete",
        "path": path,
        "command_operation_id": command_operation_id,
        "binding_name": "workspace",
        "global_seq": global_seq,
    }


# ---------------------------------------------------------------------------
# Protocol conformance + identity properties
# ---------------------------------------------------------------------------


def test_adapter_satisfies_capture_adapter_protocol() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    assert isinstance(adapter, CaptureAdapter)


def test_adapter_identity_properties() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    assert adapter.adapter_id == PYTHON_RUNTIME_ADAPTER_ID
    assert adapter.adapter_version == PYTHON_RUNTIME_ADAPTER_VERSION
    assert adapter.mechanism == PYTHON_RUNTIME_MECHANISM
    assert adapter.mechanism == Mechanism.PYTHON_RUNTIME
    assert adapter.evidence_kinds == PYTHON_RUNTIME_EVIDENCE_KINDS


def test_evidence_kinds_are_mechanism_prefixed() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    for kind in adapter.evidence_kinds:
        assert kind.startswith(f"{Mechanism.PYTHON_RUNTIME}:"), f"evidence_kind {kind!r} is not mechanism-prefixed"


# ---------------------------------------------------------------------------
# Parse — happy path
# ---------------------------------------------------------------------------


def test_parse_emits_write_observation_with_evidence_kind_write() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    result = adapter.parse(_ctx(), [_write_event()], sink)
    assert isinstance(result, ParseResult)
    assert result.parsed_count == 1
    assert result.diagnostic_count == 0
    assert not result.skipped
    assert len(sink.observations) == 1
    obs = sink.observations[0]
    assert isinstance(obs, ObservationDraft)
    assert obs.evidence_kind == EvidenceKind.PYTHON_RUNTIME_WRITE
    assert obs.mechanism == PYTHON_RUNTIME_MECHANISM
    assert obs.stable_observation["op"] == "write"
    assert obs.stable_observation["path"] == "src/app.py"
    assert obs.stable_observation["content_digest"] == "sha256:abcd"
    assert obs.stable_observation["mode"] == 0o100644
    assert obs.correlation_id == "op-cmd"


def test_parse_emits_delete_observation_with_evidence_kind_delete() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    result = adapter.parse(_ctx(), [_delete_event()], sink)
    assert result.parsed_count == 1
    assert len(sink.observations) == 1
    obs = sink.observations[0]
    assert obs.evidence_kind == EvidenceKind.PYTHON_RUNTIME_DELETE
    assert obs.stable_observation["op"] == "delete"
    # Delete events don't carry content_digest or mode.
    assert "content_digest" not in obs.stable_observation
    assert "mode" not in obs.stable_observation


def test_parse_emits_one_observation_per_event_preserving_order() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    events = [
        _write_event(path="a.py", global_seq=1),
        _delete_event(path="b.py", global_seq=2),
        _write_event(path="c.py", global_seq=3),
    ]
    result = adapter.parse(_ctx(), events, sink)
    assert result.parsed_count == 3
    paths = [obs.stable_observation["path"] for obs in sink.observations]
    assert paths == ["a.py", "b.py", "c.py"]


# ---------------------------------------------------------------------------
# Parse — skip / filter
# ---------------------------------------------------------------------------


def test_parse_skips_when_no_python_runtime_events() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    other_events = [
        {"type": "CaptureEvent", "op": "write_close"},
        {"type": "SomeOtherEffect", "op": "write"},
    ]
    result = adapter.parse(_ctx(), other_events, sink)
    assert result.skipped is True
    assert result.parsed_count == 0
    assert len(sink.observations) == 0


def test_parse_skips_empty_event_list() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    result = adapter.parse(_ctx(), [], sink)
    assert result.skipped is True
    assert result.parsed_count == 0


# ---------------------------------------------------------------------------
# Parse — malformed-event diagnostics
# ---------------------------------------------------------------------------


def test_parse_emits_diagnostic_on_unknown_op() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    bad = {
        "type": PYTHON_RUNTIME_EFFECT_TYPE,
        "op": "frobnicate",
        "path": "weird.py",
    }
    result = adapter.parse(_ctx(), [bad, _write_event()], sink)
    assert result.parsed_count == 1
    assert result.diagnostic_count == 1
    assert len(sink.diagnostics) == 1
    diagnostic = sink.diagnostics[0]
    assert isinstance(diagnostic, Diagnostic)
    assert diagnostic.code == "python-runtime:unknown_op"
    assert "frobnicate" in diagnostic.detail.get("op", "")


def test_parse_emits_diagnostic_on_missing_path() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    bad = {
        "type": PYTHON_RUNTIME_EFFECT_TYPE,
        "op": "write",
        # No path
    }
    result = adapter.parse(_ctx(), [bad], sink)
    assert result.parsed_count == 0
    assert result.diagnostic_count == 1
    assert sink.diagnostics[0].code == "python-runtime:malformed_event"


# ---------------------------------------------------------------------------
# Deterministic observation ids
# ---------------------------------------------------------------------------


def test_observation_ids_are_deterministic_across_reparses() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    events = [
        _write_event(path="a.py", global_seq=1, command_operation_id="op-X"),
        _delete_event(path="b.py", global_seq=2, command_operation_id="op-X"),
    ]
    sink_one = TupleSink()
    sink_two = TupleSink()
    adapter.parse(_ctx(), events, sink_one)
    adapter.parse(_ctx(), events, sink_two)
    ids_one = [obs.observation_id for obs in sink_one.observations]
    ids_two = [obs.observation_id for obs in sink_two.observations]
    assert ids_one == ids_two


def test_observation_ids_are_unique_within_a_parse() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink = TupleSink()
    # Multiple writes to the same path within one operation must produce
    # distinct ids via the array index suffix.
    events = [
        _write_event(path="a.py", global_seq=1),
        _write_event(path="a.py", global_seq=1),
        _write_event(path="a.py", global_seq=1),
    ]
    adapter.parse(_ctx(), events, sink)
    ids = [obs.observation_id for obs in sink.observations]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Q5b fan-out delivery
# ---------------------------------------------------------------------------


def test_parse_delivers_to_fanout_sinks_in_emission_order() -> None:
    adapter = PythonRuntimeCaptureAdapter()
    sink_a = TupleSink()
    sink_b = TupleSink()
    fanout = FanOutSink([sink_a, sink_b])
    events = [
        _write_event(path="a.py", global_seq=1),
        _write_event(path="b.py", global_seq=2),
        _delete_event(path="c.py", global_seq=3),
    ]
    result = adapter.parse(_ctx(), events, fanout)
    assert result.parsed_count == 3
    paths_a = [obs.stable_observation["path"] for obs in sink_a.observations]
    paths_b = [obs.stable_observation["path"] for obs in sink_b.observations]
    assert paths_a == ["a.py", "b.py", "c.py"]
    assert paths_b == paths_a


def test_parse_isolates_per_sink_emit_failure() -> None:
    adapter = PythonRuntimeCaptureAdapter()

    class _RaisingSink(TupleSink):
        def emit(self, observation: ObservationDraft) -> None:
            raise RuntimeError("boom")

    sink_a = TupleSink()
    raising = _RaisingSink()
    sink_c = TupleSink()
    fanout = FanOutSink([sink_a, raising, sink_c])

    events = [_write_event(path="ok.py", global_seq=1)]
    adapter.parse(_ctx(), events, fanout)

    # Good sinks still received the observation.
    assert len(sink_a.observations) == 1
    assert len(sink_c.observations) == 1
    # Failure isolated and recorded.
    assert len(fanout.failures) == 1
    assert fanout.failures[0].operation == "emit"


# ---------------------------------------------------------------------------
# Context invariance (v0.1 contract)
# ---------------------------------------------------------------------------


def test_parse_does_not_read_driver_context_in_v01() -> None:
    """Per docstring, the v0.1 adapter is context-invariant.

    Different contexts produce byte-identical observation ids for the
    same input events. T2c may revisit this when wiring the adapter to
    real runtime ingress; until then, observation identity is purely
    event-derived.
    """
    adapter = PythonRuntimeCaptureAdapter()
    sink_one = TupleSink()
    sink_two = TupleSink()

    ctx_one = _ctx()
    ctx_two = DriverContext(
        operation_id="completely-different-op",
        binding="workspace",
        role="shepherd.WorkspaceRef",
        store_identity=SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="ws:other"),
    )

    event = _write_event(path="a.py", global_seq=1, command_operation_id="op-cmd")
    adapter.parse(ctx_one, [event], sink_one)
    adapter.parse(ctx_two, [event], sink_two)
    assert sink_one.observations[0].observation_id == sink_two.observations[0].observation_id
