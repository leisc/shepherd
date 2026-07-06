# under-test: vcs_core._overlay_capture_adapter
"""Unit tests for ``OverlayCaptureAdapter``.

Validates the SPI v0.1 ``CaptureAdapter`` Protocol against the overlay
adapter implementation: parse contract, evidence-kind coverage,
single-sink and multi-sink fan-out under the Q5b discipline.
"""

from __future__ import annotations

from typing import Any

from vcs_core._overlay_capture_adapter import (
    OVERLAY_ADAPTER_ID,
    OVERLAY_ADAPTER_VERSION,
    OVERLAY_EVIDENCE_KIND_METADATA_CHANGE,
    OVERLAY_EVIDENCE_KIND_UNLINK,
    OVERLAY_EVIDENCE_KIND_WRITE_CLOSE,
    OVERLAY_EVIDENCE_KIND_WRITE_OBSERVED,
    OVERLAY_EVIDENCE_KIND_WRITE_OPEN,
    OVERLAY_EVIDENCE_KINDS,
    OVERLAY_MECHANISM,
    OverlayCaptureAdapter,
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


def _capture_event(
    *,
    op: str = "write_close",
    path: str = "src/app.py",
    global_seq: int = 1,
    proc_seq: int = 1,
    event_seq: int = 1,
    command_operation_id: str = "op-cmd",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "CaptureEvent",
        "capture_record": "raw_event",
        "capture_status": "journaled",
        "capture_mode": "direct",
        "capture_mechanism": "preload",
        "op": op,
        "path": path,
        "command_operation_id": command_operation_id,
        "binding_name": "workspace",
        "capture_scope": "scope-1",
        "capture_scope_instance_id": "scope-instance-1",
        "pid": 1000,
        "proc_seq": proc_seq,
        "global_seq": global_seq,
        "event_seq": event_seq,
    }
    if extras:
        base.update(extras)
    return base


def test_overlay_adapter_satisfies_capture_adapter_protocol() -> None:
    adapter = OverlayCaptureAdapter()
    assert isinstance(adapter, CaptureAdapter)
    assert adapter.adapter_id == OVERLAY_ADAPTER_ID
    assert adapter.adapter_version == OVERLAY_ADAPTER_VERSION
    assert adapter.mechanism == OVERLAY_MECHANISM
    assert set(adapter.evidence_kinds) == set(OVERLAY_EVIDENCE_KINDS)


def test_parse_emits_typed_observations_for_recognized_events() -> None:
    adapter = OverlayCaptureAdapter()
    sink = TupleSink()
    raw = [
        _capture_event(op="write_close", path="src/a.py", global_seq=1),
        _capture_event(op="write_open", path="src/b.py", global_seq=2),
        _capture_event(op="unlink", path="src/c.py", global_seq=3),
        _capture_event(op="metadata_change", path="src/d.py", global_seq=4),
        _capture_event(op="write_observed", path="src/e.py", global_seq=5),
    ]

    result = adapter.parse(_ctx(), raw, sink)

    assert isinstance(result, ParseResult)
    assert result.parsed_count == 5
    assert result.diagnostic_count == 0
    assert result.skipped is False
    assert len(sink.observations) == 5
    kinds = [o.evidence_kind for o in sink.observations]
    assert kinds == [
        OVERLAY_EVIDENCE_KIND_WRITE_CLOSE,
        OVERLAY_EVIDENCE_KIND_WRITE_OPEN,
        OVERLAY_EVIDENCE_KIND_UNLINK,
        OVERLAY_EVIDENCE_KIND_METADATA_CHANGE,
        OVERLAY_EVIDENCE_KIND_WRITE_OBSERVED,
    ]
    for o in sink.observations:
        assert o.mechanism == OVERLAY_MECHANISM
        assert o.correlation_id == "op-cmd"
        assert "path" in o.stable_observation


def test_parse_returns_skip_when_no_recognized_events_present() -> None:
    adapter = OverlayCaptureAdapter()
    sink = TupleSink()
    raw = [
        {"type": "FileCreate", "path": "x"},
        {"type": "OperationEnvelope"},
        {},
    ]
    result = adapter.parse(_ctx(), raw, sink)
    assert result.skipped is True
    assert result.parsed_count == 0
    assert sink.observations == []
    assert sink.diagnostics == []


def test_parse_skips_unknown_capture_op_kinds_silently() -> None:
    adapter = OverlayCaptureAdapter()
    sink = TupleSink()
    # A capture-event commit with an op the overlay vocabulary doesn't
    # recognize is silently dropped — capture_event_from_metadata returns
    # None for unknown ops. The result is still "relevant" because a
    # CaptureEvent commit was present.
    raw = [
        _capture_event(op="write_close", path="src/a.py"),
        {**_capture_event(op="write_close", path="x"), "op": "future_kind"},
    ]
    result = adapter.parse(_ctx(), raw, sink)
    assert result.parsed_count == 1
    assert result.diagnostic_count == 0
    assert result.skipped is False
    assert len(sink.observations) == 1


def test_parse_reports_diagnostic_on_malformed_event_metadata() -> None:
    adapter = OverlayCaptureAdapter()
    sink = TupleSink()
    # ``capture_event_from_metadata`` raises ValueError when required
    # string fields are missing or empty; the adapter catches and
    # converts to a diagnostic.
    malformed = _capture_event(op="write_close")
    malformed["command_operation_id"] = ""
    raw = [_capture_event(op="write_close", path="src/a.py"), malformed]

    result = adapter.parse(_ctx(), raw, sink)

    assert result.parsed_count == 1
    assert result.diagnostic_count == 1
    assert len(sink.diagnostics) == 1
    diag = sink.diagnostics[0]
    assert isinstance(diag, Diagnostic)
    assert diag.code == "overlay:parse_error"


def test_parse_observation_ids_are_deterministic() -> None:
    adapter = OverlayCaptureAdapter()
    sink_a, sink_b = TupleSink(), TupleSink()
    raw = [_capture_event(op="write_close", path="src/a.py", global_seq=42)]

    adapter.parse(_ctx(), raw, sink_a)
    adapter.parse(_ctx(), raw, sink_b)

    assert sink_a.observations[0].observation_id == sink_b.observations[0].observation_id


def test_parse_with_fan_out_sink_delivers_to_every_consumer() -> None:
    adapter = OverlayCaptureAdapter()
    coordinator = TupleSink()
    supervisor = TupleSink()
    fan = FanOutSink([coordinator, supervisor])
    raw = [
        _capture_event(op="write_close", path="src/a.py", global_seq=1),
        _capture_event(op="unlink", path="src/b.py", global_seq=2),
    ]

    result = adapter.parse(_ctx(), raw, fan)

    assert result.parsed_count == 2
    assert len(coordinator.observations) == 2
    assert len(supervisor.observations) == 2
    assert [o.observation_id for o in coordinator.observations] == [o.observation_id for o in supervisor.observations]


def test_parse_fan_out_failure_isolation_does_not_block_other_sinks() -> None:
    class _RaisingSink:
        def __init__(self) -> None:
            self.seen: list[ObservationDraft] = []

        def emit(self, observation: ObservationDraft) -> None:
            self.seen.append(observation)
            if len(self.seen) == 2:
                raise RuntimeError("supervisor exploded")

        def diagnostic(self, diagnostic: Diagnostic) -> None:
            return None

    adapter = OverlayCaptureAdapter()
    coordinator = TupleSink()
    raising = _RaisingSink()
    fan = FanOutSink([coordinator, raising])

    raw = [
        _capture_event(op="write_close", path="src/a.py", global_seq=1),
        _capture_event(op="write_close", path="src/b.py", global_seq=2),
        _capture_event(op="write_close", path="src/c.py", global_seq=3),
    ]
    result = adapter.parse(_ctx(), raw, fan)

    assert result.parsed_count == 3
    assert len(coordinator.observations) == 3  # coordinator never affected
    # Raising sink saw all three (it raises after recording the 2nd, but
    # FanOutSink calls emit on the 3rd unconditionally per the discipline).
    assert len(raising.seen) == 3
    assert len(fan.failures) == 1
    assert fan.failures[0].operation == "emit"
