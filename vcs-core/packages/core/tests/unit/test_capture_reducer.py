# under-test: vcs_core._capture_reducer
"""Capture journal parsing and reducer helper tests."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from vcs_core._capture_reducer import (
    CAPTURE_EVENT_EFFECT,
    capture_event_from_metadata,
    capture_event_metadata,
    covered_capture_paths,
    ordered_capture_events,
    reduction_operation_id,
)
from vcs_core._fs_capture import FsCaptureEvent


def test_capture_event_metadata_round_trips() -> None:
    metadata = capture_event_metadata(
        command_operation_id="cmd-1",
        capture_epoch="cap-1",
        binding_name="filesystem",
        event=FsCaptureEvent(
            op="write_close",
            scope="task",
            scope_instance_id="iid-1",
            path="src/app.py",
            pid=10,
            proc_seq=2,
            ppid=9,
            exe="/bin/bash",
            cwd="/workspace",
        ),
        global_seq=7,
        event_seq=3,
        capture_mechanism="preload",
    )
    metadata["type"] = CAPTURE_EVENT_EFFECT

    event = capture_event_from_metadata(metadata)

    assert event is not None
    assert event.command_operation_id == "cmd-1"
    assert event.capture_epoch == "cap-1"
    assert event.path == "src/app.py"
    assert event.global_seq == 7
    assert event.ppid == 9


@pytest.mark.parametrize("op", ["write_open", "write_observed"])
def test_capture_event_from_metadata_accepts_mutation_time_write_ops(op: str) -> None:
    metadata = {**_metadata("a.txt", global_seq=1), "type": CAPTURE_EVENT_EFFECT, "op": op}

    event = capture_event_from_metadata(metadata)

    assert event is not None
    assert event.op == op
    assert event.path == "a.txt"


def test_capture_event_from_metadata_rejects_shell_finish_as_raw_path_event() -> None:
    metadata = {**_metadata("a.txt", global_seq=1), "type": CAPTURE_EVENT_EFFECT, "op": "shell_command_finish"}

    assert capture_event_from_metadata(metadata) is None


def test_ordered_capture_events_uses_daemon_global_sequence() -> None:
    commits = (
        SimpleNamespace(metadata={**_metadata("b.txt", global_seq=2), "type": CAPTURE_EVENT_EFFECT}),
        SimpleNamespace(metadata={**_metadata("a.txt", global_seq=1), "type": CAPTURE_EVENT_EFFECT}),
    )

    events = ordered_capture_events(commits)

    assert [event.path for event in events] == ["a.txt", "b.txt"]
    assert covered_capture_paths(events) == ("a.txt", "b.txt")


def test_ordered_capture_events_deduplicates_same_path_after_global_ordering() -> None:
    commits = (
        SimpleNamespace(metadata={**_metadata("same.txt", global_seq=5, pid=202), "type": CAPTURE_EVENT_EFFECT}),
        SimpleNamespace(metadata={**_metadata("other.txt", global_seq=3, pid=303), "type": CAPTURE_EVENT_EFFECT}),
        SimpleNamespace(metadata={**_metadata("same.txt", global_seq=1, pid=101), "type": CAPTURE_EVENT_EFFECT}),
    )

    events = ordered_capture_events(commits)

    assert [(event.path, event.pid) for event in events] == [
        ("same.txt", 101),
        ("other.txt", 303),
        ("same.txt", 202),
    ]
    assert covered_capture_paths(events) == ("same.txt", "other.txt")


def test_ordered_capture_events_stress_same_path_multi_pid_jitter() -> None:
    pids = (101, 102, 103, 104)
    iterations = 100

    for iteration in range(iterations):
        rng = random.Random(iteration)
        commits = [
            SimpleNamespace(
                metadata={
                    **_metadata("shared.txt", global_seq=global_seq, pid=pid, proc_seq=proc_seq),
                    "type": CAPTURE_EVENT_EFFECT,
                }
            )
            for global_seq, (pid, proc_seq) in enumerate(
                ((pid, proc_seq) for proc_seq in range(1, 5) for pid in pids),
                start=1,
            )
        ]
        rng.shuffle(commits)

        events = ordered_capture_events(tuple(commits))

        assert [event.global_seq for event in events] == list(range(1, 17))
        assert covered_capture_paths(events) == ("shared.txt",)
        assert events[-1].pid == 104
        assert events[-1].proc_seq == 4


def test_capture_event_from_metadata_rejects_malformed_ints() -> None:
    metadata = {**_metadata("a.txt", global_seq=True), "type": CAPTURE_EVENT_EFFECT}

    with pytest.raises(ValueError, match="global_seq"):
        capture_event_from_metadata(metadata)


def test_reduction_operation_id_is_linked_to_command_id() -> None:
    assert reduction_operation_id("cmd-abc") == "red_cmd-abc"


def _metadata(path: str, *, global_seq: object, pid: int = 10, proc_seq: int = 1) -> dict[str, object]:
    return {
        "command_operation_id": "cmd-1",
        "binding_name": "filesystem",
        "capture_record": "raw_event",
        "capture_mechanism": "preload",
        "op": "write_close",
        "path": path,
        "capture_scope": "task",
        "capture_scope_instance_id": "iid-1",
        "pid": pid,
        "proc_seq": proc_seq,
        "global_seq": global_seq,
        "event_seq": 1,
    }
