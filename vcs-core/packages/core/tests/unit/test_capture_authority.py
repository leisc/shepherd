# under-test: vcs_core._capture_authority
"""Capture authority state-machine tests."""

from __future__ import annotations

import random
import threading
import time

from vcs_core._capture_authority import CaptureAuthority


def test_capture_authority_drains_processed_command_events() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")

    accepted = authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    authority.mark_processed("cmd-1", global_seq=1)
    drained = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert accepted.accepted
    assert drained.complete
    assert drained.high_water_by_pid == {10: 1}
    assert authority.status("cmd-1") == "complete"


def test_capture_authority_tolerates_out_of_order_per_pid_arrival() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")

    second = authority.accept_event("cmd-1", pid=10, proc_seq=2, global_seq=2)
    first = authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    authority.mark_processed("cmd-1", global_seq=2)
    authority.mark_processed("cmd-1", global_seq=1)
    drained = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert second.accepted
    assert first.accepted
    assert drained.complete
    assert drained.high_water_by_pid == {10: 2}
    assert authority.status("cmd-1") == "complete"


def test_capture_authority_detects_per_pid_sequence_gap_at_drain_deadline() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")

    accepted = authority.accept_event("cmd-1", pid=10, proc_seq=2, global_seq=1)
    authority.mark_processed("cmd-1", global_seq=1)
    drained = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert accepted.accepted
    assert not drained.complete
    assert drained.reason == "hook_proc_seq_gap"
    assert drained.high_water_by_pid == {10: 0}
    assert authority.status("cmd-1") == "incomplete"


def test_capture_authority_detects_duplicate_per_pid_sequence() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")

    authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    accepted = authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=2)
    authority.mark_processed("cmd-1", global_seq=1)
    authority.mark_processed("cmd-1", global_seq=2)
    drained = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert accepted.accepted
    assert not drained.complete
    assert drained.reason == "hook_proc_seq_duplicate"
    assert drained.high_water_by_pid == {10: 1}
    assert authority.status("cmd-1") == "incomplete"


def test_capture_authority_marks_failed_event_incomplete() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")

    accepted = authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    authority.mark_failed("cmd-1", global_seq=1, reason="capture_persist_failed")
    drained = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert accepted.accepted
    assert not drained.complete
    assert drained.reason == "capture_persist_failed"
    assert drained.accepted_count == 1
    assert drained.processed_count == 0
    assert authority.status("cmd-1") == "incomplete"


def test_capture_authority_stress_drains_interleaved_multi_pid_events() -> None:
    pids = (101, 102, 103, 104)
    iterations = 100

    for iteration in range(iterations):
        rng = random.Random(iteration)
        authority = CaptureAuthority()
        authority.begin("cmd-1")
        events = [
            (global_seq, pid, proc_seq, rng.random() / 10000, rng.random() / 10000)
            for global_seq, (pid, proc_seq) in enumerate(
                ((pid, proc_seq) for proc_seq in range(1, 5) for pid in pids),
                start=1,
            )
        ]
        rng.shuffle(events)

        def deliver(
            global_seq: int,
            pid: int,
            proc_seq: int,
            accept_delay: float,
            process_delay: float,
            authority: CaptureAuthority = authority,
        ) -> None:
            time.sleep(accept_delay)
            accepted = authority.accept_event("cmd-1", pid=pid, proc_seq=proc_seq, global_seq=global_seq)
            assert accepted.accepted
            time.sleep(process_delay)
            authority.mark_processed("cmd-1", global_seq=global_seq)

        threads = [threading.Thread(target=deliver, args=event) for event in events]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        drained = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

        assert drained.complete
        assert drained.high_water_by_pid == dict.fromkeys(pids, 4)
        assert authority.status("cmd-1") == "complete"


def test_capture_authority_rejects_late_events_after_complete() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")
    authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0).complete

    late = authority.accept_event("cmd-1", pid=10, proc_seq=2, global_seq=2)

    assert not late.accepted
    assert late.reason == "capture_complete"


def test_capture_authority_finalize_compacts_completed_state() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1")
    authority.accept_event("cmd-1", pid=10, proc_seq=1, global_seq=1)
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0).complete

    authority.finalize("cmd-1")
    late = authority.accept_event("cmd-1", pid=10, proc_seq=2, global_seq=2)

    assert authority.active_count() == 0
    assert authority.status("cmd-1") == "complete"
    assert not late.accepted
    assert late.reason == "capture_complete"


def test_capture_authority_lifecycle_markers_complete_after_quiet_period() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", require_lifecycle=True)

    assert authority.register_process("cmd-1", pid=101).accepted
    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_process("cmd-1", pid=101, last_proc_seq=1).accepted

    result = authority.drain("cmd-1", timeout_seconds=0.1, quiet_period_seconds=0.0)

    assert result.complete is True
    assert result.registered_count == 1
    assert result.finished_count == 1


def test_capture_authority_lifecycle_requires_process_start() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", require_lifecycle=True)

    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_process_start"


def test_capture_authority_lifecycle_requires_process_finish() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", require_lifecycle=True)

    assert authority.register_process("cmd-1", pid=101).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_process_finish"


def test_capture_authority_lifecycle_finish_high_water_detects_missing_event() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", require_lifecycle=True)

    assert authority.register_process("cmd-1", pid=101).accepted
    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_process("cmd-1", pid=101, last_proc_seq=2).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "hook_proc_seq_gap"


def test_capture_authority_lifecycle_rejects_finish_without_start() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", require_lifecycle=True)

    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_process("cmd-1", pid=101, last_proc_seq=1).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_process_start"


def test_capture_authority_shell_command_requires_finish_barrier() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_shell_command_finish"
    assert result.capture_policy == "shell_command"


def test_capture_authority_shell_command_allows_shell_only_capture_without_process_start() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=2).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert result.complete is True
    assert result.capture_policy == "shell_command"


def test_capture_authority_shell_command_requires_observed_child_finish() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.register_process("cmd-1", pid=202).accepted
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=2).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "background_process_still_running"


def test_capture_authority_shell_command_requires_observed_child_finish_without_process_start() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=202, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=1).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "background_process_still_running"


def test_capture_authority_shell_command_rejects_child_finish_without_process_start() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=202, proc_seq=1, global_seq=1).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    assert authority.finish_process("cmd-1", pid=202, last_proc_seq=1).accepted
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=1).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_process_start"


def test_capture_authority_shell_command_rejects_child_lifecycle_without_start_even_without_events() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.finish_process("cmd-1", pid=202, last_proc_seq=0).accepted
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=1).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "missing_process_start"


def test_capture_authority_shell_command_finish_barrier_detects_missing_prior_shell_event() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=2).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "hook_proc_seq_gap"


def test_capture_authority_shell_command_finish_rejects_wrong_pid() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.finish_shell_command("cmd-1", pid=202, proc_seq=1).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.01, quiet_period_seconds=0.0)

    assert result.complete is False
    assert result.reason == "shell_pid_mismatch"


def test_capture_authority_shell_command_completes_after_child_finish() -> None:
    authority = CaptureAuthority()
    authority.begin("cmd-1", capture_policy="shell_command", shell_pid=101)

    assert authority.accept_event("cmd-1", pid=101, proc_seq=1, global_seq=1).accepted
    assert authority.accept_event("cmd-1", pid=202, proc_seq=1, global_seq=2).accepted
    authority.mark_processed("cmd-1", global_seq=1)
    authority.mark_processed("cmd-1", global_seq=2)
    assert authority.register_process("cmd-1", pid=202).accepted
    assert authority.finish_process("cmd-1", pid=202, last_proc_seq=1).accepted
    assert authority.finish_shell_command("cmd-1", pid=101, proc_seq=2).accepted
    result = authority.drain("cmd-1", timeout_seconds=0.05, quiet_period_seconds=0.0)

    assert result.complete is True
    assert result.high_water_by_pid == {101: 1, 202: 1}
