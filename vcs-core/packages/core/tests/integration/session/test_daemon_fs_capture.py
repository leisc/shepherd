# under-test: vcs_core._session
"""Session daemon filesystem hook routing tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from vcs_core import build_builtin_substrate_context
from vcs_core._substrate_runtime import BuiltInRuntimeBinding
from vcs_core.substrates import FilesystemSubstrate
from vcs_core.testing import HookManager
from vcs_core.types import BoundSubstrate, EffectRecord, ScopeInfo


def test_wait_for_hook_drain_reports_complete_and_incomplete(workspace: Path) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))

    assert daemon._wait_for_hook_drain(timeout_seconds=0.05, quiet_period_seconds=0.0)

    seq1 = daemon._hook_frontier.accept_next()
    daemon._hook_frontier.accept_next()
    daemon._hook_frontier.mark_terminal(seq1)
    daemon._hook_accepted_seq = daemon._hook_frontier.accepted_seq
    daemon._hook_processed_seq = daemon._hook_frontier.processed_seq

    assert not daemon._wait_for_hook_drain(timeout_seconds=0.01, quiet_period_seconds=0.0)


def test_wait_for_hook_drain_waits_through_required_snapshot(workspace: Path) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    seq1 = daemon._hook_frontier.accept_next()
    seq2 = daemon._hook_frontier.accept_next()
    daemon._hook_frontier.mark_terminal(seq1)
    daemon._hook_accepted_seq = daemon._hook_frontier.accepted_seq
    daemon._hook_processed_seq = daemon._hook_frontier.processed_seq

    def finish_processing() -> None:
        time.sleep(0.02)
        with daemon._lock:
            daemon._hook_frontier.mark_terminal(seq2)
            daemon._hook_processed_seq = daemon._hook_frontier.processed_seq

    thread = threading.Thread(target=finish_processing)
    thread.start()
    try:
        assert daemon._wait_for_hook_drain(
            min_accepted_seq=2,
            timeout_seconds=0.5,
            quiet_period_seconds=0.0,
        )
    finally:
        thread.join()


def test_wait_for_hook_drain_requires_contiguous_processed_frontier(workspace: Path) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    with daemon._lock:
        seq1 = daemon._hook_frontier.accept_next()
        seq2 = daemon._hook_frontier.accept_next()
        assert (seq1, seq2) == (1, 2)
        daemon._hook_accepted_seq = daemon._hook_frontier.accepted_seq
        daemon._hook_frontier.mark_terminal(seq2)
        daemon._hook_processed_seq = daemon._hook_frontier.processed_seq

    assert not daemon._wait_for_hook_drain(
        min_accepted_seq=2,
        timeout_seconds=0.01,
        quiet_period_seconds=0.0,
    )

    with daemon._lock:
        daemon._hook_frontier.mark_terminal(seq1)
        daemon._hook_processed_seq = daemon._hook_frontier.processed_seq

    assert daemon._wait_for_hook_drain(
        min_accepted_seq=2,
        timeout_seconds=0.05,
        quiet_period_seconds=0.0,
    )


def _install_filesystem_hook_manager(
    daemon,  # type: ignore[no-untyped-def]
    workspace: Path,
    fs: FilesystemSubstrate,
    *,
    scope: ScopeInfo,
    recorded: list[tuple[str, EffectRecord, ScopeInfo]],
    capture_records: list[tuple[str, str, int, int]] | None = None,
    capture_diagnostics: list[tuple[str, str, int, int, str]] | None = None,
    fail_capture_record: bool = False,
) -> None:
    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (BoundSubstrate(binding_name="filesystem", substrate_type="filesystem", instance=fs),)

        def working_directory_for_scope(self, resolved_scope: ScopeInfo) -> Path:
            assert resolved_scope is scope
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return scope if name == scope.name else None

        def record(self, binding_name: str, effect: EffectRecord, *, scope: ScopeInfo) -> list[str]:
            recorded.append((binding_name, effect, scope))
            return ["oid"]

        def _record_capture_event(
            self,
            binding_name: str,
            event: object,
            *,
            command_operation_id: str,
            capture_epoch: str | None = None,
            global_seq: int,
            event_seq: int,
            capture_mechanism: str,
        ) -> str:
            del event, capture_epoch, capture_mechanism
            if fail_capture_record:
                raise RuntimeError("capture append failed")
            if capture_records is not None:
                capture_records.append((binding_name, command_operation_id, global_seq, event_seq))
            return "oid"

        def _record_capture_diagnostic(
            self,
            binding_name: str,
            event: object,
            *,
            command_operation_id: str,
            capture_epoch: str | None = None,
            global_seq: int,
            event_seq: int,
            capture_mechanism: str,
            reason: str,
        ) -> str:
            del event, capture_epoch, capture_mechanism
            if capture_diagnostics is not None:
                capture_diagnostics.append((binding_name, command_operation_id, global_seq, event_seq, reason))
            return "oid"

    fs.bind_runtime(
        BuiltInRuntimeBinding(
            pipeline=fs._pipeline,
            is_scope_or_ancestor_isolated=lambda _scope: False,
            overlay_base_scope_name=lambda _scope: "ground",
            working_directory_for_scope=lambda _scope: workspace,
            lookup_scope=lambda name: scope if name == scope.name else None,
        )
    )

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)


def test_handle_hook_line_rejects_late_command_capture_after_complete(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-complete")
    daemon._capture_authority.accept_event("cmd-complete", pid=123, proc_seq=1, global_seq=1)
    daemon._capture_authority.mark_processed("cmd-complete", global_seq=1)
    assert daemon._capture_authority.drain("cmd-complete", timeout_seconds=0.05, quiet_period_seconds=0.0).complete

    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []
    capture_diagnostics: list[tuple[str, str, int, int, str]] = []
    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
        capture_diagnostics=capture_diagnostics,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 2,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-complete",
                "payload": {
                    "op": "write_close",
                    "path": "late.txt",
                    "seq": 2,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert capture_records == []
    assert capture_diagnostics == [("filesystem", "cmd-complete", 1, 2, "late_event_after_finalization")]
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_records_shim_context_missing_diagnostic(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_diagnostics: list[tuple[str, str, int, int, str]] = []
    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_diagnostics=capture_diagnostics,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "payload": {
                    "op": "write_close",
                    "path": "missing-context.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert capture_diagnostics == [("filesystem", "uncorrelated", 1, 1, "shim_context_missing")]
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_records_unknown_command_as_uncorrelated_diagnostic(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_diagnostics: list[tuple[str, str, int, int, str]] = []
    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_diagnostics=capture_diagnostics,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-missing",
                "payload": {
                    "op": "write_close",
                    "path": "unknown-command.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert capture_diagnostics == [("filesystem", "cmd-missing", 1, 1, "uncorrelated_capture_event")]
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_ignored_payload_does_not_poison_command_capture(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-ignored")

    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []
    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 2,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-ignored",
                "payload": {
                    "op": "rename",
                    "path": "ignored.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    drained = daemon._capture_authority.drain("cmd-ignored", timeout_seconds=0.05, quiet_period_seconds=0.0)
    assert drained.complete
    assert drained.accepted_count == 0
    assert capture_records == []


def test_handle_hook_line_marks_command_incomplete_when_capture_persist_fails(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-fail")

    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        fail_capture_record=True,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-fail",
                "payload": {
                    "op": "write_close",
                    "path": "failed.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    drained = daemon._capture_authority.drain("cmd-fail", timeout_seconds=0.05, quiet_period_seconds=0.0)
    assert not drained.complete
    assert drained.reason == "capture_persist_failed"
    assert drained.accepted_count == 1
    assert drained.processed_count == 0
    assert daemon._hook_outcomes["failed"] == 1


def test_handle_hook_line_records_command_capture_and_advances_hook_seq(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-note")
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []

    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-note",
                "capture_epoch": "cap-note",
                "payload": {
                    "op": "write_close",
                    "path": "note.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert recorded == []
    assert capture_records == [("filesystem", "cmd-note", 1, 1)]
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_records_command_correlated_mutation_time_write(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-write")
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []

    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-write",
                "capture_epoch": "cap-write",
                "payload": {
                    "op": "write_observed",
                    "path": "note.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert recorded == []
    assert capture_records == [("filesystem", "cmd-write", 1, 1)]
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_shell_command_finish_completes_shell_capture_policy(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-shell", capture_policy="shell_command", shell_pid=123)
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []

    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
    )

    base_event = {
        "binding_name": "filesystem",
        "hook_id": "filesystem-direct",
        "kind": "ld_preload",
        "phase": "point",
        "scope": "edit",
        "scope_instance_id": "edit-iid",
        "pid": 123,
        "timestamp_ns": 42,
        "command_operation_id": "cmd-shell",
        "capture_epoch": "cap-shell",
    }
    daemon._handle_hook_line(
        json.dumps(
            {
                **base_event,
                "proc_seq": 1,
                "payload": {
                    "op": "write_observed",
                    "path": "note.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )
    daemon._handle_hook_line(
        json.dumps(
            {
                **base_event,
                "proc_seq": 2,
                "payload": {
                    "op": "shell_command_finish",
                    "seq": 2,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    drained = daemon._capture_authority.drain("cmd-shell", timeout_seconds=0.05, quiet_period_seconds=0.0)
    assert drained.complete
    assert drained.capture_policy == "shell_command"
    assert recorded == []
    assert capture_records == [("filesystem", "cmd-shell", 1, 1)]
    assert daemon._hook_accepted_seq == 2
    assert daemon._hook_processed_seq == 2
    assert daemon._hook_persisted_seq == 2
    assert daemon._hook_outcomes["persisted"] == 2


def test_handle_hook_line_records_command_correlated_metadata_change(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    daemon._capture_authority.begin("cmd-meta")
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    capture_records: list[tuple[str, str, int, int]] = []

    _install_filesystem_hook_manager(
        daemon,
        workspace,
        fs,
        scope=task,
        recorded=recorded,
        capture_records=capture_records,
    )

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-meta",
                "capture_epoch": "cap-meta",
                "payload": {
                    "op": "metadata_change",
                    "path": "script.sh",
                    "seq": 2,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert recorded == []
    assert capture_records == [("filesystem", "cmd-meta", 1, 2)]
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_drops_stale_filesystem_scope_instance(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="live-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []

    monkeypatch.setattr(
        fs,
        "effect_for_captured_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not record")),
    )
    _install_filesystem_hook_manager(daemon, workspace, fs, scope=task, recorded=recorded)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "stale-iid",
                "pid": 123,
                "proc_seq": 4,
                "timestamp_ns": 42,
                "payload": {
                    "op": "unlink",
                    "path": "note.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert recorded == []
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 0
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["ignored_stale_scope"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "rename", "path": "note.txt", "seq": 1},
        {"op": "write_close", "path": "../outside.txt", "seq": 1},
        {"op": "write_close", "path": "/tmp/outside.txt", "seq": 1},
        {"op": "write_close", "path": ".vcscore/config.toml", "seq": 1},
        {"op": [], "path": "note.txt", "seq": 1},
        {"op": {}, "path": "note.txt", "seq": 1},
        {"op": "write_close", "path": "bad\0path", "seq": 1},
        {"op": "write_close", "path": "note.txt", "seq": True},
        {"op": "write_close", "path": "note.txt", "seq": 1, "capture_mechanism": 42},
        {"op": "write_close", "path": "note.txt", "seq": 1, "capture_mechanism": ""},
    ],
)
def test_handle_hook_line_counts_unsupported_filesystem_capture_payloads(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []

    monkeypatch.setattr(
        fs,
        "effect_for_captured_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not record")),
    )
    _install_filesystem_hook_manager(daemon, workspace, fs, scope=task, recorded=recorded)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 4,
                "timestamp_ns": 42,
                "payload": payload,
            }
        )
    )

    assert recorded == []
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 0
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["ignored_unsupported"] == 1


def test_handle_hook_line_records_commandless_filesystem_capture_diagnostic(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []

    monkeypatch.setattr(
        fs,
        "effect_for_captured_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not record")),
    )
    _install_filesystem_hook_manager(daemon, workspace, fs, scope=task, recorded=recorded)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "scope_instance_id": "edit-iid",
                "pid": 123,
                "proc_seq": 4,
                "timestamp_ns": 42,
                "payload": {
                    "op": "unlink",
                    "path": "note.txt",
                    "seq": 1,
                    "capture_mechanism": "preload",
                },
            }
        )
    )

    assert recorded == []
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_rejects_missing_scope_instance_id(
    workspace: Path,
    store,  # type: ignore[no-untyped-def]
) -> None:
    from vcs_core._session import SessionDaemon

    daemon = SessionDaemon(str(workspace))
    fs = FilesystemSubstrate(build_builtin_substrate_context(store, workspace=workspace))
    task = ScopeInfo(name="edit", ref=store.GROUND_REF, instance_id="edit-iid", creation_oid="")
    recorded: list[tuple[str, EffectRecord, ScopeInfo]] = []
    _install_filesystem_hook_manager(daemon, workspace, fs, scope=task, recorded=recorded)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "filesystem",
                "hook_id": "filesystem-direct",
                "kind": "ld_preload",
                "phase": "point",
                "scope": "edit",
                "pid": 123,
                "proc_seq": 4,
                "timestamp_ns": 42,
                "payload": {"op": "unlink", "path": "note.txt", "seq": 1},
            }
        )
    )

    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 0
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["malformed"] == 1
    assert recorded == []
