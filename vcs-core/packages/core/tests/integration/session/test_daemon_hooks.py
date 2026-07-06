# under-test: vcs_core._session
"""Session daemon generic hook runtime tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vcs_core._hooks import HookEffects, HookEvent, SystemHook
from vcs_core.types import BoundSubstrate, EffectRecord, ScopeInfo


class _FakeGitHookSubstrate:
    name = "git"
    commands = {"status": object()}
    effects = {}

    def bind_pipeline(self, pipeline, *, scope_queries=None) -> None:  # type: ignore[no-untyped-def]
        del pipeline, scope_queries

    def activate(self) -> None:
        return None

    def deactivate(self) -> None:
        return None

    def authority(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def python_patches(self):  # type: ignore[no-untyped-def]
        return ()

    def system_hooks(self):
        return (
            SystemHook(
                hook_id="git-cli",
                kind="path_wrapper",
                config={"binary": "git"},
                translator=self._translate,
            ),
        )

    def _translate(self, event: HookEvent):
        if event.exit_code != 0:
            return None
        return HookEffects(effects=(EffectRecord(effect_type="GitStatus", metadata={"cwd": event.cwd or "."}),))


def test_handle_hook_line_rejects_stale_scope_instance(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    task = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="live-scope", creation_oid="")

    seen_records: list[tuple[str, EffectRecord, ScopeInfo]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return task if name == "task" else None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def record(self, binding_name: str, effect: EffectRecord, *, scope: ScopeInfo) -> str:
            seen_records.append((binding_name, effect, scope))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "task",
                "scope_instance_id": "stale-scope",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert seen_records == []
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 0
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["ignored_stale_scope"] == 1


def test_handle_hook_line_records_effect_for_live_scope(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    task = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="live-scope", creation_oid="")

    seen_records: list[tuple[str, EffectRecord, ScopeInfo, str, str, dict]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return task if name == "task" else None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def _record_in_child_operation(
            self,
            binding_name: str,
            effect: EffectRecord,
            *,
            scope: ScopeInfo,
            operation_id: str,
            operation_kind: str,
            operation_metadata: dict,
        ) -> str:
            seen_records.append((binding_name, effect, scope, operation_id, operation_kind, operation_metadata))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "task",
                "scope_instance_id": "live-scope",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert len(seen_records) == 1
    binding_name, effect, scope, operation_id, operation_kind, operation_metadata = seen_records[0]
    assert binding_name == "git"
    assert scope is task
    assert effect.effect_type == "GitStatus"
    assert effect.metadata["cwd"] == str(workspace)
    assert operation_id == "hook-git-git-cli-123-1"
    assert operation_kind == "hook.git.git-cli"
    assert operation_metadata["hook_kind"] == "path_wrapper"
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_prefers_child_operation_path_for_live_scope(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    task = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="live-scope", creation_oid="")

    seen_records: list[tuple[str, EffectRecord, ScopeInfo]] = []
    seen_child_records: list[tuple[str, EffectRecord, ScopeInfo, str, str, dict]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return task if name == "task" else None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def _record_in_child_operation(
            self,
            binding_name: str,
            effect: EffectRecord,
            *,
            scope: ScopeInfo,
            operation_id: str,
            operation_kind: str,
            operation_metadata: dict,
        ) -> str:
            seen_child_records.append((binding_name, effect, scope, operation_id, operation_kind, operation_metadata))
            return "oid"

        def record(self, binding_name: str, effect: EffectRecord, *, scope: ScopeInfo) -> str:
            seen_records.append((binding_name, effect, scope))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "task",
                "scope_instance_id": "live-scope",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert seen_records == []
    assert len(seen_child_records) == 1
    binding_name, effect, scope, operation_id, operation_kind, operation_metadata = seen_child_records[0]
    assert binding_name == "git"
    assert scope is task
    assert effect.effect_type == "GitStatus"
    assert effect.metadata["cwd"] == str(workspace)
    assert operation_id == "hook-git-git-cli-123-1"
    assert operation_kind == "hook.git.git-cli"
    assert operation_metadata["hook_kind"] == "path_wrapper"
    assert operation_metadata["hook_phase"] == "finish"
    assert operation_metadata["hook_pid"] == 123
    assert operation_metadata["hook_proc_seq"] == 1
    assert operation_metadata["hook_cwd"] == str(workspace)
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_rejects_stale_ground_scope_instance(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    seen_records: list[tuple[str, EffectRecord, ScopeInfo]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground-live", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            del name
            return None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def record(self, binding_name: str, effect: EffectRecord, *, scope: ScopeInfo) -> str:
            seen_records.append((binding_name, effect, scope))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "ground",
                "scope_instance_id": "ground-stale",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert seen_records == []
    assert daemon._hook_accepted_seq == 1
    assert daemon._hook_processed_seq == 1
    assert daemon._hook_persisted_seq == 0
    assert daemon._hook_failed_seq == 0
    assert daemon._hook_outcomes["ignored_stale_scope"] == 1


def test_handle_hook_line_records_effect_for_live_ground_scope(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    seen_records: list[tuple[str, EffectRecord, ScopeInfo, str, str, dict]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground-live", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            del name
            return None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def _record_in_child_operation(
            self,
            binding_name: str,
            effect: EffectRecord,
            *,
            scope: ScopeInfo,
            operation_id: str,
            operation_kind: str,
            operation_metadata: dict,
        ) -> str:
            seen_records.append((binding_name, effect, scope, operation_id, operation_kind, operation_metadata))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "ground",
                "scope_instance_id": "ground-live",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert len(seen_records) == 1
    binding_name, effect, scope, operation_id, operation_kind, operation_metadata = seen_records[0]
    assert binding_name == "git"
    assert scope is daemon._mg.ground
    assert effect.effect_type == "GitStatus"
    assert effect.metadata["cwd"] == str(workspace)
    assert operation_id == "hook-git-git-cli-123-1"
    assert operation_kind == "hook.git.git-cli"
    assert operation_metadata["hook_kind"] == "path_wrapper"
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_outcomes["persisted"] == 1


def test_handle_hook_line_prefers_child_operation_path_for_live_ground_scope(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcs_core._session import SessionDaemon
    from vcs_core.testing import HookManager

    daemon = SessionDaemon(str(workspace))
    seen_records: list[tuple[str, EffectRecord, ScopeInfo]] = []
    seen_child_records: list[tuple[str, EffectRecord, ScopeInfo, str, str, dict]] = []

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground-live", creation_oid="")
        bindings = (BoundSubstrate(binding_name="git", substrate_type="git", instance=_FakeGitHookSubstrate()),)

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            return workspace

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            del name
            return None

        def execute_recorded(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("command dispatch should not be used in this test")

        def _record_in_child_operation(
            self,
            binding_name: str,
            effect: EffectRecord,
            *,
            scope: ScopeInfo,
            operation_id: str,
            operation_kind: str,
            operation_metadata: dict,
        ) -> str:
            seen_child_records.append((binding_name, effect, scope, operation_id, operation_kind, operation_metadata))
            return "oid"

        def record(self, binding_name: str, effect: EffectRecord, *, scope: ScopeInfo) -> str:
            seen_records.append((binding_name, effect, scope))
            return "oid"

    daemon._mg = _FakeVcsCore()
    daemon._hook_manager = HookManager(
        daemon._mg,
        workspace=workspace,
        repo_path=workspace / ".vcscore",
        socket_path=str(workspace / ".vcscore" / "session-hook.sock"),
    )
    daemon._hook_manager.install_bindings(daemon._mg.bindings)

    daemon._handle_hook_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "ground",
                "scope_instance_id": "ground-live",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": str(workspace),
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert seen_records == []
    assert len(seen_child_records) == 1
    binding_name, effect, scope, operation_id, operation_kind, operation_metadata = seen_child_records[0]
    assert binding_name == "git"
    assert scope.instance_id == "ground-live"
    assert effect.effect_type == "GitStatus"
    assert effect.metadata["cwd"] == str(workspace)
    assert operation_id == "hook-git-git-cli-123-1"
    assert operation_kind == "hook.git.git-cli"
    assert operation_metadata["hook_kind"] == "path_wrapper"
    assert operation_metadata["hook_phase"] == "finish"
    assert operation_metadata["hook_pid"] == 123
    assert operation_metadata["hook_proc_seq"] == 1
    assert operation_metadata["hook_cwd"] == str(workspace)
    assert daemon._hook_persisted_seq == 1
    assert daemon._hook_outcomes["persisted"] == 1
