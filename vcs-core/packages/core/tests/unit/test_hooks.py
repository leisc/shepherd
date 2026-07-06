# under-test: vcs_core._hooks
"""Tests for internal hook runtime helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from vcs_core import build_builtin_substrate_context
from vcs_core._hooks import (
    HookBinding,
    HookEffects,
    HookManager,
    PathWrapperInstaller,
    SystemHook,
    _path_wrapper_script,
    parse_hook_event,
    parse_hook_event_line,
)
from vcs_core.git_substrate import GitSubstrate
from vcs_core.substrates import FilesystemSubstrate
from vcs_core.types import BoundSubstrate, EffectRecord, ScopeInfo


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content)
    path.chmod(0o755)
    return path


def _child_script(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "child.py",
        """#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys


def main() -> int:
    mode = sys.argv[1]
    if mode == "exit":
        return int(sys.argv[2])
    if mode == "signal":
        sig = getattr(signal, sys.argv[2])
        os.kill(os.getpid(), sig)
        return 99
    raise ValueError(mode)


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )


def _wrapper_script(tmp_path: Path, *, real_binary: str, binary_name: str = "wrapped-python") -> Path:
    wrapper = tmp_path / binary_name
    wrapper.write_text(
        _path_wrapper_script(
            real_binary=real_binary,
            binding_name="git",
            hook_id="git-cli",
            binary_name=binary_name,
        )
    )
    wrapper.chmod(0o755)
    return wrapper


def _path_wrapper_binding(binary: str) -> HookBinding:
    return HookBinding(
        binding_name="git",
        substrate_type="git",
        substrate=object(),
        hook=SystemHook(
            hook_id="git-cli",
            kind="path_wrapper",
            config={"binary": binary},
            translator=lambda _event: None,
        ),
    )


def test_parse_hook_event_line_accepts_path_wrapper_finish_event() -> None:
    event = parse_hook_event_line(
        json.dumps(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "task",
                "scope_instance_id": "scope-123",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "cwd": "/tmp/work",
                "argv": ["git", "status"],
                "exit_code": 0,
                "payload": {},
            }
        )
    )

    assert event.binding_name == "git"
    assert event.scope_instance_id == "scope-123"
    assert event.argv == ("git", "status")


def test_hook_manager_preserves_command_operation_id_in_recorded_metadata(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    scope = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="iid-1", creation_oid="")
    recorded: dict[str, object] = {}

    class _HookedSubstrate:
        def system_hooks(self) -> tuple[SystemHook, ...]:
            return (
                SystemHook(
                    hook_id="tool-finish",
                    kind="http_proxy",
                    config={},
                    translator=lambda _event: HookEffects(
                        effects=(EffectRecord(effect_type="Observed", metadata={"label": "observed"}),)
                    ),
                ),
            )

    class _FakeVcsCore:
        bindings = (
            BoundSubstrate(
                binding_name="tool",
                substrate_type="tool",
                instance=_HookedSubstrate(),
            ),
        )

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return scope if name == "task" else None

        def _record_in_child_operation(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args
            recorded.update(kwargs)
            return "oid-1"

    manager = HookManager(
        _FakeVcsCore(),
        workspace=tmp_path,
        repo_path=repo_path,
        socket_path=str(repo_path / "session-hook.sock"),
    )
    manager.install_bindings(_FakeVcsCore.bindings)

    result = manager.ingest_line(
        json.dumps(
            {
                "binding_name": "tool",
                "hook_id": "tool-finish",
                "kind": "http_proxy",
                "phase": "finish",
                "scope": "task",
                "scope_instance_id": "iid-1",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "command_operation_id": "cmd-session-exec",
                "payload": {},
            }
        )
    )

    assert result.outcome == "persisted"
    assert recorded["operation_metadata"] == {
        "hook_kind": "http_proxy",
        "hook_phase": "finish",
        "hook_pid": 123,
        "hook_proc_seq": 1,
        "command_operation_id": "cmd-session-exec",
    }


def test_parse_hook_event_rejects_missing_scope_instance_id() -> None:
    with pytest.raises(ValueError, match="scope_instance_id"):
        parse_hook_event(
            {
                "binding_name": "git",
                "hook_id": "git-cli",
                "kind": "path_wrapper",
                "phase": "finish",
                "scope": "task",
                "pid": 123,
                "proc_seq": 1,
                "timestamp_ns": 42,
                "payload": {},
            }
        )


def test_hook_manager_splits_static_and_scope_env(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    scope = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="iid-1", creation_oid="")

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (
            BoundSubstrate(
                binding_name="filesystem",
                substrate_type="filesystem",
                instance=FilesystemSubstrate(
                    build_builtin_substrate_context(
                        type("Store", (), {"repo_path": str(repo_path)})(), workspace=tmp_path
                    )
                ),
            ),
            BoundSubstrate(
                binding_name="git",
                substrate_type="git",
                instance=GitSubstrate(
                    build_builtin_substrate_context(
                        type("Store", (), {"repo_path": str(repo_path)})(), workspace=tmp_path
                    )
                ),
            ),
        )

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            assert scope.name == "task"
            return tmp_path / "overlay" / "task"

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return scope if name == "task" else None

    manager = HookManager(
        _FakeVcsCore(),
        workspace=tmp_path,
        repo_path=repo_path,
        socket_path=str(repo_path / "session-hook.sock"),
    )
    manager.install_bindings(_FakeVcsCore.bindings)

    static_env = manager.static_env()
    scope_env = manager.scope_env(scope)
    assert static_env.env["VCS_CORE_HOOK_SOCKET"].endswith("session-hook.sock")
    assert "VCS_CORE_SCOPE" not in static_env.env
    assert static_env.prepend_path
    assert scope_env.env["VCS_CORE_SCOPE"] == "task"
    assert scope_env.env["VCS_CORE_SCOPE_INSTANCE_ID"] == "iid-1"
    assert scope_env.env["VCS_CORE_WORKSPACE"].endswith("/overlay/task")
    assert not scope_env.prepend_path


def test_hook_manager_keeps_path_wrappers_baseline_and_preload_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    scope = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="iid-1", creation_oid="")

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (
            BoundSubstrate(
                binding_name="filesystem",
                substrate_type="filesystem",
                instance=FilesystemSubstrate(
                    build_builtin_substrate_context(
                        type("Store", (), {"repo_path": str(repo_path)})(), workspace=tmp_path
                    )
                ),
            ),
            BoundSubstrate(
                binding_name="git",
                substrate_type="git",
                instance=GitSubstrate(
                    build_builtin_substrate_context(
                        type("Store", (), {"repo_path": str(repo_path)})(), workspace=tmp_path
                    )
                ),
            ),
        )

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            assert scope.name == "task"
            return tmp_path / "overlay" / "task"

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return scope if name == "task" else None

    manager = HookManager(
        _FakeVcsCore(),
        workspace=tmp_path,
        repo_path=repo_path,
        socket_path=str(repo_path / "session-hook.sock"),
    )
    manager.install_bindings(_FakeVcsCore.bindings)

    no_caps = manager.activation([])
    no_caps_static = manager.static_env(activation=no_caps)
    no_caps_scope = manager.scope_env(scope, activation=no_caps)
    assert no_caps_static.prepend_path, "baseline path wrappers should remain active without optional capabilities"
    assert "LD_PRELOAD" not in no_caps_scope.prepend_env

    monkeypatch.setattr("vcs_core._hooks.ensure_fs_capture_shim", lambda repo_path: "/tmp/fs_capture_shim.so")
    fs_capture = manager.activation(["fs_capture"])
    fs_scope = manager.scope_env(scope, activation=fs_capture)
    assert fs_scope.prepend_env["LD_PRELOAD"]


def test_hook_manager_validates_requested_capabilities(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    scope = ScopeInfo(name="task", ref="refs/vcscore/scopes/task", instance_id="iid-1", creation_oid="")

    class _FakeVcsCore:
        ground = ScopeInfo(name="ground", ref="refs/vcscore/ground", instance_id="ground", creation_oid="")
        bindings = (
            BoundSubstrate(
                binding_name="filesystem",
                substrate_type="filesystem",
                instance=FilesystemSubstrate(
                    build_builtin_substrate_context(
                        type("Store", (), {"repo_path": str(repo_path)})(), workspace=tmp_path
                    )
                ),
            ),
        )

        def working_directory_for_scope(self, scope: ScopeInfo) -> Path:
            assert scope.name == "task"
            return tmp_path / "overlay" / "task"

        def lookup_scope(self, name: str) -> ScopeInfo | None:
            return scope if name == "task" else None

    manager = HookManager(
        _FakeVcsCore(),
        workspace=tmp_path,
        repo_path=repo_path,
        socket_path=str(repo_path / "session-hook.sock"),
    )
    manager.install_bindings(_FakeVcsCore.bindings)

    activation = manager.activation(["fs_capture"])
    assert activation.capabilities == frozenset({"fs_capture"})

    with pytest.raises(ValueError, match="Unknown hook capabilities"):
        manager.activation(["unknown_capability"])


def test_generated_wrapper_preserves_normal_exit_code(tmp_path: Path) -> None:
    child = _child_script(tmp_path)
    wrapper = _wrapper_script(tmp_path, real_binary=sys.executable)

    direct = subprocess.run(
        [sys.executable, str(child), "exit", "7"],
        check=False,
    )
    wrapped = subprocess.run(
        [str(wrapper), str(child), "exit", "7"],
        check=False,
    )

    assert direct.returncode == 7
    assert wrapped.returncode == direct.returncode


def test_generated_wrapper_preserves_signal_termination_shape(tmp_path: Path) -> None:
    child = _child_script(tmp_path)
    wrapper = _wrapper_script(tmp_path, real_binary=sys.executable)

    direct = subprocess.run(
        [sys.executable, str(child), "signal", "SIGTERM"],
        check=False,
    )
    wrapped = subprocess.run(
        [str(wrapper), str(child), "signal", "SIGTERM"],
        check=False,
    )

    assert direct.returncode < 0
    assert wrapped.returncode == direct.returncode


def test_generated_wrapper_handles_process_group_sigint_without_parent_keyboardinterrupt(tmp_path: Path) -> None:
    child = _write_executable(
        tmp_path / "sleeper.py",
        """#!/usr/bin/env python3
from __future__ import annotations

import time

print("child-start", flush=True)
time.sleep(30)
""",
    )
    wrapper = _wrapper_script(tmp_path, real_binary=sys.executable, binary_name="git")
    env = {
        **os.environ,
        "VCS_CORE_SCOPE": "ground",
        "VCS_CORE_SCOPE_INSTANCE_ID": "ground-live",
        "VCS_CORE_HOOK_SOCKET": str(tmp_path / "missing.sock"),
    }
    proc = subprocess.Popen(
        [str(wrapper), str(child)],
        cwd=tmp_path,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdout.readline()
        os.killpg(proc.pid, signal.SIGINT)
        _stdout, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == -signal.SIGINT
    assert f'File "{wrapper}"' not in stderr


def test_path_wrapper_installer_excludes_managed_wrapper_dirs_from_resolution(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()

    managed_old_dir = repo_path / "runtime" / "hooks" / "old-session" / "bin"
    managed_old_dir.mkdir(parents=True)
    old_wrapper = _write_executable(
        managed_old_dir / "demo-bin",
        "#!/bin/sh\nexit 0\n",
    )

    source_dir = tmp_path / "source-bin"
    source_dir.mkdir()
    source_binary = _write_executable(
        source_dir / "demo-bin",
        "#!/bin/sh\nexit 0\n",
    )

    installer = PathWrapperInstaller(socket_path=str(repo_path / "session-hook.sock"))
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = (
        os.pathsep.join([str(managed_old_dir), str(source_dir), original_path])
        if original_path
        else os.pathsep.join([str(managed_old_dir), str(source_dir)])
    )
    try:
        env = installer.install(
            [_path_wrapper_binding("demo-bin")],
            workspace=tmp_path,
            repo_path=repo_path,
        )
    finally:
        os.environ["PATH"] = original_path

    wrapper_path = Path(env.prepend_path[0]) / "demo-bin"
    script = wrapper_path.read_text()
    assert str(source_binary) in script
    assert str(old_wrapper) not in script


def test_path_wrapper_installer_uninstall_removes_generated_wrapper_dir(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()

    source_dir = tmp_path / "source-bin"
    source_dir.mkdir()
    _write_executable(
        source_dir / "demo-bin",
        "#!/bin/sh\nexit 0\n",
    )

    installer = PathWrapperInstaller(socket_path=str(repo_path / "session-hook.sock"))
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([str(source_dir), original_path]) if original_path else str(source_dir)
    try:
        env = installer.install(
            [_path_wrapper_binding("demo-bin")],
            workspace=tmp_path,
            repo_path=repo_path,
        )
    finally:
        os.environ["PATH"] = original_path

    wrapper_dir = Path(env.prepend_path[0])
    assert wrapper_dir.exists()

    installer.uninstall()

    assert not wrapper_dir.exists()
