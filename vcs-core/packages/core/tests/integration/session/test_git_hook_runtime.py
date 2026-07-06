# under-test: vcs_core._session
"""End-to-end Git PATH-wrapper hook runtime tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from vcs_core._ipc import is_session_alive, read_session_info, send_request
from vcs_core.runtime_api import adopt_workspace_baseline
from vcs_core.store import Store


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(workspace: Path) -> None:
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "vcs-core@example.com")
    _git(workspace, "config", "user.name", "Meta Git")
    (workspace / "README.md").write_text("seed\n")
    _git(workspace, "add", "README.md")
    _git(workspace, "commit", "-m", "seed")


def _init_vcscore_repo(workspace: Path) -> None:
    # The session daemon's _run() opens the store (VcsCore.from_config) and then
    # asserts the workspace is admissible. Create the store and adopt the committed
    # git-head baseline so README.md isn't an unadopted path that blocks startup.
    # git (tier="auto-detect") auto-binds because _init_repo created .git.
    store = Store(str(workspace / ".vcscore"))
    store.create_root_commit()
    adopt_workspace_baseline(store, workspace, source="git-head")


def test_session_shell_git_status_records_via_path_wrapper(requires_local_bind) -> None:
    from vcs_core._session import SessionDaemon

    workspace = Path(tempfile.mkdtemp(prefix="mg-hook-"))
    _init_repo(workspace)
    _init_vcscore_repo(workspace)

    daemon = SessionDaemon(str(workspace))
    thread = threading.Thread(target=daemon._run, daemon=True)
    thread.start()

    repo_path = str(workspace / ".vcscore")
    deadline = time.time() + 5
    while time.time() < deadline:
        if is_session_alive(repo_path):
            break
        time.sleep(0.1)

    info = read_session_info(repo_path)
    assert info is not None

    try:
        resp = send_request(info.socket_path, "fork", {"name": "git-shell", "parent": "ground", "isolated": False})
        assert resp["ok"], resp.get("error")
        scope_ref = resp["result"]["ref"]
        mount_path = resp["result"]["mount_path"]

        state_resp = send_request(info.socket_path, "switch", {"name": "git-shell"})
        assert state_resp["ok"], state_resp.get("error")
        state_resp = send_request(info.socket_path, "get_state")
        assert state_resp["ok"], state_resp.get("error")
        hook_static_env = state_resp["result"]["hook_static_env"]
        hook_scope_env = state_resp["result"]["hook_scope_env"]
        hook_static_prepend_path = state_resp["result"]["hook_static_prepend_path"]
        hook_scope_prepend_path = state_resp["result"]["hook_scope_prepend_path"]

        env = {**os.environ, "VCS_CORE_SESSION": "1", **hook_static_env, **hook_scope_env}
        env["PATH"] = os.pathsep.join([*hook_static_prepend_path, *hook_scope_prepend_path, env.get("PATH", "")])
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=mount_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        drain_deadline = time.time() + 5
        persisted = 0
        failed = 0
        while time.time() < drain_deadline:
            hook_state = send_request(info.socket_path, "hook_state")
            assert hook_state["ok"], hook_state.get("error")
            persisted = hook_state["result"]["persisted_seq"]
            failed = hook_state["result"]["failed_seq"]
            if persisted >= 1 and failed == 0:
                break
            time.sleep(0.1)
        assert persisted >= 1, "expected hook event to persist"
        assert failed == 0

        store = Store(repo_path)
        effects = [
            effect
            for effect in store.filter_effects(effect_type="GitStatusObserved", ref=scope_ref, max_count=20)
            if effect.metadata.get("substrate") == "git"
        ]
        assert effects
        assert any(effect.metadata["clean"] is True for effect in effects)
    finally:
        send_request(info.socket_path, "stop")
        thread.join(timeout=5)
