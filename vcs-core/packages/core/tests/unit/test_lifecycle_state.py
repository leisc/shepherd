# under-test: vcs_core._lifecycle_state
from __future__ import annotations

from pathlib import Path

import pytest
from vcs_core._lifecycle_run import LifecycleRun, LifecycleScopeState, read_lifecycle_run
from vcs_core._lifecycle_state import LifecycleRunState


class _CurrentRun:
    def __init__(self, run: LifecycleRun | None = None) -> None:
        self.run = run

    def get(self) -> LifecycleRun | None:
        return self.run

    def set(self, run: LifecycleRun | None) -> None:
        self.run = run


def _run(*, phase: str = "commit_substrates") -> LifecycleRun:
    ground = LifecycleScopeState(
        name="ground",
        ref="refs/vcscore/ground",
        instance_id="ground-test",
        creation_oid="",
        world_id="world-ground",
    )
    task = LifecycleScopeState(
        name="task",
        ref="refs/vcscore/scopes/task",
        instance_id="task-test",
        creation_oid="abc",
        world_id="world-task",
    )
    return LifecycleRun(
        session_id="session-test",
        operation="merge",
        phase=phase,
        scope=task,
        parent=ground,
    )


def _state(tmp_path: Path, current: _CurrentRun) -> LifecycleRunState:
    return LifecycleRunState(
        repo_path=str(tmp_path),
        current=current.get,
        set_current=current.set,
    )


def test_lifecycle_run_state_persists_and_sets_current(tmp_path: Path) -> None:
    current = _CurrentRun()
    state = _state(tmp_path, current)
    run = _run()

    assert state.persist(run) == run

    assert current.get() == run
    assert read_lifecycle_run(str(tmp_path)) == run


def test_lifecycle_run_state_current_or_read_hydrates_current(tmp_path: Path) -> None:
    current = _CurrentRun()
    state = _state(tmp_path, current)
    run = _run()
    state.persist(run)
    current.set(None)

    assert state.current_or_read() == run
    assert current.get() == run


def test_lifecycle_run_state_update_preserves_unspecified_fields(tmp_path: Path) -> None:
    current = _CurrentRun()
    state = _state(tmp_path, current)
    run = state.persist(_run())

    updated = state.update(phase="merge_registry", completed_substrates=("filesystem",))

    assert updated.phase == "merge_registry"
    assert updated.completed_substrates == ("filesystem",)
    assert updated.operation == run.operation
    assert updated.scope == run.scope
    assert current.get() == updated
    assert read_lifecycle_run(str(tmp_path)) == updated


def test_lifecycle_run_state_update_requires_current_run(tmp_path: Path) -> None:
    state = _state(tmp_path, _CurrentRun())

    with pytest.raises(RuntimeError, match="No lifecycle recovery run is active"):
        state.update(phase="merge_store")


def test_lifecycle_run_state_clear_removes_file_and_current(tmp_path: Path) -> None:
    current = _CurrentRun()
    state = _state(tmp_path, current)
    state.persist(_run())

    state.clear()

    assert current.get() is None
    assert read_lifecycle_run(str(tmp_path)) is None
