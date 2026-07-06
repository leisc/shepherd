# under-test: vcs_core._lifecycle_progress
from __future__ import annotations

from pathlib import Path

from vcs_core._lifecycle_progress import LifecycleProgress
from vcs_core._lifecycle_run import LifecycleRun, LifecycleScopeState, read_lifecycle_run
from vcs_core._lifecycle_state import LifecycleRunState


class _CurrentRun:
    def __init__(self, run: LifecycleRun | None = None) -> None:
        self.run = run

    def get(self) -> LifecycleRun | None:
        return self.run

    def set(self, run: LifecycleRun | None) -> None:
        self.run = run


def _run() -> LifecycleRun:
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
        operation="discard",
        phase="prepare_discard_effects",
        scope=task,
        parent=ground,
    )


def _progress(tmp_path: Path, current: _CurrentRun) -> LifecycleProgress:
    state = LifecycleRunState(
        repo_path=str(tmp_path),
        current=current.get,
        set_current=current.set,
    )
    return LifecycleProgress(state)


def test_lifecycle_progress_marks_completed_substrate_once(tmp_path: Path) -> None:
    current = _CurrentRun()
    progress = _progress(tmp_path, current)
    progress.state.persist(_run())

    progress.mark_completed_substrate("filesystem")
    progress.mark_completed_substrate("filesystem")

    run = current.get()
    assert run is not None
    assert run.completed_substrates == ("filesystem",)
    assert read_lifecycle_run(str(tmp_path)) == run


def test_lifecycle_progress_marks_prepared_substrate_once(tmp_path: Path) -> None:
    current = _CurrentRun()
    progress = _progress(tmp_path, current)
    progress.state.persist(_run())

    progress.mark_prepared_substrate("sqlite")
    progress.mark_prepared_substrate("sqlite")

    run = current.get()
    assert run is not None
    assert run.prepared_substrates == ("sqlite",)
    assert read_lifecycle_run(str(tmp_path)) == run


def test_lifecycle_progress_tracks_prepared_effect_counts(tmp_path: Path) -> None:
    current = _CurrentRun()
    progress = _progress(tmp_path, current)
    progress.state.persist(_run())

    assert progress.prepared_effect_count("marker") == 0
    progress.mark_prepared_effect_count("marker", 1)
    progress.mark_prepared_effect_count("marker", 2)

    run = current.get()
    assert run is not None
    assert run.prepared_effect_counts == (("marker", 2),)
    assert progress.prepared_effect_count("marker") == 2
    assert read_lifecycle_run(str(tmp_path)) == run


def test_lifecycle_progress_is_noop_without_active_run(tmp_path: Path) -> None:
    progress = _progress(tmp_path, _CurrentRun())

    progress.mark_completed_substrate("filesystem")
    progress.mark_prepared_substrate("filesystem")
    progress.mark_prepared_effect_count("filesystem", 1)

    assert progress.prepared_effect_count("filesystem") == 0
    assert read_lifecycle_run(str(tmp_path)) is None
