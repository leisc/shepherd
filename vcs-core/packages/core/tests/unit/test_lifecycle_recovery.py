# under-test: vcs_core._lifecycle_recovery
from __future__ import annotations

from pathlib import Path

import pytest
from vcs_core._lifecycle_progress import LifecycleProgress
from vcs_core._lifecycle_recovery import LifecycleRecovery, LifecycleRecoveryDependencies
from vcs_core._lifecycle_run import LifecycleRun, LifecycleScopeState
from vcs_core._lifecycle_state import LifecycleRunState
from vcs_core.types import ScopeInfo


class _CurrentRun:
    def __init__(self, run: LifecycleRun | None = None) -> None:
        self.run = run

    def get(self) -> LifecycleRun | None:
        return self.run

    def set(self, run: LifecycleRun | None) -> None:
        self.run = run


class _MergeSubstrate:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self._calls = calls

    def commit_merge(self, scope_id: str, *, parent_scope: ScopeInfo) -> None:
        self._calls.append(f"commit:{self.name}:{scope_id}:{parent_scope.name}")


class _DiscardSubstrate:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False) -> None:
        self.name = name
        self._calls = calls
        self._fail = fail

    def discard(self, scope_id: str) -> None:
        self._calls.append(f"discard:{self.name}:{scope_id}")
        if self._fail:
            raise RuntimeError(f"discard failure from {self.name}")


class _PassiveSubstrate:
    name = "passive"


def _scope_state(name: str) -> LifecycleScopeState:
    return LifecycleScopeState(
        name=name,
        ref=f"refs/vcscore/scopes/{name}",
        instance_id=f"{name}-instance",
        creation_oid="abc",
        world_id=f"world-{name}",
    )


def _run(
    *,
    operation: str = "merge",
    phase: str = "commit_substrates",
    completed_substrates: tuple[str, ...] = (),
) -> LifecycleRun:
    return LifecycleRun(
        session_id="session-test",
        operation=operation,
        phase=phase,
        scope=_scope_state("task"),
        parent=_scope_state("ground"),
        completed_substrates=completed_substrates,
    )


def _scope(name: str) -> ScopeInfo:
    return ScopeInfo(
        name=name,
        ref=f"refs/vcscore/scopes/{name}",
        instance_id=f"{name}-instance",
        creation_oid="abc",
        world_id=f"world-{name}",
    )


def _state(tmp_path: Path, current: _CurrentRun) -> LifecycleRunState:
    return LifecycleRunState(
        repo_path=str(tmp_path),
        current=current.get,
        set_current=current.set,
    )


def _recovery(
    tmp_path: Path,
    current: _CurrentRun,
    calls: list[str],
    *,
    substrates: tuple[object, ...] = (),
    scope_exists: bool = True,
) -> LifecycleRecovery:
    state = _state(tmp_path, current)
    progress = LifecycleProgress(state)
    scope = _scope("task")
    parent = _scope("ground")
    return LifecycleRecovery(
        LifecycleRecoveryDependencies(
            state=state,
            progress=progress,
            substrates=substrates,
            scope_ref_exists=lambda observed: scope_exists and observed.ref == scope.ref,
            load_context=lambda run: (scope, parent),
            restore_substrate_state=lambda run, observed, observed_parent: calls.append(
                f"restore:{run.operation}:{observed.name}:{observed_parent.name}"
            ),
            snapshot_merge_effects=lambda observed, observed_parent: calls.append(
                f"snapshot_merge:{observed.name}:{observed_parent.name}"
            ),
            snapshot_discard_effects=lambda observed, observed_parent: calls.append(
                f"snapshot_discard:{observed.name}:{observed_parent.name}"
            ),
            complete_merge=lambda observed, observed_parent: (
                calls.append(f"complete_merge:{observed.name}:{observed_parent.name}") or observed.name
            ),
            complete_discard=lambda observed, observed_parent: (
                calls.append(f"complete_discard:{observed.name}:{observed_parent.name}") or observed.name
            ),
            complete_seal=lambda observed, observed_parent: (
                calls.append(f"complete_seal:{observed.name}:{observed_parent.name}") or observed.name
            ),
        )
    )


def test_lifecycle_recovery_resumes_merge_and_skips_completed_substrates(tmp_path: Path) -> None:
    current = _CurrentRun()
    calls: list[str] = []
    state = _state(tmp_path, current)
    state.persist(_run(completed_substrates=("already",)))
    recovery = _recovery(
        tmp_path,
        current,
        calls,
        substrates=(
            _MergeSubstrate("merge-a", calls),
            _MergeSubstrate("already", calls),
            _PassiveSubstrate(),
        ),
    )

    result = recovery.recover()

    assert result.callback_kind == "merge"
    assert result.scope_name == "task"
    assert calls == [
        "restore:merge:task:ground",
        "commit:merge-a:task:ground",
        "complete_merge:task:ground",
    ]
    run = current.get()
    assert run is not None
    assert run.completed_substrates == ("already", "merge-a")


def test_lifecycle_recovery_discard_snapshots_before_substrates_and_reports_failures(tmp_path: Path) -> None:
    current = _CurrentRun()
    calls: list[str] = []
    state = _state(tmp_path, current)
    state.persist(_run(operation="discard", phase="prepare_discard_effects"))
    recovery = _recovery(
        tmp_path,
        current,
        calls,
        substrates=(
            _DiscardSubstrate("ok", calls),
            _DiscardSubstrate("failing", calls, fail=True),
        ),
    )

    with pytest.raises(RuntimeError, match="Scope remains active for recovery"):
        recovery.recover()

    assert calls == [
        "restore:discard:task:ground",
        "snapshot_discard:task:ground",
        "discard:failing:task",
        "discard:ok:task",
    ]
    run = current.get()
    assert run is not None
    assert run.phase == "discard_substrates"
    assert run.completed_substrates == ("ok",)


def test_lifecycle_recovery_resumes_seal_without_merge_or_discard_substrates(tmp_path: Path) -> None:
    current = _CurrentRun()
    calls: list[str] = []
    state = _state(tmp_path, current)
    state.persist(_run(operation="seal", phase="seal_runtime_close"))
    recovery = _recovery(
        tmp_path,
        current,
        calls,
        substrates=(
            _MergeSubstrate("merge-a", calls),
            _DiscardSubstrate("discard-a", calls),
            _PassiveSubstrate(),
        ),
    )

    result = recovery.recover()

    assert result.callback_kind == "seal"
    assert result.scope_name == "task"
    assert calls == [
        "restore:seal:task:ground",
        "complete_seal:task:ground",
    ]


def test_lifecycle_recovery_raises_for_invalid_mode_before_loading_context(tmp_path: Path) -> None:
    current = _CurrentRun(_run())
    calls: list[str] = []
    recovery = _recovery(tmp_path, current, calls)

    with pytest.raises(ValueError, match="Unknown lifecycle recovery mode"):
        recovery.recover(mode="force")

    assert calls == []


def test_lifecycle_recovery_requires_active_run(tmp_path: Path) -> None:
    calls: list[str] = []
    recovery = _recovery(tmp_path, _CurrentRun(), calls)

    with pytest.raises(RuntimeError, match="No lifecycle recovery run is active"):
        recovery.recover()

    assert calls == []


def test_lifecycle_recovery_raises_for_unknown_operation(tmp_path: Path) -> None:
    current = _CurrentRun()
    calls: list[str] = []
    state = _state(tmp_path, current)
    state.persist(_run(operation="rebase"))
    recovery = _recovery(tmp_path, current, calls, scope_exists=False)

    with pytest.raises(RuntimeError, match="Unknown lifecycle operation"):
        recovery.recover()

    assert calls == []
