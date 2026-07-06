# under-test: vcs_core._materialization_recovery
from __future__ import annotations

from pathlib import Path

from vcs_core._materialization_recovery import probe_materialization_recovery_state
from vcs_core._materialization_run import MaterializationRun, clear_materialization_run, write_materialization_run
from vcs_core._recovery_inventory import recovery_inventory_snapshot_for_store
from vcs_core.store import Store
from vcs_core.testing import write_dirty_flag


def test_materialization_recovery_probe_reports_corrupt_dirty_flag(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    (repo_path / "dirty").write_text('{"session_id": "crashed"}')

    state = probe_materialization_recovery_state(repo_path)

    assert state.required is True
    assert state.dirty.presence == "present"
    assert state.dirty.validity == "corrupt"


def test_materialization_recovery_probe_reports_corrupt_run_ledger(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    (repo_path / "materialization-run.json").write_text('{"session_id": "crashed"}')

    state = probe_materialization_recovery_state(repo_path)

    assert state.required is True
    assert state.run.presence == "present"
    assert state.run.validity == "corrupt"


def test_recovery_inventory_uses_tolerant_materialization_state(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    store = Store(str(repo_path))
    store.create_root_commit()
    (repo_path / "dirty").write_text("{not json")
    (repo_path / "materialization-run.json").write_text("{not json")

    snapshot = recovery_inventory_snapshot_for_store(repo_path, store)
    items = {(item.kind, item.health.status) for item in snapshot.items if item.domain == "recovery"}

    assert ("dirty_push", "present_corrupt") in items
    assert ("materialization_run", "present_corrupt") in items


def test_clear_materialization_run_unlinks_corrupt_ledger_and_artifacts(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    artifacts_root = repo_path / "materialization-runs"
    artifacts_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_text("kept")
    (artifacts_root / "run-dir").mkdir()
    (artifacts_root / "run-dir" / "artifact.txt").write_text("artifact")
    (artifacts_root / "artifact-file").write_text("artifact")
    (artifacts_root / "artifact-link").symlink_to(outside)
    (repo_path / "materialization-run.json").write_text("{not json")

    clear_materialization_run(str(repo_path))

    assert not (repo_path / "materialization-run.json").exists()
    assert not (artifacts_root / "run-dir").exists()
    assert not (artifacts_root / "artifact-file").exists()
    assert not (artifacts_root / "artifact-link").exists()
    assert (outside / "kept.txt").read_text() == "kept"


def test_clear_materialization_run_removes_only_valid_run_artifacts(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    artifacts_root = repo_path / "materialization-runs"
    (artifacts_root / "run-1").mkdir(parents=True)
    (artifacts_root / "run-1" / "artifact.txt").write_text("artifact")
    (artifacts_root / "other-run").mkdir()
    write_materialization_run(
        str(repo_path),
        MaterializationRun(
            session_id="crashed",
            run_id="run-1",
            timestamp=1.0,
            planned_unit_ids=("unit-1",),
        ),
    )

    clear_materialization_run(str(repo_path))

    assert not (repo_path / "materialization-run.json").exists()
    assert not (artifacts_root / "run-1").exists()
    assert (artifacts_root / "other-run").exists()


def test_clear_materialization_run_does_not_follow_malicious_run_id(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_text("kept")
    write_materialization_run(
        str(repo_path),
        MaterializationRun(
            session_id="crashed",
            run_id="../outside",
            timestamp=1.0,
            planned_unit_ids=("unit-1",),
        ),
    )

    clear_materialization_run(str(repo_path))

    assert not (repo_path / "materialization-run.json").exists()
    assert (outside / "kept.txt").read_text() == "kept"


def test_materialization_recovery_probe_reports_valid_state(tmp_path: Path) -> None:
    repo_path = tmp_path / ".vcscore"
    repo_path.mkdir()
    write_dirty_flag(str(repo_path), "crashed")
    write_materialization_run(
        str(repo_path),
        MaterializationRun(
            session_id="crashed",
            run_id="run-1",
            timestamp=1.0,
            planned_unit_ids=("unit-1",),
        ),
    )

    state = probe_materialization_recovery_state(repo_path)

    assert state.required is True
    assert state.dirty.validity == "valid"
    assert state.dirty.session_id == "crashed"
    assert state.run.validity == "valid"
    assert state.run.run is not None
    assert state.run.run.run_id == "run-1"
