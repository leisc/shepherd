# under-test: vcs_core._parent_tree_manifest
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from vcs_core import MergePreconditionError, ParentWorkingTreeDivergedError, _vcscore_lifecycle
from vcs_core._parent_tree_manifest import capture_parent_tree_manifest

from ...support.builders import make_marker_filesystem_vcscore
from ...support.overlays import MockOverlayBackend


class PathOverlayBackend(MockOverlayBackend):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.diffed: list[str] = []

    def create_layer(self, scope_id: str, *, parent_scope_id: str | None) -> None:
        super().create_layer(scope_id, parent_scope_id=parent_scope_id)
        path = self.working_path(scope_id)
        if parent_scope_id is not None and self.working_path(parent_scope_id).exists():
            shutil.copytree(self.working_path(parent_scope_id), path, dirs_exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)

    def working_path(self, scope_id: str) -> Path:
        return self.root / scope_id

    def write_file(self, scope_id: str, path: str, content: bytes, *, mode: int = 0o100644) -> None:
        super().write_file(scope_id, path, content, mode=mode)
        target = self.working_path(scope_id) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(mode)

    def diff_layer(self, scope_id: str) -> list[tuple[str, bytes | None, int]]:
        self.diffed.append(scope_id)
        return super().diff_layer(scope_id)


def test_clean_parent_manifest_allows_merge_and_drops_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "clean-parent-child", hints={"isolated": True})
        manifest_key = (child.ref, child.instance_id)
        assert manifest_key in mg._parent_tree_manifests

        mg.merge(child, mg.ground)

        assert backend.diffed == [child.name]
        assert backend.committed == [(child.name, "ground")]
        assert manifest_key not in mg._parent_tree_manifests
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_unavailable_virtual_parent_root_skips_manifest_and_preserves_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = MockOverlayBackend()
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "virtual-root-child", hints={"isolated": True})
        assert mg._parent_tree_manifests == {}
        backend.write_file(child.name, "child.txt", b"child\n")

        mg.merge(child, mg.ground)

        assert backend.committed == [(child.name, "ground")]
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_prefork_parent_write_is_manifest_baseline_not_dirt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        backend.write_file("ground", "prefork.txt", b"already here\n")
        child = mg.fork(mg.ground, "prefork-parent-child", hints={"isolated": True})
        manifest_key = (child.ref, child.instance_id)
        manifest = mg._parent_tree_manifests[manifest_key]
        assert manifest.entries["prefork.txt"].sha256 is not None

        mg.merge(child, mg.ground)

        assert backend.diffed == [child.name]
        assert backend.committed == [(child.name, "ground")]
        assert manifest_key not in mg._parent_tree_manifests
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_parent_effective_carrier_mutation_refuses_before_merge_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "dirty-parent-child", hints={"isolated": True})
        manifest_key = (child.ref, child.instance_id)
        (backend.working_path("ground") / "unrecorded.txt").write_text("parent dirt\n", encoding="utf-8")

        with pytest.raises(ParentWorkingTreeDivergedError, match=r"unrecorded\.txt"):
            mg.merge(child, mg.ground)

        assert manifest_key in mg._parent_tree_manifests
        assert backend.diffed == []
        assert backend.committed == []
        mg.discard(child)
        assert manifest_key not in mg._parent_tree_manifests
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_parent_ref_precondition_runs_before_parent_dirt_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "stale-parent-child", hints={"isolated": True})
        manifest_key = (child.ref, child.instance_id)
        mg.store._emit_effect(mg.ground, "TestParentAdvance", {}, substrate="test")
        (backend.working_path("ground") / "unrecorded.txt").write_text("parent dirt\n", encoding="utf-8")

        with pytest.raises(MergePreconditionError, match="advanced past fork point"):
            mg.merge(child, mg.ground)

        assert manifest_key in mg._parent_tree_manifests
        assert backend.diffed == []
        assert backend.committed == []
        mg.discard(child)
        assert manifest_key not in mg._parent_tree_manifests
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_parent_tree_manifest_flag_off_preserves_merge_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "0")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "flag-off-child", hints={"isolated": True})
        assert mg._parent_tree_manifests == {}
        (backend.working_path("ground") / "unrecorded.txt").write_text("parent dirt\n", encoding="utf-8")

        mg.merge(child, mg.ground)

        assert backend.diffed == [child.name]
        assert backend.committed == [(child.name, "ground")]
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_missing_exact_manifest_skips_parent_dirt_check_and_ignores_stale_same_name_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        child = mg.fork(mg.ground, "fresh-session-child", hints={"isolated": True})
        real_key = (child.ref, child.instance_id)
        stale_key = (child.ref, "stale-instance")
        assert real_key in mg._parent_tree_manifests
        mg._parent_tree_manifests.clear()
        mg._parent_tree_manifests[stale_key] = capture_parent_tree_manifest(
            backend.working_path("ground"),
            layer_name="ground",
        )
        (backend.working_path("ground") / "unrecorded.txt").write_text("parent dirt\n", encoding="utf-8")

        mg.merge(child, mg.ground)

        assert backend.diffed == [child.name]
        assert backend.committed == [(child.name, "ground")]
        assert stale_key not in mg._parent_tree_manifests
    finally:
        mg.deactivate(warn_on_open_scopes=False)


def test_parent_tree_manifest_discard_and_deactivate_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )
    try:
        discarded = mg.fork(mg.ground, "discarded-manifest-child", hints={"isolated": True})
        discarded_key = (discarded.ref, discarded.instance_id)
        assert discarded_key in mg._parent_tree_manifests

        mg.discard(discarded)

        assert discarded_key not in mg._parent_tree_manifests
        assert backend.discarded == [discarded.name]

        active = mg.fork(mg.ground, "deactivate-manifest-child", hints={"isolated": True})
        active_key = (active.ref, active.instance_id)
        assert active_key in mg._parent_tree_manifests

        mg.deactivate(warn_on_open_scopes=False)

        assert mg._parent_tree_manifests == {}
    finally:
        if not backend.deactivated:
            mg.deactivate(warn_on_open_scopes=False)


def test_scope_registry_publish_failure_drops_captured_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    backend = PathOverlayBackend(tmp_path / "overlay")
    mg = make_marker_filesystem_vcscore(
        tmp_path / "ws",
        declarative=False,
        backend=backend,
        activate=True,
    )

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("publish failed after manifest capture")

    monkeypatch.setattr(_vcscore_lifecycle, "_publish_scope_registry_fork_locked", fail_publish)
    try:
        with pytest.raises(RuntimeError, match="publish failed after manifest capture"):
            mg.fork(mg.ground, "publish-failure-child", hints={"isolated": True})

        assert mg._parent_tree_manifests == {}
        assert backend.discarded == ["publish-failure-child"]
    finally:
        mg.deactivate(warn_on_open_scopes=False)
