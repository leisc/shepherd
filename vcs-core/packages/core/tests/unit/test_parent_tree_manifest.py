# under-test: vcs_core._parent_tree_manifest
from __future__ import annotations

import os

import pytest
from vcs_core._parent_tree_manifest import capture_parent_tree_manifest, diff_parent_tree_manifest


def test_parent_tree_manifest_detects_same_size_content_change_with_restored_mtime(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("alpha\n", encoding="utf-8")
    before_stat = target.stat()
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    target.write_text("bravo\n", encoding="utf-8")
    os.utime(target, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))

    divergences = diff_parent_tree_manifest(manifest, root)

    assert [(divergence.path, divergence.reason) for divergence in divergences] == [("note.txt", "content_changed")]


def test_parent_tree_manifest_ignores_touch_without_content_change(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("alpha\n", encoding="utf-8")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10_000_000))

    assert diff_parent_tree_manifest(manifest, root) == ()


def test_parent_tree_manifest_verifies_unchanged_symlink_target(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    link = root / "link"
    link.symlink_to("missing-target")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    divergences = diff_parent_tree_manifest(manifest, root)

    assert manifest.entries["link"].kind == "symlink"
    assert manifest.entries["link"].link_target == "missing-target"
    assert divergences == ()


def test_parent_tree_manifest_detects_symlink_target_change(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    link = root / "link"
    link.symlink_to("before-target")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    link.unlink()
    link.symlink_to("after-target")

    divergences = diff_parent_tree_manifest(manifest, root)

    assert [(divergence.path, divergence.reason) for divergence in divergences] == [("link", "target_changed")]


def test_parent_tree_manifest_records_unsupported_entries_fail_closed(tmp_path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable on this platform")
    root = tmp_path / "parent"
    root.mkdir()
    os.mkfifo(root / "pipe")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    divergences = diff_parent_tree_manifest(manifest, root)

    assert manifest.entries["pipe"].kind == "unsupported"
    assert [(divergence.path, divergence.reason) for divergence in divergences] == [("pipe", "unverifiable")]


def test_parent_tree_manifest_detects_added_and_deleted_paths(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    deleted = root / "deleted.txt"
    deleted.write_text("before\n", encoding="utf-8")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    deleted.unlink()
    (root / "added.txt").write_text("after\n", encoding="utf-8")

    divergences = diff_parent_tree_manifest(manifest, root)

    assert [(divergence.path, divergence.reason) for divergence in divergences] == [
        ("deleted.txt", "deleted"),
        ("added.txt", "added"),
    ]


def test_parent_tree_manifest_detects_file_mode_change(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    target = root / "script.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o644)
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    target.chmod(0o755)

    divergences = diff_parent_tree_manifest(manifest, root)

    assert [(divergence.path, divergence.reason) for divergence in divergences] == [("script.sh", "mode_changed")]


def test_parent_tree_manifest_detects_kind_change(tmp_path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    target = root / "entry"
    target.write_text("file\n", encoding="utf-8")
    manifest = capture_parent_tree_manifest(root, layer_name="ground")

    target.unlink()
    target.mkdir()

    divergences = diff_parent_tree_manifest(manifest, root)

    assert [(divergence.path, divergence.reason) for divergence in divergences] == [("entry", "kind_changed")]
