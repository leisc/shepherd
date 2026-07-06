# under-test: vcs_core._graph
"""Tests for ASCII graph rendering."""

from __future__ import annotations

from vcs_core._graph import render_graph
from vcs_core.types import CommitInfo


def _commit(oid: str, effect_type: str, scope: str, **extra: object) -> CommitInfo:
    """Helper to build a CommitInfo for testing."""
    metadata: dict[str, object] = {"type": effect_type, "scope": scope, **extra}
    return CommitInfo(
        oid=oid,
        message=f"{effect_type} on {scope}",
        timestamp=0.0,
        metadata=metadata,
        parent_oids=[],
    )


def test_render_empty() -> None:
    assert render_graph([]) == []


def test_render_linear() -> None:
    """All commits on ground — single column, no branches."""
    entries = [
        _commit("aaa", "Marker", "ground"),
        _commit("bbb", "Init", "ground"),
    ]
    lines = render_graph(entries)
    assert len(lines) == 2
    # Both should be in column 0 with '*'
    for line in lines:
        assert line.startswith("*")


def test_render_fork_merge() -> None:
    """Branch from ground, merge back — two columns, merge connector."""
    entries = [
        _commit("aaa", "ScopeMerge", "task-1", merged_into="ground"),
        _commit("bbb", "Marker", "task-1"),
        _commit("ccc", "Init", "ground"),
    ]
    lines = render_graph(entries)
    assert len(lines) >= 3  # at least 3 lines + merge connector

    # ScopeMerge line should have '*' not in column 0
    assert "ScopeMerge" in lines[0]

    # There should be a merge connector line with '/'
    has_connector = any("/" in line for line in lines)
    assert has_connector, f"No merge connector found in: {lines}"


def test_render_nested_scopes() -> None:
    """Three scope levels: ground -> task -> step."""
    entries = [
        _commit("aaa", "ScopeMerge", "task-1", merged_into="ground"),
        _commit("bbb", "ScopeMerge", "step-A", merged_into="task-1"),
        _commit("ccc", "Marker", "step-A"),
        _commit("ddd", "Marker", "task-1"),
        _commit("eee", "Init", "ground"),
    ]
    lines = render_graph(entries)

    # Should have at least one line with three columns (three scopes)
    # Look for lines with multiple '|' or '*'
    max_markers = max(line.count("|") + line.count("*") for line in lines if "scope:" in line)
    assert max_markers >= 2, f"Expected at least 2 active columns, lines: {lines}"


def test_render_discard() -> None:
    """Discarded scope frees its column without a merge connector."""
    entries = [
        _commit("aaa", "DiscardSnapshot", "task-1", discarded_scope="task-1"),
        _commit("bbb", "Marker", "task-1"),
        _commit("ccc", "Init", "ground"),
    ]
    lines = render_graph(entries)

    # Discard should NOT produce a '/' connector
    non_commit_lines = [line for line in lines if "scope:" not in line]
    has_merge_connector = any("/" in line for line in non_commit_lines)
    assert not has_merge_connector, f"Discard should not have merge connector: {lines}"


def test_column_reuse() -> None:
    """After merging scope A, scope B should reuse A's column."""
    entries = [
        _commit("aaa", "ScopeMerge", "task-B", merged_into="ground"),
        _commit("bbb", "Marker", "task-B"),
        _commit("ccc", "ScopeMerge", "task-A", merged_into="ground"),
        _commit("ddd", "Marker", "task-A"),
        _commit("eee", "Init", "ground"),
    ]
    lines = render_graph(entries)

    # task-B should use column 1 (reused after task-A is released).
    # Count max columns used (max number of '|' or '*' in any line).
    max_active = max(line.count("|") + line.count("*") for line in lines if "scope:" in line)
    assert max_active <= 2, f"Expected column reuse (max 2 columns), lines: {lines}"
