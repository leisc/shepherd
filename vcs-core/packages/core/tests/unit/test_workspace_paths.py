# under-test: vcs_core._workspace_paths
"""Guard for the consolidated workspace-relative path check (P4 rider 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from vcs_core._workspace_paths import normalize_workspace_relative_path


@pytest.mark.parametrize(
    "path",
    ["", "/etc/passwd", "..", "../escape", "a/../../b", "/", "a/b/../../.."],
)
def test_rejects_empty_absolute_and_traversal(path: str) -> None:
    with pytest.raises(ValueError, match="Invalid workspace-relative path"):
        normalize_workspace_relative_path(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.py", Path("a.py")),
        ("src/a.py", Path("src/a.py")),
        ("a/b/c.txt", Path("a/b/c.txt")),
    ],
)
def test_normalizes_valid_relative_paths(path: str, expected: Path) -> None:
    assert normalize_workspace_relative_path(path) == expected


def test_rejects_paths_that_normalize_to_nothing() -> None:
    # "." has no path components once normalized — degenerate, refused.
    with pytest.raises(ValueError, match="Invalid workspace-relative path"):
        normalize_workspace_relative_path(".")
