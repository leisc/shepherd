"""Contract guard (issue 01): every advertised ``examples=`` string in a built-in
substrate source parses as a valid ``vcs-core exec`` invocation, so the help text
cannot drift from the ``exec`` grammar (``-p/--param key=value``).

This is a *guard*, not a structural collapse: the examples are curated, user-facing
help (e.g. ``git``'s ``-p name=feature/demo``) whose illustrative values generation
from the param schema would degrade. The example-vs-parser invariant is a
cross-boundary case where a guard is the right tool (see
``DESIGN-derived-fact-authorities.md``).
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "vcs_core"
# Inventory: substrate sources that advertise exec examples. If a new substrate
# source is added with examples, add it here (the collection guard below fires).
_SUBSTRATE_SOURCES = ("substrates.py", "git_substrate.py", "sqlite_substrate.py")


def _advertised_examples() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for fname in _SUBSTRATE_SOURCES:
        tree = ast.parse((_SRC / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "examples"
                and isinstance(node.value, (ast.Tuple, ast.List))
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        found.append((fname, elt.value))
    return found


def _assert_valid_exec_example(example: str) -> None:
    tokens = shlex.split(example)
    assert tokens[:2] == ["vcs-core", "exec"], f"not a `vcs-core exec` example: {example!r}"
    assert len(tokens) >= 4, f"example missing substrate/command: {example!r}"
    rest = tokens[4:]  # everything after `vcs-core exec <substrate> <command>`
    i = 0
    while i < len(rest):
        assert rest[i] in ("-p", "--param"), f"{example!r}: exec takes `-p/--param key=value`, not {rest[i]!r}"
        assert i + 1 < len(rest), f"{example!r}: {rest[i]} must be followed by a key=value pair"
        assert "=" in rest[i + 1], f"{example!r}: {rest[i]} must be followed by a key=value pair"
        i += 2


def test_substrate_example_inventory_nonempty() -> None:
    # Drift guard: if a source is renamed/dropped, this collapses and flags it.
    assert len(_advertised_examples()) >= 8


@pytest.mark.parametrize(("source", "example"), _advertised_examples())
def test_builtin_substrate_examples_match_exec_grammar(source: str, example: str) -> None:
    _assert_valid_exec_example(example)
