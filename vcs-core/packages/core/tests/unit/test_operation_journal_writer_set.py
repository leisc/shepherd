"""Guard: the closed set of ``ops/open/*`` membership writers stays enumerated and co-written.

The open-journal index is EXACT *only* because the atomic co-write is the sole producer/consumer
of open-ref membership mutations in the normal writer model. That rests on a closed, enumerated set
of writers — verified here in code, not assumed. Every membership change (create/delete of an
``ops/open/*`` ref) flows through a ``RefMove`` (the only carrier into ``_commit_moves`` / the
manager's ``atomic_co_write``), and those ``RefMove``s are constructed in exactly three prepare
methods. If a fourth membership writer appears un-co-written, this test fails — forcing a conscious
decision (route it through the co-write, then add it to the allowlist) rather than silent drift that
would let the index miss a live open ref (the corrupting *stale under-report* direction).

See ``260622-admission-tier-open-ops-index.md`` (Part A, "Writer model — the boundary of exact").
"""

from __future__ import annotations

import ast
from pathlib import Path

import vcs_core._world_operation_journal as journal_module

# The enumerated membership writers. A genuinely non-membership RefMove (e.g. a future
# closed->archived move that does NOT change open-set membership) added here would be a conscious
# allowlist edit, paired with confirming it need not co-write the open-journal index.
_SANCTIONED_OPEN_MEMBERSHIP_WRITERS = frozenset({"prepare_open", "prepare_terminal", "prepare_cleanup_stale_open_ref"})

_MODULE_SOURCE = Path(journal_module.__file__).read_text()
_MODULE_TREE = ast.parse(_MODULE_SOURCE)


def _nearest_enclosing_function(parents: dict[ast.AST, ast.AST], node: ast.AST) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _refmove_producing_functions() -> set[str]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(_MODULE_TREE):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    producers: set[str] = set()
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RefMove":
            enclosing = _nearest_enclosing_function(parents, node)
            if enclosing is not None:
                producers.add(enclosing)
    return producers


def test_open_ref_membership_refmoves_originate_only_in_the_sanctioned_writers() -> None:
    # RefMove is the ONLY carrier of journal-ref membership moves into _commit_moves / the manager
    # co-write, and the journal module constructs RefMoves solely for journal membership. So the set
    # of functions building a RefMove must equal the enumerated writer set; a fourth fails here.
    assert _refmove_producing_functions() == set(_SANCTIONED_OPEN_MEMBERSHIP_WRITERS)


def test_no_raw_open_ref_deletion_bypasses_the_refmove_co_write() -> None:
    # Backstop: the old non-atomic raw delete helper is gone, and no direct ref deletion (pygit2 or
    # `git update-ref -d`) may reappear in the journal store — open-ref deletion must ride a RefMove
    # so it is always co-written with the index tombstone.
    assert "_delete_ref_if_targets" not in _MODULE_SOURCE
    assert "references.delete(" not in _MODULE_SOURCE
    assert '"-d"' not in _MODULE_SOURCE
    assert "'-d'" not in _MODULE_SOURCE
