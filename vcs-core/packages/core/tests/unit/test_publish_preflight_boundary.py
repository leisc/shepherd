# under-test: vcs_core._world_storage_manager
"""Guard: the prior-lineage retention preflight stays OFF the publish hot path.

Trust-by-default (``260623-0640-plan.md``, Part A) removed
``_validate_authority_retention_preflight`` from ``prepare_publication`` and ``fork_world_ref`` —
the O(prior-worlds) re-walk that made publishing O(N^2) (2N-1 closure computations per publish,
Sigma = N^2). The detector method *survives* (Part B runs it on demand in ``fsck_world(mode="deep")``),
so a future edit could silently re-add it to the hot path and quietly restore the wall. This test
fails if any publish hot-path function calls the preflight again — forcing a conscious decision
rather than silent drift.

Pairs with the *behavioral* count contract in ``test_world_storage_manager.py``
(``test_world_storage_manager_publish_computes_closure_a_constant_number_of_times``): this guards the
call *site*; that guards the *cost*. Mirrors the open-ref writer-set guard
(``test_operation_journal_writer_set.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import vcs_core._world_storage_manager as wsm_module

# Publish hot-path methods. None of these may call the prior-lineage preflight; deep lineage
# integrity runs on demand in fsck_world(mode="deep") instead. Re-adding a call in any of them is
# the regression this guards — it would restore the O(N) per-publish re-walk. (Part B adds an
# *fsck* caller, which is deliberately NOT in this set, so this guard stays green across A->B.)
_PUBLISH_HOT_PATH_FUNCTIONS = frozenset(
    {
        "prepare_publication",
        "fork_world_ref",
        "advance_publication",
        "publish_root_world",
        "advance_world_ref",
        "_publish_world",
    }
)

_PREFLIGHT = "_validate_authority_retention_preflight"

_MODULE_SOURCE = Path(wsm_module.__file__).read_text()
_MODULE_TREE = ast.parse(_MODULE_SOURCE)


def _nearest_enclosing_function(parents: dict[ast.AST, ast.AST], node: ast.AST) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _preflight_calling_functions() -> set[str]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(_MODULE_TREE):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    callers: set[str] = set()
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == _PREFLIGHT:
            enclosing = _nearest_enclosing_function(parents, node)
            if enclosing is not None:
                callers.add(enclosing)
    return callers


def test_prior_lineage_preflight_has_no_publish_hot_path_caller() -> None:
    # The detector survives for fsck (Part B), but no publish hot-path function may invoke it —
    # that is exactly the O(N) per-publish re-walk trust-by-default removed.
    offenders = _preflight_calling_functions() & _PUBLISH_HOT_PATH_FUNCTIONS
    assert offenders == set(), f"prior-lineage preflight re-added to the publish hot path: {sorted(offenders)}"
