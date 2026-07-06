# under-test: vcs_core._vcscore_runtime
"""The nested-ops authorization object — the graduation gate, discharged.

``nested-operations.md``'s gate: the former ``allow_nested_parent: bool``
must not graduate as a bare bool — a future call site could pass ``True``
without the ancestry proof and silently defeat the same-scope guard.
``begin_operation`` now takes a proof-carrying ``NestedParentAuthorization``
that it re-checks against the live parent operation; the bool is gone, so a
bare ``True`` is no longer expressible at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vcs_core._vcscore_runtime import _nested_parent_authorization
from vcs_core.recording import NestedParentAuthorization

from ..support.builders import make_marker_filesystem_vcscore

if TYPE_CHECKING:
    from pathlib import Path

    from vcs_core.vcscore import VcsCore


@pytest.fixture
def mg(tmp_path: Path):
    vcscore = make_marker_filesystem_vcscore(tmp_path / "ws", activate=True)
    yield vcscore
    vcscore.deactivate(warn_on_open_scopes=False)


def _open_parent_op(mg: VcsCore):
    a = mg.fork(mg.ground, "A")
    b = mg.fork(a, "B")
    mg._pipeline.set_execution_context(a)
    mg._pipeline.begin_operation(handle_id="op-a", kind="test.run", scope=a)
    return a, b


def test_bare_bool_is_no_longer_expressible(mg: VcsCore) -> None:
    """The old escape hatch is gone: allow_nested_parent=True is a TypeError."""
    _a, b = _open_parent_op(mg)
    with pytest.raises(TypeError, match="allow_nested_parent"):
        mg._pipeline.begin_operation(handle_id="op-b", kind="test.run", scope=b, allow_nested_parent=True)


def test_cross_scope_blocked_without_authorization(mg: VcsCore) -> None:
    """Default (no authorization): the same-scope invariant holds unchanged."""
    _a, b = _open_parent_op(mg)
    with pytest.raises(RuntimeError, match="belongs to"):
        mg._pipeline.begin_operation(handle_id="op-b", kind="test.run", scope=b)


def test_wrong_pair_authorization_is_rejected(mg: VcsCore) -> None:
    """A hand-rolled authorization naming the wrong (parent, child) pair is
    re-checked against the live parent operation and rejected — the proof must
    cover the actual pair, not merely exist."""
    _a, b = _open_parent_op(mg)
    forged = NestedParentAuthorization(
        parent_scope_ref="refs/vcscore/scopes/not-the-parent",
        child_scope_ref=b.ref,
        ancestry_chain=("refs/vcscore/scopes/not-the-parent",),
    )
    with pytest.raises(RuntimeError, match="belongs to"):
        mg._pipeline.begin_operation(handle_id="op-b", kind="test.run", scope=b, nested_parent=forged)


def test_flag_off_walk_returns_no_authorization(mg: VcsCore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag off ⇒ the coordinator's walk authorizes nothing (byte-identical default)."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "0")
    _a, b = _open_parent_op(mg)
    assert _nested_parent_authorization(mg, b) is None


def test_authorized_nested_op_opens_and_finalizes(mg: VcsCore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag on + genuine ancestry: the walk constructs the proof, the guard
    admits it, the nested op opens under the parent and finalizes into it
    (increment 1's spike behavior, now unit-pinned)."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    a, b = _open_parent_op(mg)
    auth = _nested_parent_authorization(mg, b)
    assert auth is not None
    assert auth.parent_scope_ref == a.ref
    assert auth.child_scope_ref == b.ref
    assert a.ref in auth.ancestry_chain
    op_b = mg._pipeline.begin_operation(handle_id="op-b", kind="test.run", scope=b, nested_parent=auth)
    assert op_b.parent_op_ref is not None
    assert len(mg._pipeline.context.operation_stack) == 2
    mg._pipeline.end_operation(handle_id="op-b", scope=b, metadata={})
    assert len(mg._pipeline.context.operation_stack) == 1
    assert mg._pipeline.context.world == a
    assert mg._pipeline.context.span is not None
    assert mg._pipeline.context.span.scope_ref == a.ref


def test_runtime_boundary_persists_nested_edge_and_disposition(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    a, b = _open_parent_op(mg)
    with mg._opened_runtime_operation(
        scope=b,
        default_label="child",
        default_kind="test.child",
        operation_id="child-op",
        operation_metadata={"world_disposition": "release"},
    ) as operation:
        metadata = mg.store._read_operation_start_metadata(operation.ref)

    op_metadata = metadata["mg"]["operation"]
    assert operation.world_disposition == "release"
    assert operation.nested_parent_scope_ref == a.ref
    assert operation.nested_child_scope_ref == b.ref
    assert op_metadata["world_disposition"] == "release"
    assert op_metadata["nested"] == {
        "parent_scope_ref": a.ref,
        "child_scope_ref": b.ref,
        "ancestry_chain": [a.ref],
    }


def test_caller_cannot_forge_nested_metadata(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    _a, b = _open_parent_op(mg)

    with (
        pytest.raises(ValueError, match="nested operation metadata is store-owned"),
        mg._opened_runtime_operation(
            scope=b,
            default_label="child",
            default_kind="test.child",
            operation_id="forged-child-op",
            operation_metadata={"nested": {"parent_scope_ref": mg.ground.ref}},
        ),
    ):
        pass


def test_world_disposition_requires_nested_operation(mg: VcsCore) -> None:
    with (
        pytest.raises(ValueError, match="world_disposition is only valid"),
        mg._opened_runtime_operation(
            scope=mg.ground,
            default_label="root",
            default_kind="test.root",
            operation_id="root-op",
            operation_metadata={"world_disposition": "release"},
        ),
    ):
        pass
