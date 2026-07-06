from __future__ import annotations

import pytest
from vcs_core import InvalidRepositoryStateError
from vcs_core.vcscore import VcsCore


def test_high_level_nested_child_operation_restores_parent_context(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    child = mg.fork(mg.ground, "nested-run-child")

    with mg.runtime_activity(
        scope=mg.ground,
        operation_label="parent",
        operation_kind="test.parent",
    ) as parent:
        assert parent is not None

        outcome = mg._execute_recorded_in_child_operation(
            "marker",
            "mark",
            scope=child,
            operation_id="child-op",
            operation_kind="marker.mark",
            label="child",
        )

        assert outcome.oids
        assert mg._pipeline.current_operation() is not None
        assert mg._pipeline.current_operation().ref == parent.ref
        assert mg._pipeline.context.world is not None
        assert mg._pipeline.context.world.ref == mg.ground.ref

    assert mg._pipeline.current_operation() is None
    assert mg._pipeline.context.world is not None
    assert mg._pipeline.context.world.ref == child.ref
    mg.discard(child)


def test_child_operation_flush_observes_restored_parent_context(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child operation's workspace flush must run AFTER the parent context is
    restored, not merely after the boundary exits.

    ``opened_runtime_operation`` restores the enclosing parent context *between*
    ``end_operation`` and ``_flush_workspace_state_for_runtime_operation``
    (``_vcscore_runtime.py`` :454-457). If the restore were only an outer-``finally``
    cleanup it would run *after* the flush, so the flush would execute while the
    pipeline still pointed at the child world. We spy the flush and capture the
    ambient world at the instant it is invoked: under the correct ordering it is
    already the parent world (ground); a restore-on-exit-only regression would
    capture the child world here. This pins restore-before-flush, which the sibling
    ``..._restores_parent_context`` test (asserting only after the boundary exits)
    cannot distinguish.
    """
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    child = mg.fork(mg.ground, "flush-window-child")

    flush_ambient_refs: list[str | None] = []
    original_flush = mg._flush_workspace_state_for_runtime_operation

    def spy_flush(operation_id: str) -> None:
        ambient = mg._pipeline.context.world.ref if mg._pipeline.context.world is not None else None
        flush_ambient_refs.append(ambient)
        original_flush(operation_id)

    monkeypatch.setattr(mg, "_flush_workspace_state_for_runtime_operation", spy_flush)

    with mg.runtime_activity(
        scope=mg.ground,
        operation_label="parent",
        operation_kind="test.parent",
    ):
        mg._execute_recorded_in_child_operation(
            "marker",
            "mark",
            scope=child,
            operation_id="flush-window-child-op",
            operation_kind="marker.mark",
            label="child",
        )

    # The child operation closes inward (before the parent activity exits), so its
    # flush is the first one observed. It must see the restored parent world.
    assert flush_ambient_refs, "expected the child operation flush to be invoked"
    assert flush_ambient_refs[0] == mg.ground.ref
    # No flush may run while the pipeline still points at the child world — that
    # would be the restore-on-exit-only regression this test exists to catch.
    assert child.ref not in flush_ambient_refs
    mg.discard(child)


def test_nested_runtime_activity_entry_restores_parent_context(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``runtime_activity`` is itself a wired nested entry: a child activity opened on
    a descendant scope under a live parent op nests, installs the child world, and
    restores the parent world on exit."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    parent_scope = mg.fork(mg.ground, "ra-parent")
    child_scope = mg.fork(parent_scope, "ra-child")

    with mg.runtime_activity(
        scope=parent_scope,
        operation_label="parent",
        operation_kind="test.parent",
    ) as parent_op:
        assert parent_op is not None
        assert mg._pipeline.context.world is not None
        assert mg._pipeline.context.world.ref == parent_scope.ref

        with mg.runtime_activity(
            scope=child_scope,
            operation_label="child",
            operation_kind="test.child",
        ) as child_op:
            assert child_op is not None
            assert child_op.ref != parent_op.ref
            assert mg._pipeline.current_operation().ref == child_op.ref
            assert mg._pipeline.context.world is not None
            assert mg._pipeline.context.world.ref == child_scope.ref

        # Inner activity exited: parent op + parent world restored.
        assert mg._pipeline.current_operation().ref == parent_op.ref
        assert mg._pipeline.context.world is not None
        assert mg._pipeline.context.world.ref == parent_scope.ref

    assert mg._pipeline.current_operation() is None
    mg.discard(child_scope)
    mg.discard(parent_scope)


def test_depth_two_nested_runtime_activities_finalize_inward(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A three-scope chain nests two levels deep and finalizes inward, restoring the
    enclosing operation and world at each hop (execution-boundary §5 recursion)."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    a = mg.fork(mg.ground, "depth-a")
    b = mg.fork(a, "depth-b")
    c = mg.fork(b, "depth-c")

    with mg.runtime_activity(scope=a, operation_label="a", operation_kind="test.a") as op_a:
        with mg.runtime_activity(scope=b, operation_label="b", operation_kind="test.b") as op_b:
            with mg.runtime_activity(scope=c, operation_label="c", operation_kind="test.c") as op_c:
                assert mg._pipeline.current_operation().ref == op_c.ref
                assert mg._pipeline.context.world is not None
                assert mg._pipeline.context.world.ref == c.ref
            assert mg._pipeline.current_operation().ref == op_b.ref
            assert mg._pipeline.context.world is not None
            assert mg._pipeline.context.world.ref == b.ref
        assert mg._pipeline.current_operation().ref == op_a.ref
        assert mg._pipeline.context.world is not None
        assert mg._pipeline.context.world.ref == a.ref

    assert mg._pipeline.current_operation() is None
    mg.discard(c)
    mg.discard(b)
    mg.discard(a)


def test_nested_child_operation_persists_scope_edge_metadata(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store records the nested scope edge + world disposition on the open child
    operation — store-owned ancestry proof that does not rely on operation-history
    parent linkage (T3)."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "1")
    parent_scope = mg.fork(mg.ground, "edge-parent")
    child_scope = mg.fork(parent_scope, "edge-child")

    with (
        mg.runtime_activity(scope=parent_scope, operation_label="parent", operation_kind="test.parent"),
        mg.runtime_activity(scope=child_scope, operation_label="child", operation_kind="test.child"),
    ):
        open_ops = mg.store.list_open_operations()
        child_entries = [op for op in open_ops if op.nested_parent_scope_ref is not None]
        assert len(child_entries) == 1
        edge = child_entries[0]
        assert edge.nested_parent_scope_ref == parent_scope.ref
        assert edge.nested_child_scope_ref == child_scope.ref
        assert edge.world_disposition == "adopt"

    mg.discard(child_scope)
    mg.discard(parent_scope)


def test_flag_off_high_level_nested_entry_refuses_with_readiness_text(
    mg: VcsCore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-off, a high-level child entry on a descendant scope under a live op is
    refused by readiness guard #4 (which fires before boundary guard #3 on the
    high-level entries), with today's byte-exact message."""
    monkeypatch.setenv("VCS_CORE_NESTED_OPERATIONS", "0")
    child = mg.fork(mg.ground, "flag-off-child")

    with (
        mg.runtime_activity(scope=mg.ground, operation_label="parent", operation_kind="test.parent"),
        pytest.raises(
            InvalidRepositoryStateError,
            match=r"Cannot execute marker\.mark: readiness blocked by",
        ),
    ):
        mg._execute_recorded_in_child_operation(
            "marker",
            "mark",
            scope=child,
            operation_id="flag-off-child-op",
            operation_kind="marker.mark",
            label="child",
        )
    mg.discard(child)
