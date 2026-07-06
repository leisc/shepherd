# under-test: vcs_core._clonefile_carrier
"""Property-based tests for the clonefile carrier using Hypothesis stateful testing.

Sibling of ``test_store_state_machine.py``, aimed at the carrier layer: random
sequences of create/write/delete/commit/discard over a *tree* of clone layers,
with a reference model verifying after every step that each live layer's
working tree and ``diff_layer`` match the model.

Why this exists: B3c-3's richer run semantics shook two fidelity bugs out of
this backend (children of a clone-less parent cloned an empty tree instead of
the base; ``commit_layer`` into a missing destination applied a delta onto an
empty dir) — both are *state-space* bugs a property sweep finds and pins. The
machine deliberately allows shapes the coordinator does not yet produce
(N live children per parent; commits into clone-less destinations), because
the fan-out work (parallel-subtasks.md axis B/D) will produce them.

Modeled semantics are the CURRENT ones: ``diff_layer`` compares against the
parent's *live* tree (or the base snapshot when the parent has no clone) —
the ``_base_dir_for`` fork-point/3-way question for concurrent siblings is a
named open item in parallel-subtasks.md axis D, not asserted here.

``push_layer`` (workspace materialization + ground reset) is out of scope for
this machine; it composes with workspace-authority state that has its own
suites.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "darwin", reason="clonefile carrier requires macOS"),
]

GROUND = "ground"

#: Small, collision-friendly path / content pools — overlap is the point
#: (create-over, delete-of-shared, re-create are the interesting transitions).
PATHS = st.sampled_from(["a.txt", "b.txt", "dir/c.txt", "dir/sub/d.txt"])
CONTENTS = st.sampled_from([b"one", b"two", b"three", b""])


class ClonefileCarrierStateMachine(RuleBasedStateMachine):
    """Stateful property test for ``ClonefileCarrierBackend``.

    ``self.model`` mirrors what every layer's working tree must contain;
    ``self.parents`` mirrors the backend's parent links. Layers may exist in
    the parent map without a clone (the clone-less-parent shape) — the model
    resolves their effective tree through the same base-fallback rule the
    backend uses.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend = None
        self.base: dict[str, bytes] = {}
        self.model: dict[str, dict[str, bytes]] = {}
        self.parents: dict[str, str | None] = {GROUND: None}
        self._counter = 0

    @initialize()
    def init_backend(self) -> None:
        from vcs_core._clonefile_carrier import ClonefileCarrierBackend

        self._tmpdir = Path(tempfile.mkdtemp(prefix="carrier-sm-"))
        workspace = self._tmpdir / "ws"
        workspace.mkdir()
        # A non-trivial base snapshot: the clone-less-parent fidelity fix is
        # only observable when the base is non-empty.
        (workspace / "base.txt").write_bytes(b"base")
        (workspace / "dir").mkdir()
        (workspace / "dir" / "seed.txt").write_bytes(b"seed")
        self.base = {"base.txt": b"base", "dir/seed.txt": b"seed"}
        self.backend = ClonefileCarrierBackend(workspace, self._tmpdir / "state")

    def teardown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # --- model helpers -----------------------------------------------------

    def _live_layers(self) -> list[str]:
        return sorted(self.model)

    def _known_scopes(self) -> list[str]:
        return sorted(set(self.parents) | set(self.model))

    def _tree_or_base(self, scope_id: str | None) -> dict[str, bytes]:
        """The backend's one fallback rule: a scope's clone if it exists,
        else the base snapshot (``_source_dir`` / ``_base_dir_for``)."""
        if scope_id is not None and scope_id in self.model:
            return dict(self.model[scope_id])
        return dict(self.base)

    def _base_for(self, scope_id: str) -> dict[str, bytes]:
        """The diff/materialization base: the scope's recorded parent's
        tree-or-base (one level, never recursive — mirroring ``_base_dir_for``)."""
        return self._tree_or_base(self.parents.get(scope_id))

    def _model_diff(self, scope_id: str) -> dict[str, bytes | None]:
        mine = self.model[scope_id]
        base = self._base_for(scope_id)
        diff: dict[str, bytes | None] = {path: content for path, content in mine.items() if base.get(path) != content}
        diff.update({path: None for path in base if path not in mine})
        return diff

    # --- rules ---------------------------------------------------------------

    @rule(pick=st.integers(min_value=0, max_value=7))
    def create_child_layer(self, pick: int) -> None:
        """Fork a child of any known scope — including clone-less parents
        (the first B3c-3 fidelity-fix shape: the child must see the parent's
        effective tree, never an empty one)."""
        known = self._known_scopes()
        parent = known[pick % len(known)]
        self._counter += 1
        child = f"s{self._counter}"
        self.backend.create_layer(child, parent_scope_id=parent)
        self.parents[child] = parent
        # The backend's source rule: the parent's clone if it exists, else the
        # base snapshot — the model mirrors it through _tree_or_base.
        self.model[child] = self._tree_or_base(parent)

    @rule(path=PATHS, content=CONTENTS)
    @precondition(lambda self: self._live_layers())
    def write_file(self, path: str, content: bytes) -> None:
        layers = self._live_layers()
        layer = layers[self._counter % len(layers)]
        self.backend.write_file(layer, path, content)
        self.model[layer][path] = content
        self._counter += 1

    @rule(path=PATHS)
    @precondition(lambda self: self._live_layers())
    def delete_file(self, path: str) -> None:
        layers = self._live_layers()
        layer = layers[self._counter % len(layers)]
        self.backend.delete_file(layer, path)
        self.model[layer].pop(path, None)
        self._counter += 1

    @rule()
    @precondition(lambda self: [s for s in self._live_layers() if s != GROUND])
    def commit_into_parent(self) -> None:
        """Commit a live layer into its parent — including a parent with no
        clone (the second fidelity-fix shape: the destination must materialize
        as a full tree, not receive a bare delta)."""
        candidates = [s for s in self._live_layers() if s != GROUND and self.parents.get(s) is not None]
        if not candidates:
            return
        scope = candidates[self._counter % len(candidates)]
        parent = self.parents[scope]
        diff = self._model_diff(scope)
        self.backend.commit_layer(scope, into_scope_id=parent)
        # Model: a live destination takes the diff in place; a clone-less
        # destination first materializes from ITS effective base (the same
        # one-level fallback rule), then takes the diff.
        dest_tree = self.model[parent] if parent in self.model else self._base_for(parent)
        for path, content in diff.items():
            if content is None:
                dest_tree.pop(path, None)
            else:
                dest_tree[path] = content
        self.model[parent] = dest_tree
        self.parents.setdefault(parent, None)
        del self.model[scope]
        self.parents.pop(scope, None)
        self._counter += 1

    @rule()
    @precondition(lambda self: [s for s in self._live_layers() if s != GROUND])
    def discard_layer(self) -> None:
        candidates = [s for s in self._live_layers() if s != GROUND]
        scope = candidates[self._counter % len(candidates)]
        self.backend.discard_layer(scope)
        del self.model[scope]
        self.parents.pop(scope, None)
        self._counter += 1

    # --- invariants ----------------------------------------------------------

    @invariant()
    def working_trees_match_model(self) -> None:
        """Every live layer's on-disk working tree is exactly its model tree —
        no cross-layer bleed, no lost writes, no delta-only trees."""
        if self.backend is None:
            return
        for scope_id, expected in self.model.items():
            root = self.backend.working_path(scope_id)
            actual = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            assert actual == expected, (
                f"layer {scope_id!r} diverged from model:\n actual={sorted(actual)}\n expected={sorted(expected)}"
            )

    @invariant()
    def diffs_match_model(self) -> None:
        """``diff_layer`` agrees with the model's diff against the effective base."""
        if self.backend is None:
            return
        for scope_id in self.model:
            actual = {path: content for path, content, _mode in self.backend.diff_layer(scope_id)}
            expected = self._model_diff(scope_id)
            assert actual == expected, f"diff_layer({scope_id!r}) diverged:\n actual={actual}\n expected={expected}"


TestCarrierStandard = ClonefileCarrierStateMachine.TestCase
TestCarrierStandard.settings = settings(max_examples=60, stateful_step_count=25, deadline=None)


class TestCarrierStress(ClonefileCarrierStateMachine.TestCase):  # type: ignore[misc]
    settings = settings(max_examples=20, stateful_step_count=60, deadline=None)
