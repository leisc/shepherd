"""Shadow commons-vcs recording for vcs-core Store commits.

The existing Store remains authoritative. This module mirrors already-created
Store carrier commits into commons-vcs Objects so the shared kernel path can be
validated before it becomes a write/read substrate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import pygit2
from commons_vcs import Object, Repo
from commons_vcs.backends.git import GitBackend

from vcs_core._errors import VcsCoreError
from vcs_core.profiles.committed_view import reachable_from_heads
from vcs_core.profiles.commons_refs import (
    carrier_commit_ref,
    pending_projection_prefix,
    pending_projection_ref,
    scope_head_ref,
    workspace_tree_pin_name,
)
from vcs_core.profiles.commons_vcs import profile as vcscore_profile
from vcs_core.profiles.projection import (
    is_scope_creation_boundary,
    project_commit_object,
    project_effect_object,
    project_scope_object,
)

if TYPE_CHECKING:
    from vcs_core.store import Store
    from vcs_core.types import ScopeInfo


_UNSET: Final = object()


class CommonsShadowConflictError(VcsCoreError, RuntimeError):
    """Raised when shadow state was concurrently advanced."""


class CommonsShadowRecoveryError(VcsCoreError, RuntimeError):
    """Raised when shadow pending state cannot be repaired safely."""


class CommonsShadowUnsupportedError(VcsCoreError, RuntimeError):
    """Raised when a caller attempts an unsupported shadow recording path."""


@dataclass(frozen=True)
class CommonsShadowRecord:
    """Result of projecting one Store carrier commit into commons-vcs.

    `effect_id` and `commit_id` identify the projected Objects for
    `carrier_oid`. `previous_head` and `new_head` describe the committed shadow
    head observed before and after this call. For an idempotent re-recording of
    an older reachable carrier, `commit_id` may differ from `new_head`.
    """

    carrier_oid: str
    scope_id: str
    effect_id: str
    commit_id: str
    previous_head: str | None
    new_head: str
    shadow_head_ref: str
    carrier_commit_ref: str


@dataclass(frozen=True)
class _ProjectionPlan:
    carrier_oid: str
    scope_id: str
    expected_head: str | None
    parent_carrier_oid: str | None
    parent_commit_id: str | None
    scope_object: Object
    effect_object: Object
    commit_object: Object
    effect_id: str
    commit_id: str
    workspace_tree: str


@dataclass(frozen=True)
class _PendingProjection:
    scope_id: str
    carrier_oid: str
    expected_head: str | None
    effect_id: str
    commit_id: str
    workspace_tree: str
    parent_carrier_oid: str | None
    parent_commit_id: str | None


class CommonsShadowRecorder:
    """Project Store commits into commons-vcs shadow state."""

    def __init__(self, store: Store) -> None:
        self._store = store
        git_dir = Path(store._repo.path).resolve()
        self._backend = GitBackend.open(git_dir)
        self._repo = Repo(profiles=[vcscore_profile], backend=self._backend)

    @property
    def repo(self) -> Repo:
        """The commons-vcs Repo used by this recorder."""
        return self._repo

    @property
    def backend(self) -> GitBackend:
        """The GitBackend used by this recorder."""
        return self._backend

    def record_carrier_commit(
        self,
        scope: ScopeInfo,
        carrier_oid: str,
        *,
        expected_head: str | None | object = _UNSET,
    ) -> CommonsShadowRecord:
        """Shadow-project one already-created Store carrier commit.

        If `expected_head` is omitted, the current shadow head is used. Passing
        an explicit value turns this into a stale-writer check for tests and
        future coordinator wiring.
        """
        carrier_commit = self._carrier_commit(carrier_oid)
        scope_object = project_scope_object(scope)
        scope_id = scope_object.id
        head_ref = self.shadow_head_ref(scope_id)

        with self._backend.scope_lock(scope_id):
            self._recover_pending(scope, scope_id)
            current_head = self._backend.get_ref(head_ref)
            previous_head = current_head if expected_head is _UNSET else cast("str | None", expected_head)
            if current_head != previous_head:
                raise CommonsShadowConflictError(
                    f"shadow head for {scope.name!r} is {current_head!r}, expected {previous_head!r}"
                )

            carrier_ref = self.carrier_commit_ref(carrier_oid)
            existing_commit_id = self._backend.get_ref(carrier_ref)
            if existing_commit_id is not None:
                plan = self._build_projection_plan(
                    scope,
                    scope_object,
                    carrier_commit,
                    expected_head=self._expected_head_for_existing_mapping(scope, carrier_commit),
                )
                if existing_commit_id != plan.commit_id:
                    raise CommonsShadowConflictError(
                        f"carrier commit {carrier_oid} projects to {existing_commit_id}, "
                        f"but Store truth recomputes {plan.commit_id}"
                    )
                if current_head is None or not self._is_reachable(current_head, existing_commit_id):
                    raise CommonsShadowConflictError(
                        f"carrier commit {carrier_oid} projects to {existing_commit_id}, "
                        "but that commit is not reachable from the committed shadow head"
                    )
                return CommonsShadowRecord(
                    carrier_oid=carrier_oid,
                    scope_id=scope_id,
                    effect_id=self._effect_id_for_commons_commit(existing_commit_id),
                    commit_id=existing_commit_id,
                    previous_head=current_head,
                    new_head=current_head,
                    shadow_head_ref=head_ref,
                    carrier_commit_ref=carrier_ref,
                )

            plan = self._build_projection_plan(scope, scope_object, carrier_commit, expected_head=previous_head)
            pending = self._pending_from_plan(plan)
            pending_json = self._encode_pending(pending)
            pending_ref = self.pending_projection_ref(scope_id, carrier_oid)
            if not self._backend.compare_and_swap_ref(pending_ref, None, pending_json):
                raise CommonsShadowRecoveryError(f"pending projection already exists for {carrier_oid}")

            self._publish_plan_objects(plan)
            if not self._backend.compare_and_swap_ref(head_ref, previous_head, plan.commit_id):
                raise CommonsShadowConflictError(
                    f"shadow head for {scope.name!r} changed while projecting {carrier_oid}"
                )
            if not self._backend.compare_and_swap_ref(carrier_ref, None, plan.commit_id):
                existing = self._backend.get_ref(carrier_ref)
                if existing != plan.commit_id:
                    raise CommonsShadowRecoveryError(
                        f"carrier commit {carrier_oid} was concurrently projected to {existing!r}"
                    )
            self._delete_pending_if_unchanged(pending_ref, pending_json)
            return CommonsShadowRecord(
                carrier_oid=carrier_oid,
                scope_id=scope_id,
                effect_id=plan.effect_id,
                commit_id=plan.commit_id,
                previous_head=previous_head,
                new_head=plan.commit_id,
                shadow_head_ref=head_ref,
                carrier_commit_ref=carrier_ref,
            )

    @staticmethod
    def shadow_head_ref(scope_id: str) -> str:
        """Backend ref name for one projected vcs-core scope head."""
        return scope_head_ref(scope_id)

    @staticmethod
    def pending_projection_ref(scope_id: str, carrier_oid: str) -> str:
        """Backend ref name for one in-flight carrier projection."""
        return pending_projection_ref(scope_id, carrier_oid)

    @staticmethod
    def pending_projection_prefix(scope_id: str) -> str:
        """Backend ref prefix for in-flight projections for one scope."""
        return pending_projection_prefix(scope_id)

    @staticmethod
    def carrier_commit_ref(carrier_oid: str) -> str:
        """Backend ref name mapping a Store carrier commit to its commons commit."""
        return carrier_commit_ref(carrier_oid)

    @staticmethod
    def workspace_tree_pin_name(commit_id: str) -> str:
        """Pin name for the Git workspace tree cited by a commons commit."""
        return workspace_tree_pin_name(commit_id)

    def _carrier_commit(self, carrier_oid: str) -> pygit2.Commit:
        obj = self._store._repo[pygit2.Oid(hex=carrier_oid)]
        if not isinstance(obj, pygit2.Commit):
            raise TypeError(f"carrier object is not a commit: {carrier_oid}")
        return obj

    def _build_projection_plan(
        self,
        scope: ScopeInfo,
        scope_object: Object,
        carrier_commit: pygit2.Commit,
        *,
        expected_head: str | None,
    ) -> _ProjectionPlan:
        parent_carrier_oid, parent_commit_id = self._parent_projection(scope, carrier_commit)
        if parent_commit_id != expected_head:
            raise CommonsShadowConflictError(
                f"carrier commit {carrier_commit.id} is out of order: parent projects to "
                f"{parent_commit_id!r}, but shadow head is {expected_head!r}"
            )
        return self._projection_plan_from_parent(
            scope_object,
            carrier_commit,
            expected_head=expected_head,
            parent_carrier_oid=parent_carrier_oid,
            parent_commit_id=parent_commit_id,
        )

    def _projection_plan_from_parent(
        self,
        scope_object: Object,
        carrier_commit: pygit2.Commit,
        *,
        expected_head: str | None,
        parent_carrier_oid: str | None,
        parent_commit_id: str | None,
    ) -> _ProjectionPlan:
        effect_object = project_effect_object(self._store._repo, carrier_commit)
        effect_id = effect_object.id
        commit_object = project_commit_object(
            self._store._repo,
            carrier_commit,
            effect_id=effect_id,
            scope_id=scope_object.id,
            parent_id=parent_commit_id,
        )
        workspace_tree = str(commit_object.body["workspace_tree"])
        return _ProjectionPlan(
            carrier_oid=str(carrier_commit.id),
            scope_id=scope_object.id,
            expected_head=expected_head,
            parent_carrier_oid=parent_carrier_oid,
            parent_commit_id=parent_commit_id,
            scope_object=scope_object,
            effect_object=effect_object,
            commit_object=commit_object,
            effect_id=effect_id,
            commit_id=commit_object.id,
            workspace_tree=workspace_tree,
        )

    def _parent_projection(self, scope: ScopeInfo, carrier_commit: pygit2.Commit) -> tuple[str | None, str | None]:
        carrier_oid = str(carrier_commit.id)
        if not carrier_commit.parent_ids:
            if self._is_scope_creation_boundary(scope, carrier_oid):
                return None, None
            raise CommonsShadowConflictError(f"carrier commit {carrier_commit.id} has no first parent")
        parent_carrier_oid = str(carrier_commit.parent_ids[0])
        if scope.creation_oid and parent_carrier_oid == scope.creation_oid:
            return None, None
        parent_commit_id = self._backend.get_ref(self.carrier_commit_ref(parent_carrier_oid))
        if parent_commit_id is not None:
            return parent_carrier_oid, parent_commit_id
        if self._is_scope_creation_boundary(scope, parent_carrier_oid):
            return None, None
        raise CommonsShadowConflictError(
            f"carrier commit {carrier_commit.id} first parent {parent_carrier_oid} is not projected"
        )

    def _is_scope_creation_boundary(self, scope: ScopeInfo, parent_carrier_oid: str) -> bool:
        return is_scope_creation_boundary(self._store._repo, scope, parent_carrier_oid)

    def _expected_head_for_existing_mapping(self, scope: ScopeInfo, carrier_commit: pygit2.Commit) -> str | None:
        _parent_carrier_oid, parent_commit_id = self._parent_projection(scope, carrier_commit)
        return parent_commit_id

    def _publish_plan_objects(self, plan: _ProjectionPlan) -> None:
        self._repo.append(plan.scope_object)
        self._repo.append(plan.effect_object)
        self._repo.append(plan.commit_object)
        self._pin_workspace_tree(plan.commit_id, plan.workspace_tree)

    def _pin_workspace_tree(self, commit_id: str, tree_oid: str) -> None:
        self._backend.pin_git_object(self.workspace_tree_pin_name(commit_id), tree_oid)

    def _is_reachable(self, head: str, commit_id: str) -> bool:
        try:
            return commit_id in reachable_from_heads(self._repo, (head,), schema_ref="vcscore/commit/v1")
        except ValueError as exc:
            raise CommonsShadowConflictError(str(exc)) from exc

    def _effect_id_for_commons_commit(self, commit_id: str) -> str:
        obj = self._repo.get(commit_id)
        if obj is None:
            raise CommonsShadowConflictError(f"projected commons commit is missing: {commit_id}")
        effect_edges = [edge.target for edge in obj.edges if edge.role == "effect"]
        if len(effect_edges) != 1:
            raise CommonsShadowConflictError(f"projected commons commit has invalid effect edges: {commit_id}")
        return effect_edges[0]

    def _pending_from_plan(self, plan: _ProjectionPlan) -> _PendingProjection:
        return _PendingProjection(
            scope_id=plan.scope_id,
            carrier_oid=plan.carrier_oid,
            expected_head=plan.expected_head,
            effect_id=plan.effect_id,
            commit_id=plan.commit_id,
            workspace_tree=plan.workspace_tree,
            parent_carrier_oid=plan.parent_carrier_oid,
            parent_commit_id=plan.parent_commit_id,
        )

    def _recover_pending(self, scope: ScopeInfo, scope_id: str) -> None:
        pending_refs = list(self._backend.list_refs(self.pending_projection_prefix(scope_id)))
        if not pending_refs:
            return
        if len(pending_refs) > 1:
            raise CommonsShadowRecoveryError(f"multiple pending commons shadow projections for scope {scope_id}")
        pending_ref = pending_refs[0]
        pending_json = self._backend.get_ref(pending_ref)
        if pending_json is None:
            return
        pending = self._decode_pending(pending_json)
        if pending.scope_id != scope_id:
            raise CommonsShadowRecoveryError(
                f"pending projection {pending_ref} claims scope {pending.scope_id}, expected {scope_id}"
            )
        carrier_commit = self._carrier_commit(pending.carrier_oid)
        scope_object = project_scope_object(scope)
        plan = self._build_recovery_plan(scope, scope_object, carrier_commit, pending)
        self._recover_pending_plan(pending_ref, pending_json, pending, plan)

    def _build_recovery_plan(
        self,
        scope: ScopeInfo,
        scope_object: Object,
        carrier_commit: pygit2.Commit,
        pending: _PendingProjection,
    ) -> _ProjectionPlan:
        parent_carrier_oid, parent_commit_id = self._parent_projection(scope, carrier_commit)
        if parent_commit_id != pending.expected_head:
            raise CommonsShadowRecoveryError(
                f"pending projection {pending.carrier_oid} expected head {pending.expected_head!r}, "
                f"but Store parent projects to {parent_commit_id!r}"
            )
        plan = self._projection_plan_from_parent(
            scope_object,
            carrier_commit,
            expected_head=pending.expected_head,
            parent_carrier_oid=parent_carrier_oid,
            parent_commit_id=parent_commit_id,
        )
        recomputed = self._pending_from_plan(plan)
        if recomputed != pending:
            raise CommonsShadowRecoveryError(
                f"pending projection {pending.carrier_oid} no longer matches Store projection identity"
            )
        return plan

    def _recover_pending_plan(
        self,
        pending_ref: str,
        pending_json: str,
        pending: _PendingProjection,
        plan: _ProjectionPlan,
    ) -> None:
        head_ref = self.shadow_head_ref(pending.scope_id)
        carrier_ref = self.carrier_commit_ref(pending.carrier_oid)
        shadow_head = self._backend.get_ref(head_ref)
        carrier_mapping = self._backend.get_ref(carrier_ref)

        # The head advanced, but the carrier mapping was not published.
        if shadow_head == pending.commit_id and carrier_mapping is None:
            self._publish_plan_objects(plan)
            if not self._backend.compare_and_swap_ref(carrier_ref, None, pending.commit_id):
                existing = self._backend.get_ref(carrier_ref)
                if existing != pending.commit_id:
                    raise CommonsShadowRecoveryError(
                        f"cannot recover carrier mapping for {pending.carrier_oid}: found {existing!r}"
                    )
            self._delete_pending_if_unchanged(pending_ref, pending_json)
            return

        # Both durable refs are complete; only the stale pending ref remains.
        if shadow_head == pending.commit_id and carrier_mapping == pending.commit_id:
            self._publish_plan_objects(plan)
            self._delete_pending_if_unchanged(pending_ref, pending_json)
            return

        # The pending ref was created before either durable ref advanced.
        if shadow_head == pending.expected_head and carrier_mapping is None:
            self._publish_plan_objects(plan)
            if not self._backend.compare_and_swap_ref(head_ref, pending.expected_head, pending.commit_id):
                raise CommonsShadowRecoveryError(f"cannot recover shadow head for {pending.carrier_oid}")
            if not self._backend.compare_and_swap_ref(carrier_ref, None, pending.commit_id):
                existing = self._backend.get_ref(carrier_ref)
                if existing != pending.commit_id:
                    raise CommonsShadowRecoveryError(
                        f"cannot recover carrier mapping for {pending.carrier_oid}: found {existing!r}"
                    )
            self._delete_pending_if_unchanged(pending_ref, pending_json)
            return

        # A later head may already reach the mapped commit after another writer
        # completed this projection and advanced the scope.
        if (
            carrier_mapping == pending.commit_id
            and shadow_head is not None
            and self._is_reachable(shadow_head, pending.commit_id)
        ):
            self._delete_pending_if_unchanged(pending_ref, pending_json)
            return

        raise CommonsShadowRecoveryError(
            f"cannot recover pending projection for {pending.carrier_oid}: "
            f"shadow_head={shadow_head!r}, carrier_mapping={carrier_mapping!r}"
        )

    def _delete_pending_if_unchanged(self, pending_ref: str, expected_json: str) -> None:
        if not self._backend.compare_and_delete_ref(pending_ref, expected_json):
            raise CommonsShadowRecoveryError(f"pending projection changed while completing {pending_ref}")

    @staticmethod
    def _encode_pending(record: _PendingProjection) -> str:
        payload = {
            "version": 1,
            "scope_id": record.scope_id,
            "carrier_oid": record.carrier_oid,
            "expected_head": record.expected_head,
            "effect_id": record.effect_id,
            "commit_id": record.commit_id,
            "workspace_tree": record.workspace_tree,
            "parent_carrier_oid": record.parent_carrier_oid,
            "parent_commit_id": record.parent_commit_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _decode_pending(cls, raw: str) -> _PendingProjection:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CommonsShadowRecoveryError("pending projection is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise CommonsShadowRecoveryError("pending projection must be a JSON object")
        if payload.get("version") != 1:
            raise CommonsShadowRecoveryError("pending projection has unsupported version")
        allowed = {
            "version",
            "scope_id",
            "carrier_oid",
            "expected_head",
            "effect_id",
            "commit_id",
            "workspace_tree",
            "parent_carrier_oid",
            "parent_commit_id",
        }
        if set(payload) != allowed:
            raise CommonsShadowRecoveryError("pending projection has invalid fields")
        record = _PendingProjection(
            scope_id=cls._required_str(payload, "scope_id"),
            carrier_oid=cls._required_str(payload, "carrier_oid"),
            expected_head=cls._optional_str(payload, "expected_head"),
            effect_id=cls._required_str(payload, "effect_id"),
            commit_id=cls._required_str(payload, "commit_id"),
            workspace_tree=cls._required_str(payload, "workspace_tree"),
            parent_carrier_oid=cls._optional_str(payload, "parent_carrier_oid"),
            parent_commit_id=cls._optional_str(payload, "parent_commit_id"),
        )
        if cls._encode_pending(record) != raw:
            raise CommonsShadowRecoveryError("pending projection JSON is not canonical")
        return record

    @staticmethod
    def _required_str(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise CommonsShadowRecoveryError(f"pending projection field {key} must be a non-empty string")
        return value

    @staticmethod
    def _optional_str(payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise CommonsShadowRecoveryError(f"pending projection field {key} must be null or a non-empty string")
        return value
