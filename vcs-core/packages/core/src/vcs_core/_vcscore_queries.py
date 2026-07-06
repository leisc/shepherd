"""QueryService — read-only query/inspection operations for one VcsCore session.

P3 (V3.2): the former ``owner: VcsCore`` free-function module is now a collaborator
constructed in ``VcsCore.__init__``. It takes a :class:`SessionState` plus a few
injected callables (``ground``, ``scope_world_id``, ``recovery_inventory``) for the
non-stable / sibling-function dependencies, and imports nothing from ``vcscore`` —
so the ``_vcscore_queries <-> vcscore`` cycle dissolves. Pure helpers stay
module-level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vcs_core._errors import StaleScopeError
from vcs_core._recovery_inventory import (
    recovery_orphaned_operation_items,
    recovery_orphaned_scope_refs,
)
from vcs_core._workspace_authority_inventory import probe_workspace_authority_pending, workspace_authority_pending_label
from vcs_core.store import GROUND_REF
from vcs_core.types import (
    CommitInfo,
    DiffSummary,
    OperationHistory,
    OperationSummary,
    OperationVisibility,
    RecoverySnapshot,
    ScopeInfo,
    Status,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vcs_core._query_inventory import InventoryItem, InventorySnapshot
    from vcs_core._session_state import SessionState


class QueryService:
    """Read-only queries over durable/session state for one VcsCore session.

    Injected dependencies (no ``VcsCore`` back-reference):
      * ``state`` — the session's :class:`SessionState`;
      * ``ground`` — accessor for the (lazily-set) ground scope;
      * ``scope_world_id`` — owner's scope→world-id resolver;
      * ``recovery_inventory`` — owner-bound recovery inventory snapshot.
    """

    def __init__(
        self,
        *,
        state: SessionState,
        ground: Callable[[], ScopeInfo | None],
        scope_world_id: Callable[[ScopeInfo], str],
        recovery_inventory: Callable[[], InventorySnapshot],
    ) -> None:
        self._state = state
        self._ground = ground
        self._scope_world_id = scope_world_id
        self._recovery_inventory = recovery_inventory

    def status(self) -> Status:
        return self._state.store.status()

    def diff(self) -> DiffSummary:
        return self._state.store.diff()

    def log(self, *, ref: str | None = None, max_count: int = 50) -> list[CommitInfo]:
        return self._state.store.log(ref=ref, max_count=max_count)

    def filter_effects(
        self,
        *,
        effect_type: str | None = None,
        substrate: str | None = None,
        ref: str | None = None,
        max_count: int = 100,
        scope: str | None = None,
    ) -> list[CommitInfo]:
        return self._state.store.filter_effects(
            effect_type=effect_type,
            substrate=substrate,
            ref=ref,
            max_count=max_count,
            scope=scope,
        )

    def visible_operations(self, *, ref: str | None = None, max_count: int = 50) -> list[OperationSummary]:
        return self._state.store.visible_operations(ref=ref, max_count=max_count)

    def open_operations(
        self,
        *,
        scope: ScopeInfo | None = None,
        session_id: str | None = None,
    ) -> list[OperationSummary]:
        scope_ref = scope.ref if scope is not None else None
        return self._state.store.open_operations(scope_ref=scope_ref, session_id=session_id)

    def archived_operations(
        self,
        *,
        max_count: int = 50,
        world_id: str | None = None,
        operation_id: str | None = None,
    ) -> list[OperationSummary]:
        return self._state.store.archived_operations(
            max_count=max_count,
            world_id=world_id,
            operation_id=operation_id,
        )

    def operation_history(self, ref: str) -> OperationHistory:
        return self._state.store.read_operation_history(ref)

    def recovery_snapshot(self, *, archived_max_count: int = 50) -> RecoverySnapshot:
        inventory = self._recovery_inventory()
        return RecoverySnapshot(
            orphaned_scope_refs=recovery_orphaned_scope_refs(inventory),
            open_operations=tuple(self.open_operations()),
            archived_recovery_operations=tuple(
                self._state.store.archived_recovery_operations(max_count=archived_max_count)
            ),
            orphaned_operations=self._hydrate_orphaned_operation_summaries(
                recovery_orphaned_operation_items(inventory),
            ),
            workspace_authority_pending=tuple(
                workspace_authority_pending_label(item)
                for item in probe_workspace_authority_pending(self._state.repo_path)
            ),
        )

    def orphaned_operations(self) -> tuple[OperationSummary, ...]:
        """Return orphaned operations selected by recovery inventory."""
        inventory = self._recovery_inventory()
        return self._hydrate_orphaned_operation_summaries(recovery_orphaned_operation_items(inventory))

    def _hydrate_orphaned_operation_summaries(
        self,
        items: tuple[InventoryItem, ...],
    ) -> tuple[OperationSummary, ...]:
        return tuple(self._hydrate_orphaned_operation_summary(item) for item in items)

    def _hydrate_orphaned_operation_summary(self, item: InventoryItem) -> OperationSummary:
        if item.locator is not None and self._state.store.ref_exists(item.locator):
            return self._state.store.read_operation_history(item.locator).summary
        operation_id = _field_str(item, "operation_id") or _field_str(item, "operation_label") or item.id
        scope_ref = _field_str(item, "scope_ref") or "refs/vcscore/ground"
        return OperationSummary(
            operation_id=operation_id,
            label=_field_str(item, "operation_label"),
            kind=_field_str(item, "operation_kind") or "unknown",
            status=_field_str(item, "status") or "open",
            visibility=cast("OperationVisibility", _field_str(item, "visibility") or "staged"),
            world_id=_field_str(item, "world_id") or self._orphaned_operation_world_id(scope_ref),
            world_name=_field_str(item, "world_name") or _scope_name_for_ref(scope_ref),
            world_ref=scope_ref,
            carrier_ref=item.locator or item.id,
            parent_operation_id=_field_str(item, "parent_operation_id"),
        )

    def _orphaned_operation_world_id(self, scope_ref: str) -> str:
        ground = self._ground()
        if ground is not None and scope_ref == ground.ref:
            return self._scope_world_id(ground)
        for scope in self._state.active_scopes.values():
            if scope.ref == scope_ref:
                return self._scope_world_id(scope)
        return "unknown"

    def resolve_operation_history(
        self,
        selector: str,
        *,
        scope: ScopeInfo | None = None,
        max_count: int = 200,
    ) -> OperationHistory:
        if selector.startswith(("refs/vcscore/ops/", "refs/vcscore/archive/ops/")):
            return self.operation_history(selector)

        direct_matches = self.operation_direct_matches(selector, scope=scope)
        identity_matches = self.operation_id_matches(selector, scope=scope, max_count=max_count)

        if len(direct_matches) == 1:
            return self.read_operation_summary_history(direct_matches[0])
        if len(direct_matches) > 1:
            labels = ", ".join(sorted(describe_operation_selector_match(item) for item in direct_matches)[:5])
            msg = f"Ambiguous operation selector {selector!r}. Matches: {labels}"
            raise ValueError(msg)

        if len(identity_matches) == 1:
            return self.read_operation_summary_history(next(iter(identity_matches.values())))
        if len(identity_matches) > 1:
            labels = ", ".join(
                sorted(describe_operation_selector_match(item) for item in identity_matches.values())[:5]
            )
            msg = f"Ambiguous operation selector {selector!r}. Matches: {labels}"
            raise ValueError(msg)
        msg = f"No operation matches {selector!r}."
        raise ValueError(msg)

    def operation_direct_matches(
        self,
        selector: str,
        *,
        scope: ScopeInfo | None,
    ) -> list[OperationSummary]:
        try:
            summaries = self._state.store.committed_carrier_operations(selector, max_count=1_000_000)
        except (StaleScopeError, ValueError):
            return []
        if scope is None:
            return summaries
        world_id = self._scope_world_id(scope)
        return [summary for summary in summaries if summary.world_id == world_id]

    def operation_id_matches(
        self,
        selector: str,
        *,
        scope: ScopeInfo | None,
        max_count: int = 200,
    ) -> dict[str, OperationSummary]:
        matches: dict[str, OperationSummary] = {}
        visible_refs = (
            [scope.ref] if scope is not None else ["refs/vcscore/ground", *self._state.store.list_scope_refs()]
        )
        for ref in visible_refs:
            try:
                summary = self._state.store.read_visible_operation_history(ref, operation_id=selector).summary
            except StaleScopeError:
                continue
            matches[summary.carrier_ref] = summary

        operations = self.open_operations(scope=scope) if scope is not None else self.open_operations()
        for summary in operations:
            if summary.operation_id == selector:
                matches[summary.carrier_ref] = summary

        world_id = self._scope_world_id(scope) if scope is not None else None
        for summary in self._state.store.archived_operations(
            max_count=max(max_count, 1_000_000),
            world_id=world_id,
            operation_id=selector,
        ):
            matches[summary.carrier_ref] = summary
        return matches

    def read_operation_summary_history(self, summary: OperationSummary) -> OperationHistory:
        if summary.visibility == "visible":
            return self._state.store.read_visible_operation_history(
                summary.carrier_ref,
                operation_id=summary.operation_id,
            )
        if summary.archived_via == "discarded_world_ref":
            return self._state.store.read_discarded_world_operation_history(
                summary.carrier_ref,
                operation_id=summary.operation_id,
            )
        return self.operation_history(summary.carrier_ref)


def _field_str(item: InventoryItem, key: str) -> str | None:
    value = item.fields.get(key)
    return value if isinstance(value, str) and value else None


def _scope_name_for_ref(ref: str) -> str:
    if ref == GROUND_REF:
        return "ground"
    return ref.rsplit("/", 1)[-1]


def describe_operation_selector_match(summary: OperationSummary) -> str:
    label = summary.label or summary.operation_id
    carrier_ref = summary.carrier_ref
    return (
        f"{summary.operation_id} ({label}) [{summary.visibility}/{summary.status}] "
        f"world:{summary.world_name} carrier:{carrier_ref.rsplit('/', 1)[-1]}"
    )
