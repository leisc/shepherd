"""Total inventory probes for private v2 world operation journals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygit2

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._pygit2_helpers import require_commit
from vcs_core._query_inventory import (
    OPEN_OPERATION_JOURNAL_INDEX_CORRUPT,
    OPERATION_JOURNAL_CHAIN_INVALID,
    OPERATION_JOURNAL_IDENTITY_MISMATCH,
    OPERATION_JOURNAL_MISSING_REF,
    OPERATION_JOURNAL_PAYLOAD_CORRUPT,
    OPERATION_JOURNAL_REF_UNREADABLE,
    OPERATION_JOURNAL_SCHEMA_MISMATCH,
    OPERATION_JOURNAL_UNSUPPORTED_FAMILY,
    Disposition,
    Health,
    HealthIssue,
    HealthLifecycle,
    InventoryIssue,
    InventoryItem,
    issue_id,
    missing,
    present_invalid,
    present_valid,
)
from vcs_core._query_locators import LocatorComponent, classify_locator_component
from vcs_core._world_operation_journal import (
    OPERATION_JOURNAL_PATH,
    OPERATION_JOURNAL_REF_FAMILIES,
    OPERATION_JOURNAL_SCHEMA,
    OperationJournalStore,
)
from vcs_core._world_refs import (
    operation_journal_family_prefix,
    operation_journal_ref,
    world_open_operation_journal_index_ref,
)
from vcs_core._world_types import load_canonical_json

if TYPE_CHECKING:
    from vcs_core._world_storage_manager import WorldStorageManager

OPEN_OPERATION_JOURNAL_INDEX_KIND = "open_operation_journal_index"

_JOURNAL_REF_PREFIX = "refs/vcscore/ops"


def probe_operation_journals(repo: pygit2.Repository, *, family: str | None = None) -> tuple[InventoryItem, ...]:
    """Enumerate present v2 journal refs without dropping invalid records."""
    if family is None:
        prefix = f"{_JOURNAL_REF_PREFIX}/"
        return tuple(
            probe_operation_journal_ref(repo, ref)
            for ref in sorted(
                name for name in repo.references if name.startswith(prefix) and _is_v2_journal_ref_shape(name)
            )
        )
    families = (family,)
    items: list[InventoryItem] = []
    for current_family in families:
        prefix = operation_journal_family_prefix(current_family)
        # Apply the same v2-shape boundary as the family=None branch above, so a family filter
        # cannot make a malformed deeper ref (under the same encoded family prefix) newly visible.
        for ref in sorted(
            name for name in repo.references if name.startswith(prefix) and _is_v2_journal_ref_shape(name)
        ):
            items.append(probe_operation_journal_ref(repo, ref, expected_family=current_family))
    return tuple(items)


def probe_operation_journal(repo: pygit2.Repository, operation_id: str, *, family: str) -> InventoryItem:
    """Classify one expected operation journal ref, including targeted absence."""
    return probe_operation_journal_ref(
        repo,
        operation_journal_ref(family, operation_id),
        expected_family=family,
        expected_operation_id=operation_id,
    )


def admission_operation_journal_items(manager: WorldStorageManager) -> tuple[InventoryItem, ...]:
    """Bounded open-journal admission inventory: read the durable index, probe ONLY those refs.

    The admission tier's ``operation_journal`` source. With the index present this NEVER enumerates
    the operation-journal ref namespace — one blob read for the open-ref set, then O(open) direct
    ref probes. Three cases:

    * **present** → probe only the indexed open refs (bounded, no namespace scan);
    * **missing** → a cold-start/post-corrupt fallback scan. Admission is STRICTLY READ-ONLY (it
      runs on every runtime gate, including read-only execs), so it does NOT rebuild here — the
      index self-heals on the next co-write (open/close folds it from authority) or via recovery;
    * **corrupt** → a single fail-closed *blocking* fact — admission must not probe an unreadable
      accelerator nor silently fall back to a scan that would mask the corruption.
    """
    repo = manager.world_store.repo
    try:
        open_refs = manager.read_open_operation_journal_index()
    except InvalidRepositoryStateError as exc:
        store_id = manager.world_store.world_store_id
        return (
            _open_operation_journal_index_corrupt_item(
                store_id=store_id, ref=world_open_operation_journal_index_ref(store_id), detail=str(exc)
            ),
        )
    if open_refs is None:
        return probe_operation_journals(repo, family="open")  # cold-start / post-corrupt fallback scan (read-only)
    return tuple(probe_operation_journal_ref(repo, ref, expected_family="open") for ref in sorted(open_refs))


def _open_operation_journal_index_corrupt_item(*, store_id: str, ref: str, detail: str) -> InventoryItem:
    """The fail-closed BLOCKING admission fact for a corrupt open-journal accelerator.

    Declares ``disposition="blocking"`` (Part C), so ``_derive_blockers`` blocks ordinary mutating
    readiness on it, while targeted ``vcscore.recover`` can rebuild it (the issue code is in
    ``_RECOVERABLE_ISSUES`` and the item is a recovery target). Distinct from the lease index, whose
    corruption is non-blocking ``diagnostic`` because that index is a superset GC-protection set, not
    an exact mutation gate.
    """
    item_id = f"{OPEN_OPERATION_JOURNAL_INDEX_KIND}:{store_id}"
    issue = InventoryIssue(
        id=issue_id(item_id, OPEN_OPERATION_JOURNAL_INDEX_CORRUPT),
        code=OPEN_OPERATION_JOURNAL_INDEX_CORRUPT,
        message=f"open-operation-journal accelerator is corrupt: {detail}",
        subject_id=item_id,
        locator=ref,
        recovery_hint=(
            "Rebuild the open-operation-journal accelerator via targeted recovery; "
            "the authoritative open operation-journal refs are unaffected."
        ),
    )
    return InventoryItem(
        id=item_id,
        domain="operation_journal",
        kind=OPEN_OPERATION_JOURNAL_INDEX_KIND,
        locator=ref,
        source_kind="git_ref",
        source_store="coordinator",
        health=present_invalid(
            primary_issue="corrupt",
            issue_codes=(OPEN_OPERATION_JOURNAL_INDEX_CORRUPT,),
            lifecycle="recoverable",
            authority_role="projection",
            status="present_corrupt",
        ),
        role=("journal", "recovery", "blocker"),
        fields={"store_id": store_id},
        source_identity={"ref": ref},
        issues=(issue,),
        disposition="blocking",  # gates ordinary mutation; targeted vcscore.recover is exempted to rebuild
    )


def probe_operation_journal_ref(
    repo: pygit2.Repository,
    ref: str,
    *,
    expected_family: str | None = None,
    expected_operation_id: str | None = None,
) -> InventoryItem:
    """Classify one concrete operation journal ref as valid, absent, or present-invalid."""
    family_component, operation_component = _ref_components(ref)
    family_locator = classify_locator_component(family_component or "")
    operation_locator = classify_locator_component(operation_component or "")
    item_id = _item_id(family_component=family_component, operation_component=operation_component)
    base_fields = _base_fields(
        family_component=family_component,
        family_locator=family_locator,
        operation_component=operation_component,
        operation_locator=operation_locator,
    )
    family = family_locator.decoded_value if family_locator.reversible else None
    if family is not None:
        base_fields["family"] = family
    if operation_locator.reversible and operation_locator.decoded_value is not None:
        base_fields["locator_operation_id"] = operation_locator.decoded_value

    if ref not in repo.references:
        issue = _issue(
            item_id,
            OPERATION_JOURNAL_MISSING_REF,
            f"operation journal ref is missing: {ref}",
            locator=ref,
        )
        return _item(
            item_id=item_id,
            ref=ref,
            health=missing(
                issue_codes=(OPERATION_JOURNAL_MISSING_REF,),
                authority_role="authoritative",
            ),
            fields=base_fields,
            issues=(issue,),
        )

    source_identity: dict[str, object] = {"ref": ref}
    try:
        target_oid = str(repo.references[ref].target)
        source_identity["ref_target_oid"] = target_oid
        commit = require_commit(repo, pygit2.Oid(hex=target_oid), context=f"operation journal ref {ref}")
        source_identity["tip_tree_oid"] = str(commit.tree_id)
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="unreadable",
            code=OPERATION_JOURNAL_REF_UNREADABLE,
            message=str(exc),
            fields=base_fields,
            source_identity=source_identity,
        )

    if family not in OPERATION_JOURNAL_REF_FAMILIES:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="unsupported_schema",
            code=OPERATION_JOURNAL_UNSUPPORTED_FAMILY,
            message=f"operation journal ref family is unsupported: {family_component!r}",
            fields=base_fields,
            source_identity=source_identity,
        )
    if expected_family is not None and family != expected_family:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="identity_mismatch",
            code=OPERATION_JOURNAL_IDENTITY_MISMATCH,
            message=f"operation journal family {family!r} disagrees with expected family {expected_family!r}",
            fields=base_fields,
            source_identity=source_identity,
        )

    try:
        payload = load_canonical_json(_read_blob_bytes(repo, commit.tree, OPERATION_JOURNAL_PATH))
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="corrupt",
            code=OPERATION_JOURNAL_PAYLOAD_CORRUPT,
            message=str(exc),
            fields=base_fields,
            source_identity=source_identity,
        )
    if payload.get("schema") != OPERATION_JOURNAL_SCHEMA:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="schema_mismatch",
            code=OPERATION_JOURNAL_SCHEMA_MISMATCH,
            message=f"unsupported operation journal schema: {payload.get('schema')!r}",
            fields=_fields_with_payload(base_fields, payload),
            source_identity=source_identity,
        )

    payload_operation_id = payload.get("operation_id")
    fields = _fields_with_payload(base_fields, payload)
    if isinstance(payload_operation_id, str) and payload_operation_id:
        fields["payload_operation_id"] = payload_operation_id
    if expected_operation_id is not None:
        fields["expected_operation_id"] = expected_operation_id
    locator_operation_id = operation_locator.decoded_value if operation_locator.reversible else None
    if expected_operation_id is not None and payload_operation_id != expected_operation_id:
        fields["identity_match"] = False
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="identity_mismatch",
            code=OPERATION_JOURNAL_IDENTITY_MISMATCH,
            message="operation journal operation_id disagrees with expected operation id",
            fields=fields,
            source_identity=source_identity,
        )
    if locator_operation_id is not None and payload_operation_id != locator_operation_id:
        fields["identity_match"] = False
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="identity_mismatch",
            code=OPERATION_JOURNAL_IDENTITY_MISMATCH,
            message="operation journal operation_id disagrees with ref",
            fields=fields,
            source_identity=source_identity,
        )
    if locator_operation_id is not None:
        fields["identity_match"] = True

    history_operation_id = expected_operation_id or locator_operation_id or payload_operation_id
    if not isinstance(history_operation_id, str) or not history_operation_id:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="identity_mismatch",
            code=OPERATION_JOURNAL_IDENTITY_MISMATCH,
            message="operation journal operation_id is unavailable from locator and payload",
            fields=fields,
            source_identity=source_identity,
        )

    try:
        history = OperationJournalStore(repo).read_ref(
            ref,
            expected_family=family,
            expected_operation_id=history_operation_id,
            tip_oid=target_oid,
        )
    except (InvalidRepositoryStateError, KeyError, TypeError, ValueError, pygit2.GitError) as exc:
        return _invalid_item(
            item_id=item_id,
            ref=ref,
            primary_issue="corrupt",
            code=OPERATION_JOURNAL_CHAIN_INVALID,
            message=str(exc),
            fields=fields,
            source_identity=source_identity,
        )

    tip = history.tip.payload
    fields = _fields_with_payload(fields, tip)
    lifecycle: HealthLifecycle = "terminal" if family in {"closed", "archived"} else "active"
    # An open (lifecycle=="active") journal is the canonical BLOCKING admission fact (it gates a
    # mutation mid-operation). Declared here for Tier-2 observability; the block itself is still
    # driven by the legacy `_derive_blockers` open-journal rule (which handles it before the
    # disposition branch), so this is regression-free description, not new enforcement.
    disposition: Disposition | None = "blocking" if lifecycle == "active" else None
    return _item(
        item_id=item_id,
        ref=ref,
        health=present_valid(lifecycle=lifecycle, authority_role="authoritative"),
        fields=fields,
        source_identity=source_identity,
        disposition=disposition,
    )


def _ref_components(ref: str) -> tuple[str | None, str | None]:
    prefix = f"{_JOURNAL_REF_PREFIX}/"
    if not ref.startswith(prefix):
        return None, None
    remainder = ref.removeprefix(prefix)
    parts = remainder.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return parts[0] if parts else None, parts[1] if len(parts) > 1 else None
    return parts[0], parts[1]


def _is_v2_journal_ref_shape(ref: str) -> bool:
    prefix = f"{_JOURNAL_REF_PREFIX}/"
    remainder = ref.removeprefix(prefix)
    parts = remainder.split("/")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


def _base_fields(
    *,
    family_component: str | None,
    family_locator: LocatorComponent,
    operation_component: str | None,
    operation_locator: LocatorComponent,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    if family_component is not None:
        fields.update(family_locator.to_fields("locator_family"))
    if operation_component is not None:
        fields.update(operation_locator.to_fields("locator_operation"))
    return fields


def _fields_with_payload(fields: dict[str, object], payload: dict[str, Any]) -> dict[str, object]:
    merged = dict(fields)
    for key in (
        "operation_id",
        "operation_kind",
        "status",
        "seq",
        "target_ref",
        "input_world_oid",
        "parent_operation_id",
        "previous_journal_oid",
        "world_oid",
        "operation_final_digest",
        "error",
    ):
        value = payload.get(key)
        if isinstance(value, (str, int)) or value is None:
            merged[key] = value
    return merged


def _item_id(*, family_component: str | None, operation_component: str | None) -> str:
    family = family_component or "malformed"
    operation = operation_component or "malformed"
    return f"operation_journal:{family}:{operation}"


def _item(
    *,
    item_id: str,
    ref: str,
    health: Health,
    fields: dict[str, object],
    source_identity: dict[str, object] | None = None,
    issues: tuple[InventoryIssue, ...] = (),
    disposition: Disposition | None = None,
) -> InventoryItem:
    return InventoryItem(
        id=item_id,
        domain="operation_journal",
        kind="v2_world_operation_journal",
        locator=ref,
        source_kind="git_ref",
        source_store="coordinator",
        health=health,
        role=("journal", "recovery", "authority"),
        fields=fields,
        source_identity=dict(source_identity or {"ref": ref}),
        issues=issues,
        disposition=disposition,
    )


def _invalid_item(
    *,
    item_id: str,
    ref: str,
    primary_issue: HealthIssue,
    code: str,
    message: str,
    fields: dict[str, object],
    source_identity: dict[str, object],
) -> InventoryItem:
    issue = _issue(item_id, code, message, locator=ref)
    return _item(
        item_id=item_id,
        ref=ref,
        health=present_invalid(
            primary_issue=primary_issue,
            issue_codes=(code,),
            authority_role="authoritative",
        ),
        fields=fields,
        source_identity=source_identity,
        issues=(issue,),
    )


def _issue(subject_id: str, code: str, message: str, *, locator: str) -> InventoryIssue:
    return InventoryIssue(
        id=issue_id(subject_id, code),
        code=code,
        message=message,
        subject_id=subject_id,
        locator=locator,
        recovery_hint="Inspect or archive the operation journal before retrying.",
    )


def _read_blob_bytes(repo: pygit2.Repository, tree: pygit2.Tree, path: str) -> bytes:
    obj: pygit2.Object = tree
    for component in path.split("/"):
        if not isinstance(obj, pygit2.Tree):
            raise TypeError(f"{path!r} did not resolve to a blob")
        obj = repo[obj[component].id]
    if not isinstance(obj, pygit2.Blob):
        raise TypeError(f"{path!r} did not resolve to a blob")
    return bytes(obj.data)
