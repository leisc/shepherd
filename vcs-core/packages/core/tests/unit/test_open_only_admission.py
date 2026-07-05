"""Step 2 — bounded (open-only) readiness admission.

Admission scopes its operation-journal scan to the `open` family: a healthy non-`open`
terminal journal can never block readiness (the blocker rule fires on lifecycle=="active"
or tip status in {failed,recovery_required}, and a terminal-family ref structurally cannot
carry those — `_world_operation_journal.py:395-399`), so terminals are dropped from the hot
path. Their integrity moves to the explicit, off-hot-path `fsck_operation_journals()`.

These tests pin: (1) the scoped probe never blob-reads terminal refs; (2) the v2-shape
boundary is preserved under the family filter; (3) open-journal blocking is unchanged;
(4) terminals are absent from the admission/recovery hot surfaces but visible to fsck; and
(5) `fsck_operation_journals()` preserves invalid/unknown-family refs and runs targeted fsck
on valid ones. See `260622-step2-open-only-admission.md`.
"""

from __future__ import annotations

import pygit2
from vcs_core import _operation_journal_inventory as oji
from vcs_core._operation_journal_inventory import probe_operation_journals
from vcs_core._query_inventory import OPERATION_JOURNAL_PAYLOAD_CORRUPT, OPERATION_JOURNAL_UNSUPPORTED_FAMILY
from vcs_core._query_readiness import _admission_operation_journal_items
from vcs_core._world_operation_journal import OPERATION_JOURNAL_PATH
from vcs_core._world_refs import (
    encode_ref_component,
    operation_journal_family_prefix,
    operation_journal_ref,
    world_open_operation_journal_index_ref,
)
from vcs_core._world_storage_installation import open_or_init_default_world_storage
from vcs_core._world_storage_manager import DEFAULT_GROUND_REF, SubstrateStoreSpec, WorldStorageManager
from vcs_core._world_types import SubstrateStoreIdentity
from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry
from vcs_core.vcscore import VcsCore


def _workspace_identity() -> SubstrateStoreIdentity:
    return SubstrateStoreIdentity(store_id="store_workspace", kind="filesystem", resource_id="fs:repo-main")


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="world-store",
        stores=(SubstrateStoreSpec(identity=_workspace_identity(), locator="workspace-store.git"),),
    )


def _open_valid_journal(manager: WorldStorageManager, operation_id: str) -> None:
    manager.open_operation_journal(
        operation_id=operation_id,
        operation_kind="shepherd.task",
        target_ref=DEFAULT_GROUND_REF,
        input_world_oid=None,
    )


def _unknown_family_ref(operation_id: str) -> str:
    return f"refs/vcscore/ops/{encode_ref_component('retrying')}/{encode_ref_component(operation_id)}"


def _write_manual_journal_commit(manager: WorldStorageManager, *, payload_bytes: bytes, ref: str) -> str:
    repo = manager.world_store.repo
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        OPERATION_JOURNAL_PATH.split("/")[-1],
        repo.create_blob(payload_bytes),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root_builder = repo.TreeBuilder()
    insert_tree_entry(repo, root_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("vcs-core operation journal", "vcs-core@example.invalid")
    oid = create_commit_with_recovery(repo, None, signature, signature, "manual journal", root_builder.write(), [])
    repo.references.create(ref, oid, force=True)
    return str(oid)


# --- Part A: the scoped probe (the perf fix) ---


def test_open_scoped_probe_does_not_blob_read_terminal_journals(tmp_path, monkeypatch):
    """Analog of the lease 'hot read does not scan the namespace': the admission probe must
    blob-read only `open` refs, even with many accumulated terminals (the per-journal read term
    is O(open), not O(total))."""
    manager = _manager(tmp_path)
    _open_valid_journal(manager, "op-open")
    for n in range(200):  # accumulated terminals admission must not read
        _write_manual_journal_commit(
            manager, payload_bytes=b"terminal", ref=operation_journal_ref("closed", f"op-closed-{n}")
        )

    probed: list[str] = []
    real = oji.probe_operation_journal_ref

    def spy(repo, ref, **kwargs):
        probed.append(ref)
        return real(repo, ref, **kwargs)

    monkeypatch.setattr(oji, "probe_operation_journal_ref", spy)
    probe_operation_journals(manager.world_store.repo, family="open")

    # Journal refs are base64url-encoded (closed -> b64u_Y2xvc2Vk), so a substring like "/closed/"
    # never appears — assert against the *encoded* open-family prefix instead, so a real terminal
    # probe (under the encoded closed/archived prefix) would be caught.
    open_prefix = f"refs/vcscore/ops/{encode_ref_component('open')}/"
    assert operation_journal_ref("open", "op-open") in probed  # the open ref IS read
    assert all(ref.startswith(open_prefix) for ref in probed)  # only open refs blob-read; no terminal probed


def test_open_scoped_probe_preserves_v2_shape_boundary(tmp_path):
    """The family filter keeps the same v2-shape boundary as the family=None path: terminals,
    unknown families, and malformed deeper open-prefixed refs are not admission-visible."""
    manager = _manager(tmp_path)
    _open_valid_journal(manager, "op-open")
    # A malformed deeper ref UNDER the (encoded) open-family prefix: it matches the family prefix,
    # so only the v2-shape filter keeps it out of admission (the A2 "freebie").
    malformed_deep = f"refs/vcscore/ops/{encode_ref_component('open')}/foo/bar"
    _write_manual_journal_commit(manager, payload_bytes=b"x", ref=malformed_deep)
    _write_manual_journal_commit(manager, payload_bytes=b"x", ref=operation_journal_ref("closed", "op-closed"))
    _write_manual_journal_commit(manager, payload_bytes=b"x", ref=operation_journal_ref("archived", "op-arch"))
    _write_manual_journal_commit(manager, payload_bytes=b"x", ref=_unknown_family_ref("op-retry"))

    locators = {item.locator for item in probe_operation_journals(manager.world_store.repo, family="open")}

    assert locators == {operation_journal_ref("open", "op-open")}
    assert malformed_deep not in locators  # the family branch did not drop the v2-shape filter


def test_corrupt_open_journal_still_visible_to_admission(tmp_path):
    """Open-only must not weaken open-journal blocking: a corrupt *open* ref is still in the
    admission inventory (it blocks via the validity=="invalid" fallback, unchanged)."""
    manager = _manager(tmp_path)
    _write_manual_journal_commit(manager, payload_bytes=b"not json", ref=operation_journal_ref("open", "op-broken"))

    items = probe_operation_journals(manager.world_store.repo, family="open")

    assert [item.locator for item in items] == [operation_journal_ref("open", "op-broken")]
    assert items[0].health.validity == "invalid"


# --- Part B: terminal integrity is explicit, off-hot-path (diagnostic-only) ---


def test_terminal_absent_from_admission_but_present_in_all_families_scan(tmp_path):
    """Behavioral-scope change: a terminal ref is *absent* from the open-scoped admission
    inventory, yet still discoverable via the all-families scan that inspect/fsck use."""
    manager = _manager(tmp_path)
    closed_ref = operation_journal_ref("closed", "op-closed")
    _write_manual_journal_commit(manager, payload_bytes=b"terminal", ref=closed_ref)

    admission = {item.locator for item in probe_operation_journals(manager.world_store.repo, family="open")}
    all_families = {item.locator for item in probe_operation_journals(manager.world_store.repo)}

    assert closed_ref not in admission
    assert closed_ref in all_families  # the inspect/fsck source still sees it


def test_fsck_operation_journals_preserves_invalid_and_unknown_family_refs(tmp_path):
    """The store-level fsck surface reports corrupt terminals and unknown-family refs (the cases
    moved off admission), and it must PRESERVE them — it walks the inventory probe, not list(),
    which would silently skip unparseable refs. Admission never sees any of them."""
    manager = _manager(tmp_path)
    corrupt_closed = operation_journal_ref("closed", "op-corrupt")
    unknown_family = _unknown_family_ref("op-unknown")
    _write_manual_journal_commit(manager, payload_bytes=b"not json", ref=corrupt_closed)
    _write_manual_journal_commit(manager, payload_bytes=b"x", ref=unknown_family)

    report = manager.fsck_operation_journals()

    assert not report.ok
    reported_refs = {issue.ref for issue in report.issue_details}
    assert corrupt_closed in reported_refs  # corrupt terminal preserved, not silently dropped
    assert unknown_family in reported_refs  # unknown family preserved
    reported_codes = {issue.code for issue in report.issue_details}
    assert {OPERATION_JOURNAL_PAYLOAD_CORRUPT, OPERATION_JOURNAL_UNSUPPORTED_FAMILY} <= reported_codes
    # invalid refs carry the diagnostic-only hint (overridden, not an inherited "recover this" hint)
    assert all("not auto-recoverable" in (issue.recovery_hint or "") for issue in report.issue_details)
    # ...and admission (open-scoped) sees none of them
    assert probe_operation_journals(manager.world_store.repo, family="open") == ()


def test_fsck_operation_journals_runs_targeted_fsck_on_valid_journals(tmp_path, monkeypatch):
    """A valid known-family journal is routed through the deeper targeted fsck (and a healthy
    one yields a clean report)."""
    manager = _manager(tmp_path)
    _open_valid_journal(manager, "op-valid")

    called: list[tuple[str, str]] = []
    real = manager.fsck_operation_journal

    def spy(operation_id, *, family):
        called.append((operation_id, family))
        return real(operation_id, family=family)

    monkeypatch.setattr(manager, "fsck_operation_journal", spy)
    report = manager.fsck_operation_journals()

    assert ("op-valid", "open") in called  # valid -> deeper targeted fsck
    assert report.scanned == 1
    assert report.ok


def test_fsck_operation_journals_surfaces_open_index_stale_drift(tmp_path):
    """The store-global journal fsck detects out-of-model open-journal index drift: an open ref a
    manual/private-ref writer created bypassing the co-write, which the index misses (the stale
    under-report direction). Atomic co-write precludes phantoms, so this stale case is the residual
    hazard fsck must catch, with the rebuild recovery hint."""
    manager = _manager(tmp_path)
    _open_valid_journal(manager, "op-valid")  # co-written into the index (verify fresh so far)
    valid_tip = manager.read_operation_journal("op-valid", family="open").tip.oid
    manager.world_store.repo.references.create(
        operation_journal_ref("open", "op-out-of-model"), pygit2.Oid(hex=valid_tip)
    )

    report = manager.fsck_operation_journals()

    assert not report.ok
    index_issues = [issue for issue in report.issue_details if issue.code == "open_operation_journal_index_stale"]
    assert len(index_issues) == 1
    assert index_issues[0].ref == world_open_operation_journal_index_ref(manager.world_store.world_store_id)
    assert "rebuild_open_operation_journal_index" in (index_issues[0].recovery_hint or "")


# --- hot-surface guards (the real readiness / recovery surfaces) ---


def test_admission_journal_inventory_is_open_scoped(mg: VcsCore):
    """The readiness/admission journal inventory source is open-scoped end-to-end: a terminal ref
    is absent while the open ref is present."""
    manager = open_or_init_default_world_storage(mg._repo_path)
    _open_valid_journal(manager, "op-open")
    _write_manual_journal_commit(manager, payload_bytes=b"terminal", ref=operation_journal_ref("closed", "op-closed"))

    locators = {item.locator for item in _admission_operation_journal_items(mg._repo_path)}

    assert operation_journal_ref("open", "op-open") in locators
    assert operation_journal_ref("closed", "op-closed") not in locators


def test_recovery_inventory_does_not_enumerate_terminal_journals(mg: VcsCore, monkeypatch):
    """The recovery inventory is the *other* readiness hot surface (reachable under shepherd.run's
    _ALL_DOMAINS). It must never enumerate terminal journals — locking in that an all-families
    journal scan is not reintroduced there (which would re-add the O(history) cost Part A removes)."""
    manager = open_or_init_default_world_storage(mg._repo_path)
    _write_manual_journal_commit(manager, payload_bytes=b"terminal", ref=operation_journal_ref("closed", "op-closed"))
    _write_manual_journal_commit(manager, payload_bytes=b"terminal", ref=operation_journal_ref("archived", "op-arch"))

    # Encoding-independent guard: any all-families journal probe on the recovery hot path explodes.
    def _boom(*_args, **_kwargs):
        raise AssertionError("recovery inventory must not enumerate operation journals")

    monkeypatch.setattr(oji, "probe_operation_journals", _boom)
    locators = {item.locator or "" for item in mg.recovery_inventory().items}

    # Defense-in-depth, by ref: no terminal journal appears — using the ENCODED family prefixes
    # (a raw, unencoded family prefix never matches the base64url-encoded refs; the vacuous-guard trap).
    terminal_prefixes = (operation_journal_family_prefix("closed"), operation_journal_family_prefix("archived"))
    assert not any(loc.startswith(terminal_prefixes) for loc in locators)


def test_no_raw_operation_journal_family_ref_literals():
    """Operation-journal family refs are base64url-encoded, so a raw ``ops/<family>/`` literal never
    matches a real ref — an always-a-bug footgun that has recurred four times (two test guards, the
    shape test, the enumeration spike). The ref builders (``operation_journal_ref`` /
    ``operation_journal_family_prefix``) are the only construction path; this fails if a raw family
    ref literal reappears anywhere in product or test code.
    """
    import pathlib

    import vcs_core

    core_root = pathlib.Path(vcs_core.__file__).parents[2]  # .../packages/core
    # built from parts so this guard does not flag its own source
    raw_family_refs = tuple(f"refs/vcscore/ops/{family}/" for family in ("open", "closed", "archived"))
    offenders = []
    for base in (core_root / "src", core_root / "tests"):
        for path in base.rglob("*.py"):
            if path.name == "_world_refs.py":  # the builder home — the one legitimate place
                continue
            if any(token in path.read_text() for token in raw_family_refs):
                offenders.append(str(path.relative_to(core_root)))
    assert not offenders, (
        "raw operation-journal family ref literal(s) found — construct via operation_journal_ref(...) "
        f"/ operation_journal_family_prefix(...) instead: {offenders}"
    )
