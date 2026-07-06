# under-test: vcs_core._incremental
"""1b.2a — the open-operation-journal index (third DeltaIndex customer), standalone.

Tested without the journal store/manager: a bare repo, manual open-ref sets, a fake rebuild_source.
The store/manager atomic co-write wiring lands in 1b.2b; here we pin the index's read/rebuild/verify
+ the fail-closed entry validation, and that its prepared ref-moves apply via a --stdin transaction.
"""

from __future__ import annotations

import pygit2
import pytest
from vcs_core import InvalidRepositoryStateError
from vcs_core import _ref_txn as rt
from vcs_core._incremental import OPEN_OPERATION_JOURNAL_INDEX_SCHEMA, OpenOperationJournalIndex
from vcs_core._incremental import _git_record as gr
from vcs_core._world_refs import world_open_operation_journal_index_ref
from vcs_core.testing import operation_journal_ref


@pytest.fixture
def repo(tmp_path) -> pygit2.Repository:
    return pygit2.init_repository(str(tmp_path / "repo"), bare=True)


def _open_ref(operation_id: str) -> str:
    return operation_journal_ref("open", operation_id)


def _index(repo: pygit2.Repository, open_refs) -> OpenOperationJournalIndex:
    # rebuild_source returns the live set of open refs (a manager would scan ops/open/*)
    return OpenOperationJournalIndex(repo, "store", rebuild_source=lambda: set(open_refs))


def test_contract_is_exact_atomic():
    contract = OpenOperationJournalIndex.CONTRACT
    assert (contract.read_safety, contract.crash_lag) == ("exact", "atomic")


def test_missing_read_is_none(repo):
    assert _index(repo, []).read_open_refs() is None  # missing -> caller falls back


def test_rebuild_then_read_open_refs(repo):
    refs = {_open_ref("op1"), _open_ref("op2")}
    idx = _index(repo, refs)
    idx.rebuild_from_durable_history()
    assert idx.read_open_refs() == frozenset(refs)
    assert idx.verify_against_authority().ok


def test_prepare_add_and_remove_apply_via_stdin(repo):
    idx = _index(repo, set())
    idx.rebuild_from_durable_history()  # empty index present
    ref1 = _open_ref("op1")

    added = idx.prepare_add(ref1)
    assert rt.run_update_ref_stdin(repo, [added.ref_move()]).ok is True
    assert idx.read_open_refs() == {ref1}

    removed = idx.prepare_remove(ref1)
    assert rt.run_update_ref_stdin(repo, [removed.ref_move()]).ok is True
    assert idx.read_open_refs() == frozenset()


def test_prepare_refuses_non_open_ref(repo):
    # the WRITE boundary rejects a non-open ref, so a normal writer can't self-corrupt the index
    idx = _index(repo, set())
    idx.rebuild_from_durable_history()
    with pytest.raises(InvalidRepositoryStateError):
        idx.prepare_add(operation_journal_ref("closed", "op"))
    with pytest.raises(InvalidRepositoryStateError):
        idx.prepare_remove(operation_journal_ref("archived", "op"))


def test_forged_non_open_entry_is_corrupt(repo):
    # a digest-valid record whose entry key is NOT a v2-shaped open ref must fail closed (corrupt),
    # so a malformed accelerator can't smuggle an arbitrary string onto the admission probe.
    bad = gr.with_self_digest(
        {
            "schema": OPEN_OPERATION_JOURNAL_INDEX_SCHEMA,
            "generation": 1,
            "base_segment_ref": None,
            "entries": {operation_journal_ref("closed", "op-x"): {}},  # a CLOSED ref, not open
            "delta_added": {},
            "delta_removed": [],
        },
        digest_field="index_digest",
    )
    commit = gr.write_record(repo, meta_name="open-operation-journal-index.json", payload=bad, message="forged")
    repo.references.create(world_open_operation_journal_index_ref("store"), pygit2.Oid(hex=commit))
    idx = _index(repo, [])

    with pytest.raises(InvalidRepositoryStateError):
        idx.read_open_refs()
    assert idx.verify_against_authority().status == "corrupt"
