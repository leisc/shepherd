# under-test: vcs_core._incremental
"""Conformance suite for the vcs-core incremental-accelerator contract.

Runs the generic ``SingleSegmentDeltaIndex`` (the lease customer's engine) against
every gate from ``260621-1730-incremental-frontier-primitive.md`` rev2. A future
backend (retention, journal) reuses these gates by building over the same
``DeltaIndex`` / ``RebuildableAccelerator`` protocols.

Gates: missing->fallback, corrupt->fail-closed, durable-across-process,
read-bounded (no authority/namespace scan on the hot read), CAS + crash +
concurrency, rebuild bit-for-bit.
"""

from __future__ import annotations

import pygit2
import pytest
from vcs_core import InvalidRepositoryStateError, canonical_bytes
from vcs_core import _ref_txn as rt
from vcs_core._incremental import SingleSegmentDeltaIndex, atomic_co_write
from vcs_core._incremental import _delta_index as di
from vcs_core._incremental import _git_record as gr
from vcs_core._incremental._delta_index import TOMBSTONE

_REF = "refs/vcscore/test/incremental-idx"
_SCHEMA = "vcscore/test-index/v1"
_META = "test-index.json"


@pytest.fixture
def repo(tmp_path) -> pygit2.Repository:
    return pygit2.init_repository(str(tmp_path / "repo"), bare=True)


def _index(repo: pygit2.Repository, authority: dict) -> SingleSegmentDeltaIndex:
    return SingleSegmentDeltaIndex(
        repo,
        _REF,
        schema=_SCHEMA,
        meta_name=_META,
        message_prefix="test idx",
        rebuild_source=lambda: dict(authority),
    )


def _corrupt_record(repo: pygit2.Repository) -> None:
    bad = {
        "schema": _SCHEMA,
        "generation": 1,
        "base_segment_ref": None,
        "entries": {"z": 1},
        "delta_added": {},
        "delta_removed": [],
        "index_digest": "sha256:deadbeef",
    }
    meta = repo.TreeBuilder()
    meta.insert(_META, repo.create_blob(canonical_bytes(bad)), pygit2.GIT_FILEMODE_BLOB)
    root = repo.TreeBuilder()
    root.insert("meta", meta.write(), pygit2.GIT_FILEMODE_TREE)
    sig = pygit2.Signature("t", "t@e.invalid")
    repo.references[_REF].set_target(repo.create_commit(None, sig, sig, "bad", root.write(), []))


def _members(idx: SingleSegmentDeltaIndex) -> frozenset[str]:
    """Membership via the present-segment read surface; asserts the index is present.

    The engine has no missing->empty accessor by design, so reads go through read() and
    the caller handles None. These helpers encode "I expect a present segment here".
    """
    segment = idx.read()
    assert segment is not None
    return segment.members()


def _query(idx: SingleSegmentDeltaIndex, key: str) -> object | None:
    segment = idx.read()
    assert segment is not None
    return segment.query(key)


# --- basic semantics ---


def test_missing_read_is_none(repo):
    # Missing means *unknown*: read() returns None and the caller must fall back / rebuild.
    # There is deliberately no engine members()/query() that could collapse missing into "empty".
    idx = _index(repo, {})
    assert idx.read() is None


def test_extend_query_members(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    idx.extend({"b": 2})
    assert _members(idx) == {"a", "b"}
    assert _query(idx, "a") == 1


def test_idempotent_extend_does_not_churn_generation(repo):
    idx = _index(repo, {})
    first = idx.extend({"a": 1})
    again = idx.extend({"a": 1})
    assert again.generation == first.generation


def test_tombstone_retracts(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1, "b": 2})
    idx.extend({"a": TOMBSTONE})
    assert _members(idx) == {"b"}


# --- the contract gates ---


def test_corrupt_fails_closed(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    _corrupt_record(repo)
    with pytest.raises(InvalidRepositoryStateError):
        idx.read()
    assert idx.verify_against_authority().status == "corrupt"


def test_durable_across_process(tmp_path):
    path = str(tmp_path / "repo")
    repo_writer = pygit2.init_repository(path, bare=True)
    _index(repo_writer, {}).extend({"a": {"world_oid": "W"}})
    # a fresh repo handle on the same path stands in for a fresh process
    repo_reader = pygit2.Repository(path)
    assert _members(_index(repo_reader, {})) == {"a"}


def test_read_never_scans_authority(repo):
    """The hot read must answer from the record alone — never call the authority/scan."""

    def exploding_authority():
        raise AssertionError("hot read must not enumerate the authority")

    # establish a record via a benign sibling, then read via an exploding-authority index
    _index(repo, {}).extend({"a": 1})
    idx = SingleSegmentDeltaIndex(
        repo, _REF, schema=_SCHEMA, meta_name=_META, message_prefix="t", rebuild_source=exploding_authority
    )
    assert _members(idx) == {"a"}
    assert _query(idx, "a") == 1
    assert idx.read() is not None


def test_read_bounded_under_many_unrelated_refs(repo):
    idx = _index(repo, {})
    idx.extend({"a": {"world_oid": "W"}})
    sig = pygit2.Signature("t", "t@e.invalid")
    empty = repo.create_commit(None, sig, sig, "junk", repo.TreeBuilder().write(), [])
    for n in range(500):
        repo.references.create(f"refs/vcscore/junk/{n}", empty)
    assert _members(idx) == {"a"}  # correct regardless of namespace size


def test_rebuild_bit_for_bit_and_verify_fresh(repo):
    authority = {"x": {"world_oid": "WX"}}
    idx = _index(repo, authority)
    idx.rebuild_from_durable_history()
    assert _members(idx) == {"x"}
    assert idx.verify_against_authority().ok


def test_verify_detects_stale(repo):
    authority = {"x": 1, "y": 2}
    idx = _index(repo, authority)
    idx.rebuild_from_durable_history()
    authority.pop("y")
    assert idx.verify_against_authority().status == "stale"


def test_malformed_but_digest_valid_record_is_corrupt(repo):
    # a structurally malformed payload (no "entries") carrying a *matching* self-digest:
    # the digest check passes, so the shape check must fail it closed, not raise KeyError
    bad = gr.with_self_digest({"schema": _SCHEMA, "generation": 1}, digest_field="index_digest")
    commit = gr.write_record(repo, meta_name=_META, payload=bad, message="malformed")
    repo.references.create(_REF, pygit2.Oid(hex=commit))
    idx = _index(repo, {})
    with pytest.raises(InvalidRepositoryStateError):
        idx.read()
    assert idx.verify_against_authority().status == "corrupt"


def test_rebuild_overwrites_corrupt_prior(repo):
    authority = {"x": 1}
    idx = _index(repo, authority)
    idx.extend({"stale": 9})
    _corrupt_record(repo)
    idx.rebuild_from_durable_history()  # tolerant of the corrupt prior record
    assert _members(idx) == {"x"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_added", "not-a-map"),  # dict("not-a-map") would raise a raw ValueError
        ("delta_added", [["a", 1]]),
        ("delta_removed", 123),  # tuple(123) would raise a raw TypeError
        ("delta_removed", [1, 2]),  # non-str members
    ],
)
def test_malformed_delta_fields_are_corrupt(repo, field, value):
    # delta_added/delta_removed are provenance fields; a digest-valid record with a
    # malformed one must fail closed (corrupt), not raise a raw ValueError/TypeError.
    payload = {
        "schema": _SCHEMA,
        "generation": 1,
        "base_segment_ref": None,
        "entries": {"a": 1},
        "delta_added": {},
        "delta_removed": [],
    }
    payload[field] = value
    bad = gr.with_self_digest(payload, digest_field="index_digest")
    commit = gr.write_record(repo, meta_name=_META, payload=bad, message="malformed-delta")
    repo.references.create(_REF, pygit2.Oid(hex=commit))
    idx = _index(repo, {})
    with pytest.raises(InvalidRepositoryStateError):
        idx.read()
    assert idx.verify_against_authority().status == "corrupt"


# --- missing = rebuild from authority, never empty (derived-view invariant) ---


def test_extend_add_over_missing_seeds_from_authority(repo):
    # The authority already holds entries the index never recorded (the ref was reset,
    # or predates the data). A write must reconstruct from the authority first, not
    # materialize a delta-only SUBSET that the hot read would then trust.
    authority = {"x": {"world_oid": "WX"}, "y": {"world_oid": "WY"}}
    idx = _index(repo, authority)
    assert idx.read() is None  # index missing
    idx.extend({"z": {"world_oid": "WZ"}})
    assert _members(idx) == {"x", "y", "z"}  # authority + add, never just {"z"}


def test_extend_tombstone_over_missing_seeds_from_authority(repo):
    # A retraction over a missing index must preserve the rest of the authority,
    # not write an empty record (the round-2 repro).
    authority = {"x": {"world_oid": "WX"}, "y": {"world_oid": "WY"}}
    idx = _index(repo, authority)
    assert idx.read() is None
    idx.extend({"x": TOMBSTONE})
    assert _members(idx) == {"y"}  # authority minus removed, never empty


def test_cas_stale_expected_loses_and_leaves_ref(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    head = gr.current_ref_target(repo, _REF)
    payload = gr.with_self_digest(
        {
            "schema": _SCHEMA,
            "generation": 99,
            "base_segment_ref": None,
            "entries": {},
            "delta_added": {},
            "delta_removed": [],
        },
        digest_field="index_digest",
    )
    orphan = gr.write_record(repo, meta_name=_META, payload=payload, message="x")
    assert gr.cas_update_ref(repo, _REF, orphan, expected_oid="0" * 40) is False
    assert gr.current_ref_target(repo, _REF) == head  # ref unchanged on CAS loss


def test_crash_mid_advance_leaves_prior_generation_authoritative(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    head = gr.current_ref_target(repo, _REF)
    # write a new record commit but never CAS the ref — simulates a crash mid-extend
    payload = gr.with_self_digest(
        {
            "schema": _SCHEMA,
            "generation": 2,
            "base_segment_ref": head,
            "entries": {"a": 1, "b": 2},
            "delta_added": {"b": 2},
            "delta_removed": [],
        },
        digest_field="index_digest",
    )
    gr.write_record(repo, meta_name=_META, payload=payload, message="crash")
    assert _members(idx) == {"a"}  # the inert orphan is ignored; prior generation stands
    assert gr.current_ref_target(repo, _REF) == head


def test_concurrent_extend_folds_onto_a_real_competing_write(repo, monkeypatch):
    # On the first CAS, a *real* competing writer lands {a, c} (not just a fake CAS=False).
    # The retry must re-read that new base and fold our {b} on top -> {a, b, c}: the
    # competitor's write is preserved AND ours is added. Nothing lost in either direction.
    idx = _index(repo, {})
    idx.extend({"a": 1})
    head = gr.current_ref_target(repo, _REF)  # gen-0 {a}
    real_cas = di.cas_update_ref
    calls = {"n": 0}

    def flaky_cas(repo_, ref, new_oid, *, expected_oid):
        calls["n"] += 1
        if calls["n"] == 1:
            competing = gr.with_self_digest(
                {
                    "schema": _SCHEMA,
                    "generation": 1,
                    "base_segment_ref": head,
                    "entries": {"a": 1, "c": 3},
                    "delta_added": {"c": 3},
                    "delta_removed": [],
                },
                digest_field="index_digest",
            )
            competing_oid = gr.write_record(repo, meta_name=_META, payload=competing, message="competitor")
            assert real_cas(repo, _REF, competing_oid, expected_oid=head) is True  # competitor wins the race
            return False  # so our CAS loses: the ref now targets {a, c}, not `head`
        return real_cas(repo_, ref, new_oid, expected_oid=expected_oid)

    monkeypatch.setattr(di, "cas_update_ref", flaky_cas)
    idx.extend({"b": 2})
    assert _members(idx) == {"a", "b", "c"}  # folded onto the competitor's base; nothing lost
    assert calls["n"] >= 2  # the first CAS lost to the real competitor, the extend retried


# --- batchable / atomic co-write (prepare_extend; crash_lag="atomic") ---


def test_prepare_extend_then_stdin_apply_matches_extend(repo):
    """prepare_extend writes the record but does NOT move the ref; the customer commits it by
    folding ref_move() into its own `git update-ref --stdin`, after which the segment is live."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    prepared = idx.prepare_extend({"b": 2})

    assert prepared.idempotent_noop is False
    assert "b" not in idx.read().entries  # not yet committed — the ref hasn't moved
    assert rt.run_update_ref_stdin(repo, [prepared.ref_move()]).ok is True
    assert _members(idx) == {"a", "b"}  # now live
    assert dict(idx.read().entries) == dict(prepared.segment.entries)


def test_prepare_extend_idempotent_noop_has_no_ref_move(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    prepared = idx.prepare_extend({"a": 1})  # does not change the set
    assert prepared.idempotent_noop is True
    assert prepared.segment.members() == {"a"}  # the customer skips ref_move() for a no-op


def test_prepare_extend_over_missing_rebuilds_from_authority(repo):
    # the batched path preserves the missing=rebuild invariant: it seeds from authority, not {}
    authority = {"x": {"world_oid": "WX"}, "y": {"world_oid": "WY"}}
    idx = _index(repo, authority)
    assert idx.read() is None
    prepared = idx.prepare_extend({"z": {"world_oid": "WZ"}})
    assert prepared.segment.members() == {"x", "y", "z"}  # authority + add, never just {"z"}
    assert rt.run_update_ref_stdin(repo, [prepared.ref_move()]).ok is True  # create (expected=None)
    assert _members(idx) == {"x", "y", "z"}


def test_prepare_extend_stdin_cas_loss_then_reprepare(repo):
    """A concurrent advance between prepare and apply makes the batched CAS lose; the prepared write
    moves nothing, and re-preparing folds onto the new base (the seed of the co-write retry loop)."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    prepared = idx.prepare_extend({"b": 2})  # expected_oid = gen-0
    idx.extend({"c": 3})  # a competitor advances the ref to gen-1 {a, c}

    assert rt.run_update_ref_stdin(repo, [prepared.ref_move()]).ok is False  # stale expected -> rejected
    assert _members(idx) == {"a", "c"}  # our stale apply moved nothing

    reprepared = idx.prepare_extend({"b": 2})  # re-fold onto the new base
    assert rt.run_update_ref_stdin(repo, [reprepared.ref_move()]).ok is True
    assert _members(idx) == {"a", "b", "c"}  # competitor's c preserved AND our b added


def test_co_write_is_atomic_all_or_none(repo):
    """The load-bearing co-write property: batch the (valid) index ref-move with a sibling authority
    ref-move whose precondition is WRONG; git rejects the WHOLE transaction, so the index ref does
    NOT move despite being valid. This is what makes index+authority both-or-neither on a crash."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    index_head = gr.current_ref_target(repo, _REF)

    sig = pygit2.Signature("t", "t@e.invalid")
    sibling_oid = str(repo.create_commit(None, sig, sig, "auth", repo.TreeBuilder().write(), []))
    sibling_ref = "refs/vcscore/test/sibling-authority"
    repo.references.create(sibling_ref, pygit2.Oid(hex=sibling_oid))  # make the create-only precondition fail

    prepared = idx.prepare_extend({"b": 2})  # a VALID index advance
    committed = rt.run_update_ref_stdin(
        repo,
        [prepared.ref_move(), rt.RefMove(sibling_ref, sibling_oid, expected_oid=None)],
    )

    assert committed.ok is False  # the sibling create-only fails -> the whole batch is rejected
    assert _members(idx) == {"a"}  # the valid index ref-move did NOT land (atomic all-or-none)
    assert gr.current_ref_target(repo, _REF) == index_head


def test_ref_move_rejects_impossible_shapes():
    ref = "refs/vcscore/test/x"
    oid = "a" * 40
    assert rt.RefMove(ref, oid, None).command().startswith("create ")  # create: new, no expected
    assert rt.RefMove(ref, oid, "b" * 40).command().startswith("update ")  # update: new + expected
    assert rt.RefMove(ref, None, "b" * 40).command().startswith("delete ")  # delete: expected only
    with pytest.raises(InvalidRepositoryStateError):
        rt.RefMove(ref, None, None)  # neither create/update nor delete -> impossible
    with pytest.raises(InvalidRepositoryStateError):
        rt.RefMove("not a valid ref", oid, None)  # invalid ref name


# --- atomic_co_write orchestrator (advance accelerator + authority together) ---

_AUTH_REF = "refs/vcscore/test/authority"


def _new_authority_create_move(repo: pygit2.Repository) -> tuple[rt.RefMove, str]:
    """A throwaway authority ref-move (create-only at _AUTH_REF) to co-write alongside the index."""
    sig = pygit2.Signature("t", "t@e.invalid")
    oid = str(repo.create_commit(None, sig, sig, "auth", repo.TreeBuilder().write(), []))
    return rt.RefMove(_AUTH_REF, oid, expected_oid=None), oid


def test_atomic_co_write_commits_authority_and_index_together(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    auth_move, auth_oid = _new_authority_create_move(repo)

    segment = atomic_co_write(repo, authority_moves=[auth_move], prepare=lambda: idx.prepare_extend({"b": 2}))

    assert segment.members() == {"a", "b"}
    assert _members(idx) == {"a", "b"}  # index advanced
    assert gr.current_ref_target(repo, _AUTH_REF) == auth_oid  # authority created — together, atomically


def test_atomic_co_write_retries_on_index_contention(repo):
    """A competitor advances the index AFTER we prepared: the first batch loses the index CAS (so the
    authority create is rolled back too), and the retry re-folds onto the new base and commits."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    auth_move, auth_oid = _new_authority_create_move(repo)
    calls = {"n": 0}

    def prepare():
        calls["n"] += 1
        prepared = idx.prepare_extend({"b": 2})  # reads the current base
        if calls["n"] == 1:
            idx.extend({"c": 3})  # competitor advances the index after we prepared -> our batch will lose
        return prepared

    segment = atomic_co_write(repo, authority_moves=[auth_move], prepare=prepare)

    assert calls["n"] >= 2  # attempt 1 lost the index CAS and retried
    assert segment.members() == {"a", "b", "c"}  # re-folded onto the competitor's base; nothing lost
    assert _members(idx) == {"a", "b", "c"}
    assert gr.current_ref_target(repo, _AUTH_REF) == auth_oid  # authority committed once, on the winning attempt


def test_atomic_co_write_surfaces_authority_conflict_without_retry(repo):
    """An authority-ref precondition failure is a real conflict: surface immediately, never retry,
    and leave the index unmoved (atomic all-or-none)."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    index_head = gr.current_ref_target(repo, _REF)
    auth_move, auth_oid = _new_authority_create_move(repo)
    repo.references.create(_AUTH_REF, pygit2.Oid(hex=auth_oid))  # make the create-only precondition fail
    calls = {"n": 0}

    def prepare():
        calls["n"] += 1
        return idx.prepare_extend({"b": 2})

    with pytest.raises(InvalidRepositoryStateError, match="authority ref precondition"):
        atomic_co_write(repo, authority_moves=[auth_move], prepare=prepare)

    assert calls["n"] == 1  # surfaced immediately, NOT retried
    assert gr.current_ref_target(repo, _REF) == index_head  # index NOT advanced (atomic)


def test_atomic_co_write_surfaces_authority_conflict_even_when_index_also_moved(repo):
    """Stricter contract: if BOTH an authority precondition AND the accelerator ref fail in the same
    rejected batch, the authority conflict surfaces immediately — never a wasted retry."""
    idx = _index(repo, {})
    idx.extend({"a": 1})
    auth_move, auth_oid = _new_authority_create_move(repo)
    repo.references.create(_AUTH_REF, pygit2.Oid(hex=auth_oid))  # the authority create-only will fail
    calls = {"n": 0}

    def prepare():
        calls["n"] += 1
        prepared = idx.prepare_extend({"b": 2})
        idx.extend({f"competitor{calls['n']}": calls["n"]})  # ALSO move the index on every attempt
        return prepared

    with pytest.raises(InvalidRepositoryStateError, match="authority ref precondition"):
        atomic_co_write(repo, authority_moves=[auth_move], prepare=prepare)
    assert calls["n"] == 1  # surfaced immediately despite the index also moving; NOT retried


def test_atomic_co_write_idempotent_noop_still_commits_authority(repo):
    idx = _index(repo, {})
    idx.extend({"a": 1})
    index_head = gr.current_ref_target(repo, _REF)
    auth_move, auth_oid = _new_authority_create_move(repo)

    segment = atomic_co_write(
        repo,
        authority_moves=[auth_move],
        prepare=lambda: idx.prepare_extend({"a": 1}),  # no-op (a present)
    )

    assert segment.members() == {"a"}
    assert gr.current_ref_target(repo, _REF) == index_head  # no index ref-move batched
    assert gr.current_ref_target(repo, _AUTH_REF) == auth_oid  # authority still committed


def test_derived_view_contract_rejects_incoherent_pairings():
    from vcs_core._incremental import DerivedViewContract

    # the two pairings we use construct fine
    assert DerivedViewContract(read_safety="exact", crash_lag="atomic").crash_lag == "atomic"
    assert DerivedViewContract(read_safety="superset", crash_lag="index-leads").read_safety == "superset"
    # incoherent pairings fail closed -> the policy can't drift back to prose-only
    with pytest.raises(ValueError, match="incoherent DerivedViewContract"):
        DerivedViewContract(read_safety="exact", crash_lag="index-leads")
    with pytest.raises(ValueError, match="incoherent DerivedViewContract"):
        DerivedViewContract(read_safety="superset", crash_lag="atomic")


def test_vcs_core_core_has_no_shepherd2_import():
    """The accelerator library reuses the TraceStore vocabulary, not a dependency."""
    import pathlib
    import re

    import vcs_core

    root = pathlib.Path(vcs_core.__file__).parent
    pattern = re.compile(r"^\s*(from|import)\s+shepherd2(\b|\.)", re.MULTILINE)
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py") if pattern.search(p.read_text())]
    assert not offenders, f"vcs-core core must not import shepherd2: {offenders}"
