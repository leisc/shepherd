# under-test: vcs_core._transition_kernel_records
"""Unit tests for private transition-kernel canonical records."""

from __future__ import annotations

import pytest
from vcs_core import canonical_digest
from vcs_core._transition_kernel_records import (
    CandidateCommitRecord,
    CandidateOutcomeRecord,
    EvidenceRecord,
    EvidenceRef,
    HeadSelectionEvidence,
    HeadSelectionRecord,
    LogicalTransition,
    PayloadDescriptorClaim,
    PreparedRevisionPlan,
    PreparedTransitionBundle,
    RetentionPolicyRequirement,
    RevisionPreparationRecord,
    ValidatedPayloadDescriptor,
    validate_head_selection,
)


def _sha(label: str) -> str:
    return canonical_digest({"label": label})


def _evidence_ref(operation_id: str, ref: str) -> EvidenceRef:
    record = EvidenceRecord(
        operation_id=operation_id,
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
        ingress_kind="scan",
        evidence_kind="diff_scan",
        payload_digest=_sha("payload"),
        stable_observation={"path": "artifact.txt", "sha256": _sha("payload")},
        observed_at_unix_ns=123,
        mechanism="cli_scan",
    )
    return EvidenceRef(
        ref=ref,
        evidence_digest=record.evidence_digest(),
        record_digest=record.record_digest(),
        payload_digest=record.payload_digest,
    )


def _transition(evidence_ref: EvidenceRef, *, ingress_kind: str = "scan") -> LogicalTransition:
    return LogicalTransition(
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        substrate_kind="filesystem",
        driver="builtin.filesystem",
        driver_version="test",
        base_heads=("1" * 40,),
        ingress_kind=ingress_kind,
        semantic_op="FilePatch",
        payload_digest=_sha("workspace payload"),
        evidence_digests=(evidence_ref.evidence_digest,),
    )


def _plan(transition: LogicalTransition) -> PreparedRevisionPlan:
    return PreparedRevisionPlan(
        binding=transition.binding,
        store_id=transition.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=transition.base_heads,
        content_digest=_sha("workspace content"),
        materialization_class="external",
        entries=({"path": "artifact.txt", "payload_digest": transition.payload_digest},),
    )


def _preparation(
    *,
    operation_id: str,
    transition: LogicalTransition,
    plan: PreparedRevisionPlan,
    evidence_ref: EvidenceRef,
    cited_evidence_refs: tuple[EvidenceRef, ...] = (),
) -> RevisionPreparationRecord:
    return RevisionPreparationRecord(
        operation_id=operation_id,
        binding=transition.binding,
        store_id=transition.store_id,
        resource_id=transition.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence_ref,),
        cited_evidence_refs=cited_evidence_refs,
    )


def _legacy_v1_preparation_json(preparation: RevisionPreparationRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "vcscore/revision-preparation/v1",
        "operation_id": preparation.operation_id,
        "binding": preparation.binding,
        "store_id": preparation.store_id,
        "resource_id": preparation.resource_id,
        "transition_digest": preparation.transition_digest,
        "revision_plan_digest": preparation.revision_plan_digest,
        "content_digest": preparation.content_digest,
        "evidence_digests": sorted(preparation.evidence_digests),
        "evidence_refs": sorted((ref.to_json() for ref in preparation.evidence_refs), key=canonical_digest),
        "relationship_requirements": sorted(
            (requirement.to_json() for requirement in preparation.relationship_requirements),
            key=canonical_digest,
        ),
    }
    payload["revision_preparation_digest"] = canonical_digest(payload)
    return payload


def test_evidence_record_round_trips_canonically_and_recomputes_digests() -> None:
    record = EvidenceRecord(
        operation_id="op-a",
        binding="workspace",
        evidence_kind="diff_scan",
        payload_digest=_sha("payload"),
        stable_observation={"path": "artifact.txt"},
        observed_at_unix_ns=123,
    )

    decoded = EvidenceRecord.from_canonical_bytes(record.canonical_bytes())

    assert decoded == record
    mutated = record.to_json()
    mutated["extra"] = "bad"
    with pytest.raises(ValueError, match="unexpected evidence record fields"):
        EvidenceRecord.from_json(mutated)
    mutated_digest = record.to_json()
    mutated_digest["evidence_digest"] = _sha("wrong")
    with pytest.raises(ValueError, match="evidence_digest disagrees"):
        EvidenceRecord.from_json(mutated_digest)
    with pytest.raises(ValueError, match="canonical record is missing"):
        EvidenceRecord.from_canonical_bytes(b"{}")


def test_validated_payload_descriptor_claim_recomputes_json_digest() -> None:
    payload = {"schema": "example/workspace", "label": "candidate"}
    descriptor = ValidatedPayloadDescriptor.for_json_payload(payload)

    assert descriptor.payload_digest == canonical_digest(payload)
    assert ValidatedPayloadDescriptor.from_json(descriptor.to_json()) == descriptor

    bad_claim = PayloadDescriptorClaim(
        codec_id="vcscore.json",
        codec_version="v1",
        authority_mode="coordinator-native",
        payload_digest=_sha("wrong"),
        canonical_manifest={"payload_format": "canonical-json-v1"},
    )
    with pytest.raises(ValueError, match="claim digest disagrees"):
        bad_claim.validate(expected_payload_digest=canonical_digest(payload))


def test_retry_stable_transition_differs_from_operation_bound_preparation() -> None:
    ref_a = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    ref_b = _evidence_ref("op-b", "refs/vcscore/evidence/op-b/1")
    transition_a = _transition(ref_a)
    transition_b = _transition(ref_b)
    plan_a = _plan(transition_a)
    plan_b = _plan(transition_b)
    preparation_a = _preparation(operation_id="op-a", transition=transition_a, plan=plan_a, evidence_ref=ref_a)
    preparation_b = _preparation(operation_id="op-b", transition=transition_b, plan=plan_b, evidence_ref=ref_b)

    assert ref_a.evidence_digest == ref_b.evidence_digest
    assert ref_a.record_digest != ref_b.record_digest
    assert transition_a.transition_digest() == transition_b.transition_digest()
    assert plan_a.revision_plan_digest() == plan_b.revision_plan_digest()
    assert preparation_a.revision_preparation_digest() != preparation_b.revision_preparation_digest()


def test_revision_preparation_v2_serializes_cited_evidence_refs() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)
    plan = _plan(transition)
    preparation = _preparation(operation_id="op-a", transition=transition, plan=plan, evidence_ref=ref)
    serialized = preparation.to_json()

    assert serialized["schema"] == "vcscore/revision-preparation/v2"
    assert serialized["cited_evidence_refs"] == []
    assert RevisionPreparationRecord.from_json(serialized) == preparation


def test_revision_preparation_v2_requires_cited_evidence_refs() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)
    plan = _plan(transition)
    serialized = _preparation(operation_id="op-a", transition=transition, plan=plan, evidence_ref=ref).to_json()
    del serialized["cited_evidence_refs"]

    with pytest.raises(TypeError, match="cited_evidence_refs must be a list"):
        RevisionPreparationRecord.from_json(serialized)


def test_revision_preparation_rejects_legacy_v1_schema() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)
    plan = _plan(transition)
    preparation = _preparation(operation_id="op-a", transition=transition, plan=plan, evidence_ref=ref)
    legacy = _legacy_v1_preparation_json(preparation)

    with pytest.raises(ValueError, match="schema"):
        RevisionPreparationRecord.from_json(legacy)


def test_revision_preparation_citation_split_is_part_of_v2_digest() -> None:
    local_ref = _evidence_ref("op-reduce", "refs/vcscore/evidence/op-reduce/1")
    cited_ref = _evidence_ref("op-command", "refs/vcscore/evidence/op-command/1")
    transition = _transition(local_ref)
    plan = _plan(transition)
    base = _preparation(operation_id="op-reduce", transition=transition, plan=plan, evidence_ref=local_ref)
    common = {
        **{key: value for key, value in base.__dict__.items() if key != "cited_evidence_refs"},
        "evidence_digests": (local_ref.evidence_digest, cited_ref.evidence_digest),
        "evidence_refs": (local_ref, cited_ref),
    }

    with_citation = RevisionPreparationRecord(**common, cited_evidence_refs=(cited_ref,))
    without_citation = RevisionPreparationRecord(**common, cited_evidence_refs=())

    assert with_citation.revision_preparation_digest() != without_citation.revision_preparation_digest()


def test_plan_entry_ordering_is_part_of_revision_plan_identity() -> None:
    transition_digest = _sha("trace transition")
    plan_ab = PreparedRevisionPlan(
        binding="trace",
        store_id="store_trace",
        transition_digest=transition_digest,
        base_heads=("t1",),
        expected_parent_heads=("t1",),
        content_digest=_sha("trace content"),
        materialization_class="noop",
        entry_ordering="sequence",
        entries=({"event": "call"}, {"event": "result"}),
    )
    plan_ba = PreparedRevisionPlan(
        binding="trace",
        store_id="store_trace",
        transition_digest=transition_digest,
        base_heads=("t1",),
        expected_parent_heads=("t1",),
        content_digest=_sha("trace content"),
        materialization_class="noop",
        entry_ordering="sequence",
        entries=({"event": "result"}, {"event": "call"}),
    )

    assert plan_ab.revision_plan_digest() != plan_ba.revision_plan_digest()
    as_set_ab = PreparedRevisionPlan(**{**plan_ab.__dict__, "entry_ordering": "set"})
    as_set_ba = PreparedRevisionPlan(**{**plan_ba.__dict__, "entry_ordering": "set"})
    assert as_set_ab.revision_plan_digest() == as_set_ba.revision_plan_digest()


def test_revision_plan_digest_excludes_git_tree_oid() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)
    without_tree = _plan(transition)
    with_tree_a = PreparedRevisionPlan(**{**without_tree.__dict__, "git_tree_oid": "a" * 40})
    with_tree_b = PreparedRevisionPlan(**{**without_tree.__dict__, "git_tree_oid": "b" * 40})

    assert with_tree_a.revision_plan_digest() == without_tree.revision_plan_digest()
    assert with_tree_a.revision_plan_digest() == with_tree_b.revision_plan_digest()
    assert PreparedRevisionPlan.from_json(with_tree_a.to_json()) == with_tree_a


def test_head_selection_record_rejects_producer_world_oid() -> None:
    selection = HeadSelectionRecord(
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head="2" * 40,
        selection_kind="child-produced",
        retention_policy_requirements=(RetentionPolicyRequirement(kind="selected-head-pin", target="2" * 40),),
        selection_policy_digest=_sha("select child-produced workspace"),
    )
    serialized = selection.to_json()

    assert "producer_world_oid" not in serialized
    assert HeadSelectionRecord.from_json(serialized) == selection
    with pytest.raises(ValueError, match="unexpected head selection fields"):
        HeadSelectionRecord.from_json({**serialized, "producer_world_oid": "3" * 40})


def test_candidate_outcome_record_has_canonical_record_shape() -> None:
    outcome = CandidateOutcomeRecord(
        binding="workspace",
        candidate="2" * 40,
        outcome="selected",
        producer_world_oid="3" * 40,
    )
    changed_producer_world = CandidateOutcomeRecord(**{**outcome.__dict__, "producer_world_oid": "4" * 40})

    assert outcome.outcome_digest(final_operation_id="op-a") == changed_producer_world.outcome_digest(
        final_operation_id="op-a"
    )
    assert outcome.record_digest(final_operation_id="op-a") != changed_producer_world.record_digest(
        final_operation_id="op-a"
    )
    assert CandidateOutcomeRecord.from_record_json(outcome.to_record_json(final_operation_id="op-a")) == outcome

    legacy = outcome.to_json(final_operation_id="op-a")
    assert "schema" not in legacy
    assert CandidateOutcomeRecord.from_operation_final_json(legacy) == outcome
    with pytest.raises(ValueError, match="unsupported candidate outcome schema"):
        CandidateOutcomeRecord.from_record_json(legacy)

    mutated = outcome.to_record_json(final_operation_id="op-a")
    mutated["extra"] = "bad"
    with pytest.raises(ValueError, match="unexpected candidate outcome fields"):
        CandidateOutcomeRecord.from_record_json(mutated)
    mutated_digest = outcome.to_record_json(final_operation_id="op-a")
    mutated_digest["outcome_digest"] = _sha("wrong")
    with pytest.raises(ValueError, match="outcome_digest disagrees"):
        CandidateOutcomeRecord.from_record_json(mutated_digest)


def test_prepared_transition_bundle_requires_payload_or_ref() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)

    with pytest.raises(ValueError, match="payload or payload_ref"):
        PreparedTransitionBundle(transition=transition).require_resolvable_payload()

    PreparedTransitionBundle(transition=transition, payload={"ok": True}).require_resolvable_payload()
    PreparedTransitionBundle(transition=transition, payload_ref="refs/vcscore/payloads/1").require_resolvable_payload()


def test_candidate_and_selection_records_validate_round_trips() -> None:
    ref = _evidence_ref("op-a", "refs/vcscore/evidence/op-a/1")
    transition = _transition(ref)
    plan = _plan(transition)
    preparation = _preparation(operation_id="op-a", transition=transition, plan=plan, evidence_ref=ref)
    commit = CandidateCommitRecord(
        operation_id="op-a",
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        candidate_head="2" * 40,
        candidate_ref="refs/vcscore/candidates/op-a/workspace",
        revision_preparation_digest=preparation.revision_preparation_digest(),
    )
    selection = HeadSelectionRecord(
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head=commit.candidate_head,
        selection_kind="new-candidate",
        retention_policy_requirements=(
            RetentionPolicyRequirement(kind="selected-head-pin", target=commit.candidate_head),
        ),
        selection_policy_digest=_sha("select workspace"),
    )
    evidence = HeadSelectionEvidence(
        operation_id="op-a",
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head=commit.candidate_head,
        selection_digest=selection.selection_digest(),
        revision_preparation_digest=preparation.revision_preparation_digest(),
        candidate_commit_digest=commit.candidate_commit_digest(),
        candidate_ref=commit.candidate_ref,
        evidence_refs=(ref,),
        retention_policy_requirements=selection.retention_policy_requirements,
    )

    assert RevisionPreparationRecord.from_json(preparation.to_json()) == preparation
    assert CandidateCommitRecord.from_json(commit.to_json()) == commit
    assert HeadSelectionRecord.from_json(selection.to_json()) == selection
    assert HeadSelectionEvidence.from_json(evidence.to_json()) == evidence
    validate_head_selection(selection, evidence)

    second_requirement = RetentionPolicyRequirement(kind="world-pin", target=commit.candidate_head)
    reordered_selection = HeadSelectionRecord(
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head=commit.candidate_head,
        selection_kind="checkpoint",
        retention_policy_requirements=(selection.retention_policy_requirements[0], second_requirement),
    )
    reordered_evidence = HeadSelectionEvidence(
        operation_id="op-a",
        binding="workspace",
        store_id="store_workspace",
        resource_id="fs:repo-main",
        selected_head=commit.candidate_head,
        selection_digest=reordered_selection.selection_digest(),
        retention_policy_requirements=(second_requirement, selection.retention_policy_requirements[0]),
    )
    validate_head_selection(reordered_selection, reordered_evidence)

    checkpoint = HeadSelectionRecord(
        binding="session",
        store_id="store_session",
        resource_id="session:child",
        selected_head="7" * 40,
        selection_kind="checkpoint",
    )
    bad_checkpoint_evidence = HeadSelectionEvidence(
        operation_id="op-a",
        binding="session",
        store_id="store_session",
        resource_id="session:child",
        selected_head="7" * 40,
        selection_digest=checkpoint.selection_digest(),
        revision_preparation_digest=preparation.revision_preparation_digest(),
        candidate_commit_digest=commit.candidate_commit_digest(),
        candidate_ref=commit.candidate_ref,
    )
    with pytest.raises(ValueError, match="non-candidate selection"):
        validate_head_selection(checkpoint, bad_checkpoint_evidence)
