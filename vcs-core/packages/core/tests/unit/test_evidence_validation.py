# under-test: vcs_core._evidence_validation
"""Unit tests for transition-kernel evidence authority validation."""

from __future__ import annotations

import pytest
from vcs_core import EvidenceRef, InvalidRepositoryStateError, canonical_digest
from vcs_core._evidence_validation import EvidenceCitationScope, validate_preparation_evidence_refs
from vcs_core._transition_kernel_records import EvidenceRecord


def test_preparation_evidence_validation_accepts_local_first_cited_suffix() -> None:
    local_record = _record("op-reduce", "local")
    cited_record = _record("op-command", "cited")
    local_ref = _ref(local_record)
    cited_ref = _ref(cited_record)

    records = validate_preparation_evidence_refs(
        (local_ref, cited_ref),
        cited_evidence_refs=(cited_ref,),
        expected_digests=(local_ref.evidence_digest, cited_ref.evidence_digest),
        scope=_scope("op-reduce"),
        resolver=_resolver(local_record, cited_record),
    )

    assert records == (local_record, cited_record)


def test_preparation_evidence_validation_rejects_cited_ref_before_local_ref() -> None:
    local_record = _record("op-reduce", "local")
    cited_record = _record("op-command", "cited")
    local_ref = _ref(local_record)
    cited_ref = _ref(cited_record)

    with pytest.raises(InvalidRepositoryStateError, match="cited_evidence_refs must be a suffix"):
        validate_preparation_evidence_refs(
            (cited_ref, local_ref),
            cited_evidence_refs=(cited_ref,),
            expected_digests=(local_ref.evidence_digest, cited_ref.evidence_digest),
            scope=_scope("op-reduce"),
            resolver=_resolver(local_record, cited_record),
        )


def test_preparation_evidence_validation_rejects_reordered_cited_suffix() -> None:
    local_record = _record("op-reduce", "local")
    cited_record_a = _record("op-command", "cited-a")
    cited_record_b = _record("op-command", "cited-b")
    local_ref = _ref(local_record)
    cited_ref_a = _ref(cited_record_a)
    cited_ref_b = _ref(cited_record_b)

    with pytest.raises(InvalidRepositoryStateError, match="cited_evidence_refs must be a suffix"):
        validate_preparation_evidence_refs(
            (local_ref, cited_ref_a, cited_ref_b),
            cited_evidence_refs=(cited_ref_b, cited_ref_a),
            expected_digests=(local_ref.evidence_digest, cited_ref_a.evidence_digest, cited_ref_b.evidence_digest),
            scope=_scope("op-reduce"),
            resolver=_resolver(local_record, cited_record_a, cited_record_b),
        )


def _record(operation_id: str, label: str) -> EvidenceRecord:
    payload_digest = canonical_digest({"payload": label})
    return EvidenceRecord(
        operation_id=operation_id,
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
        ingress_kind="command",
        evidence_kind="command_envelope",
        payload_digest=payload_digest,
        stable_observation={"command": "write", "label": label},
    )


def _ref(record: EvidenceRecord) -> EvidenceRef:
    label = str(record.stable_observation["label"])
    return EvidenceRef(
        ref=f"refs/vcscore/evidence/{record.operation_id}/{label}",
        evidence_digest=record.evidence_digest(),
        record_digest=record.record_digest(),
        payload_digest=record.payload_digest,
    )


def _scope(operation_id: str) -> EvidenceCitationScope:
    return EvidenceCitationScope(
        operation_id=operation_id,
        binding="workspace",
        store_id="store_workspace",
        substrate_kind="filesystem",
    )


def _resolver(*records: EvidenceRecord):
    by_digest = {record.evidence_digest(): record for record in records}
    return lambda ref: by_digest.get(ref.evidence_digest)
