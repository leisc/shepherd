from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from shepherd2.canonical import witness_body_digest

from shepherd2 import (
    ABI_VERSION,
    CANONICAL_VERSION,
    Cut,
    CutSpec,
    Fact,
    FactBody,
    FactDraft,
    FactEnvelope,
    FactShape,
    OwnerCutoff,
    OwnerCutoffSpec,
    Record,
    RecordBody,
    RecordDraft,
    RecordEnvelope,
    RecordShape,
    canonical_digest,
    root_witness_body,
    root_witness_body_digest,
    root_witness_record_id,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "kernel_abi_v0.json"


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def test_kernel_abi_v0_versions_are_frozen() -> None:
    golden = _golden()

    assert ABI_VERSION == "shepherd.kernel.abi.v0"
    assert CANONICAL_VERSION == "shepherd.kernel.canonical.v2"
    assert golden["abi_version"] == ABI_VERSION
    assert golden["canonical_version"] == CANONICAL_VERSION


def test_kernel_abi_v0_witness_golden_vectors_are_exact() -> None:
    golden = _golden()

    assert root_witness_body() == golden["root_witness"]["body"]
    assert root_witness_body_digest() == golden["root_witness"]["body_digest"]
    assert root_witness_record_id() == golden["root_witness"]["record_id"]
    assert canonical_digest(golden["root_witness"]["record_input"]) == golden["root_witness"]["record_id"]
    assert (
        witness_body_digest(
            schema_ref=golden["ordinary_witness"]["schema_ref"],
            body=golden["ordinary_witness"]["body"],
        )
        == golden["ordinary_witness"]["body_digest"]
    )
    assert canonical_digest(golden["ordinary_witness"]["record_input"]) == golden["ordinary_witness"]["record_id"]
    assert (
        witness_body_digest(
            schema_ref=golden["alternate_witness"]["schema_ref"],
            body=golden["alternate_witness"]["body"],
        )
        == golden["alternate_witness"]["body_digest"]
    )
    assert canonical_digest(golden["alternate_witness"]["record_input"]) == golden["alternate_witness"]["record_id"]


def test_kernel_abi_v0_witness_record_ids_are_not_body_digests() -> None:
    golden = _golden()

    assert golden["root_witness"]["record_id"] != golden["root_witness"]["body_digest"]
    assert golden["ordinary_witness"]["record_id"] != golden["ordinary_witness"]["body_digest"]
    assert golden["alternate_witness"]["record_id"] != golden["alternate_witness"]["body_digest"]
    assert golden["ordinary_witness"]["record_input"]["witness"] == golden["root_witness"]["record_id"]
    assert golden["alternate_witness"]["record_input"]["witness"] == golden["root_witness"]["record_id"]


def test_kernel_abi_v0_record_vectors_use_retained_witness_record_ids() -> None:
    golden = _golden()

    for key in (
        "capture_record",
        "declaration_record",
        "ordered_parent_record",
        "reversed_parent_record",
    ):
        assert golden[key]["input"]["witness"] == golden["ordinary_witness"]["record_id"]
        assert golden[key]["input"]["witness"] != golden["ordinary_witness"]["body_digest"]
    assert golden["alternate_witness_record"]["input"]["witness"] == golden["alternate_witness"]["record_id"]
    assert golden["alternate_witness_record"]["input"]["witness"] != golden["alternate_witness"]["body_digest"]


def test_kernel_abi_v0_record_golden_vectors_are_exact() -> None:
    golden = _golden()

    for key in (
        "capture_record",
        "declaration_record",
        "ordered_parent_record",
        "reversed_parent_record",
        "alternate_witness_record",
    ):
        assert canonical_digest(golden[key]["input"]) == golden[key]["digest"]


def test_kernel_abi_v0_golden_vectors_cover_field_sensitivity() -> None:
    golden = _golden()

    assert golden["capture_record"]["digest"] != golden["declaration_record"]["digest"]
    assert golden["ordered_parent_record"]["digest"] != golden["reversed_parent_record"]["digest"]
    assert golden["capture_record"]["digest"] != golden["alternate_witness_record"]["digest"]


def test_kernel_abi_v0_fact_names_are_record_aliases() -> None:
    assert FactDraft is RecordDraft
    assert FactBody is RecordBody
    assert FactEnvelope is RecordEnvelope
    assert Fact is Record
    assert FactShape is RecordShape
    assert OwnerCutoff is Cut
    assert OwnerCutoffSpec is CutSpec


def test_kernel_abi_v0_record_draft_requires_mode() -> None:
    with pytest.raises(TypeError, match="mode"):
        RecordDraft(kind_label="step", schema_ref="example.step.v1", payload={})  # type: ignore[call-arg]
