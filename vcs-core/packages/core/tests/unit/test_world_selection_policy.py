# under-test: vcs_core._world_selection_policy
"""Unit tests for private world-vector selection policy helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest
from vcs_core import EvidenceRef, InvalidRepositoryStateError
from vcs_core._world_selection_policy import (
    allowed_existing_head_semantic_ops,
    resolve_candidate_selection_kind,
    validate_root_selection_policy,
    validate_unchanged_head_identity,
    validate_unchanged_selection_policy,
)
from vcs_core._world_types import SubstrateHead


def test_root_selection_policy_rejects_unchanged_selection() -> None:
    validate_root_selection_policy(input_world_oid=None, selection_kind="bootstrap")
    validate_root_selection_policy(input_world_oid=None, selection_kind="checkpoint")
    validate_root_selection_policy(input_world_oid=None, selection_kind="import")
    validate_root_selection_policy(input_world_oid=None, selection_kind="revert")
    validate_root_selection_policy(input_world_oid=None, selection_kind="new-candidate")
    validate_root_selection_policy(input_world_oid="a" * 40, selection_kind="unchanged")

    with pytest.raises(InvalidRepositoryStateError, match="explicit bootstrap/import/checkpoint/revert"):
        validate_root_selection_policy(input_world_oid=None, selection_kind="unchanged")


def test_unchanged_selection_policy_requires_input_world_and_no_evidence_refs() -> None:
    validate_unchanged_selection_policy(input_world_oid="a" * 40, evidence_refs=())

    with pytest.raises(InvalidRepositoryStateError, match="explicit bootstrap/import/checkpoint/revert"):
        validate_unchanged_selection_policy(input_world_oid=None, evidence_refs=())
    with pytest.raises(InvalidRepositoryStateError, match="must not carry evidence refs"):
        validate_unchanged_selection_policy(
            input_world_oid="a" * 40,
            evidence_refs=(
                EvidenceRef(
                    ref="refs/vcscore/evidence/op/1",
                    evidence_digest="sha256:" + "1" * 64,
                    record_digest="sha256:" + "2" * 64,
                    payload_digest="sha256:" + "3" * 64,
                ),
            ),
        )


def test_unchanged_head_identity_requires_exact_substrate_head() -> None:
    input_head = SubstrateHead(
        binding="workspace",
        kind="filesystem",
        role="shepherd.WorkspaceRef",
        store_id="store_workspace",
        store_scope="local",
        resource_id="fs:repo-main",
        head="1" * 40,
    )

    validate_unchanged_head_identity(input_head=input_head, selected_head=input_head)

    with pytest.raises(InvalidRepositoryStateError, match="input world head identity"):
        validate_unchanged_head_identity(
            input_head=input_head,
            selected_head=replace(input_head, role="shepherd.OtherWorkspaceRef"),
        )
    with pytest.raises(InvalidRepositoryStateError, match="input world head identity"):
        validate_unchanged_head_identity(
            input_head=input_head,
            selected_head=replace(input_head, store_scope="imported"),
        )


def test_existing_head_selection_semantic_ops_are_narrow() -> None:
    assert allowed_existing_head_semantic_ops("bootstrap") == {"bootstrap"}
    assert allowed_existing_head_semantic_ops("checkpoint") == {"checkpoint"}
    assert allowed_existing_head_semantic_ops("import") == {
        "bootstrap",
        "import",
        "workspace-adoption",
        "workspace-capture-reduction",
        "workspace-overlay-merge",
        "workspace-scan",
    }
    assert allowed_existing_head_semantic_ops("revert") == {"revert"}


def test_candidate_selection_kind_resolution_is_operation_scoped() -> None:
    assert (
        resolve_candidate_selection_kind(
            operation_id="op-parent",
            producer_operation_id="op-parent",
            producer_world_oid=None,
            requested_kind=None,
        )
        == "new-candidate"
    )
    assert (
        resolve_candidate_selection_kind(
            operation_id="op-parent",
            producer_operation_id="op-child",
            producer_world_oid="1" * 40,
            requested_kind=None,
        )
        == "child-produced"
    )

    with pytest.raises(InvalidRepositoryStateError, match="current operation producer"):
        resolve_candidate_selection_kind(
            operation_id="op-parent",
            producer_operation_id="op-child",
            producer_world_oid=None,
            requested_kind="new-candidate",
        )
    with pytest.raises(InvalidRepositoryStateError, match="producer_world_oid"):
        resolve_candidate_selection_kind(
            operation_id="op-parent",
            producer_operation_id="op-parent",
            producer_world_oid="1" * 40,
            requested_kind="new-candidate",
        )
    with pytest.raises(InvalidRepositoryStateError, match="producer_world_oid"):
        resolve_candidate_selection_kind(
            operation_id="op-parent",
            producer_operation_id="op-child",
            producer_world_oid=None,
            requested_kind="child-produced",
        )
