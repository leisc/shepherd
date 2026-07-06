# under-test: vcs_core._world_substrate_adapters
"""Unit tests for narrow v2 world substrate adapters."""

from __future__ import annotations

from dataclasses import replace

import pytest
from vcs_core import WORLD_TRANSITION_SCHEMA, InvalidRepositoryStateError, WorldSnapshot, canonical_digest
from vcs_core._capture_reducer import CaptureJournalEvent
from vcs_core._transition_kernel_records import RetentionPolicyRequirement
from vcs_core._world_substrate_adapters import (
    SESSION_STATE_REVISION_SCHEMA,
    TRACE_REVISION_SCHEMA,
    WORKSPACE_REVISION_SCHEMA,
    WORKSPACE_STATE_MANIFEST_SCHEMA,
    RoleSubstrateAdapter,
    SessionStateSubstrateAdapter,
    SessionStateSubstrateDriver,
    TaskTraceSubstrateAdapter,
    TaskTraceSubstrateDriver,
    WorkspaceSubstrateAdapter,
    WorkspaceSubstrateDriver,
    WorldRefSubstrateAdapter,
    WorldRefSubstrateDriver,
    workspace_state_manifest_payload,
    workspace_state_revision_payload,
)
from vcs_core._world_types import WORLD_REF_SUBSTRATE_KIND, WorldRefPayload
from vcs_core.runtime_api import CommandRequest, DriverIngressResult
from vcs_core.spi import ObservationDraft, PayloadDescriptorClaim, SubstrateDriver, SubstrateStoreIdentity
from vcs_core.testing import (
    DEFAULT_GROUND_REF,
    CandidateSelection,
    OperationFinalBuilder,
    SubstrateStoreSpec,
    WorldStorageManager,
)


def _manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_workspace",
                    kind="filesystem",
                    resource_id="fs:repo-main",
                ),
                locator="substrates/workspace.git",
            ),
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_session",
                    kind="shepherd.session_state",
                    resource_id="shepherd-session:child-baseline",
                ),
                locator="substrates/session.git",
            ),
        ),
    )


def _world_ref_manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_workspace",
                    kind="filesystem",
                    resource_id="fs:repo-main",
                ),
                locator="substrates/workspace.git",
            ),
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_child_world_ref",
                    kind=WORLD_REF_SUBSTRATE_KIND,
                    resource_id="world-ref:child-task",
                ),
                locator="substrates/child-world-ref.git",
            ),
        ),
    )


def _trace_manager(tmp_path) -> WorldStorageManager:
    return WorldStorageManager.open_or_init(
        tmp_path / ".vcscore",
        world_store_id="store_world_test",
        stores=(
            SubstrateStoreSpec(
                identity=SubstrateStoreIdentity(
                    store_id="store_trace",
                    kind="shepherd.trace",
                    resource_id="shepherd-trace:parent",
                ),
                locator="substrates/trace.git",
            ),
        ),
    )


def _capture_event(
    command_operation_id: str,
    path: str,
    *,
    global_seq: int,
    proc_seq: int = 1,
    event_seq: int = 1,
    op: str = "write_observed",
) -> CaptureJournalEvent:
    return CaptureJournalEvent(
        command_operation_id=command_operation_id,
        binding_name="workspace",
        op=op,  # type: ignore[arg-type]
        path=path,
        scope="task",
        scope_instance_id="scope-1",
        pid=101,
        proc_seq=proc_seq,
        global_seq=global_seq,
        event_seq=event_seq,
        capture_mechanism="preload",
        capture_epoch="cap-1",
        cwd="/workspace",
    )


def _empty_world(manager: WorldStorageManager, operation_id: str = "op-child-world") -> str:
    return manager.create_unsafe_world(
        snapshot=WorldSnapshot(),
        transition={
            "schema": WORLD_TRANSITION_SCHEMA,
            "operation_id": operation_id,
            "parent_worlds": [],
        },
        operation_final={
            "schema": "vcscore/operation-final/v2",
            "operation_id": operation_id,
            "selected": {},
            "candidate_commits": [],
            "candidate_outcomes": [],
            "head_selections": [],
            "selection_evidence": [],
        },
    )


def test_workspace_and_session_adapters_create_role_aware_heads_and_candidates(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    session = SessionStateSubstrateAdapter(manager)

    w42 = workspace.create_bootstrap_revision(
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-workspace",
    )
    s7 = session.create_checkpoint(
        "refs/checkpoints/S7",
        {"label": "session S7"},
        operation_id="op-checkpoint-session",
    )
    w43 = workspace.create_candidate(
        operation_id="op-child",
        payload={"label": "workspace W43"},
        parents=(w42,),
    )
    s8 = session.create_candidate(
        operation_id="op-child",
        payload={"label": "session S8"},
        parents=(s7,),
    )

    workspace_head = workspace.head(w43.candidate.head)
    session_head = session.head(s8.candidate.head)
    workspace_plan = workspace.plan_candidate_selection(w43)
    session_plan = session.plan_candidate_selection(s8)

    assert workspace_head.role == "shepherd.WorkspaceRef"
    assert workspace_head.binding == "workspace"
    assert session_head.role == "shepherd.SessionState"
    assert session_head.binding == "session"
    assert workspace_plan.operation_id == "op-child"
    assert workspace_plan.selection.candidate == w43.candidate
    assert session_plan.selection.candidate == s8.candidate
    assert str(manager.store("store_workspace").repo.references[w43.candidate.ref].target) == w43.candidate.head
    assert str(manager.store("store_session").repo.references[s8.candidate.ref].target) == s8.candidate.head
    assert w43.candidate_commit.candidate_head == w43.candidate.head
    assert s8.candidate_commit.candidate_head == s8.candidate.head
    assert (
        manager.store("store_workspace").read_revision_metadata(w43.candidate.head).materialization_class == "external"
    )
    assert (
        manager.store("store_session").read_revision_metadata(s8.candidate.head).produced_by_operation_id == "op-child"
    )
    workspace_provenance = manager.store("store_workspace").validate_prepared_candidate(
        w43.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    session_provenance = manager.store("store_session").validate_prepared_candidate(
        s8.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    assert workspace_provenance.transition.ingress_kind == "command"
    assert workspace_provenance.transition.semantic_op == "workspace-json-revision"
    assert workspace_provenance.transition.driver == "shepherd.workspace_ref"
    assert (
        manager.store("store_workspace")
        .validate_prepared_revision(
            w42,
            evidence_resolver=manager.world_store.resolve_evidence_ref,
        )
        .transition.semantic_op
        == "bootstrap"
    )
    assert (
        manager.store("store_session")
        .validate_prepared_revision(
            s7,
            evidence_resolver=manager.world_store.resolve_evidence_ref,
        )
        .transition.semantic_op
        == "checkpoint"
    )
    assert session_provenance.transition.semantic_op == "session-state-json-revision"
    assert session_provenance.transition.driver == "shepherd.session_state"


def test_workspace_driver_prepares_json_state_drafts_without_manager(tmp_path) -> None:
    manager = _manager(tmp_path)
    adapter = WorkspaceSubstrateAdapter(manager)
    driver = WorkspaceSubstrateDriver()

    bootstrap = driver.prepare(
        adapter._context(operation_id="op-workspace", parents=()),
        CommandRequest(command="bootstrap", params={"payload": {"label": "workspace W42"}}),
    )
    imported = driver.prepare(
        adapter._context(operation_id="op-workspace-import", parents=("1" * 40,)),
        CommandRequest(command="import", params={"payload": {"label": "workspace W43"}}),
    )
    candidate = driver.prepare(
        adapter._context(operation_id="op-workspace-candidate", parents=("2" * 40,)),
        CommandRequest(command="create-candidate", params={"payload": {"label": "workspace W44"}}),
    )

    assert isinstance(driver, SubstrateDriver)
    # T1a: workspace driver only declares CommandRequest acceptance in its
    # CapabilitySet; T1c expands ``accepts`` to the full typed family and
    # surfaces the legacy 8 string commands through ``describe()``.
    assert CommandRequest in driver.capabilities.accepts
    assert driver.capabilities.materializable is True
    assert not hasattr(driver, "manager")
    assert not hasattr(driver, "store")
    assert bootstrap.transitions[0].semantic_op == "bootstrap"
    assert imported.transitions[0].semantic_op == "import"
    assert candidate.transitions[0].semantic_op == "workspace-json-revision"
    assert bootstrap.transitions[0].payload == {
        "schema": WORKSPACE_REVISION_SCHEMA,
        "label": "workspace W42",
    }
    with pytest.raises(ValueError, match="unsupported workspace command"):
        driver.prepare(
            adapter._context(operation_id="op-workspace-legacy", parents=()),
            CommandRequest(command="create-revision", params={"payload": {"label": "workspace legacy"}}),
        )


def test_workspace_driver_revisions_are_selectable_for_bootstrap_and_import(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)

    bootstrap = workspace.create_bootstrap_revision(
        "refs/heads/bootstrap",
        {"label": "workspace bootstrap"},
        operation_id="op-bootstrap-workspace",
    )
    imported = workspace.create_import_revision(
        "refs/heads/import",
        {"label": "workspace import"},
        operation_id="op-import-workspace",
        parents=(bootstrap,),
    )

    bootstrap_plan = manager.plan_existing_head_selection(
        operation_id="op-select-bootstrap",
        head=workspace.head(bootstrap),
        selection_kind="bootstrap",
    )
    import_plan = manager.plan_existing_head_selection(
        operation_id="op-select-import",
        head=workspace.head(imported),
        selection_kind="import",
    )

    assert bootstrap_plan.selected_head == bootstrap
    assert import_plan.selected_head == imported
    assert (
        manager.store("store_workspace")
        .validate_prepared_revision(imported, evidence_resolver=manager.world_store.resolve_evidence_ref)
        .transition.semantic_op
        == "import"
    )


def test_workspace_state_manifest_payload_validates_and_sorts_entries() -> None:
    payload = workspace_state_manifest_payload(
        (
            {"path": "src/tool.sh", "state": "present", "mode": 0o100755, "content_digest": "sha256:" + "2" * 64},
            {"path": "README.md", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "1" * 64},
            {"path": "old.txt", "state": "deleted"},
        )
    )

    assert payload["schema"] == WORKSPACE_STATE_MANIFEST_SCHEMA
    assert payload["byte_authority"] == "digest-only"
    assert [entry["path"] for entry in payload["entries"]] == ["README.md", "old.txt", "src/tool.sh"]


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"path": "../escape", "state": "deleted"}, "escapes"),
        ({"path": "/abs", "state": "deleted"}, "relative"),
        ({"path": "run.sh", "state": "present", "mode": 0o100777, "content_digest": "sha256:" + "1" * 64}, "mode"),
        ({"path": "run.sh", "state": "present", "mode": 0o100644}, "content_digest"),
    ],
)
def test_workspace_state_manifest_rejects_invalid_entries(entry, match) -> None:
    with pytest.raises(ValueError, match=match):
        workspace_state_manifest_payload((entry,))


def test_workspace_state_manifest_rejects_unsupported_byte_authority() -> None:
    with pytest.raises(ValueError, match="unsupported workspace manifest byte_authority"):
        workspace_state_manifest_payload(
            (
                {
                    "path": "src/app.py",
                    "state": "present",
                    "mode": 0o100644,
                    "content_digest": "sha256:" + "1" * 64,
                },
            ),
            byte_authority="content-blob",
        )


def test_workspace_state_manifest_revision_remains_json_workspace_payload(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    payload = workspace_state_revision_payload(
        (
            {"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "3" * 64},
            {"path": "deleted.txt", "state": "deleted"},
        )
    )

    bundle = workspace.create_candidate(operation_id="op-workspace-manifest", payload=payload)
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        bundle.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert payload["schema"] == WORKSPACE_REVISION_SCHEMA
    assert payload["state_manifest"]["schema"] == WORKSPACE_STATE_MANIFEST_SCHEMA
    assert payload["state_manifest"]["byte_authority"] == "digest-only"
    assert provenance.payload_digest == canonical_digest(payload)


def test_workspace_scan_and_adoption_candidates_use_native_driver_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "5" * 64},)
    )

    scan = workspace.create_scan_candidate(operation_id="op-workspace-scan", payload=payload)
    adoption = workspace.create_adoption_candidate(operation_id="op-workspace-adopt", payload=payload)
    scan_provenance = manager.store("store_workspace").validate_prepared_candidate(
        scan.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    adoption_provenance = manager.store("store_workspace").validate_prepared_candidate(
        adoption.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert scan_provenance.transition.semantic_op == "workspace-scan"
    assert adoption_provenance.transition.semantic_op == "workspace-adoption"
    assert scan_provenance.transition.ingress_kind == "scan"
    assert adoption_provenance.transition.ingress_kind == "scan"
    assert scan_provenance.payload_digest == canonical_digest(payload)
    assert adoption_provenance.payload_digest == canonical_digest(payload)


def test_workspace_overlay_merge_candidate_uses_native_driver_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    payload = workspace_state_revision_payload(
        (
            {"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "6" * 64},
            {"path": "old.txt", "state": "deleted"},
        )
    )

    bundle = workspace.create_overlay_merge_candidate(operation_id="op-workspace-overlay", payload=payload)
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        bundle.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert provenance.transition.semantic_op == "workspace-overlay-merge"
    assert provenance.transition.ingress_kind == "merge"
    assert provenance.payload_digest == canonical_digest(payload)


def test_workspace_capture_reduction_lowers_capture_events_to_evidence(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (
        _capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),
        _capture_event("op-command", "src/app.py", global_seq=2, proc_seq=2, op="write_close"),
    )
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    bundle = workspace.create_capture_reduction_candidate_from_evidence(
        operation_id="op-reduce-capture",
        command_operation_id="op-command",
        payload=payload,
        reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
        reduced_state_proof={
            "byte_authority": "digest-only",
            "manifest_digest": canonical_digest(payload["state_manifest"]),
        },
    )
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        bundle.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    evidence_records = tuple(manager.world_store.resolve_evidence_ref(ref) for ref in bundle.preparation.evidence_refs)

    assert provenance.transition.semantic_op == "workspace-capture-reduction"
    assert provenance.transition.ingress_kind == "reduce"
    assert provenance.transition.driver == "shepherd.workspace_ref"
    assert len(evidence_records) == 3
    assert [record.ingress_kind for record in evidence_records] == ["reduce", "capture", "capture"]
    assert [record.evidence_kind for record in evidence_records] == [
        "reduce:reduced-state-proof",
        "capture:filesystem-event",
        "capture:filesystem-event",
    ]
    assert [record.stable_observation.get("global_seq") for record in evidence_records] == [None, 1, 2]
    assert evidence_records[1].stable_observation["path"] == "src/app.py"
    assert evidence_records[1].stable_observation["command_operation_id"] == "op-command"
    assert {record.operation_id for record in evidence_records[1:]} == {"op-command"}
    assert evidence_records[0].operation_id == "op-reduce-capture"
    assert "operation_id" not in evidence_records[1].stable_observation
    assert "parent_heads" not in evidence_records[1].stable_observation
    assert "payload_digest" not in evidence_records[1].stable_observation
    assert evidence_records[1].payload_digest == canonical_digest(evidence_records[1].stable_observation)
    assert evidence_records[1].payload_digest != provenance.transition.payload_digest


def test_workspace_capture_history_persists_command_owned_evidence_only(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (
        _capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),
        _capture_event("op-command", "src/app.py", global_seq=2, proc_seq=2, op="write_close"),
    )

    persisted = workspace.persist_capture_history_evidence(
        command_operation_id="op-command",
        capture_events=events,
    )

    assert len(persisted.evidence_refs) == 2
    envelope = manager.world_store.resolve_evidence_only_envelope(
        persisted.envelope_ref,
        expected_operation_id="op-command",
    )
    assert envelope.evidence_refs == persisted.evidence_refs
    records = tuple(manager.world_store.resolve_evidence_ref(ref) for ref in persisted.evidence_refs)
    assert {record.operation_id for record in records} == {"op-command"}
    assert [record.stable_observation["global_seq"] for record in records] == [1, 2]
    assert not any(ref.startswith("refs/vcscore/candidates/") for ref in manager.world_store.repo.references)


def test_workspace_capture_reduction_from_evidence_cites_raw_and_state_proof(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)
    batch = manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw")
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    bundle = workspace.create_capture_reduction_candidate_from_evidence(
        operation_id="op-reduce",
        command_operation_id="op-command",
        payload=payload,
        reduction_batch=batch,
        reduced_state_proof={
            "byte_authority": "digest-only",
            "manifest_digest": canonical_digest(payload["state_manifest"]),
        },
    )
    records = tuple(manager.world_store.resolve_evidence_ref(ref) for ref in bundle.preparation.evidence_refs)

    assert bundle.transition.semantic_op == "workspace-capture-reduction"
    assert bundle.transition.evidence_digests == tuple(ref.evidence_digest for ref in bundle.preparation.evidence_refs)
    assert [record.operation_id for record in records] == ["op-reduce", "op-command"]
    assert [record.evidence_kind for record in records] == [
        "reduce:reduced-state-proof",
        "capture:filesystem-event",
    ]


def test_workspace_capture_reduction_from_evidence_rejects_duplicate_cited_ref(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)
    citation = manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw").citations[0]
    duplicate = replace(citation, citation_id="raw-duplicate")
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    with pytest.raises(InvalidRepositoryStateError, match="duplicate evidence ref"):
        workspace.create_capture_reduction_candidate_from_evidence(
            operation_id="op-reduce",
            command_operation_id="op-command",
            payload=payload,
            reduction_batch=replace(
                manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
                citations=(citation, duplicate),
            ),
            reduced_state_proof={
                "byte_authority": "digest-only",
                "manifest_digest": canonical_digest(payload["state_manifest"]),
            },
        )


def test_workspace_capture_reduction_from_evidence_rejects_diagnostic_citation(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    diagnostic = {"command_operation_id": "op-command", "reason": "unsupported_capture_event"}
    diagnostic_ref = manager.persist_driver_diagnostics(
        "store_workspace",
        operation_id="op-command-diagnostic",
        binding="workspace",
        result=DriverIngressResult(
            observations=(
                ObservationDraft(
                    observation_id="diagnostic",
                    evidence_kind="diagnostic:capture:unsupported-event",
                    stable_observation=diagnostic,
                    evidence_payload_descriptor_claim=PayloadDescriptorClaim.for_json_payload(diagnostic),
                ),
            )
        ),
        ingress_kind="capture",
        driver_id=workspace.driver.driver_id,
        driver_version=workspace.driver.driver_version,
    ).evidence_refs[0]
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    with pytest.raises(InvalidRepositoryStateError, match="capture-mechanism evidence"):
        workspace.create_capture_reduction_candidate_from_evidence(
            operation_id="op-reduce",
            command_operation_id="op-command",
            payload=payload,
            reduction_batch=manager.build_reduction_batch((diagnostic_ref,), citation_prefix="diagnostic"),
            reduced_state_proof={
                "byte_authority": "digest-only",
                "manifest_digest": canonical_digest(payload["state_manifest"]),
            },
        )


def test_workspace_capture_reduction_from_evidence_rejects_bad_state_proof(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    with pytest.raises(ValueError, match="manifest_digest disagrees"):
        workspace.create_capture_reduction_candidate_from_evidence(
            operation_id="op-reduce",
            command_operation_id="op-command",
            payload=payload,
            reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
            reduced_state_proof={
                "byte_authority": "digest-only",
                "manifest_digest": "sha256:" + "0" * 64,
            },
        )


def test_workspace_capture_reduction_from_evidence_requires_state_manifest_payload(tmp_path) -> None:
    # T3-callers: the typed ReduceRequest path passes the caller's payload
    # through unmodified. Pre-T3 the legacy prepare_command path wrapped the
    # payload with WORKSPACE_REVISION_SCHEMA before validation, so a payload
    # missing both schema AND state_manifest failed on the second check
    # (TypeError, "requires state_manifest"). Post-T3 it fails on the first
    # check (ValueError, schema mismatch) — clearer error semantics.
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)

    with pytest.raises(ValueError, match="payload schema"):
        workspace.create_capture_reduction_candidate_from_evidence(
            operation_id="op-reduce",
            command_operation_id="op-command",
            payload={"label": "not a workspace state revision"},
            reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
            reduced_state_proof={
                "byte_authority": "digest-only",
                "manifest_digest": "sha256:" + "0" * 64,
            },
        )


def test_workspace_capture_reduction_from_evidence_rejects_proof_command_mismatch(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)
    payload = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )

    with pytest.raises(ValueError, match="command_operation_id disagrees"):
        workspace.create_capture_reduction_candidate_from_evidence(
            operation_id="op-reduce",
            command_operation_id="op-command",
            payload=payload,
            reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
            reduced_state_proof={
                "byte_authority": "digest-only",
                "manifest_digest": canonical_digest(payload["state_manifest"]),
                "command_operation_id": "op-other",
            },
        )


def test_workspace_capture_reduction_rejects_inline_raw_event_command(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)

    with pytest.raises(ValueError, match="unsupported workspace command"):
        workspace.driver.prepare(
            workspace._context(operation_id="op-reduce-capture", parents=()),
            CommandRequest(
                command="capture-reduction",
                params={
                    "payload": {"label": "workspace after capture"},
                    "command_operation_id": "op-command",
                    "capture_events": events,
                },
            ),
        )


def test_workspace_capture_raw_evidence_identity_ignores_reducer_operation_and_payload(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    events = (_capture_event("op-command", "src/app.py", global_seq=1, proc_seq=1),)
    payload_a = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "4" * 64},)
    )
    payload_b = workspace_state_revision_payload(
        ({"path": "src/app.py", "state": "present", "mode": 0o100644, "content_digest": "sha256:" + "5" * 64},)
    )
    raw = workspace.persist_capture_history_evidence(command_operation_id="op-command", capture_events=events)

    first = workspace.create_capture_reduction_candidate_from_evidence(
        operation_id="op-reduce-a",
        command_operation_id="op-command",
        payload=payload_a,
        reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
        reduced_state_proof={
            "byte_authority": "digest-only",
            "manifest_digest": canonical_digest(payload_a["state_manifest"]),
        },
    )
    second = workspace.create_capture_reduction_candidate_from_evidence(
        operation_id="op-reduce-b",
        command_operation_id="op-command",
        payload=payload_b,
        reduction_batch=manager.build_reduction_batch(raw.evidence_refs, citation_prefix="raw"),
        reduced_state_proof={
            "byte_authority": "digest-only",
            "manifest_digest": canonical_digest(payload_b["state_manifest"]),
        },
    )

    assert first.preparation.evidence_refs[1].evidence_digest == second.preparation.evidence_refs[1].evidence_digest
    assert first.preparation.evidence_refs[1].record_digest == second.preparation.evidence_refs[1].record_digest
    assert first.preparation.evidence_refs[1].ref == second.preparation.evidence_refs[1].ref
    assert first.preparation.evidence_refs[0].evidence_digest != second.preparation.evidence_refs[0].evidence_digest


def test_workspace_capture_history_rejects_empty_or_mismatched_events(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)

    with pytest.raises(ValueError, match="capture_events"):
        workspace.persist_capture_history_evidence(
            command_operation_id="op-command",
            capture_events=(),
        )
    with pytest.raises(ValueError, match="command_operation_id"):
        workspace.persist_capture_history_evidence(
            command_operation_id="op-command",
            capture_events=(_capture_event("other-command", "src/app.py", global_seq=1),),
        )


def test_session_state_driver_prepares_provider_neutral_drafts_without_manager(tmp_path) -> None:
    manager = _manager(tmp_path)
    adapter = SessionStateSubstrateAdapter(manager)
    driver = SessionStateSubstrateDriver()

    result = driver.prepare(
        adapter._context(operation_id="op-session", parents=()),
        CommandRequest(command="checkpoint", params={"payload": {"label": "session S7"}}),
    )

    assert isinstance(driver, SubstrateDriver)
    assert not hasattr(driver, "manager")
    assert not hasattr(driver, "store")
    assert result.transitions[0].semantic_op == "checkpoint"
    assert result.transitions[0].payload == {
        "schema": SESSION_STATE_REVISION_SCHEMA,
        "label": "session S7",
    }


def test_task_trace_driver_creates_provider_neutral_checkpoints_and_candidates(tmp_path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)

    t0 = trace.create_checkpoint(
        "refs/heads/parent",
        {
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T0",
        },
        operation_id="op-checkpoint-trace",
    )
    t1 = trace.create_candidate(
        operation_id="op-append-trace",
        payload={
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T1",
            "child_world": "1" * 40,
        },
        parents=(t0,),
    )

    trace_head = trace.head(t1.candidate.head)
    plan = trace.plan_candidate_selection(t1)
    checkpoint_provenance = manager.store("store_trace").validate_prepared_revision(
        t0,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    candidate_provenance = manager.store("store_trace").validate_prepared_candidate(
        t1.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert trace_head.role == "shepherd.TraceState"
    assert trace_head.binding == "trace"
    assert trace_head.kind == "shepherd.trace"
    assert plan.operation_id == "op-append-trace"
    assert plan.selection.candidate == t1.candidate
    assert checkpoint_provenance.transition.driver == "shepherd.task_trace"
    assert checkpoint_provenance.transition.semantic_op == "checkpoint"
    assert candidate_provenance.transition.driver == "shepherd.task_trace"
    assert candidate_provenance.transition.semantic_op == "task-trace-append"
    assert candidate_provenance.payload_digest == canonical_digest(
        {
            "schema": TRACE_REVISION_SCHEMA,
            "kind": "shepherd.trace",
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T1",
            "child_world": "1" * 40,
        }
    )


def test_task_trace_driver_prepares_drafts_without_manager(tmp_path) -> None:
    manager = _trace_manager(tmp_path)
    adapter = TaskTraceSubstrateAdapter(manager)
    driver = TaskTraceSubstrateDriver()

    result = driver.prepare(
        adapter._context(operation_id="op-trace", parents=()),
        CommandRequest(
            command="append",
            params={
                "payload": {
                    "trace_runtime": "shepherd.trace.provider-neutral.v1",
                    "trace_owner_id": "shepherd-run:parent",
                    "frontier_id": "frontier:T1",
                }
            },
        ),
    )

    assert isinstance(driver, SubstrateDriver)
    assert not hasattr(driver, "manager")
    assert not hasattr(driver, "store")
    assert result.transitions[0].semantic_op == "task-trace-append"
    assert result.transitions[0].payload == {
        "schema": TRACE_REVISION_SCHEMA,
        "kind": "shepherd.trace",
        "trace_runtime": "shepherd.trace.provider-neutral.v1",
        "trace_owner_id": "shepherd-run:parent",
        "frontier_id": "frontier:T1",
    }


def test_task_trace_candidate_selection_can_publish_selected_trace_head(tmp_path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)
    bundle = trace.create_candidate(
        operation_id="op-select-trace",
        payload={
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T1",
        },
    )
    head = trace.head(bundle.candidate.head)
    plan = trace.plan_candidate_selection(bundle)

    prepared = (
        OperationFinalBuilder("op-select-trace")
        .select_candidate_plan(plan=plan)
        .build_prepared(
            operation_kind="select-trace",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=WorldSnapshot((head,)),
            transition={
                "schema": WORLD_TRANSITION_SCHEMA,
                "operation_id": "op-select-trace",
                "parent_worlds": [],
            },
        )
    )

    assert manager.create_world_from_prepared(prepared)


def test_task_trace_candidate_can_be_archived_without_selecting_trace_head(tmp_path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)
    t0 = trace.create_checkpoint(
        "refs/heads/parent",
        {
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:parent",
            "frontier_id": "frontier:T0",
        },
        operation_id="op-checkpoint-parent-trace",
    )
    discarded = trace.create_candidate(
        operation_id="op-child-discard",
        payload={
            "trace_runtime": "shepherd.trace.provider-neutral.v1",
            "trace_owner_id": "shepherd-run:child",
            "frontier_id": "frontier:T1",
        },
        parents=(t0,),
    )
    selected_head = trace.head(t0)
    existing_plan = manager.plan_existing_head_selection(
        operation_id="op-archive-trace",
        head=selected_head,
        selection_kind="checkpoint",
    )

    prepared = (
        OperationFinalBuilder("op-archive-trace")
        .select_existing(plan=existing_plan)
        .archive_candidate(selection=CandidateSelection.from_bundle(discarded))
        .build_prepared(
            operation_kind="archive-trace",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=WorldSnapshot((selected_head,)),
            transition={
                "schema": WORLD_TRANSITION_SCHEMA,
                "operation_id": "op-archive-trace",
                "parent_worlds": [],
            },
        )
    )

    world_oid = manager.create_world_from_prepared(prepared)
    world = manager.read_world(world_oid)

    assert world.snapshot.head_for("trace").head == t0
    assert prepared.candidate_outcomes[0].outcome == "archived"
    assert prepared.candidate_outcomes[0].candidate == discarded.candidate.head


def test_task_trace_driver_rejects_invalid_payload_shape(tmp_path) -> None:
    manager = _trace_manager(tmp_path)
    trace = TaskTraceSubstrateAdapter(manager)

    with pytest.raises(ValueError, match="trace_runtime"):
        trace.create_checkpoint(
            "refs/heads/missing-runtime",
            {"trace_owner_id": "shepherd-run:parent"},
            operation_id="op-bad-trace-runtime",
        )
    with pytest.raises(ValueError, match="trace_owner_id"):
        trace.create_checkpoint(
            "refs/heads/missing-owner",
            {"trace_runtime": "shepherd.trace.provider-neutral.v1"},
            operation_id="op-bad-trace-owner",
        )
    with pytest.raises(ValueError, match="frontier_id"):
        trace.create_checkpoint(
            "refs/heads/missing-frontier",
            {
                "trace_runtime": "shepherd.trace.provider-neutral.v1",
                "trace_owner_id": "shepherd-run:parent",
            },
            operation_id="op-bad-trace-frontier",
        )
    with pytest.raises(ValueError, match=TRACE_REVISION_SCHEMA):
        trace.create_checkpoint(
            "refs/heads/wrong-schema",
            {
                "schema": "other/trace-revision/v1",
                "kind": "shepherd.trace",
                "trace_runtime": "shepherd.trace.provider-neutral.v1",
                "trace_owner_id": "shepherd-run:parent",
            },
            operation_id="op-bad-trace-schema",
        )
    with pytest.raises(ValueError, match=r"shepherd\.trace"):
        trace.create_checkpoint(
            "refs/heads/wrong-kind",
            {
                "kind": "other.trace",
                "trace_runtime": "shepherd.trace.provider-neutral.v1",
                "trace_owner_id": "shepherd-run:parent",
            },
            operation_id="op-bad-trace-kind",
        )


def test_world_ref_driver_creates_role_aware_candidates_and_import_revisions(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)
    snapshot_digest = manager.read_world(child_world_oid).snapshot.digest()

    candidate = world_ref.create_candidate(
        operation_id="op-parent-link-child",
        world_oid=child_world_oid,
        expected_snapshot_digest=snapshot_digest,
    )
    imported = world_ref.create_import_revision(
        "refs/heads/imported-child",
        operation_id="op-import-child",
        world_oid=child_world_oid,
        expected_snapshot_digest=snapshot_digest,
    )

    candidate_head = world_ref.head(candidate.candidate.head)
    import_head = world_ref.head(imported)
    candidate_provenance = manager.store("store_child_world_ref").validate_prepared_candidate(
        candidate.candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    import_provenance = manager.store("store_child_world_ref").validate_prepared_revision(
        imported,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert candidate_head.role == "vcscore.WorldRef"
    assert candidate_head.kind == WORLD_REF_SUBSTRATE_KIND
    assert import_head.binding == "child"
    assert candidate.candidate_commit.candidate_head == candidate.candidate.head
    assert candidate_provenance.metadata.materialization_class == "internal"
    assert candidate_provenance.transition.driver == "vcscore.world_ref"
    assert import_provenance.transition.driver == "vcscore.world_ref"
    assert candidate_provenance.transition.semantic_op == "world-ref-json-revision"
    assert import_provenance.transition.semantic_op == "import"
    assert candidate_provenance.transition.payload_digest == canonical_digest(
        WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world_oid,
            snapshot_digest=snapshot_digest,
        ).to_json()
    )


def test_world_ref_driver_prepares_drafts_only_through_read_only_context(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    adapter = WorldRefSubstrateAdapter(manager)
    driver = WorldRefSubstrateDriver()
    child_world_oid = _empty_world(manager)
    snapshot_digest = manager.read_world(child_world_oid).snapshot.digest()

    result = driver.prepare(
        adapter._context(operation_id="op-parent-link-child", parents=()),
        CommandRequest(
            command="create-candidate",
            params={
                "world_oid": child_world_oid,
                "expected_snapshot_digest": snapshot_digest,
            },
        ),
    )

    assert isinstance(driver, SubstrateDriver)
    assert not hasattr(driver, "manager")
    assert not hasattr(driver, "store")
    assert result.transitions[0].semantic_op == "world-ref-json-revision"
    assert (
        result.transitions[0].payload
        == WorldRefPayload(
            world_store_id=manager.world_store.world_store_id,
            world_oid=child_world_oid,
            snapshot_digest=snapshot_digest,
        ).to_json()
    )


def test_world_ref_candidate_selection_plan_retains_child_world(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)
    snapshot_digest = manager.read_world(child_world_oid).snapshot.digest()
    bundle = world_ref.create_candidate(
        operation_id="op-parent-link-child",
        world_oid=child_world_oid,
        expected_snapshot_digest=snapshot_digest,
    )
    head = world_ref.head(bundle.candidate.head)
    plan = world_ref.plan_candidate_selection(bundle)
    retention_by_kind = {requirement.kind: requirement for requirement in plan.retention_policy_requirements}

    assert retention_by_kind["selected-head-pin"].target == bundle.candidate.head
    assert retention_by_kind["child-world-retention"].target == f"world:{child_world_oid}"
    assert retention_by_kind["child-world-retention"].digest == snapshot_digest

    prepared = (
        OperationFinalBuilder("op-parent-link-child")
        .select_candidate_plan(plan=plan)
        .build_prepared(
            operation_kind="link-child",
            target_ref=DEFAULT_GROUND_REF,
            input_world_oid=None,
            snapshot=WorldSnapshot((head,)),
            transition={
                "schema": WORLD_TRANSITION_SCHEMA,
                "operation_id": "op-parent-link-child",
                "parent_worlds": [],
            },
        )
    )

    assert manager.create_world_from_prepared(prepared)


def test_world_ref_candidate_selection_plan_preserves_mandatory_retention_with_explicit_policies(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)
    snapshot_digest = manager.read_world(child_world_oid).snapshot.digest()
    bundle = world_ref.create_candidate(
        operation_id="op-parent-link-child-explicit-retention",
        world_oid=child_world_oid,
        expected_snapshot_digest=snapshot_digest,
    )

    plan = world_ref.plan_candidate_selection(
        bundle,
        retention_policy_requirements=(
            RetentionPolicyRequirement(kind="selected-head-pin", target=bundle.candidate.head),
        ),
    )
    retention_by_kind = {requirement.kind: requirement for requirement in plan.retention_policy_requirements}

    assert retention_by_kind["selected-head-pin"].target == bundle.candidate.head
    assert retention_by_kind["child-world-retention"].target == f"world:{child_world_oid}"
    assert retention_by_kind["child-world-retention"].digest == snapshot_digest


def test_candidate_selection_planning_requires_prepared_tuple(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)
    bundle = world_ref.create_candidate(operation_id="op-parent-link-child", world_oid=child_world_oid)

    with pytest.raises(ValueError, match="prepared candidate tuple"):
        CandidateSelection(bundle.candidate, bundle.candidate_commit, None)  # type: ignore[arg-type]


def test_candidate_selection_planning_rejects_invalid_operation_binding(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)
    bundle = world_ref.create_candidate(operation_id="op-child-producer", world_oid=child_world_oid)

    with pytest.raises(InvalidRepositoryStateError, match="role"):
        manager.plan_candidate_selection(
            operation_id="op-child-producer",
            selection=CandidateSelection.from_bundle(bundle),
        )

    with pytest.raises(InvalidRepositoryStateError, match="new-candidate"):
        world_ref.plan_candidate_selection(
            bundle,
            operation_id="op-parent-consumer",
            selection_kind="new-candidate",
            producer_operation_id="op-child-producer",
        )

    with pytest.raises(InvalidRepositoryStateError, match="producer_world_oid"):
        world_ref.plan_candidate_selection(
            bundle,
            operation_id="op-parent-consumer",
            producer_operation_id="op-child-producer",
        )


def test_world_ref_driver_rejects_unresolved_or_mismatched_child_world_payloads(tmp_path) -> None:
    manager = _world_ref_manager(tmp_path)
    world_ref = WorldRefSubstrateAdapter(manager)
    child_world_oid = _empty_world(manager)

    with pytest.raises(InvalidRepositoryStateError, match="expected_snapshot_digest"):
        world_ref.create_candidate(
            operation_id="op-parent-link-wrong-digest",
            world_oid=child_world_oid,
            expected_snapshot_digest=canonical_digest({"snapshot": "wrong"}),
        )
    with pytest.raises(InvalidRepositoryStateError, match="world_store_id"):
        world_ref.create_candidate(
            operation_id="op-parent-link-wrong-store",
            world_oid=child_world_oid,
            world_store_id="store_other",
        )
    with pytest.raises((KeyError, ValueError)):
        world_ref.create_candidate(
            operation_id="op-parent-link-missing-world",
            world_oid="1" * 40,
        )


def test_substrate_adapters_reject_conflicting_payload_schema(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    session = SessionStateSubstrateAdapter(manager)

    with pytest.raises(ValueError, match=WORKSPACE_REVISION_SCHEMA):
        workspace.create_import_revision("refs/heads/main", {"schema": "other/v1"}, operation_id="op-bad-workspace")
    with pytest.raises(ValueError, match=SESSION_STATE_REVISION_SCHEMA):
        session.create_checkpoint("refs/checkpoints/S7", {"schema": "other/v1"}, operation_id="op-bad-session")


def test_manager_prepared_json_candidate_supports_capture_ingress(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = WorkspaceSubstrateAdapter(manager)
    w42 = workspace.create_bootstrap_revision(
        "refs/heads/main",
        {"label": "workspace W42"},
        operation_id="op-bootstrap-workspace",
    )

    candidate, commit = manager.create_prepared_json_candidate(
        "store_workspace",
        operation_id="op-capture",
        binding="workspace",
        payload={"schema": WORKSPACE_REVISION_SCHEMA, "label": "workspace W43"},
        parents=(w42,),
        ingress_kind="capture",
        semantic_op="filesystem-capture",
    )
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        candidate.head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )

    assert commit.candidate_head == candidate.head
    assert provenance.transition.ingress_kind == "capture"
    assert provenance.transition.semantic_op == "filesystem-capture"


def test_role_substrate_adapter_command_wiring() -> None:
    """The three state adapters are RoleSubstrateAdapter leaves whose only
    variation is the driver + the two command-name fields.
    """
    session = SessionStateSubstrateAdapter(manager=None)  # type: ignore[arg-type]
    trace = TaskTraceSubstrateAdapter(manager=None)  # type: ignore[arg-type]
    world_ref = WorldRefSubstrateAdapter(manager=None)  # type: ignore[arg-type]

    for adapter in (session, trace, world_ref):
        assert isinstance(adapter, RoleSubstrateAdapter)

    assert (session.revision_command, session.candidate_command) == ("checkpoint", "create-candidate")
    assert (trace.revision_command, trace.candidate_command) == ("checkpoint", "append")
    assert (world_ref.revision_command, world_ref.candidate_command) == ("import", "create-candidate")

    assert isinstance(session.driver, SessionStateSubstrateDriver)
    assert isinstance(trace.driver, TaskTraceSubstrateDriver)
    assert isinstance(world_ref.driver, WorldRefSubstrateDriver)


def test_role_substrate_adapter_generic_core_builds_revision_and_candidate(tmp_path) -> None:
    """The generic build_revision / build_candidate work when RoleSubstrateAdapter
    is used directly (the path a new in-tree state substrate takes), with the
    command names supplied as fields rather than hard-coded per method.
    """
    manager = _manager(tmp_path)
    adapter = RoleSubstrateAdapter(
        manager=manager,
        driver=SessionStateSubstrateDriver(),
        revision_command="checkpoint",
        candidate_command="create-candidate",
    )

    head = adapter.build_revision(
        "refs/checkpoints/generic",
        operation_id="op-generic-checkpoint",
        params={"payload": {"label": "generic R1"}},
    )
    candidate = adapter.build_candidate(
        operation_id="op-generic-child",
        params={"payload": {"label": "generic C1"}},
        parents=(head,),
    )
    plan = adapter.plan_candidate_selection(candidate)
    resolved_head = adapter.head(candidate.candidate.head)

    assert resolved_head.role == "shepherd.SessionState"
    assert resolved_head.binding == "session"
    assert plan.selection.candidate == candidate.candidate
    revision_provenance = manager.store("store_session").validate_prepared_revision(
        head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    assert revision_provenance.transition.semantic_op == "checkpoint"
