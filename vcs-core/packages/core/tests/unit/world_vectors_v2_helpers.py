"""Test-only helpers for the v2 world-vector capability spikes."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
from dataclasses import replace
from typing import Any

import pygit2
from vcs_core import EvidenceRef
from vcs_core._transition_kernel_records import (
    CandidateCommitRecord,
    CandidateOutcomeRecord,
    EvidenceRecord,
    HeadSelectionEvidence,
    HeadSelectionRecord,
    LogicalTransition,
    PreparedRevisionPlan,
    RetentionPolicyRequirement,
    RevisionPreparationRecord,
    ValidatedPayloadDescriptor,
)
from vcs_core._world_types import CandidateRevision
from vcs_core.spi import RelationshipRequirement

SIG = pygit2.Signature("vcs-core v2 test", "test@example.invalid")
FILEMODE_COMMIT = getattr(pygit2, "GIT_FILEMODE_COMMIT", 0o160000)
CANONICAL_PREFIX = b"vcscore.canonical.v2\n"
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def canonical_bytes(value: object) -> bytes:
    return CANONICAL_PREFIX + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def update_ref(repo: pygit2.Repository, ref: str, oid: pygit2.Oid) -> None:
    repo.references.create(ref, oid, force=True)


def publish_cas(repo: pygit2.Repository, ref: str, target: pygit2.Oid, expected: pygit2.Oid | None) -> bool:
    cmd = ["git", "update-ref", ref, str(target), str(expected) if expected is not None else ""]
    result = subprocess.run(cmd, cwd=repo.path, capture_output=True, check=False, text=True)
    return result.returncode == 0


def commit_json(
    repo: pygit2.Repository,
    ref: str,
    payload: dict[str, Any],
    *,
    parents: tuple[pygit2.Oid, ...] = (),
    message: str | None = None,
) -> pygit2.Oid:
    blob = repo.create_blob(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    tree_builder = repo.TreeBuilder()
    tree_builder.insert("revision.json", blob, pygit2.GIT_FILEMODE_BLOB)
    tree = tree_builder.write()
    oid = repo.create_commit(None, SIG, SIG, message or str(payload.get("label", "revision")), tree, list(parents))
    update_ref(repo, ref, oid)
    return oid


def kind_for_binding(binding: str) -> str:
    if binding.startswith("session"):
        return "shepherd.session_state"
    if binding.startswith("trace"):
        return "shepherd.trace"
    return "filesystem"


def role_for_binding(binding: str) -> str:
    if binding.startswith("session"):
        return "shepherd.SessionState"
    if binding.startswith("trace"):
        return "shepherd.TraceState"
    return "shepherd.WorkspaceRef"


def resource_id_for_binding(binding: str) -> str:
    if binding.startswith("session"):
        return "shepherd-session:child-baseline"
    if binding.startswith("trace"):
        return "shepherd-trace:parent"
    if binding.startswith("workspace"):
        return "fs:repo-main"
    return f"resource:{binding}"


def operation_final_with_head_selections(
    operation_id: str,
    selected: dict[str, str],
    *,
    outcomes: list[dict[str, object]] | None = None,
    candidate_commits: list[CandidateCommitRecord] | None = None,
    store_ids: dict[str, str] | None = None,
    resource_ids: dict[str, str] | None = None,
    selection_kinds: dict[str, str] | None = None,
    relationship_requirements: dict[str, tuple[RelationshipRequirement, ...]] | None = None,
    retention_policy_requirements: dict[str, tuple[RetentionPolicyRequirement, ...]] | None = None,
) -> dict[str, object]:
    resolved_store_ids = store_ids or {}
    resolved_resource_ids = resource_ids or {}
    resolved_selection_kinds = selection_kinds or {}
    resolved_relationship_requirements = relationship_requirements or {}
    resolved_retention_policy_requirements = retention_policy_requirements or {}
    resolved_outcomes = outcomes or []
    resolved_candidate_commits = candidate_commits or []
    commits_by_selected_binding = {
        commit.binding: commit
        for commit in resolved_candidate_commits
        if any(
            outcome.get("binding") == commit.binding
            and outcome.get("candidate") == commit.candidate_head
            and outcome.get("outcome") == "selected"
            for outcome in resolved_outcomes
        )
    }
    selected_outcomes_by_binding = {
        str(outcome["binding"]): outcome
        for outcome in resolved_outcomes
        if outcome.get("outcome") == "selected" and isinstance(outcome.get("binding"), str)
    }
    head_selections: list[dict[str, object]] = []
    selection_evidence: list[dict[str, object]] = []
    for binding, head in sorted(selected.items()):
        store_id = resolved_store_ids.get(binding, f"store_{binding}")
        resource_id = resolved_resource_ids.get(binding, resource_id_for_binding(binding))
        commit = commits_by_selected_binding.get(binding)
        outcome = selected_outcomes_by_binding.get(binding)
        selection_kind = resolved_selection_kinds.get(binding)
        if selection_kind is None:
            selection_kind = "unchanged" if commit is None else _selection_kind_for_candidate(operation_id, outcome)
        selection = HeadSelectionRecord(
            binding=binding,
            store_id=store_id,
            resource_id=resource_id,
            selected_head=head,
            selection_kind=selection_kind,  # type: ignore[arg-type]
            relationship_requirements=resolved_relationship_requirements.get(binding, ()),
            retention_policy_requirements=resolved_retention_policy_requirements.get(
                binding,
                (RetentionPolicyRequirement(kind="selected-head-pin", target=head),),
            ),
            selection_policy_digest=canonical_digest({"selection": binding, "head": head}),
        )
        evidence = HeadSelectionEvidence(
            operation_id=operation_id,
            binding=binding,
            store_id=store_id,
            resource_id=resource_id,
            selected_head=head,
            selection_digest=selection.selection_digest(),
            revision_preparation_digest=commit.revision_preparation_digest if commit is not None else None,
            candidate_commit_digest=commit.candidate_commit_digest() if commit is not None else None,
            candidate_ref=commit.candidate_ref if commit is not None else None,
            producer_operation_id=_producer_operation_id(operation_id, outcome) if commit is not None else None,
            retention_policy_requirements=selection.retention_policy_requirements,
        )
        head_selections.append(selection.to_json())
        selection_evidence.append(evidence.to_json())
    return {
        "schema": "vcscore/operation-final/v2",
        "operation_id": operation_id,
        "candidate_outcomes": resolved_outcomes,
        "candidate_commits": [commit.to_json() for commit in resolved_candidate_commits],
        "selected": selected,
        "head_selections": head_selections,
        "selection_evidence": selection_evidence,
    }


def selection_evidence_ref(
    world_store,
    *,
    operation_id: str,
    binding: str,
    store,
    head: str,
    evidence_kind: str,
    selected_from: str | None = None,
) -> EvidenceRef:
    stable_observation = {
        "binding": binding,
        "store_id": store.identity.store_id,
        "resource_id": store.identity.resource_id,
        "substrate_kind": store.identity.kind,
        "head": head,
        "kind": evidence_kind,
    }
    if selected_from is not None:
        stable_observation["selected_from"] = selected_from
    return world_store.store_evidence_record(
        EvidenceRecord(
            operation_id=operation_id,
            binding=binding,
            store_id=store.identity.store_id,
            substrate_kind=store.identity.kind,
            ingress_kind="coordinator",
            observed_head=head,
            evidence_kind=evidence_kind,
            payload_digest=canonical_digest(stable_observation),
            stable_observation=stable_observation,
        )
    )


def attach_selection_evidence_ref(
    final: dict[str, object],
    *,
    binding: str,
    evidence_ref: EvidenceRef,
    selected_from: str | None = None,
) -> dict[str, object]:
    selections = final["head_selections"]
    evidences = final["selection_evidence"]
    assert isinstance(selections, list)
    assert isinstance(evidences, list)
    for index, item in enumerate(selections):
        selection = HeadSelectionRecord.from_json(item)
        if selection.binding != binding:
            continue
        updated_selection = replace(selection, selected_from=selected_from)
        selections[index] = updated_selection.to_json()
        evidence = HeadSelectionEvidence.from_json(evidences[index])
        evidences[index] = replace(
            evidence,
            selection_digest=updated_selection.selection_digest(),
            evidence_refs=(evidence_ref,),
        ).to_json()
        return final
    raise AssertionError(f"missing selection for binding {binding!r}")


def create_prepared_candidate(
    store,
    *,
    operation_id: str,
    binding: str,
    candidate_id: str = "primary",
    payload: dict[str, object],
    parents: tuple[str, ...] = (),
    driver: str = "test.driver",
    semantic_op: str = "test-op",
    world_store=None,
    relationship_requirements: tuple[RelationshipRequirement, ...] = (),
) -> tuple[CandidateRevision, CandidateCommitRecord]:
    evidence_record = _evidence_record(
        operation_id=operation_id,
        binding=binding,
        store_id=store.identity.store_id,
        substrate_kind=store.identity.kind,
        payload_digest=canonical_digest(payload),
    )
    evidence = (
        world_store.store_evidence_record(evidence_record)
        if world_store is not None
        else _evidence_ref_from_record(evidence_record)
    )
    transition = LogicalTransition(
        binding=binding,
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        substrate_kind=store.identity.kind,
        driver=driver,
        driver_version="test",
        base_heads=parents,
        ingress_kind="command",
        semantic_op=semantic_op,
        payload_digest=canonical_digest(payload),
        evidence_digests=(evidence.evidence_digest,),
        requirements=relationship_requirements,
    )
    plan = PreparedRevisionPlan(
        binding=binding,
        store_id=store.identity.store_id,
        transition_digest=transition.transition_digest(),
        base_heads=transition.base_heads,
        expected_parent_heads=parents,
        content_digest=canonical_digest(payload),
        materialization_class="external",
        entries=({"path": "revision.json", "payload_digest": canonical_digest(payload)},),
    )
    preparation = RevisionPreparationRecord(
        operation_id=operation_id,
        binding=binding,
        store_id=store.identity.store_id,
        resource_id=store.identity.resource_id,
        transition_digest=transition.transition_digest(),
        revision_plan_digest=plan.revision_plan_digest(),
        content_digest=plan.content_digest,
        evidence_digests=transition.evidence_digests,
        evidence_refs=(evidence,),
        relationship_requirements=relationship_requirements,
    )
    evidence_resolver = world_store.resolve_evidence_ref if world_store is not None else lambda _ref: evidence_record
    candidate = store.create_candidate_from_prepared(
        transition=transition,
        plan=plan,
        preparation=preparation,
        payload_descriptor=ValidatedPayloadDescriptor.for_json_payload(payload),
        payload=payload,
        candidate_id=candidate_id,
        parents=parents,
        evidence_resolver=evidence_resolver,
    )
    return candidate, store.candidate_commit_record(candidate, evidence_resolver=evidence_resolver)


def candidate_outcome_for_commit(
    store,
    candidate_commit: CandidateCommitRecord,
    *,
    final_operation_id: str,
    world_store,
    outcome: str = "selected",
    producer_world_oid: str | None = None,
) -> dict[str, object]:
    producer_operation_id = candidate_commit.operation_id
    provenance = store.validate_prepared_candidate(
        candidate_commit.candidate_head,
        expected_revision_preparation_digest=candidate_commit.revision_preparation_digest,
        evidence_resolver=lambda evidence_ref: world_store.resolve_evidence_ref(
            evidence_ref,
            expected_operation_id=producer_operation_id,
        ),
    )
    return CandidateOutcomeRecord(
        binding=candidate_commit.binding,
        candidate=candidate_commit.candidate_head,
        outcome=outcome,  # type: ignore[arg-type]
        candidate_id=candidate_commit.candidate_id,
        store_id=candidate_commit.store_id,
        resource_id=candidate_commit.resource_id,
        transition_digest=provenance.transition.transition_digest(),
        revision_plan_digest=provenance.plan.revision_plan_digest(),
        content_digest=provenance.plan.content_digest,
        revision_preparation_digest=provenance.preparation.revision_preparation_digest(),
        candidate_commit_digest=candidate_commit.candidate_commit_digest(),
        evidence_digests=provenance.preparation.evidence_digests,
        producer_operation_id=producer_operation_id if producer_operation_id != final_operation_id else None,
        producer_world_oid=producer_world_oid,
        evidence_refs=provenance.preparation.evidence_refs,
    ).to_json(final_operation_id=final_operation_id)


def _selection_kind_for_candidate(operation_id: str, outcome: dict[str, object] | None) -> str:
    producer_operation_id = _producer_operation_id(operation_id, outcome)
    if producer_operation_id != operation_id or _producer_world_oid(outcome) is not None:
        return "child-produced"
    return "new-candidate"


def _producer_operation_id(operation_id: str, outcome: dict[str, object] | None) -> str:
    if outcome is None:
        return operation_id
    raw = outcome.get("producer_operation_id", operation_id)
    return raw if isinstance(raw, str) else operation_id


def _producer_world_oid(outcome: dict[str, object] | None) -> str | None:
    if outcome is None:
        return None
    raw = outcome.get("producer_world_oid")
    return raw if isinstance(raw, str) else None


def _evidence_record(
    *,
    operation_id: str,
    binding: str,
    store_id: str,
    substrate_kind: str,
    payload_digest: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        operation_id=operation_id,
        binding=binding,
        store_id=store_id,
        substrate_kind=substrate_kind,
        ingress_kind="command",
        evidence_kind="command_envelope",
        payload_digest=payload_digest,
        stable_observation={"command": "write", "binding": binding},
    )


def _evidence_ref_from_record(record: EvidenceRecord) -> EvidenceRef:
    return EvidenceRef(
        ref=f"refs/vcscore/evidence/{record.operation_id}/{record.binding}",
        evidence_digest=record.evidence_digest(),
        record_digest=record.record_digest(),
        payload_digest=record.payload_digest,
    )


def world_snapshot(
    heads: dict[str, pygit2.Oid],
    *,
    store_ids: dict[str, str] | None = None,
    resource_ids: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
    roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved_store_ids = store_ids or {}
    resolved_resource_ids = resource_ids or {}
    resolved_kinds = kinds or {}
    resolved_roles = roles or {}
    snapshot: dict[str, dict[str, str]] = {}
    locator_hints: dict[str, str] = {}
    for binding, oid in sorted(heads.items()):
        store_id = resolved_store_ids.get(binding, f"store_{binding}")
        snapshot[binding] = {
            "binding": binding,
            "kind": resolved_kinds.get(binding, kind_for_binding(binding)),
            "role": resolved_roles.get(binding, role_for_binding(binding)),
            "store_id": store_id,
            "store_scope": "resource",
            "resource_id": resolved_resource_ids.get(binding, f"resource:{binding}"),
            "head": str(oid),
            "object_format": "sha1",
        }
        locator_hints[store_id] = f"substrates/{binding}.git"

    return {
        "schema": "vcscore/world/v2",
        "snapshot": snapshot,
        "locator_hints": locator_hints,
    }


def world_commit(
    repo: pygit2.Repository,
    ref: str | None,
    heads: dict[str, pygit2.Oid],
    transition: dict[str, Any],
    *,
    parents: tuple[pygit2.Oid, ...] = (),
    operation_final: dict[str, Any] | None = None,
    include_gitlinks: bool = True,
    gitlink_heads: dict[str, pygit2.Oid] | None = None,
    snapshot_store_ids: dict[str, str] | None = None,
    snapshot_resource_ids: dict[str, str] | None = None,
    snapshot_kinds: dict[str, str] | None = None,
    snapshot_roles: dict[str, str] | None = None,
) -> pygit2.Oid:
    final_record = operation_final or operation_final_with_head_selections(
        str(transition["operation_id"]),
        {binding: str(head) for binding, head in sorted(heads.items())},
    )
    final_record_digest = canonical_digest(final_record)
    transition = {
        **transition,
        "operation_final": {
            "path": "meta/operation-final.json",
            "digest": final_record_digest,
        },
    }

    meta_builder = repo.TreeBuilder()
    meta_builder.insert(
        "world.json",
        repo.create_blob(
            json.dumps(
                world_snapshot(
                    heads,
                    store_ids=snapshot_store_ids,
                    resource_ids=snapshot_resource_ids,
                    kinds=snapshot_kinds,
                    roles=snapshot_roles,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        pygit2.GIT_FILEMODE_BLOB,
    )
    meta_builder.insert(
        "transition.json",
        repo.create_blob(json.dumps(transition, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        pygit2.GIT_FILEMODE_BLOB,
    )
    meta_builder.insert(
        "operation-final.json",
        repo.create_blob(canonical_bytes(final_record)),
        pygit2.GIT_FILEMODE_BLOB,
    )
    meta_tree = meta_builder.write()

    root_builder = repo.TreeBuilder()
    root_builder.insert("meta", meta_tree, pygit2.GIT_FILEMODE_TREE)
    if include_gitlinks:
        substrates_builder = repo.TreeBuilder()
        for binding, head in sorted((gitlink_heads or heads).items()):
            substrates_builder.insert(binding, head, FILEMODE_COMMIT)
        root_builder.insert("substrates", substrates_builder.write(), pygit2.GIT_FILEMODE_TREE)
    root_tree = root_builder.write()
    oid = repo.create_commit(None, SIG, SIG, str(transition["operation_id"]), root_tree, list(parents))
    if ref is not None:
        update_ref(repo, ref, oid)
    return oid


def read_blob_bytes(repo: pygit2.Repository, tree: pygit2.Tree, path: str) -> bytes:
    entry: pygit2.Tree | pygit2.Blob = tree
    for component in path.split("/"):
        entry = repo[entry[component].id]  # type: ignore[index,assignment]
    if not isinstance(entry, pygit2.Blob):
        raise TypeError(f"{path!r} did not resolve to a blob")
    return bytes(entry.data)


def read_json_blob(repo: pygit2.Repository, tree: pygit2.Tree, path: str) -> dict[str, Any]:
    return json.loads(read_blob_bytes(repo, tree, path).decode("utf-8"))


def measure_ref_resolution(repo: pygit2.Repository, ref: str, *, iterations: int = 100) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=repo.path,
            capture_output=True,
            check=True,
            text=True,
        )
    return time.perf_counter() - started


def store_identity(
    *,
    store_id: str = "store_workspace",
    kind: str = "filesystem",
    resource_id: str = "fs:repo-main",
    object_format: str = "sha1",
) -> dict[str, Any]:
    return {
        "schema": "vcscore/substrate-store-identity/v1",
        "store_id": store_id,
        "kind": kind,
        "resource_id": resource_id,
        "object_format": object_format,
        "created_at_unix_ns": 1_789_150_000_000_000_000,
    }


def validate_bound_store(manifest_head: dict[str, Any], identity: dict[str, Any]) -> None:
    for key in ("store_id", "kind", "resource_id", "object_format"):
        if manifest_head[key] != identity[key]:
            raise ValueError(f"substrate identity mismatch for {key}")


def validate_world_bindings(
    manifest: dict[str, Any],
    identities_by_store_id: dict[str, dict[str, Any]],
    *,
    allow_same_resource_alias: bool = False,
) -> None:
    seen_store_resources: dict[str, str] = {}
    seen_resource_stores: dict[str, str] = {}
    for binding, head in manifest["snapshot"].items():
        store_id = head["store_id"]
        try:
            identity = identities_by_store_id[store_id]
        except KeyError as exc:
            raise ValueError(f"missing substrate identity for {binding}") from exc
        validate_bound_store(head, identity)

        prior_resource = seen_store_resources.setdefault(store_id, head["resource_id"])
        if prior_resource != head["resource_id"]:
            raise ValueError("distinct resource_id bindings must not share one substrate store")

        prior_store = seen_resource_stores.setdefault(head["resource_id"], store_id)
        if prior_store != store_id:
            raise ValueError("one resource_id must not resolve to multiple substrate stores")
        if prior_store == store_id and head["resource_id"] in seen_resource_stores:
            aliases = [h for h in manifest["snapshot"].values() if h["resource_id"] == head["resource_id"]]
            if len(aliases) > 1 and not allow_same_resource_alias:
                raise ValueError("same-resource aliases require explicit coordinator policy")


def validate_optional_gitlinks(repo: pygit2.Repository, commit: pygit2.Commit) -> None:
    manifest = read_json_blob(repo, commit.tree, "meta/world.json")
    try:
        substrates_entry = commit.tree["substrates"]
    except KeyError:
        return
    substrates_tree = repo[substrates_entry.id]
    if not isinstance(substrates_tree, pygit2.Tree):
        raise TypeError("substrates entry is not a tree")

    for entry in substrates_tree:
        if entry.name not in manifest["snapshot"]:
            raise ValueError(f"unexpected gitlink for {entry.name}")
        if entry.filemode != FILEMODE_COMMIT:
            raise ValueError(f"gitlink for {entry.name} has wrong file mode")
        if str(entry.id) != manifest["snapshot"][entry.name]["head"]:
            raise ValueError(f"gitlink for {entry.name} disagrees with manifest")


def is_ref_safe_component(component: str) -> bool:
    return (
        bool(component)
        and _SAFE_COMPONENT_RE.fullmatch(component) is not None
        and component not in {".", ".."}
        and not component.startswith(".")
        and ".." not in component
        and "@{" not in component
        and not component.endswith(".lock")
    )


def encode_ref_component(value: str, *, max_component_length: int = 96) -> str:
    raw = value.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    candidate = f"b64u_{encoded}"
    if is_ref_safe_component(candidate) and len(candidate) <= max_component_length:
        return candidate
    digest = hashlib.sha256(raw).hexdigest()
    return f"sha256_{digest}"


def candidate_ref(operation_id: str, binding: str) -> str:
    return f"refs/vcscore/candidates/{encode_ref_component(operation_id)}/{encode_ref_component(binding)}"


def validate_candidate_ref(
    repo: pygit2.Repository,
    *,
    operation_id: str,
    binding: str,
    expected_head: pygit2.Oid,
) -> None:
    ref = candidate_ref(operation_id, binding)
    try:
        target = repo.references[ref].target
    except KeyError as exc:
        raise ValueError("operation record names a candidate without a durable candidate ref") from exc
    if target != expected_head:
        raise ValueError("candidate ref target disagrees with operation record")
