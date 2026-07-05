"""Git-backed substrate store for v2 world storage."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pygit2

from vcs_core._errors import InvalidRepositoryStateError
from vcs_core._evidence_validation import EvidenceCitationScope, validate_preparation_evidence_refs
from vcs_core._pygit2_helpers import require_blob, require_commit, require_tree
from vcs_core._substrate_driver import KeyedJsonTreeDraft, RevisionContentDraft
from vcs_core._world_refs import candidate_archive_ref, candidate_ref, world_pin_ref
from vcs_core._world_types import (
    CandidateRevision,
    SubstrateHead,
    SubstrateRevisionMetadata,
    SubstrateStoreIdentity,
    canonical_bytes,
    canonical_digest,
    compact_json_bytes,
    load_canonical_json,
)
from vcs_core.git_store import build_tree, create_commit_with_recovery, create_or_update_reference, insert_tree_entry

if TYPE_CHECKING:
    from collections.abc import Callable

    from vcs_core._transition_kernel_records import (
        CandidateCommitRecord,
        EvidenceRecord,
        EvidenceRef,
        LogicalTransition,
        PreparedRevisionPlan,
        RevisionPreparationRecord,
        ValidatedPayloadDescriptor,
    )

    EvidenceResolver = Callable[[EvidenceRef], EvidenceRecord]

IDENTITY_REF = "refs/vcscore/identity"
SUBSTRATE_REVISION_METADATA_PATH = "meta/substrate-revision.json"
PAYLOAD_DESCRIPTOR_PATH = "meta/payload-descriptor.json"
STRUCTURED_CONTENT_DESCRIPTOR_PATH = "meta/structured-content.json"
STRUCTURED_CONTENT_DESCRIPTOR_SCHEMA = "vcscore/structured-content/v1"


@dataclass(frozen=True)
class PreparedCandidateProvenance:
    """Validated transition-kernel sidecars for one prepared candidate commit."""

    head: str
    metadata: SubstrateRevisionMetadata
    transition: LogicalTransition
    plan: PreparedRevisionPlan
    preparation: RevisionPreparationRecord
    payload_descriptor: ValidatedPayloadDescriptor
    payload_digest: str
    parent_heads: tuple[str, ...]


class SubstrateStore:
    """Git repository that owns revisions for one stable substrate resource."""

    def __init__(self, repo_path: str, identity: SubstrateStoreIdentity, *, repo: pygit2.Repository) -> None:
        self._repo_path = repo_path
        self._repo = repo
        self._identity = identity
        self._mutation_lock = threading.RLock()

    @classmethod
    def open_or_init(
        cls,
        repo_path: str | Path,
        identity: SubstrateStoreIdentity,
        *,
        shared_object_repo_path: str | Path | None = None,
    ) -> SubstrateStore:
        """Open or initialize a bare substrate repo and verify its stable identity.

        When ``shared_object_repo_path`` is provided, the substrate's
        ``objects/info/alternates`` is configured to make objects in the
        coordinator (or any sibling) ODB visible from this store. This is the
        cross-store object-availability path that lets tree-backed workspace
        revisions reference Git tree/blob objects already materialized by the
        coordinator without duplicating them.
        """
        path = Path(repo_path)
        repo = _open_or_init_bare_repo(path)
        if shared_object_repo_path is not None:
            repo = _ensure_alternates(repo, path, Path(shared_object_repo_path))
        store = cls(str(path), identity, repo=repo)
        try:
            existing_identity = store.read_identity()
        except KeyError:
            store._write_identity(identity)
        else:
            if existing_identity != identity:
                raise InvalidRepositoryStateError(
                    f"Substrate store identity mismatch for {path}: "
                    f"expected {identity.to_json()}, found {existing_identity.to_json()}."
                )
        return store

    @classmethod
    def open_existing(
        cls,
        repo_path: str | Path,
        identity: SubstrateStoreIdentity,
        *,
        shared_object_repo_path: str | Path | None = None,
    ) -> SubstrateStore:
        """Open an existing bare substrate repo and verify its stable identity.

        See :meth:`open_or_init` for the ``shared_object_repo_path`` contract.
        """
        path = Path(repo_path)
        repo = _open_existing_bare_repo(path)
        if shared_object_repo_path is not None:
            repo = _ensure_alternates(repo, path, Path(shared_object_repo_path))
        store = cls(str(path), identity, repo=repo)
        try:
            existing_identity = store.read_identity()
        except KeyError as exc:
            raise InvalidRepositoryStateError(f"Substrate store at {path} is missing its identity ref") from exc
        if existing_identity != identity:
            raise InvalidRepositoryStateError(
                f"Substrate store identity mismatch for {path}: "
                f"expected {identity.to_json()}, found {existing_identity.to_json()}."
            )
        return store

    @property
    def repo_path(self) -> str:
        return self._repo_path

    @property
    def repo(self) -> pygit2.Repository:
        return self._repo

    @property
    def identity(self) -> SubstrateStoreIdentity:
        return self._identity

    def read_identity(self) -> SubstrateStoreIdentity:
        """Read the persisted substrate store identity record."""
        if IDENTITY_REF not in self._repo.references:
            raise KeyError(IDENTITY_REF)
        commit = self._repo.references[IDENTITY_REF].peel(pygit2.Commit)
        tree = require_tree(self._repo, commit.tree.id, context="substrate identity tree")
        entry = tree["identity.json"]
        blob = require_blob(self._repo, entry.id, context="substrate identity blob")
        value = json.loads(bytes(blob.data).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("substrate identity must be a JSON object")
        return SubstrateStoreIdentity.from_json(value)

    def contains(self, head: str | SubstrateHead) -> bool:
        """Return true when ``head`` is a commit present in this substrate repo."""
        oid = _head_oid(head)
        obj = self._repo.get(oid)
        return isinstance(obj, pygit2.Commit)

    def substrate_head(self, *, binding: str, head: str, role: str, store_scope: str = "resource") -> SubstrateHead:
        """Build a selected-head DTO for a commit already owned by this store."""
        if not self.contains(head):
            raise KeyError(f"substrate head {head!r} is not present in store {self.identity.store_id!r}")
        return SubstrateHead(
            binding=binding,
            kind=self.identity.kind,
            role=role,
            store_id=self.identity.store_id,
            store_scope=store_scope,
            resource_id=self.identity.resource_id,
            head=head,
            object_format=self.identity.object_format,
        )

    def create_unsafe_unprepared_json_revision(
        self,
        ref: str,
        payload: dict[str, Any],
        *,
        parents: tuple[str | pygit2.Oid, ...] = (),
        message: str | None = None,
    ) -> str:
        """Create a provenance-free JSON revision for tests and migration tools."""
        with self._mutation_lock:
            oid = self._create_json_commit(payload, parents=parents, message=message)
            create_or_update_reference(self._repo, ref, oid, force=True)
            return str(oid)

    def create_unsafe_unprepared_candidate(
        self,
        *,
        operation_id: str,
        binding: str,
        candidate_id: str = "primary",
        payload: dict[str, Any],
        parents: tuple[str | pygit2.Oid, ...] = (),
        message: str | None = None,
    ) -> CandidateRevision:
        """Create a legacy candidate ref without full transition-kernel sidecars."""
        ref = candidate_ref(operation_id, binding, candidate_id)
        with self._mutation_lock:
            if ref in self._repo.references:
                raise InvalidRepositoryStateError(f"Candidate ref already exists: {ref}")
            oid = self._create_json_commit(
                payload,
                parents=parents,
                message=message,
                produced_by_operation_id=operation_id,
            )
            create_or_update_reference(self._repo, ref, oid)
            return CandidateRevision(
                operation_id=operation_id,
                binding=binding,
                candidate_id=candidate_id,
                store_id=self.identity.store_id,
                resource_id=self.identity.resource_id,
                head=str(oid),
                ref=ref,
            )

    def create_candidate_from_prepared(
        self,
        *,
        transition: LogicalTransition,
        plan: PreparedRevisionPlan,
        preparation: RevisionPreparationRecord,
        payload_descriptor: ValidatedPayloadDescriptor,
        payload: dict[str, Any],
        content: RevisionContentDraft | None = None,
        candidate_id: str = "primary",
        parents: tuple[str | pygit2.Oid, ...] = (),
        message: str | None = None,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> CandidateRevision:
        """Create a JSON candidate from typed transition-kernel preparation records."""
        self._validate_prepared_candidate_inputs(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            content=content,
            parents=parents,
            evidence_resolver=evidence_resolver,
        )
        ref = candidate_ref(preparation.operation_id, preparation.binding, candidate_id)
        with self._mutation_lock:
            if ref in self._repo.references:
                raise InvalidRepositoryStateError(f"Candidate ref already exists: {ref}")
            oid = self._create_json_commit(
                payload,
                content=content,
                parents=parents,
                message=message,
                produced_by_operation_id=preparation.operation_id,
                transition=transition,
                plan=plan,
                preparation=preparation,
                payload_descriptor=payload_descriptor,
            )
            create_or_update_reference(self._repo, ref, oid)
            return CandidateRevision(
                operation_id=preparation.operation_id,
                binding=preparation.binding,
                candidate_id=candidate_id,
                store_id=self.identity.store_id,
                resource_id=self.identity.resource_id,
                head=str(oid),
                ref=ref,
            )

    def plan_revision_content(
        self,
        content: RevisionContentDraft | None,
        *,
        payload_digest: str,
        parents: tuple[str | pygit2.Oid, ...],
    ) -> tuple[str, tuple[dict[str, object], ...]]:
        """Return the semantic content digest and bounded plan entries for a revision draft."""
        entries: list[dict[str, object]] = [{"path": "revision.json", "payload_digest": payload_digest}]
        if content is None:
            return payload_digest, tuple(entries)
        if isinstance(content, KeyedJsonTreeDraft):
            parent_oids = [_coerce_oid(parent) for parent in parents]
            descriptor = self._structured_content_descriptor(content, parent_oids=parent_oids)
            entries.append(
                {
                    "path": STRUCTURED_CONTENT_DESCRIPTOR_PATH,
                    "payload_digest": canonical_digest(descriptor),
                }
            )
            for item in content.puts:
                entries.append(
                    {
                        "path": f"{content.content_root}/{item.path}",
                        "payload_digest": canonical_digest(item.payload),
                        "key": item.key,
                    }
                )
            for path in content.deletes:
                entries.append({"path": f"{content.content_root}/{path}", "state": "deleted"})
            return canonical_digest(descriptor), tuple(entries)
        raise InvalidRepositoryStateError(f"unsupported structured content draft: {type(content).__name__}")

    def create_revision_from_prepared(
        self,
        ref: str,
        *,
        transition: LogicalTransition,
        plan: PreparedRevisionPlan,
        preparation: RevisionPreparationRecord,
        payload_descriptor: ValidatedPayloadDescriptor,
        payload: dict[str, Any],
        content: RevisionContentDraft | None = None,
        parents: tuple[str | pygit2.Oid, ...] = (),
        message: str | None = None,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> str:
        """Create a non-candidate JSON revision from typed transition-kernel records."""
        self._validate_prepared_candidate_inputs(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            content=content,
            parents=parents,
            evidence_resolver=evidence_resolver,
        )
        with self._mutation_lock:
            oid = self._create_json_commit(
                payload,
                content=content,
                parents=parents,
                message=message,
                produced_by_operation_id=preparation.operation_id,
                transition=transition,
                plan=plan,
                preparation=preparation,
                payload_descriptor=payload_descriptor,
            )
            create_or_update_reference(self._repo, ref, oid, force=True)
            return str(oid)

    def validate_candidate_ref(
        self,
        *,
        operation_id: str,
        binding: str,
        candidate_id: str = "primary",
        expected_head: str,
    ) -> None:
        """Reject operation metadata that names a candidate without its durable candidate ref."""
        ref = candidate_ref(operation_id, binding, candidate_id)
        try:
            target = self._repo.references[ref].target
        except KeyError as exc:
            raise InvalidRepositoryStateError(
                "operation record names a candidate without a durable candidate ref"
            ) from exc
        if str(target) != expected_head:
            raise InvalidRepositoryStateError("candidate ref target disagrees with operation record")

    def pin_world_head(self, *, world_store_id: str, world_oid: str, binding: str, head: str) -> str:
        """Protect a selected substrate head for the lifetime of a reachable world commit."""
        if not self.contains(head):
            raise KeyError(f"substrate head {head!r} is not present in store {self.identity.store_id!r}")
        ref = world_pin_ref(world_store_id, world_oid, binding)
        create_or_update_reference(self._repo, ref, pygit2.Oid(hex=head), force=True)
        return ref

    def archive_candidate(self, *, operation_id: str, binding: str, head: str, candidate_id: str = "primary") -> str:
        """Retain an unselected candidate as operation evidence."""
        if not self.contains(head):
            raise KeyError(f"substrate head {head!r} is not present in store {self.identity.store_id!r}")
        ref = candidate_archive_ref(operation_id, binding, candidate_id)
        oid = pygit2.Oid(hex=head)
        if ref in self._repo.references:
            if self._repo.references[ref].target != oid:
                raise InvalidRepositoryStateError(f"Archive ref already exists for a different head: {ref}")
            return ref
        create_or_update_reference(self._repo, ref, oid)
        return ref

    def read_revision_payload(self, oid: str) -> dict[str, object]:
        """Read one revision's JSON payload, digest-validated against its metadata.

        The read-back half of the selectable dispatch route (B4b W2): consumers
        read a selected head's payload field-complete; the metadata
        ``payload_digest`` check makes a torn or tampered payload loud.
        """
        metadata = self.read_revision_metadata(oid)
        commit = require_commit(self._repo, pygit2.Oid(hex=oid), context="substrate revision")
        tree = require_tree(self._repo, commit.tree.id, context="substrate revision tree")
        revision_blob = require_blob(self._repo, tree["revision.json"].id, context="substrate revision payload blob")
        payload = json.loads(bytes(revision_blob.data).decode("utf-8"))
        assert isinstance(payload, dict)  # read_revision_metadata validated shape + digest
        del metadata
        return payload

    def read_revision_manifest(self, oid: str) -> dict[str, object]:
        """Read one revision's ``revision.json`` manifest.

        For JSON-snapshot revisions this is the complete payload. For
        structured revisions it is only the small manifest/envelope; callers
        should use addressable entry readers for records under ``data/``.
        """
        return self.read_revision_payload(oid)

    def read_revision_entry(self, oid: str, path: str) -> bytes | None:
        """Read one blob path from a revision commit after metadata validation."""
        _validate_read_path(path)
        self.read_revision_metadata(oid)
        commit = require_commit(self._repo, pygit2.Oid(hex=oid), context="substrate revision")
        tree = require_tree(self._repo, commit.tree.id, context="substrate revision tree")
        return _read_optional_blob_bytes(self._repo, tree, path)

    def read_revision_json_entry(self, oid: str, path: str) -> dict[str, object] | None:
        """Read one JSON-object blob path from a revision commit."""
        raw = self.read_revision_entry(oid, path)
        if raw is None:
            return None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise InvalidRepositoryStateError(f"revision entry {path!r} is not a JSON object")
        return value

    def read_revision_json_entries(self, oid: str, prefix: str) -> tuple[tuple[str, dict[str, object]], ...]:
        """Read JSON-object blobs under ``prefix`` from a revision commit."""
        _validate_read_path(prefix)
        self.read_revision_metadata(oid)
        commit = require_commit(self._repo, pygit2.Oid(hex=oid), context="substrate revision")
        tree = require_tree(self._repo, commit.tree.id, context="substrate revision tree")
        rows: list[tuple[str, dict[str, object]]] = []
        for path, raw in _iter_blob_bytes(self._repo, tree, prefix):
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise InvalidRepositoryStateError(f"revision entry {path!r} is not a JSON object")
            rows.append((path, value))
        return tuple(sorted(rows, key=lambda item: item[0]))

    def read_revision_metadata(self, oid: str) -> SubstrateRevisionMetadata:
        """Read and validate the private minimum metadata record for one revision."""
        commit = require_commit(self._repo, pygit2.Oid(hex=oid), context="substrate revision")
        tree = require_tree(self._repo, commit.tree.id, context="substrate revision tree")
        try:
            revision_entry = tree["revision.json"]
        except KeyError as exc:
            raise InvalidRepositoryStateError("substrate revision is missing revision.json") from exc
        revision_blob = require_blob(self._repo, revision_entry.id, context="substrate revision payload blob")
        payload = json.loads(bytes(revision_blob.data).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("substrate revision payload must be a JSON object")
        metadata_value = load_canonical_json(_read_blob_bytes(self._repo, tree, SUBSTRATE_REVISION_METADATA_PATH))
        metadata = SubstrateRevisionMetadata.from_json(metadata_value)
        if metadata.kind != self.identity.kind or metadata.resource_id != self.identity.resource_id:
            raise InvalidRepositoryStateError("substrate revision metadata identity disagrees with store identity")
        if metadata.payload_digest != canonical_digest(payload):
            raise InvalidRepositoryStateError(
                "substrate revision metadata payload_digest disagrees with revision payload"
            )
        if metadata.byte_authority == "structured-tree":
            if metadata.content_digest is None:
                raise InvalidRepositoryStateError("structured substrate revision metadata is missing content_digest")
            self._validate_structured_revision_content(
                tree,
                expected_content_digest=metadata.content_digest,
                payload=payload,
            )
        parent_heads = tuple(str(parent) for parent in commit.parent_ids)
        if metadata.parent_heads != parent_heads:
            raise InvalidRepositoryStateError("substrate revision metadata parent_heads disagree with Git parents")
        return metadata

    def validate_candidate(
        self,
        head: str,
        *,
        expected_transition_digest: str | None = None,
        expected_revision_plan_digest: str | None = None,
        expected_revision_preparation_digest: str | None = None,
        require_evidence: bool = True,
    ) -> SubstrateRevisionMetadata:
        """Validate a candidate's generic transition-kernel provenance metadata."""
        metadata = self.read_revision_metadata(head)
        if expected_transition_digest is not None and metadata.transition_digest != expected_transition_digest:
            raise InvalidRepositoryStateError("candidate transition_digest disagrees with expected value")
        if expected_revision_plan_digest is not None and metadata.revision_plan_digest != expected_revision_plan_digest:
            raise InvalidRepositoryStateError("candidate revision_plan_digest disagrees with expected value")
        if (
            expected_revision_preparation_digest is not None
            and metadata.revision_preparation_digest != expected_revision_preparation_digest
        ):
            raise InvalidRepositoryStateError("revision preparation digest disagrees with expected value")
        if require_evidence and metadata.revision_preparation_digest is not None and not metadata.evidence_digests:
            raise InvalidRepositoryStateError("prepared candidate metadata is missing evidence digests")
        return metadata

    def validate_prepared_revision(
        self,
        head: str,
        *,
        expected_transition_digest: str | None = None,
        expected_revision_plan_digest: str | None = None,
        expected_revision_preparation_digest: str | None = None,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> PreparedCandidateProvenance:
        """Validate a prepared revision by checking metadata and typed sidecar records."""
        from vcs_core._transition_kernel_records import (
            LogicalTransition,
            PreparedRevisionPlan,
            RevisionPreparationRecord,
        )

        commit = require_commit(self._repo, pygit2.Oid(hex=head), context="prepared substrate revision")
        tree = require_tree(self._repo, commit.tree.id, context="prepared substrate revision tree")
        try:
            revision_entry = tree["revision.json"]
        except KeyError as exc:
            raise InvalidRepositoryStateError("prepared revision is missing revision.json") from exc
        revision_blob = require_blob(self._repo, revision_entry.id, context="prepared revision payload blob")
        payload = json.loads(bytes(revision_blob.data).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("prepared revision payload must be a JSON object")

        metadata = self.validate_candidate(
            head,
            expected_transition_digest=expected_transition_digest,
            expected_revision_plan_digest=expected_revision_plan_digest,
            expected_revision_preparation_digest=expected_revision_preparation_digest,
        )
        if metadata.transition_digest is None:
            raise InvalidRepositoryStateError("prepared revision metadata is missing transition_digest")
        if metadata.revision_plan_digest is None:
            raise InvalidRepositoryStateError("prepared revision metadata is missing revision_plan_digest")
        if metadata.revision_preparation_digest is None:
            raise InvalidRepositoryStateError("prepared revision metadata is missing revision_preparation_digest")

        try:
            from vcs_core._transition_kernel_records import ValidatedPayloadDescriptor

            payload_descriptor = ValidatedPayloadDescriptor.from_json(
                load_canonical_json(_read_blob_bytes(self._repo, tree, PAYLOAD_DESCRIPTOR_PATH))
            )
            transition = LogicalTransition.from_json(
                load_canonical_json(_read_blob_bytes(self._repo, tree, "meta/logical-transition.json"))
            )
            plan = PreparedRevisionPlan.from_json(
                load_canonical_json(_read_blob_bytes(self._repo, tree, "meta/prepared-revision-plan.json"))
            )
            preparation = RevisionPreparationRecord.from_json(
                load_canonical_json(_read_blob_bytes(self._repo, tree, "meta/revision-preparation.json"))
            )
        except KeyError as exc:
            raise InvalidRepositoryStateError("prepared revision is missing transition-kernel sidecar records") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidRepositoryStateError(
                "prepared revision has invalid transition-kernel sidecar records"
            ) from exc

        self._validate_prepared_candidate_inputs(
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload=payload,
            parents=tuple(str(parent) for parent in commit.parent_ids),
            evidence_resolver=evidence_resolver,
        )
        if metadata.transition_digest != transition.transition_digest():
            raise InvalidRepositoryStateError("prepared revision metadata transition_digest disagrees with sidecar")
        if metadata.revision_plan_digest != plan.revision_plan_digest():
            raise InvalidRepositoryStateError("prepared revision metadata revision_plan_digest disagrees with sidecar")
        if metadata.materialization_class != plan.materialization_class:
            raise InvalidRepositoryStateError("prepared revision metadata materialization_class disagrees with sidecar")
        if metadata.content_digest != plan.content_digest:
            raise InvalidRepositoryStateError("prepared revision metadata content_digest disagrees with sidecar")
        if payload_descriptor.payload_digest != metadata.payload_digest:
            raise InvalidRepositoryStateError("prepared revision payload descriptor disagrees with metadata")
        if metadata.revision_preparation_digest != preparation.revision_preparation_digest():
            raise InvalidRepositoryStateError("prepared revision metadata preparation digest disagrees with sidecar")
        if sorted(metadata.evidence_digests) != sorted(preparation.evidence_digests):
            raise InvalidRepositoryStateError("prepared revision metadata evidence_digests disagree with sidecar")
        if metadata.produced_by_operation_id != preparation.operation_id:
            raise InvalidRepositoryStateError("prepared revision metadata producer operation_id disagrees with sidecar")
        if metadata.git_tree_oid != plan.git_tree_oid:
            raise InvalidRepositoryStateError("prepared revision metadata git_tree_oid disagrees with sidecar")
        expected_byte_authority = _byte_authority_for_plan(plan)
        if metadata.byte_authority != expected_byte_authority:
            raise InvalidRepositoryStateError("prepared revision metadata byte_authority disagrees with sidecar plan")
        _validate_plan_entries(self._repo, tree, plan, payload=payload)
        commit_tree = require_tree(self._repo, commit.tree.id, context="prepared revision commit tree")
        has_workspace_entry = False
        for entry in commit_tree:
            if entry.name == "workspace":
                if entry.filemode != pygit2.GIT_FILEMODE_TREE:
                    raise InvalidRepositoryStateError("prepared revision workspace entry is not a tree")
                if plan.git_tree_oid is None:
                    raise InvalidRepositoryStateError(
                        "digest-only prepared revision must not contain a workspace tree entry"
                    )
                if str(entry.id) != plan.git_tree_oid:
                    raise InvalidRepositoryStateError(
                        "prepared revision workspace tree oid disagrees with plan git_tree_oid"
                    )
                has_workspace_entry = True
                break
        if plan.git_tree_oid is not None and not has_workspace_entry:
            raise InvalidRepositoryStateError("tree-backed prepared revision is missing a workspace tree entry")
        parent_heads = tuple(str(parent) for parent in commit.parent_ids)
        return PreparedCandidateProvenance(
            head=head,
            metadata=metadata,
            transition=transition,
            plan=plan,
            preparation=preparation,
            payload_descriptor=payload_descriptor,
            payload_digest=canonical_digest(payload),
            parent_heads=parent_heads,
        )

    def validate_prepared_candidate(
        self,
        head: str,
        *,
        expected_transition_digest: str | None = None,
        expected_revision_plan_digest: str | None = None,
        expected_revision_preparation_digest: str | None = None,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> PreparedCandidateProvenance:
        """Validate a prepared candidate by checking metadata and typed sidecar records."""
        return self.validate_prepared_revision(
            head,
            expected_transition_digest=expected_transition_digest,
            expected_revision_plan_digest=expected_revision_plan_digest,
            expected_revision_preparation_digest=expected_revision_preparation_digest,
            evidence_resolver=evidence_resolver,
        )

    def candidate_commit_record(
        self,
        candidate: CandidateRevision,
        *,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> CandidateCommitRecord:
        """Build typed evidence for a prepared candidate protected by a candidate ref."""
        from vcs_core._transition_kernel_records import CandidateCommitRecord

        provenance = self.validate_prepared_candidate(candidate.head, evidence_resolver=evidence_resolver)
        metadata = provenance.metadata
        if metadata.revision_preparation_digest is None:
            raise InvalidRepositoryStateError("candidate is missing revision preparation metadata")
        self.validate_candidate_ref(
            operation_id=candidate.operation_id,
            binding=candidate.binding,
            candidate_id=candidate.candidate_id,
            expected_head=candidate.head,
        )
        return CandidateCommitRecord(
            operation_id=candidate.operation_id,
            binding=candidate.binding,
            store_id=candidate.store_id,
            resource_id=candidate.resource_id,
            candidate_head=candidate.head,
            candidate_ref=candidate.ref,
            revision_preparation_digest=metadata.revision_preparation_digest,
            candidate_id=candidate.candidate_id,
        )

    def _write_identity(self, identity: SubstrateStoreIdentity) -> None:
        with self._mutation_lock:
            if IDENTITY_REF in self._repo.references:
                raise InvalidRepositoryStateError(f"Substrate identity ref already exists: {IDENTITY_REF}")
            tree_builder = self._repo.TreeBuilder()
            blob = self._repo.create_blob(compact_json_bytes(identity.to_json()))
            insert_tree_entry(self._repo, tree_builder, "identity.json", blob, pygit2.GIT_FILEMODE_BLOB)
            tree = tree_builder.write()
            signature = pygit2.Signature("vcs-core substrate store", "vcs-core@example.invalid")
            oid = create_commit_with_recovery(
                self._repo,
                None,
                signature,
                signature,
                "substrate identity",
                tree,
                [],
            )
            create_or_update_reference(self._repo, IDENTITY_REF, oid)

    def _create_json_commit(
        self,
        payload: dict[str, Any],
        *,
        content: RevisionContentDraft | None = None,
        parents: tuple[str | pygit2.Oid, ...],
        message: str | None,
        produced_by_operation_id: str | None = None,
        transition: LogicalTransition | None = None,
        plan: PreparedRevisionPlan | None = None,
        preparation: RevisionPreparationRecord | None = None,
        payload_descriptor: ValidatedPayloadDescriptor | None = None,
    ) -> pygit2.Oid:
        if payload_descriptor is None:
            if transition is not None or plan is not None or preparation is not None:
                raise InvalidRepositoryStateError("prepared JSON commits require a validated payload descriptor")
            from vcs_core._transition_kernel_records import ValidatedPayloadDescriptor

            resolved_payload_descriptor = ValidatedPayloadDescriptor.for_json_payload(payload)
        else:
            resolved_payload_descriptor = payload_descriptor
        if content is not None:
            return self._create_structured_commit(
                payload,
                content=content,
                parents=parents,
                message=message,
                produced_by_operation_id=produced_by_operation_id,
                transition=transition,
                plan=plan,
                preparation=preparation,
                payload_descriptor=resolved_payload_descriptor,
            )
        tree_builder = self._repo.TreeBuilder()
        blob = self._repo.create_blob(compact_json_bytes(payload))
        insert_tree_entry(self._repo, tree_builder, "revision.json", blob, pygit2.GIT_FILEMODE_BLOB)
        parent_oids = [_coerce_oid(parent) for parent in parents]
        plan_tree_oid = plan.git_tree_oid if plan is not None else None
        if plan_tree_oid is not None:
            try:
                workspace_tree_obj = self._repo[pygit2.Oid(hex=plan_tree_oid)]
            except (KeyError, ValueError) as exc:
                raise InvalidRepositoryStateError(
                    f"prepared revision plan git_tree_oid {plan_tree_oid!r} is not present in this substrate store"
                ) from exc
            if not isinstance(workspace_tree_obj, pygit2.Tree):
                raise InvalidRepositoryStateError(
                    f"prepared revision plan git_tree_oid {plan_tree_oid!r} does not resolve to a tree"
                )
            insert_tree_entry(
                self._repo,
                tree_builder,
                "workspace",
                workspace_tree_obj.id,
                pygit2.GIT_FILEMODE_TREE,
            )
        metadata = SubstrateRevisionMetadata(
            kind=self.identity.kind,
            resource_id=self.identity.resource_id,
            materialization_class=(
                plan.materialization_class if plan is not None else _default_materialization_class(self.identity.kind)
            ),
            payload_digest=canonical_digest(payload),
            parent_heads=tuple(str(parent) for parent in parent_oids),
            produced_by_operation_id=produced_by_operation_id,
            transition_digest=transition.transition_digest() if transition is not None else None,
            revision_plan_digest=plan.revision_plan_digest() if plan is not None else None,
            content_digest=plan.content_digest if plan is not None else None,
            revision_preparation_digest=(
                preparation.revision_preparation_digest() if preparation is not None else None
            ),
            evidence_digests=preparation.evidence_digests if preparation is not None else (),
            ingress_kind=transition.ingress_kind if transition is not None else None,
            semantic_op=transition.semantic_op if transition is not None else None,
            driver=transition.driver if transition is not None else None,
            driver_version=transition.driver_version if transition is not None else None,
            byte_authority="tree-backed" if plan_tree_oid is not None else "digest-only",
            git_tree_oid=plan_tree_oid,
        )
        meta_builder = self._repo.TreeBuilder()
        metadata_blob = self._repo.create_blob(canonical_bytes(metadata.to_json()))
        insert_tree_entry(
            self._repo,
            meta_builder,
            "substrate-revision.json",
            metadata_blob,
            pygit2.GIT_FILEMODE_BLOB,
        )
        _insert_canonical_sidecar(
            self._repo,
            meta_builder,
            "payload-descriptor.json",
            resolved_payload_descriptor.to_json(),
        )
        if transition is not None:
            _insert_canonical_sidecar(self._repo, meta_builder, "logical-transition.json", transition.to_json())
        if plan is not None:
            _insert_canonical_sidecar(self._repo, meta_builder, "prepared-revision-plan.json", plan.to_json())
        if preparation is not None:
            _insert_canonical_sidecar(self._repo, meta_builder, "revision-preparation.json", preparation.to_json())
        insert_tree_entry(self._repo, tree_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
        tree = tree_builder.write()
        signature = pygit2.Signature("vcs-core substrate store", "vcs-core@example.invalid")
        return create_commit_with_recovery(
            self._repo,
            None,
            signature,
            signature,
            message or str(payload.get("label", "substrate revision")),
            tree,
            parent_oids,
        )

    def _create_structured_commit(
        self,
        payload: dict[str, Any],
        *,
        content: RevisionContentDraft,
        parents: tuple[str | pygit2.Oid, ...],
        message: str | None,
        produced_by_operation_id: str | None = None,
        transition: LogicalTransition | None = None,
        plan: PreparedRevisionPlan | None = None,
        preparation: RevisionPreparationRecord | None = None,
        payload_descriptor: ValidatedPayloadDescriptor,
    ) -> pygit2.Oid:
        if plan is None or transition is None or preparation is None:
            raise InvalidRepositoryStateError("structured commits require prepared transition sidecars")
        if plan.git_tree_oid is not None:
            raise InvalidRepositoryStateError("structured commits must not use workspace git_tree_oid")
        parent_oids = [_coerce_oid(parent) for parent in parents]
        payload_digest = canonical_digest(payload)
        expected_content_digest, _entries = self.plan_revision_content(
            content,
            payload_digest=payload_digest,
            parents=parents,
        )
        if plan.content_digest != expected_content_digest:
            raise InvalidRepositoryStateError("structured revision plan content_digest disagrees with content tree")
        parent_tree_oid = self._structured_parent_tree(parent_oids, content)
        metadata = SubstrateRevisionMetadata(
            kind=self.identity.kind,
            resource_id=self.identity.resource_id,
            materialization_class=plan.materialization_class,
            payload_digest=payload_digest,
            parent_heads=tuple(str(parent) for parent in parent_oids),
            produced_by_operation_id=produced_by_operation_id,
            transition_digest=transition.transition_digest(),
            revision_plan_digest=plan.revision_plan_digest(),
            content_digest=plan.content_digest,
            revision_preparation_digest=preparation.revision_preparation_digest(),
            evidence_digests=preparation.evidence_digests,
            ingress_kind=transition.ingress_kind,
            semantic_op=transition.semantic_op,
            driver=transition.driver,
            driver_version=transition.driver_version,
            byte_authority=_byte_authority_for_plan(plan),
        )
        changes: list[tuple[str, bytes | None] | tuple[str, bytes | None, int]] = [
            ("revision.json", compact_json_bytes(payload)),
            (SUBSTRATE_REVISION_METADATA_PATH, canonical_bytes(metadata.to_json())),
            (PAYLOAD_DESCRIPTOR_PATH, canonical_bytes(payload_descriptor.to_json())),
            ("meta/logical-transition.json", canonical_bytes(transition.to_json())),
            ("meta/prepared-revision-plan.json", canonical_bytes(plan.to_json())),
            ("meta/revision-preparation.json", canonical_bytes(preparation.to_json())),
        ]
        if isinstance(content, KeyedJsonTreeDraft):
            descriptor = self._structured_content_descriptor(content, parent_oids=parent_oids)
            changes.append((STRUCTURED_CONTENT_DESCRIPTOR_PATH, canonical_bytes(descriptor)))
        changes.extend(_structured_content_changes(content))
        tree = build_tree(self._repo, parent_tree_oid, changes)
        commit_tree = require_tree(self._repo, tree, context="structured revision commit tree")
        self._validate_structured_revision_content(
            commit_tree,
            expected_content_digest=plan.content_digest,
            payload=payload,
        )
        signature = pygit2.Signature("vcs-core substrate store", "vcs-core@example.invalid")
        return create_commit_with_recovery(
            self._repo,
            None,
            signature,
            signature,
            message or str(payload.get("label", "substrate revision")),
            tree,
            parent_oids,
        )

    def _structured_parent_tree(
        self,
        parent_oids: list[pygit2.Oid],
        content: RevisionContentDraft,
    ) -> pygit2.Oid | None:
        if isinstance(content, KeyedJsonTreeDraft):
            if content.base_head is not None and parent_oids and str(parent_oids[0]) != content.base_head:
                raise InvalidRepositoryStateError("structured content base_head disagrees with revision parent")
            if content.base_head is not None and not parent_oids:
                raise InvalidRepositoryStateError("structured content base_head requires a parent revision")
            if not parent_oids:
                return None
            parent_commit = require_commit(
                self._repo,
                parent_oids[0],
                context="structured content parent revision",
            )
            return parent_commit.tree.id
        raise InvalidRepositoryStateError(f"unsupported structured content draft: {type(content).__name__}")

    def _validate_prepared_candidate_inputs(
        self,
        *,
        transition: LogicalTransition,
        plan: PreparedRevisionPlan,
        preparation: RevisionPreparationRecord,
        payload_descriptor: ValidatedPayloadDescriptor,
        payload: dict[str, Any],
        content: RevisionContentDraft | None = None,
        parents: tuple[str | pygit2.Oid, ...],
        evidence_resolver: EvidenceResolver | None = None,
    ) -> None:
        payload_digest = canonical_digest(payload)
        if payload_descriptor.payload_digest != payload_digest:
            raise InvalidRepositoryStateError("payload descriptor digest disagrees with JSON payload")
        if transition.store_id != self.identity.store_id or plan.store_id != self.identity.store_id:
            raise InvalidRepositoryStateError("prepared candidate store_id disagrees with substrate store identity")
        if preparation.store_id != self.identity.store_id:
            raise InvalidRepositoryStateError("revision preparation store_id disagrees with substrate store identity")
        if transition.resource_id != self.identity.resource_id or preparation.resource_id != self.identity.resource_id:
            raise InvalidRepositoryStateError("prepared candidate resource_id disagrees with substrate store identity")
        if transition.substrate_kind != self.identity.kind:
            raise InvalidRepositoryStateError(
                "prepared candidate substrate kind disagrees with substrate store identity"
            )
        if plan.binding != transition.binding or preparation.binding != transition.binding:
            raise InvalidRepositoryStateError("prepared candidate binding records disagree")
        if plan.transition_digest != transition.transition_digest():
            raise InvalidRepositoryStateError("prepared revision plan transition_digest disagrees with transition")
        if preparation.transition_digest != transition.transition_digest():
            raise InvalidRepositoryStateError("revision preparation transition_digest disagrees with transition")
        if preparation.revision_plan_digest != plan.revision_plan_digest():
            raise InvalidRepositoryStateError("revision preparation revision_plan_digest disagrees with plan")
        if preparation.content_digest != plan.content_digest:
            raise InvalidRepositoryStateError("revision preparation content_digest disagrees with plan")
        if transition.payload_digest != payload_digest:
            raise InvalidRepositoryStateError("logical transition payload_digest disagrees with JSON payload")
        if content is not None:
            expected_content_digest, _entries = self.plan_revision_content(
                content,
                payload_digest=payload_digest,
                parents=parents,
            )
            if plan.content_digest != expected_content_digest:
                raise InvalidRepositoryStateError("prepared revision plan content_digest disagrees with content draft")
        elif not _plan_has_structured_entries(plan) and plan.content_digest != payload_digest:
            raise InvalidRepositoryStateError("prepared revision plan content_digest disagrees with JSON payload")
        parent_heads = tuple(str(_coerce_oid(parent)) for parent in parents)
        if plan.expected_parent_heads != parent_heads:
            raise InvalidRepositoryStateError("prepared revision plan expected_parent_heads disagree with Git parents")
        if plan.base_heads != transition.base_heads:
            raise InvalidRepositoryStateError("prepared revision plan base_heads disagree with transition")
        if sorted(preparation.evidence_digests) != sorted(transition.evidence_digests):
            raise InvalidRepositoryStateError("revision preparation evidence_digests disagree with transition")
        # Cross-operation evidence citation (the capture-shadow flow in
        # `_world_substrate_adapters`) is admitted only through explicit
        # ``cited_evidence_refs``, which must be a suffix of ``evidence_refs``.
        # The extracted helper enforces digest agreement, binding/store/
        # substrate-kind scope, and the cited-suffix rule (replaces the prior
        # inline checks + implicit cross-operation skip).
        validate_preparation_evidence_refs(
            preparation.evidence_refs,
            cited_evidence_refs=preparation.cited_evidence_refs,
            expected_digests=preparation.evidence_digests,
            scope=EvidenceCitationScope(
                operation_id=preparation.operation_id,
                binding=preparation.binding,
                store_id=preparation.store_id,
                substrate_kind=transition.substrate_kind,
            ),
            resolver=evidence_resolver,
        )
        transition_requirements = sorted(
            canonical_digest(requirement.to_json()) for requirement in transition.requirements
        )
        preparation_requirements = sorted(
            canonical_digest(requirement.to_json()) for requirement in preparation.relationship_requirements
        )
        if preparation_requirements != transition_requirements:
            raise InvalidRepositoryStateError("revision preparation relationship requirements disagree with transition")
        if plan.git_tree_oid is not None:
            if content is not None:
                raise InvalidRepositoryStateError("structured content must not use workspace git_tree_oid")
            self._validate_tree_backed_correspondence(tree_oid=plan.git_tree_oid, payload=payload)

    def _structured_content_descriptor(
        self,
        content: KeyedJsonTreeDraft,
        *,
        parent_oids: list[pygit2.Oid],
    ) -> dict[str, object]:
        content_tree_oid = self._structured_content_tree_oid(parent_oids, content)
        return {
            "schema": STRUCTURED_CONTENT_DESCRIPTOR_SCHEMA,
            "shape": "keyed-json-tree",
            "content_root": content.content_root,
            "object_format": self.identity.object_format,
            "content_tree_oid": str(content_tree_oid),
            "manifest_digest": canonical_digest(content.manifest),
        }

    def _structured_content_tree_oid(
        self,
        parent_oids: list[pygit2.Oid],
        content: KeyedJsonTreeDraft,
    ) -> pygit2.Oid:
        parent_content_tree_oid = self._structured_parent_content_tree(parent_oids, content)
        return build_tree(self._repo, parent_content_tree_oid, _structured_content_root_changes(content))

    def _structured_parent_content_tree(
        self,
        parent_oids: list[pygit2.Oid],
        content: KeyedJsonTreeDraft,
    ) -> pygit2.Oid | None:
        if content.base_head is not None and parent_oids and str(parent_oids[0]) != content.base_head:
            raise InvalidRepositoryStateError("structured content base_head disagrees with revision parent")
        if content.base_head is not None and not parent_oids:
            raise InvalidRepositoryStateError("structured content base_head requires a parent revision")
        if not parent_oids:
            return None
        parent_commit = require_commit(
            self._repo,
            parent_oids[0],
            context="structured content parent revision",
        )
        parent_tree = require_tree(self._repo, parent_commit.tree.id, context="structured content parent tree")
        try:
            entry = parent_tree[content.content_root]
        except KeyError:
            return None
        obj = self._repo[entry.id]
        if not isinstance(obj, pygit2.Tree):
            raise InvalidRepositoryStateError("structured content parent root is not a tree")
        return obj.id

    def _validate_structured_revision_content(
        self,
        tree: pygit2.Tree,
        *,
        expected_content_digest: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            descriptor = load_canonical_json(_read_blob_bytes(self._repo, tree, STRUCTURED_CONTENT_DESCRIPTOR_PATH))
        except KeyError as exc:
            raise InvalidRepositoryStateError("structured revision is missing structured-content descriptor") from exc
        if canonical_digest(descriptor) != expected_content_digest:
            raise InvalidRepositoryStateError("structured revision descriptor digest disagrees with content_digest")
        if descriptor.get("schema") != STRUCTURED_CONTENT_DESCRIPTOR_SCHEMA:
            raise InvalidRepositoryStateError("structured revision descriptor has unsupported schema")
        if descriptor.get("shape") != "keyed-json-tree":
            raise InvalidRepositoryStateError("structured revision descriptor has unsupported shape")
        if descriptor.get("object_format") != self.identity.object_format:
            raise InvalidRepositoryStateError("structured revision descriptor object_format disagrees with store")
        if descriptor.get("manifest_digest") != canonical_digest(payload):
            raise InvalidRepositoryStateError("structured revision descriptor manifest_digest disagrees with payload")
        content_root = descriptor.get("content_root")
        if not isinstance(content_root, str) or not content_root:
            raise InvalidRepositoryStateError("structured revision descriptor content_root must be a string")
        _validate_read_path(content_root)
        if "/" in content_root or content_root in {"meta", "workspace"}:
            raise InvalidRepositoryStateError("structured revision descriptor content_root is invalid")
        content_tree_oid = descriptor.get("content_tree_oid")
        if not isinstance(content_tree_oid, str):
            raise InvalidRepositoryStateError("structured revision descriptor content_tree_oid must be a string")
        try:
            root_entry = tree[content_root]
        except KeyError as exc:
            raise InvalidRepositoryStateError("structured revision is missing its content root tree") from exc
        if root_entry.filemode != pygit2.GIT_FILEMODE_TREE:
            raise InvalidRepositoryStateError("structured revision content root is not a tree")
        if str(root_entry.id) != content_tree_oid:
            raise InvalidRepositoryStateError("structured revision content root tree disagrees with descriptor")

    def _validate_tree_backed_correspondence(
        self,
        *,
        tree_oid: str,
        payload: dict[str, Any],
    ) -> None:
        """Enforce that a tree-backed plan's embedded tree matches its manifest entries.

        Walks the tree referenced by ``tree_oid`` and confirms every blob's sha256
        matches the corresponding workspace state manifest entry's content_digest,
        modes agree, deleted entries have no blob, and the tree contains no blobs
        unmentioned by the manifest. Tree assembly happens at commit time, so the
        tree oid is excluded from ``revision_plan_digest``; this write-time check
        provides the integrity claim instead.
        """
        state_manifest = payload.get("state_manifest")
        if not isinstance(state_manifest, dict):
            raise InvalidRepositoryStateError("tree-backed substrate revision payload requires a state_manifest")
        raw_entries = state_manifest.get("entries")
        if not isinstance(raw_entries, list):
            raise InvalidRepositoryStateError("tree-backed substrate revision manifest entries must be a list")
        manifest_byte_authority = state_manifest.get("byte_authority")
        if manifest_byte_authority != "tree-backed":
            raise InvalidRepositoryStateError(
                "tree-backed substrate revision requires manifest byte_authority='tree-backed'"
            )
        expected_present: dict[str, tuple[int, str]] = {}
        expected_deleted: set[str] = set()
        for entry in raw_entries:
            if not isinstance(entry, dict):
                raise InvalidRepositoryStateError("workspace manifest entries must be objects")
            path = entry.get("path")
            state = entry.get("state")
            if not isinstance(path, str):
                raise InvalidRepositoryStateError("workspace manifest entry path must be a string")
            if state == "present":
                mode = entry.get("mode")
                content_digest = entry.get("content_digest")
                if not isinstance(mode, int) or isinstance(mode, bool):
                    raise InvalidRepositoryStateError(f"workspace manifest entry mode must be an integer at {path!r}")
                if not isinstance(content_digest, str):
                    raise InvalidRepositoryStateError(
                        f"workspace manifest entry content_digest must be a string at {path!r}"
                    )
                expected_present[path] = (mode, content_digest)
            elif state == "deleted":
                expected_deleted.add(path)
            else:
                raise InvalidRepositoryStateError(
                    f"workspace manifest entry state must be 'present' or 'deleted' at {path!r}"
                )
        try:
            tree_obj = self._repo[pygit2.Oid(hex=tree_oid)]
        except (KeyError, ValueError) as exc:
            raise InvalidRepositoryStateError(
                f"tree-backed substrate revision references unknown tree {tree_oid!r}"
            ) from exc
        if not isinstance(tree_obj, pygit2.Tree):
            raise InvalidRepositoryStateError(
                f"tree-backed substrate revision git_tree_oid {tree_oid!r} does not resolve to a tree"
            )
        observed: dict[str, tuple[int, str]] = {}
        self._walk_workspace_tree(tree_obj, prefix="", observed=observed)
        observed_paths = set(observed)
        present_paths = set(expected_present)
        # Check deleted-with-blob first so the more specific error fires before
        # the generic extra-in-tree message when the manifest explicitly named
        # the path as deleted.
        deleted_with_blob = observed_paths & expected_deleted
        if deleted_with_blob:
            path = sorted(deleted_with_blob)[0]
            raise InvalidRepositoryStateError(
                f"tree-backed substrate revision tree contains blob for deleted manifest entry: {path!r}"
            )
        extra_in_tree = observed_paths - present_paths
        if extra_in_tree:
            path = sorted(extra_in_tree)[0]
            raise InvalidRepositoryStateError(
                f"tree-backed substrate revision tree contains blob without a manifest entry: {path!r}"
            )
        missing_from_tree = present_paths - observed_paths
        if missing_from_tree:
            path = sorted(missing_from_tree)[0]
            raise InvalidRepositoryStateError(
                f"tree-backed substrate revision manifest entry has no tree blob: {path!r}"
            )
        for path, (expected_mode, expected_digest) in expected_present.items():
            observed_mode, observed_digest = observed[path]
            if observed_mode != expected_mode:
                raise InvalidRepositoryStateError(
                    f"tree-backed substrate revision mode mismatch at {path!r}: "
                    f"manifest=0o{expected_mode:o}, tree=0o{observed_mode:o}"
                )
            if observed_digest != expected_digest:
                raise InvalidRepositoryStateError(f"tree-backed substrate revision content_digest mismatch at {path!r}")

    def _walk_workspace_tree(
        self,
        tree: pygit2.Tree,
        *,
        prefix: str,
        observed: dict[str, tuple[int, str]],
    ) -> None:
        for entry in tree:
            name = entry.name
            if name is None:
                raise InvalidRepositoryStateError(
                    f"tree-backed substrate revision has unnamed tree entry under {prefix!r}"
                )
            path = f"{prefix}/{name}" if prefix else name
            if entry.filemode == pygit2.GIT_FILEMODE_TREE:
                sub = self._repo[entry.id]
                if not isinstance(sub, pygit2.Tree):
                    raise InvalidRepositoryStateError(f"tree-backed substrate revision has non-tree object at {path!r}")
                self._walk_workspace_tree(sub, prefix=path, observed=observed)
            elif entry.filemode in (pygit2.GIT_FILEMODE_BLOB, pygit2.GIT_FILEMODE_BLOB_EXECUTABLE):
                blob = self._repo[entry.id]
                if not isinstance(blob, pygit2.Blob):
                    raise InvalidRepositoryStateError(f"tree-backed substrate revision has non-blob object at {path!r}")
                digest = f"sha256:{hashlib.sha256(bytes(blob.data)).hexdigest()}"
                observed[path] = (entry.filemode, digest)
            else:
                raise InvalidRepositoryStateError(
                    f"tree-backed substrate revision has unsupported file mode at {path!r}: 0o{entry.filemode:o}"
                )


def _ensure_alternates(repo: pygit2.Repository, repo_path: Path, source_repo_path: Path) -> pygit2.Repository:
    """Configure ``objects/info/alternates`` so ``repo`` can read source ODB.

    libgit2 caches the alternates list at first object access, so this function
    writes the alternates entry idempotently and returns a freshly-opened
    repository handle that will discover the new alternate on its next ODB
    lookup. The original ``repo`` handle remains valid for the caller's
    bookkeeping but should not be used for object reads that depend on the
    alternate.
    """
    source_objects = (source_repo_path / "objects").resolve()
    if not source_objects.is_dir():
        raise InvalidRepositoryStateError(
            f"shared object repo does not contain an objects directory: {source_repo_path}"
        )
    alternates_path = Path(repo.path) / "objects" / "info" / "alternates"
    alternates_path.parent.mkdir(parents=True, exist_ok=True)
    desired_line = str(source_objects)
    existing_lines: list[str] = []
    if alternates_path.exists():
        existing_lines = [
            line.strip() for line in alternates_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if desired_line in existing_lines:
            return repo
    new_lines = [*existing_lines, desired_line]
    alternates_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    # Re-open so libgit2 discovers the new alternate entry on its next lookup.
    return pygit2.Repository(str(repo_path))


def _open_or_init_bare_repo(path: Path) -> pygit2.Repository:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = pygit2.init_repository(str(path), bare=True)
    else:
        try:
            repo = pygit2.Repository(str(path))
        except (KeyError, ValueError, pygit2.GitError) as exc:
            if not path.is_dir() or any(path.iterdir()):
                raise InvalidRepositoryStateError(f"{path} exists but is not a Git repository") from exc
            repo = pygit2.init_repository(str(path), bare=True)
    if not repo.is_bare:
        raise InvalidRepositoryStateError(f"{path} is not a bare substrate store repository")
    return repo


def _open_existing_bare_repo(path: Path) -> pygit2.Repository:
    if not path.exists():
        raise InvalidRepositoryStateError(f"configured substrate store is missing: {path}")
    try:
        repo = pygit2.Repository(str(path))
    except (KeyError, ValueError, pygit2.GitError) as exc:
        raise InvalidRepositoryStateError(f"{path} exists but is not a Git repository") from exc
    if not repo.is_bare:
        raise InvalidRepositoryStateError(f"{path} is not a bare substrate store repository")
    return repo


def _coerce_oid(value: str | pygit2.Oid) -> pygit2.Oid:
    if isinstance(value, pygit2.Oid):
        return value
    return pygit2.Oid(hex=value)


def _head_oid(head: str | SubstrateHead) -> pygit2.Oid:
    if isinstance(head, SubstrateHead):
        return pygit2.Oid(hex=head.head)
    return pygit2.Oid(hex=head)


def _read_blob_bytes(repo: pygit2.Repository, tree: pygit2.Tree, path: str) -> bytes:
    obj: pygit2.Object = tree
    for component in path.split("/"):
        if not isinstance(obj, pygit2.Tree):
            raise TypeError(f"{path!r} did not resolve to a blob")
        obj = repo[obj[component].id]
    if not isinstance(obj, pygit2.Blob):
        raise TypeError(f"{path!r} did not resolve to a blob")
    return bytes(obj.data)


def _read_optional_blob_bytes(repo: pygit2.Repository, tree: pygit2.Tree, path: str) -> bytes | None:
    try:
        return _read_blob_bytes(repo, tree, path)
    except KeyError:
        return None


def _iter_blob_bytes(repo: pygit2.Repository, tree: pygit2.Tree, prefix: str) -> tuple[tuple[str, bytes], ...]:
    obj: pygit2.Object = tree
    normalized = prefix.strip("/")
    base = ""
    if normalized:
        parts = normalized.split("/")
        for component in parts:
            try:
                obj = repo[obj[component].id]  # type: ignore[index]
            except KeyError:
                return ()
        base = normalized
    if isinstance(obj, pygit2.Blob):
        return ((base, bytes(obj.data)),)
    if not isinstance(obj, pygit2.Tree):
        raise TypeError(f"{prefix!r} did not resolve to a tree or blob")
    rows: list[tuple[str, bytes]] = []
    _collect_tree_blobs(repo, obj, base=base, rows=rows)
    return tuple(rows)


def _collect_tree_blobs(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    *,
    base: str,
    rows: list[tuple[str, bytes]],
) -> None:
    for entry in tree:
        path = f"{base}/{entry.name}" if base else entry.name
        obj = repo[entry.id]
        if isinstance(obj, pygit2.Blob):
            rows.append((path, bytes(obj.data)))
        elif isinstance(obj, pygit2.Tree):
            _collect_tree_blobs(repo, obj, base=path, rows=rows)


def _validate_read_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("revision entry path must be a non-empty string")
    if path.startswith("/") or path.endswith("/") or "//" in path:
        raise ValueError("revision entry path must be relative")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("revision entry path must not contain empty, '.', or '..' segments")


def _structured_content_changes(
    content: RevisionContentDraft,
) -> list[tuple[str, bytes | None] | tuple[str, bytes | None, int]]:
    if isinstance(content, KeyedJsonTreeDraft):
        changes: list[tuple[str, bytes | None] | tuple[str, bytes | None, int]] = []
        for item in content.puts:
            changes.append((f"{content.content_root}/{item.path}", compact_json_bytes(item.payload)))
        for path in content.deletes:
            changes.append((f"{content.content_root}/{path}", None))
        return changes
    raise InvalidRepositoryStateError(f"unsupported structured content draft: {type(content).__name__}")


def _structured_content_root_changes(
    content: KeyedJsonTreeDraft,
) -> list[tuple[str, bytes | None] | tuple[str, bytes | None, int]]:
    changes: list[tuple[str, bytes | None] | tuple[str, bytes | None, int]] = []
    for item in content.puts:
        changes.append((item.path, compact_json_bytes(item.payload)))
    for path in content.deletes:
        changes.append((path, None))
    return changes


def _plan_has_structured_entries(plan: PreparedRevisionPlan) -> bool:
    return any(isinstance(entry.get("path"), str) and entry["path"] != "revision.json" for entry in plan.entries)


def _byte_authority_for_plan(plan: PreparedRevisionPlan) -> str:
    if plan.git_tree_oid is not None:
        return "tree-backed"
    if _plan_has_structured_entries(plan):
        return "structured-tree"
    return "digest-only"


def _validate_plan_entries(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    plan: PreparedRevisionPlan,
    *,
    payload: dict[str, Any],
) -> None:
    for index, entry in enumerate(plan.entries):
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise InvalidRepositoryStateError(f"prepared revision plan entry {index} has invalid path")
        state = entry.get("state", "present")
        if state == "deleted":
            if _read_optional_blob_bytes(repo, tree, path) is not None:
                raise InvalidRepositoryStateError(f"prepared revision deleted plan entry is present: {path!r}")
            continue
        if state != "present":
            raise InvalidRepositoryStateError(f"prepared revision plan entry {path!r} has invalid state")
        payload_digest = entry.get("payload_digest")
        if not isinstance(payload_digest, str):
            raise InvalidRepositoryStateError(f"prepared revision plan entry {path!r} is missing payload_digest")
        if path == "revision.json":
            observed_digest = canonical_digest(payload)
        else:
            raw = _read_optional_blob_bytes(repo, tree, path)
            if raw is None:
                raise InvalidRepositoryStateError(f"prepared revision plan entry is missing from tree: {path!r}")
            value = _load_plan_entry_json(raw)
            if not isinstance(value, dict):
                raise InvalidRepositoryStateError(f"prepared revision plan JSON entry is not an object: {path!r}")
            observed_digest = canonical_digest(value)
        if observed_digest != payload_digest:
            raise InvalidRepositoryStateError(f"prepared revision plan entry digest disagrees for {path!r}")


def _load_plan_entry_json(raw: bytes) -> object:
    try:
        return load_canonical_json(raw)
    except (TypeError, ValueError):
        return json.loads(raw.decode("utf-8"))


def _insert_canonical_sidecar(
    repo: pygit2.Repository,
    builder: pygit2.TreeBuilder,
    name: str,
    value: dict[str, object],
) -> None:
    insert_tree_entry(repo, builder, name, repo.create_blob(canonical_bytes(value)), pygit2.GIT_FILEMODE_BLOB)


def _default_materialization_class(kind: str) -> str:
    if kind == "filesystem":
        return "external"
    if kind.startswith("shepherd."):
        return "noop"
    return "internal"
