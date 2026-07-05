"""Internal DTOs and canonical JSON helpers for v2 world storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

CANONICAL_PREFIX = b"vcscore.canonical.v2\n"
WORLD_SCHEMA = "vcscore/world/v2"
WORLD_TRANSITION_SCHEMA = "vcscore/transition/v2"
OPERATION_FINAL_SCHEMA = "vcscore/operation-final/v2"
WORLD_REF_PAYLOAD_SCHEMA = "vcscore/world-ref-payload/v1"
WORLD_REF_SUBSTRATE_KIND = "vcscore.world_ref"
SUBSTRATE_STORE_IDENTITY_SCHEMA = "vcscore/substrate-store-identity/v1"
SUBSTRATE_REVISION_METADATA_SCHEMA = "vcscore/substrate-revision-metadata/v1"
MATERIALIZATION_RECEIPT_SCHEMA = "vcscore/materialization-receipt/v1"


def canonical_bytes(value: object) -> bytes:
    r"""Return the vcs-core v2 canonical byte representation for one JSON value.

    ``ensure_ascii=False`` emits non-ASCII payloads as raw UTF-8 rather than
    ``\\uXXXX`` escapes. This matches shepherd2's ``canonical_json_bytes``
    discipline (per the digest-compatibility spike at
    ``vcs-core/design/spikes/260515-world-vectors/260524-shepherd2-digest-compat/``)
    and unblocks future cross-domain non-ASCII parity. ASCII-only payloads are
    byte-stable under both settings, so existing vcs-core digests do not change.
    """
    return CANONICAL_PREFIX + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Return the sha256 digest of ``canonical_bytes(value)``."""
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def compact_json_bytes(value: object) -> bytes:
    """Return stable compact JSON bytes without the canonical domain prefix."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_canonical_json(data: bytes) -> dict[str, Any]:
    """Load a canonical JSON object with the vcs-core v2 domain prefix."""
    if not data.startswith(CANONICAL_PREFIX):
        raise ValueError("canonical record is missing the vcs-core v2 domain prefix")
    value = json.loads(data[len(CANONICAL_PREFIX) :].decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("canonical record must be a JSON object")
    if canonical_bytes(value) != data:
        raise ValueError("canonical record is not byte-canonical")
    return value


@dataclass(frozen=True)
class SubstrateStoreIdentity:
    """Stable identity for the Git repository that owns one resource's revisions."""

    store_id: str
    kind: str
    resource_id: str
    object_format: str = "sha1"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("store_id", self.store_id),
            ("kind", self.kind),
            ("resource_id", self.resource_id),
            ("object_format", self.object_format),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")

    def to_json(self) -> dict[str, str]:
        return {
            "schema": SUBSTRATE_STORE_IDENTITY_SCHEMA,
            "store_id": self.store_id,
            "kind": self.kind,
            "resource_id": self.resource_id,
            "object_format": self.object_format,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SubstrateStoreIdentity:
        schema = value.get("schema")
        if schema != SUBSTRATE_STORE_IDENTITY_SCHEMA:
            raise ValueError(f"unsupported substrate store identity schema: {schema!r}")
        return cls(
            store_id=_required_str(value, "store_id"),
            kind=_required_str(value, "kind"),
            resource_id=_required_str(value, "resource_id"),
            object_format=_required_str(value, "object_format"),
        )


@dataclass(frozen=True)
class SubstrateRevisionMetadata:
    """Minimum private metadata contract for JSON-backed substrate revisions."""

    kind: str
    resource_id: str
    materialization_class: str
    payload_digest: str
    parent_heads: tuple[str, ...] = ()
    produced_by_operation_id: str | None = None
    transition_digest: str | None = None
    revision_plan_digest: str | None = None
    content_digest: str | None = None
    revision_preparation_digest: str | None = None
    evidence_digests: tuple[str, ...] = ()
    ingress_kind: str | None = None
    semantic_op: str | None = None
    driver: str | None = None
    driver_version: str | None = None
    byte_authority: str = "digest-only"
    git_tree_oid: str | None = None
    schema: str = SUBSTRATE_REVISION_METADATA_SCHEMA

    def __post_init__(self) -> None:
        for field_name, value in (
            ("schema", self.schema),
            ("kind", self.kind),
            ("resource_id", self.resource_id),
            ("materialization_class", self.materialization_class),
            ("payload_digest", self.payload_digest),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")
        if self.schema != SUBSTRATE_REVISION_METADATA_SCHEMA:
            raise ValueError(f"unsupported substrate revision metadata schema: {self.schema!r}")
        if self.materialization_class not in {"external", "noop", "receipt-only", "ledger", "internal"}:
            raise ValueError(f"unsupported materialization class: {self.materialization_class!r}")
        if self.byte_authority not in {"digest-only", "tree-backed", "structured-tree"}:
            raise ValueError(f"unsupported substrate revision byte_authority: {self.byte_authority!r}")
        if self.byte_authority == "tree-backed" and not self.git_tree_oid:
            raise ValueError("tree-backed substrate revisions require git_tree_oid")
        if self.byte_authority in {"digest-only", "structured-tree"} and self.git_tree_oid is not None:
            raise ValueError(f"{self.byte_authority} substrate revisions must not carry git_tree_oid")
        if self.git_tree_oid is not None and not _is_git_oid(self.git_tree_oid):
            raise ValueError("substrate revision git_tree_oid must be a 40-char hex Git oid")
        _validate_sha256_digest(self.payload_digest, "payload_digest")
        for digest_field_name, digest_value in (
            ("transition_digest", self.transition_digest),
            ("revision_plan_digest", self.revision_plan_digest),
            ("content_digest", self.content_digest),
            ("revision_preparation_digest", self.revision_preparation_digest),
        ):
            if digest_value is not None:
                _validate_sha256_digest(digest_value, digest_field_name)
        for evidence_digest in self.evidence_digests:
            _validate_sha256_digest(evidence_digest, "evidence_digests")
        for parent in self.parent_heads:
            if not parent:
                raise ValueError("metadata parent_heads must be non-empty strings")

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "kind": self.kind,
            "resource_id": self.resource_id,
            "materialization_class": self.materialization_class,
            "payload_digest": self.payload_digest,
            "parent_heads": list(self.parent_heads),
            "byte_authority": self.byte_authority,
        }
        if self.produced_by_operation_id is not None:
            value["produced_by_operation_id"] = self.produced_by_operation_id
        value.update(
            {
                key: item
                for key, item in (
                    ("transition_digest", self.transition_digest),
                    ("revision_plan_digest", self.revision_plan_digest),
                    ("content_digest", self.content_digest),
                    ("revision_preparation_digest", self.revision_preparation_digest),
                    ("ingress_kind", self.ingress_kind),
                    ("semantic_op", self.semantic_op),
                    ("driver", self.driver),
                    ("driver_version", self.driver_version),
                    ("git_tree_oid", self.git_tree_oid),
                )
                if item is not None
            }
        )
        if self.evidence_digests:
            value["evidence_digests"] = list(self.evidence_digests)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SubstrateRevisionMetadata:
        _reject_unexpected_keys(
            value,
            {
                "schema",
                "kind",
                "resource_id",
                "materialization_class",
                "payload_digest",
                "parent_heads",
                "produced_by_operation_id",
                "transition_digest",
                "revision_plan_digest",
                "content_digest",
                "revision_preparation_digest",
                "evidence_digests",
                "ingress_kind",
                "semantic_op",
                "driver",
                "driver_version",
                "byte_authority",
                "git_tree_oid",
            },
            "substrate revision metadata",
        )
        schema = value.get("schema")
        if schema != SUBSTRATE_REVISION_METADATA_SCHEMA:
            raise ValueError(f"unsupported substrate revision metadata schema: {schema!r}")
        raw_parent_heads = value.get("parent_heads")
        if not isinstance(raw_parent_heads, list) or not all(isinstance(parent, str) for parent in raw_parent_heads):
            raise ValueError("substrate revision metadata parent_heads must be a string list")
        produced_by_operation_id = value.get("produced_by_operation_id")
        if produced_by_operation_id is not None and not isinstance(produced_by_operation_id, str):
            raise ValueError("produced_by_operation_id must be a string when present")
        raw_evidence_digests = value.get("evidence_digests", [])
        if not isinstance(raw_evidence_digests, list) or not all(
            isinstance(digest, str) for digest in raw_evidence_digests
        ):
            raise ValueError("substrate revision metadata evidence_digests must be a string list")
        raw_byte_authority = value.get("byte_authority", "digest-only")
        if not isinstance(raw_byte_authority, str):
            raise TypeError("substrate revision metadata byte_authority must be a string")
        return cls(
            kind=_required_str(value, "kind"),
            resource_id=_required_str(value, "resource_id"),
            materialization_class=_required_str(value, "materialization_class"),
            payload_digest=_required_str(value, "payload_digest"),
            parent_heads=tuple(raw_parent_heads),
            produced_by_operation_id=produced_by_operation_id,
            transition_digest=_optional_str(value, "transition_digest"),
            revision_plan_digest=_optional_str(value, "revision_plan_digest"),
            content_digest=_optional_str(value, "content_digest"),
            revision_preparation_digest=_optional_str(value, "revision_preparation_digest"),
            evidence_digests=tuple(raw_evidence_digests),
            ingress_kind=_optional_str(value, "ingress_kind"),
            semantic_op=_optional_str(value, "semantic_op"),
            driver=_optional_str(value, "driver"),
            driver_version=_optional_str(value, "driver_version"),
            byte_authority=raw_byte_authority,
            git_tree_oid=_optional_str(value, "git_tree_oid"),
        )


@dataclass(frozen=True)
class MaterializationReceipt:
    """Canonical private record for one materialization unit receipt."""

    materialization_id: str
    unit_id: str
    binding: str
    target_identity: str
    status: str
    idempotency_key: str | None = None
    payload_digest: str | None = None
    world_oid: str | None = None
    schema: str = MATERIALIZATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field_name, value in (
            ("schema", self.schema),
            ("materialization_id", self.materialization_id),
            ("unit_id", self.unit_id),
            ("binding", self.binding),
            ("target_identity", self.target_identity),
            ("status", self.status),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")
        if self.schema != MATERIALIZATION_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported materialization receipt schema: {self.schema!r}")
        if self.status not in {"open", "completed", "failed"}:
            raise ValueError(f"unsupported materialization receipt status: {self.status!r}")
        if self.payload_digest is not None:
            _validate_sha256_digest(self.payload_digest, "payload_digest")

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "materialization_id": self.materialization_id,
            "unit_id": self.unit_id,
            "binding": self.binding,
            "target_identity": self.target_identity,
            "status": self.status,
        }
        value.update(
            {
                key: item
                for key, item in (
                    ("idempotency_key", self.idempotency_key),
                    ("payload_digest", self.payload_digest),
                    ("world_oid", self.world_oid),
                )
                if item is not None
            }
        )
        return value

    def digest(self) -> str:
        return canonical_digest(self.to_json())

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> MaterializationReceipt:
        _reject_unexpected_keys(
            value,
            {
                "schema",
                "materialization_id",
                "unit_id",
                "binding",
                "target_identity",
                "status",
                "idempotency_key",
                "payload_digest",
                "world_oid",
            },
            "materialization receipt",
        )
        schema = value.get("schema")
        if schema != MATERIALIZATION_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported materialization receipt schema: {schema!r}")
        receipt = cls(
            materialization_id=_required_str(value, "materialization_id"),
            unit_id=_required_str(value, "unit_id"),
            binding=_required_str(value, "binding"),
            target_identity=_required_str(value, "target_identity"),
            status=_required_str(value, "status"),
            idempotency_key=_optional_str(value, "idempotency_key"),
            payload_digest=_optional_str(value, "payload_digest"),
            world_oid=_optional_str(value, "world_oid"),
        )
        if receipt.payload_digest is not None:
            _validate_sha256_digest(receipt.payload_digest, "payload_digest")
        return receipt


@dataclass(frozen=True)
class SubstrateHead:
    """A selected immutable revision inside one substrate store."""

    binding: str
    kind: str
    role: str
    store_id: str
    store_scope: str
    resource_id: str
    head: str
    object_format: str = "sha1"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("binding", self.binding),
            ("kind", self.kind),
            ("role", self.role),
            ("store_id", self.store_id),
            ("store_scope", self.store_scope),
            ("resource_id", self.resource_id),
            ("head", self.head),
            ("object_format", self.object_format),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")

    def to_json(self) -> dict[str, str]:
        return {
            "binding": self.binding,
            "kind": self.kind,
            "role": self.role,
            "store_id": self.store_id,
            "store_scope": self.store_scope,
            "resource_id": self.resource_id,
            "head": self.head,
            "object_format": self.object_format,
        }

    @classmethod
    def from_json(cls, binding: str, value: Mapping[str, object]) -> SubstrateHead:
        expected_keys = {
            "binding",
            "kind",
            "role",
            "store_id",
            "store_scope",
            "resource_id",
            "head",
            "object_format",
        }
        extra_keys = set(value) - expected_keys
        if extra_keys:
            raise ValueError(f"unexpected substrate head fields for {binding!r}: {sorted(extra_keys)!r}")
        embedded_binding = _required_str(value, "binding")
        if embedded_binding != binding:
            raise ValueError(f"snapshot binding {binding!r} disagrees with embedded binding {embedded_binding!r}")
        return cls(
            binding=binding,
            kind=_required_str(value, "kind"),
            role=_required_str(value, "role"),
            store_id=_required_str(value, "store_id"),
            store_scope=_required_str(value, "store_scope"),
            resource_id=_required_str(value, "resource_id"),
            head=_required_str(value, "head"),
            object_format=_required_str(value, "object_format"),
        )


@dataclass(frozen=True)
class WorldSnapshot:
    """Pure binding-to-substrate-head selected state for one world commit."""

    heads: tuple[SubstrateHead, ...] = ()

    def __post_init__(self) -> None:
        sorted_heads = tuple(sorted(self.heads, key=lambda head: head.binding))
        bindings = [head.binding for head in sorted_heads]
        if len(set(bindings)) != len(bindings):
            raise ValueError("world snapshot contains duplicate bindings")
        object.__setattr__(self, "heads", sorted_heads)

    @classmethod
    def from_heads(cls, heads: Mapping[str, SubstrateHead] | tuple[SubstrateHead, ...]) -> WorldSnapshot:
        if isinstance(heads, tuple):
            return cls(heads)
        values: list[SubstrateHead] = []
        for binding, head in heads.items():
            if head.binding != binding:
                raise ValueError(f"snapshot key {binding!r} disagrees with head binding {head.binding!r}")
            values.append(head)
        return cls(tuple(values))

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> WorldSnapshot:
        heads: list[SubstrateHead] = []
        for binding, raw_head in value.items():
            if not isinstance(raw_head, dict):
                raise TypeError(f"snapshot head for {binding!r} must be a JSON object")
            heads.append(SubstrateHead.from_json(binding, raw_head))
        return cls(tuple(heads))

    def to_json(self) -> dict[str, dict[str, str]]:
        return {head.binding: head.to_json() for head in self.heads}

    def digest(self) -> str:
        return canonical_digest(self.to_json())

    def by_binding(self) -> dict[str, SubstrateHead]:
        return {head.binding: head for head in self.heads}

    def head_for(self, binding: str) -> SubstrateHead:
        try:
            return self.by_binding()[binding]
        except KeyError as exc:
            raise KeyError(f"world snapshot has no binding {binding!r}") from exc


@dataclass(frozen=True)
class OperationFinalRecord:
    """Immutable final operation evidence embedded in a world commit."""

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _canonical_operation_final_payload(self.payload))

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload)

    def digest(self) -> str:
        return canonical_digest(self.payload)


@dataclass(frozen=True)
class WorldRefPayload:
    """Opaque substrate payload pointing at a world snapshot."""

    world_store_id: str
    world_oid: str
    snapshot_digest: str
    schema: str = WORLD_REF_PAYLOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORLD_REF_PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported world ref payload schema: {self.schema!r}")
        for field_name, value in (
            ("world_store_id", self.world_store_id),
            ("world_oid", self.world_oid),
            ("snapshot_digest", self.snapshot_digest),
        ):
            if not value:
                raise ValueError(f"{field_name} is required")
        _validate_sha256_digest(self.snapshot_digest, "snapshot_digest")

    def to_json(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "world_store_id": self.world_store_id,
            "world_oid": self.world_oid,
            "snapshot_digest": self.snapshot_digest,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> WorldRefPayload:
        _reject_unexpected_keys(
            value,
            {"schema", "world_store_id", "world_oid", "snapshot_digest"},
            "world ref payload",
        )
        schema = value.get("schema")
        if schema != WORLD_REF_PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported world ref payload schema: {schema!r}")
        return cls(
            world_store_id=_required_str(value, "world_store_id"),
            world_oid=_required_str(value, "world_oid"),
            snapshot_digest=_required_str(value, "snapshot_digest"),
        )


def _canonical_operation_final_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _validate_operation_final_payload(value)
    for key in ("candidate_commits", "candidate_outcomes", "head_selections", "selection_evidence"):
        raw_items = value.get(key)
        if isinstance(raw_items, list):
            value[key] = sorted(
                (dict(item) if isinstance(item, dict) else item for item in raw_items), key=canonical_digest
            )
    return value


def _validate_operation_final_payload(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "operation_id",
        "selected",
        "candidate_commits",
        "candidate_outcomes",
        "head_selections",
        "selection_evidence",
    }
    _reject_unexpected_keys(value, expected_keys, "operation-final")
    if value.get("schema") != OPERATION_FINAL_SCHEMA:
        raise ValueError(f"unsupported operation-final schema: {value.get('schema')!r}")
    _required_str(value, "operation_id")
    selected = value.get("selected")
    if not isinstance(selected, dict) or not all(
        isinstance(binding, str) and binding and isinstance(head, str) and head for binding, head in selected.items()
    ):
        raise ValueError("operation-final selected must be a string map")
    for key in ("candidate_commits", "candidate_outcomes", "head_selections", "selection_evidence"):
        raw_items = value.get(key)
        if not isinstance(raw_items, list):
            raise TypeError(f"operation-final {key} must be a list")
        if not all(isinstance(item, dict) for item in raw_items):
            raise ValueError(f"operation-final {key} entries must be objects")


@dataclass(frozen=True)
class WorldTransition:
    """Transition metadata stored beside a world snapshot."""

    payload: dict[str, Any]


@dataclass(frozen=True)
class CandidateRevision:
    """A substrate revision protected by an operation-scoped candidate ref."""

    operation_id: str
    binding: str
    candidate_id: str
    store_id: str
    resource_id: str
    head: str
    ref: str


@dataclass(frozen=True)
class WorldCommit:
    """A decoded v2 world commit from the coordinator repository."""

    oid: str
    snapshot: WorldSnapshot
    transition: dict[str, Any]
    operation_final: dict[str, Any]
    manifest: dict[str, Any]
    locator_hints: dict[str, str]
    parent_oids: tuple[str, ...]


@dataclass(frozen=True)
class StructuredIssue:
    """Stable private diagnostic emitted by v2 world inspection surfaces."""

    code: str
    message: str
    severity: str = "error"
    world_oid: str | None = None
    operation_id: str | None = None
    store_id: str | None = None
    binding: str | None = None
    ref: str | None = None
    recovery_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("issue code is required")
        if not self.message:
            raise ValueError("issue message is required")
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError(f"unsupported issue severity: {self.severity!r}")


def _required_str(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{key} is required")
    return raw


def _optional_str(value: Mapping[str, object], key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{key} must be a non-empty string when present")
    return raw


def _reject_unexpected_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    extra_keys = set(value) - expected
    if extra_keys:
        raise ValueError(f"unexpected {label} fields: {sorted(extra_keys)!r}")


def _validate_sha256_digest(value: str, field: str) -> None:
    prefix = "sha256:"
    hex_digest = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(hex_digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in hex_digest)
    ):
        raise ValueError(f"{field} must be a sha256 digest")


def _is_git_oid(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value)
