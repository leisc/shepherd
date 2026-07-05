"""Durable run argument and artifact input reference helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from shepherd_dialect.workspace_control.errors import WorkspaceControlError

if TYPE_CHECKING:
    from shepherd_dialect.workspace_control.run_outputs import RunOutput
    from shepherd_dialect.workspace_control.workspace import ShepherdWorkspace

JsonObject = dict[str, object]

RUN_ARGS_SCHEMA = "shepherd.workspace_control.run_args.v1"
RUN_ARTIFACT_INPUT_SCHEMA = "skeleton.run_artifact_input.v1"
REDACTED_ARGUMENT_SCHEMA = "shepherd.workspace_control.redacted_python_argument.v1"


@dataclass(frozen=True)
class RunArtifactInputRef:
    """Durable citation for one artifact path inside a retained run output."""

    run_ref: str
    output_id: str
    path: str
    output_name: str = "workspace"
    binding: str = "workspace"
    label: str | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("run_ref", self.run_ref),
            ("output_id", self.output_id),
            ("output_name", self.output_name),
            ("binding", self.binding),
            ("path", self.path),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"RunArtifactInputRef.{field_name} must be a non-empty string")
        _validate_relative_posix_path(self.path, field_name="RunArtifactInputRef.path")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("RunArtifactInputRef.label must be a non-empty string or None")
        if self.content_digest is not None and not _is_sha256_digest(self.content_digest):
            raise ValueError("RunArtifactInputRef.content_digest must be a sha256 digest or None")

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "kind": RUN_ARTIFACT_INPUT_SCHEMA,
            "run_ref": self.run_ref,
            "output_id": self.output_id,
            "output_name": self.output_name,
            "binding": self.binding,
            "path": self.path,
            "label": self.label,
        }
        if self.content_digest is not None:
            payload["content_digest"] = self.content_digest
        return payload

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> RunArtifactInputRef:
        kind = value.get("kind")
        if kind != RUN_ARTIFACT_INPUT_SCHEMA:
            raise ValueError(f"unsupported run artifact input ref kind: {kind!r}")
        return cls(
            run_ref=_required_str(value, "run_ref"),
            output_id=_required_str(value, "output_id"),
            output_name=_required_str(value, "output_name"),
            binding=_required_str(value, "binding"),
            path=_required_str(value, "path"),
            label=_optional_str(value, "label"),
            content_digest=_optional_str(value, "content_digest"),
        )


@dataclass(frozen=True)
class RunOutputArtifact:
    """Path-specific public view over a retained run output artifact."""

    output: RunOutput
    path: str

    def __post_init__(self) -> None:
        _validate_relative_posix_path(self.path, field_name="run output artifact path")

    def read_file(self) -> tuple[bytes, int] | None:
        """Read this artifact from its retained output."""
        return self.output.read_file(self.path)

    def read_text(self, *, encoding: str = "utf-8") -> str:
        """Read this artifact as text, failing closed if it is missing."""
        data = self.read_file()
        if data is None:
            raise WorkspaceControlError(f"run output artifact {self.path!r} is not present")
        return data[0].decode(encoding)

    def read_json(self, *, encoding: str = "utf-8") -> object:
        """Read this artifact as JSON, failing closed if it is missing or malformed."""
        return json.loads(self.read_text(encoding=encoding))

    def to_input(self, *, label: str | None = None, include_digest: bool = True) -> RunArtifactInputRef:
        """Return a durable input citation for this retained artifact."""
        data = self.read_file()
        if data is None:
            raise WorkspaceControlError(f"cannot cite missing run output artifact {self.path!r}")
        owner = self.output.owner
        if owner.kind != "run" or owner.run_id is None:
            raise WorkspaceControlError("artifact input refs require run-owned outputs")
        return RunArtifactInputRef(
            run_ref=owner.run_id,
            output_id=self.output.output_id,
            output_name=self.output.output_name,
            binding=self.output.binding,
            path=self.path,
            label=label,
            content_digest=_bytes_digest(data[0]) if include_digest else None,
        )


def build_run_args_payload(
    *,
    run_ref: str,
    args: Mapping[str, object],
    created_at: str,
) -> JsonObject:
    """Build the durable run-argument payload stored beside run records."""
    if not isinstance(run_ref, str) or not run_ref:
        raise ValueError("run_ref must be a non-empty string")
    payload = _jsonable_argument_value(dict(args))
    assert isinstance(payload, dict)
    args_digest = _json_digest(payload)
    input_refs = [ref.to_json() for ref in iter_run_artifact_input_refs(payload)]
    args_ref = run_args_ref(run_ref=run_ref, args_digest=args_digest)
    return {
        "schema": RUN_ARGS_SCHEMA,
        "args_ref": args_ref,
        "run_ref": run_ref,
        "args_digest": args_digest,
        "payload": payload,
        "payload_digest": _json_digest(payload),
        "input_refs": input_refs,
        "created_at": created_at,
    }


def run_args_ref(*, run_ref: str, args_digest: str) -> str:
    """Return the stable public reference for one run's persisted argument payload."""
    digest_tail = args_digest.split(":", 1)[1] if ":" in args_digest else args_digest
    return f"{run_ref}:args:{digest_tail[:16]}"


def iter_run_artifact_input_refs(value: object) -> tuple[RunArtifactInputRef, ...]:
    """Return all artifact input refs embedded in a JSON-shaped argument value."""
    refs: list[RunArtifactInputRef] = []
    _collect_run_artifact_input_refs(value, refs)
    return tuple(refs)


def validate_run_artifact_input_refs(workspace: ShepherdWorkspace, value: object) -> None:
    """Validate all artifact input refs embedded in a public argument payload."""
    for ref in iter_run_artifact_input_refs(value):
        _validate_run_artifact_input_ref(workspace, ref)


def _validate_run_artifact_input_ref(workspace: ShepherdWorkspace, ref: RunArtifactInputRef) -> None:
    matches = [
        output
        for output in workspace.runs.outputs(run_ref=ref.run_ref, binding=ref.binding)
        if output.output_id == ref.output_id
    ]
    if not matches:
        raise WorkspaceControlError(f"run artifact input ref cannot resolve output {ref.output_id!r}")
    output = matches[0]
    if output.output_name != ref.output_name:
        raise WorkspaceControlError("run artifact input ref output_name disagrees with retained output")
    data = output.read_file(ref.path)
    if data is None:
        raise WorkspaceControlError(f"run artifact input ref path is not present: {ref.path!r}")
    if ref.content_digest is not None and _bytes_digest(data[0]) != ref.content_digest:
        raise WorkspaceControlError(f"run artifact input ref digest mismatch for {ref.path!r}")


def _collect_run_artifact_input_refs(value: object, refs: list[RunArtifactInputRef]) -> None:
    if isinstance(value, RunArtifactInputRef):
        refs.append(value)
        return
    if isinstance(value, Mapping):
        if value.get("kind") == RUN_ARTIFACT_INPUT_SCHEMA:
            refs.append(RunArtifactInputRef.from_json(value))
            return
        for child in value.values():
            _collect_run_artifact_input_refs(child, refs)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            _collect_run_artifact_input_refs(child, refs)


def _jsonable_argument_value(value: object) -> object:
    if isinstance(value, RunArtifactInputRef):
        return value.to_json()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return {
            "kind": REDACTED_ARGUMENT_SCHEMA,
            "type": "bytes",
            "byte_length": len(value),
            "content_digest": _bytes_digest(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_argument_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable_argument_value(child) for child in value]
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        raw = to_json()
        if isinstance(raw, Mapping):
            return _jsonable_argument_value(raw)
    return {
        "kind": REDACTED_ARGUMENT_SCHEMA,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value)[:240],
    }


def _validate_relative_posix_path(path: str, *, field_name: str) -> None:
    parsed = PurePosixPath(path)
    if path in {"", ".", ".."} or parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{field_name} must be a relative POSIX path")


def _required_str(value: Mapping[str, object], field_name: str) -> str:
    raw = value.get(field_name)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field_name} must be a non-empty string")
    return raw


def _optional_str(value: Mapping[str, object], field_name: str) -> str | None:
    raw = value.get(field_name)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field_name} must be a non-empty string or null")
    return raw


def _is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    prefix, sep, digest = value.partition(":")
    return prefix == "sha256" and sep == ":" and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def _bytes_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _bytes_digest(encoded)
