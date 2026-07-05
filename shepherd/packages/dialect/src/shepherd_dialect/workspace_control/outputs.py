"""Product RunOutput reconstruction for workspace-control run ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from shepherd2.kernel.facts import TRUSTED_READ_CONTEXT, FactDraft
from shepherd2.schemas.run_outputs import (
    RUN_OUTPUT_SCHEMA,
    ProjectedRunOutputDescriptor,
    RunOutputDescriptor,
    RunOutputDescriptorLocator,
    RunOutputMaterializationKind,
    RunOutputOwner,
    RunOutputRef,
    resolve_run_output_descriptor_from_store,
    run_output_descriptor_fact,
    run_output_descriptor_locator_from_payload,
    run_output_descriptor_locator_payload,
    run_output_identity_for,
)
from vcs_core.types import RetainedOutputIdentity

from shepherd_dialect.workspace_control.schemas import RunOutputCitationRef, RunRetainedCustody, TraceRef

if TYPE_CHECKING:
    from collections.abc import Iterable

_GROUND_REF = "refs/vcscore/ground"
_DIRECT_LOOKUP_UNAVAILABLE = object()


class RunOutputResolutionError(ValueError):
    """Raised when a run-output citation cannot be resolved to product output state."""


class TraceDescriptorNotResolvedError(RunOutputResolutionError):
    """Raised when product output reconstruction lacks trace descriptor authority."""


class DescriptorResolver(Protocol):
    """Resolve a trace-owned RunOutput descriptor from its durable locator."""

    def __call__(self, locator: RunOutputDescriptorLocator) -> ProjectedRunOutputDescriptor: ...


@dataclass(frozen=True)
class RunOutputPublicationDraft:
    """Run-ledger citation material waiting on trace-store descriptor identity."""

    trace_ref: TraceRef
    output_name: str
    output_id: str
    binding: str
    store_id: str
    resource_id: str
    materialization_kind: RunOutputMaterializationKind
    custody_ref: str
    output_world_oid: str
    parent_basis_world_oid: str
    citation_payload: dict[str, object]

    def descriptor_fact(self) -> FactDraft:
        """Build the trace-owned descriptor fact for this output."""
        return run_output_descriptor_fact(
            execution_id=self.trace_ref.execution_id,
            output_name=self.output_name,
            world_binding=self.binding,
            citation=dict(self.citation_payload),
        )

    def citation_ref(self, *, descriptor_fact_id: str) -> RunOutputCitationRef:
        """Build the run-ledger citation after the descriptor fact is retained."""
        locator = RunOutputDescriptorLocator(
            execution_id=self.trace_ref.execution_id,
            output_name=self.output_name,
            frontier_id=self.trace_ref.frontier_id,
            descriptor_fact_id=descriptor_fact_id,
        )
        return RunOutputCitationRef(
            output_name=self.output_name,
            output_id=self.output_id,
            trace_ref=self.trace_ref,
            descriptor_locator=run_output_descriptor_locator_payload(locator),
            binding=self.binding,
            store_id=self.store_id,
            resource_id=self.resource_id,
            materialization_kind=self.materialization_kind,
            custody_ref=self.custody_ref,
            output_world_oid=self.output_world_oid,
            parent_basis_world_oid=self.parent_basis_world_oid,
        )


@dataclass(frozen=True)
class _RetainedOutputFields:
    scope_name: str
    scope_ref: str
    scope_instance_id: str
    parent_ref: str
    parent_scope_name: str
    parent_scope_instance_id: str | None
    binding: str
    output_world_oid: str
    handoff_ref: str
    parent_basis_world_oid: str
    store_id: str
    resource_id: str
    candidate_id: str
    candidate_ref: str
    candidate_head: str


@dataclass(frozen=True)
class RunOutputResolver:
    """Join run-ledger citations, retained custody, and trace-owned descriptors."""

    mg: Any
    parent: Any = None
    binding: str | None = None
    state: str | None = None
    trace_store: Any = None
    descriptor_resolver: DescriptorResolver | None = None
    read_context: Any = None

    def __post_init__(self) -> None:
        if self.trace_store is not None and self.descriptor_resolver is not None:
            raise TypeError("RunOutputResolver accepts either trace_store or descriptor_resolver, not both")

    def resolve(self, citations: Iterable[RunOutputCitationRef]) -> tuple[RunOutputRef, ...]:
        citations_tuple = tuple(citations)
        if not citations_tuple:
            return ()
        candidate_citations = tuple(
            citation for citation in citations_tuple if self.binding is None or citation.binding == self.binding
        )
        if not candidate_citations:
            return ()
        descriptor_resolver: DescriptorResolver | None = None
        retained_rows: tuple[Any, ...] | None = None
        refs: list[RunOutputRef] = []
        for citation in candidate_citations:
            if descriptor_resolver is None:
                descriptor_resolver = self._require_descriptor_resolver()
            descriptor_record = self._resolve_trace_descriptor(citation, descriptor_resolver)
            retained = self._direct_retained_output(citation, descriptor_record)
            if retained is _DIRECT_LOOKUP_UNAVAILABLE:
                if retained_rows is None:
                    retained_rows = self._filtered_retained_output_rows()
                retained = _matching_retained_row(citation, retained_rows)
            if retained is None:
                raise RunOutputResolutionError(
                    f"no retained-output custody row matches run output {citation.output_name!r}"
                )
            if not self._matches_filters(retained):
                continue
            refs.append(_run_output_ref_from_authorities(citation, retained, descriptor_record))
        return tuple(refs)

    def _require_descriptor_resolver(self) -> DescriptorResolver:
        if self.descriptor_resolver is not None:
            return self.descriptor_resolver
        if self.trace_store is not None:
            read_context = TRUSTED_READ_CONTEXT if self.read_context is None else self.read_context

            def _resolve(locator: RunOutputDescriptorLocator) -> ProjectedRunOutputDescriptor:
                return resolve_run_output_descriptor_from_store(self.trace_store, read_context, locator)

            return _resolve
        raise TraceDescriptorNotResolvedError(
            "product run-output queries require trace_store or descriptor_resolver; "
            "use run_output_citations() for raw run-ledger citations"
        )

    def _filtered_retained_output_rows(self) -> tuple[Any, ...] | None:
        return self._list_retained_output_rows(parent=self.parent, binding=self.binding, state=self.state)

    def _matches_filters(self, retained: Any) -> bool:
        if self.parent is not None:
            parent_ref = _required_attr_str(self.parent, "ref", "parent")
            if getattr(retained, "parent_ref", None) != parent_ref:
                return False
        if self.binding is not None and getattr(retained, "binding", None) != self.binding:
            return False
        return self.state is None or getattr(retained, "state", None) == self.state

    def _list_retained_output_rows(
        self,
        *,
        parent: Any,
        binding: str | None,
        state: str | None,
    ) -> tuple[Any, ...]:
        reader = getattr(self.mg, "list_retained_outputs", None)
        if reader is None:
            raise TypeError("workspace-control output queries require VcsCore.list_retained_outputs")
        return tuple(reader(parent=parent, binding=binding, state=state))

    def _resolve_trace_descriptor(
        self,
        citation: RunOutputCitationRef,
        descriptor_resolver: DescriptorResolver,
    ) -> ProjectedRunOutputDescriptor:
        locator = run_output_descriptor_locator_from_payload(dict(citation.descriptor_locator))
        try:
            record = descriptor_resolver(locator)
        except RunOutputResolutionError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TraceDescriptorNotResolvedError("trace-owned RunOutput descriptor could not be resolved") from exc
        if not isinstance(record, ProjectedRunOutputDescriptor):
            raise TraceDescriptorNotResolvedError("trace descriptor resolver returned an unsupported record")
        if record.locator != locator:
            raise RunOutputResolutionError("trace descriptor locator disagrees with run output citation")
        return record

    def _direct_retained_output(
        self,
        citation: RunOutputCitationRef,
        descriptor_record: ProjectedRunOutputDescriptor,
    ) -> Any:
        reader = getattr(self.mg, "get_retained_output", None)
        if not callable(reader):
            return _DIRECT_LOOKUP_UNAVAILABLE
        identity = _retained_identity_from_descriptor_payload(citation, descriptor_record)
        try:
            return reader(identity)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RunOutputResolutionError("direct retained-output lookup rejected custody identity") from exc


def run_output_publication_from_seal_handoff(
    handoff: Any,
    *,
    parent: Any,
    trace_ref: TraceRef,
    output_name: str | None = None,
    materialization_kind: RunOutputMaterializationKind = "tree",
) -> RunOutputPublicationDraft:
    """Assemble resolver-compatible output citation material from vcs-core seal custody."""
    if not isinstance(trace_ref, TraceRef):
        raise TypeError("run output publication requires TraceRef")
    if materialization_kind not in {"tree", "external"}:
        raise ValueError(f"unsupported run output materialization kind: {materialization_kind!r}")

    parent_ref = _required_attr_str(parent, "ref", "parent")
    handoff_parent_ref = _required_attr_str(handoff, "parent_ref", "seal handoff")
    if handoff_parent_ref != parent_ref:
        raise ValueError("seal handoff parent_ref disagrees with parent scope")

    binding = _required_attr_str(handoff, "binding", "seal handoff")
    stable_output_name = output_name or binding
    parent_scope_name = _required_attr_str(parent, "name", "parent")
    parent_scope_instance_id = _parent_scope_instance_id(parent_ref, parent)
    scope_name = _required_attr_str(handoff, "scope_name", "seal handoff")
    scope_ref = _required_attr_str(handoff, "scope_ref", "seal handoff")
    scope_instance_id = _required_attr_str(handoff, "scope_instance_id", "seal handoff")
    output_world_oid = _required_attr_str(handoff, "output_world_oid", "seal handoff")
    handoff_ref = _required_attr_str(handoff, "handoff_ref", "seal handoff")
    candidate_id = _required_attr_str(handoff, "candidate_id", "seal handoff")
    candidate_head = _required_attr_str(handoff, "candidate_head", "seal handoff")
    candidate_ref = _required_attr_str(handoff, "candidate_ref", "seal handoff")
    parent_basis_world_oid = _required_attr_str(handoff, "parent_basis_world_oid", "seal handoff")
    store_id = _required_attr_str(handoff, "store_id", "seal handoff")
    resource_id = _required_attr_str(handoff, "resource_id", "seal handoff")
    changed_paths = _changed_paths_tuple(getattr(handoff, "changed_paths", ()), "seal handoff changed_paths")
    identity = run_output_identity_for(
        output_name=stable_output_name,
        binding=binding,
        parent_scope_name=parent_scope_name,
        parent_ref=parent_ref,
        parent_scope_instance_id=parent_scope_instance_id,
        scope_name=scope_name,
        scope_ref=scope_ref,
        scope_instance_id=scope_instance_id,
        candidate_id=candidate_id,
        candidate_head=candidate_head,
        output_world_oid=output_world_oid,
        handoff_ref=handoff_ref,
    )
    payload: dict[str, object] = {
        "schema": RUN_OUTPUT_SCHEMA,
        "output_name": stable_output_name,
        "parent_scope_name": parent_scope_name,
        "parent_ref": parent_ref,
        "scope_name": scope_name,
        "scope_ref": scope_ref,
        "scope_instance_id": scope_instance_id,
        "binding": binding,
        "output_world_oid": output_world_oid,
        "handoff_ref": handoff_ref,
        "candidate_id": candidate_id,
        "candidate_head": candidate_head,
        "candidate_ref": candidate_ref,
        "parent_basis_world_oid": parent_basis_world_oid,
        "store_id": store_id,
        "resource_id": resource_id,
        "materialization_kind": materialization_kind,
        "retained_handle_head": candidate_head,
        "changed_paths": list(changed_paths),
        "trace_run_id": trace_ref.run_id,
        "trace_execution_id": trace_ref.execution_id,
        "trace_frontier_id": trace_ref.frontier_id,
    }
    if parent_scope_instance_id is not None:
        payload["parent_scope_instance_id"] = parent_scope_instance_id
    return RunOutputPublicationDraft(
        trace_ref=trace_ref,
        output_name=stable_output_name,
        output_id=identity.output_id,
        binding=binding,
        store_id=store_id,
        resource_id=resource_id,
        materialization_kind=materialization_kind,
        custody_ref=handoff_ref,
        output_world_oid=output_world_oid,
        parent_basis_world_oid=parent_basis_world_oid,
        citation_payload=payload,
    )


def run_output_publication_from_retained_row(
    retained: Any,
    *,
    trace_ref: TraceRef,
    output_name: str | None = None,
    materialization_kind: RunOutputMaterializationKind = "tree",
) -> RunOutputPublicationDraft:
    """Assemble resolver-compatible output citation material from retained custody inventory."""
    if not isinstance(trace_ref, TraceRef):
        raise TypeError("run output publication requires TraceRef")
    if materialization_kind not in {"tree", "external"}:
        raise ValueError(f"unsupported run output materialization kind: {materialization_kind!r}")

    parent_ref = _required_attr_str(retained, "parent_ref", "retained output")
    binding = _required_attr_str(retained, "binding", "retained output")
    stable_output_name = output_name or binding
    parent_scope_name = _required_attr_str(retained, "parent_scope_name", "retained output")
    parent_scope_instance_id = _retained_parent_scope_instance_id(parent_ref, retained)
    scope_name = _required_attr_str(retained, "scope_name", "retained output")
    scope_ref = _required_attr_str(retained, "scope_ref", "retained output")
    scope_instance_id = _required_attr_str(retained, "scope_instance_id", "retained output")
    output_world_oid = _required_attr_str(retained, "output_world_oid", "retained output")
    handoff_ref = _required_attr_str(retained, "handoff_ref", "retained output")
    candidate_id = _required_attr_str(retained, "candidate_id", "retained output")
    candidate_head = _required_attr_str(retained, "candidate_head", "retained output")
    candidate_ref = _required_attr_str(retained, "candidate_ref", "retained output")
    parent_basis_world_oid = _required_attr_str(retained, "parent_basis_world_oid", "retained output")
    store_id = _required_attr_str(retained, "store_id", "retained output")
    resource_id = _required_attr_str(retained, "resource_id", "retained output")
    changed_paths = _changed_paths_tuple(getattr(retained, "changed_paths", ()), "retained output changed_paths")
    identity = run_output_identity_for(
        output_name=stable_output_name,
        binding=binding,
        parent_scope_name=parent_scope_name,
        parent_ref=parent_ref,
        parent_scope_instance_id=parent_scope_instance_id,
        scope_name=scope_name,
        scope_ref=scope_ref,
        scope_instance_id=scope_instance_id,
        candidate_id=candidate_id,
        candidate_head=candidate_head,
        output_world_oid=output_world_oid,
        handoff_ref=handoff_ref,
    )
    payload: dict[str, object] = {
        "schema": RUN_OUTPUT_SCHEMA,
        "output_name": stable_output_name,
        "parent_scope_name": parent_scope_name,
        "parent_ref": parent_ref,
        "scope_name": scope_name,
        "scope_ref": scope_ref,
        "scope_instance_id": scope_instance_id,
        "binding": binding,
        "output_world_oid": output_world_oid,
        "handoff_ref": handoff_ref,
        "candidate_id": candidate_id,
        "candidate_head": candidate_head,
        "candidate_ref": candidate_ref,
        "parent_basis_world_oid": parent_basis_world_oid,
        "store_id": store_id,
        "resource_id": resource_id,
        "materialization_kind": materialization_kind,
        "retained_handle_head": candidate_head,
        "changed_paths": list(changed_paths),
        "trace_run_id": trace_ref.run_id,
        "trace_execution_id": trace_ref.execution_id,
        "trace_frontier_id": trace_ref.frontier_id,
    }
    if parent_scope_instance_id is not None:
        payload["parent_scope_instance_id"] = parent_scope_instance_id
    return RunOutputPublicationDraft(
        trace_ref=trace_ref,
        output_name=stable_output_name,
        output_id=identity.output_id,
        binding=binding,
        store_id=store_id,
        resource_id=resource_id,
        materialization_kind=materialization_kind,
        custody_ref=handoff_ref,
        output_world_oid=output_world_oid,
        parent_basis_world_oid=parent_basis_world_oid,
        citation_payload=payload,
    )


def _required_attr_str(value: Any, field_name: str, label: str) -> str:
    raw = getattr(value, field_name, None)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} field {field_name!r} must be a non-empty string")
    return raw


def _parent_scope_instance_id(parent_ref: str, parent: Any) -> str | None:
    if parent_ref == _GROUND_REF:
        return None
    return _required_attr_str(parent, "instance_id", "parent")


def _retained_parent_scope_instance_id(parent_ref: str, retained: Any) -> str | None:
    raw = getattr(retained, "parent_scope_instance_id", None)
    if parent_ref == _GROUND_REF:
        if raw is not None:
            raise ValueError("retained output parent_scope_instance_id must be absent for ground parent")
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("retained output field 'parent_scope_instance_id' must be a non-empty string")
    return raw


def _changed_paths_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{field_name} must be a list or tuple")
    paths: list[str] = []
    for path in value:
        if not isinstance(path, str) or not path:
            raise ValueError(f"{field_name} must contain non-empty strings")
        paths.append(path)
    return tuple(paths)


def _matching_retained_row(citation: RunOutputCitationRef, rows: tuple[Any, ...]) -> Any | None:
    for row in rows:
        if _retained_row_matches(citation, row):
            return row
    return None


def _retained_identity_from_descriptor_payload(
    citation: RunOutputCitationRef,
    descriptor_record: ProjectedRunOutputDescriptor,
) -> RetainedOutputIdentity:
    payload = dict(descriptor_record.citation_payload)
    parent_ref = _required_payload_str(payload, "parent_ref")
    raw_parent_scope_instance_id = payload.get("parent_scope_instance_id")
    if parent_ref == _GROUND_REF:
        parent_scope_instance_id = None
    elif isinstance(raw_parent_scope_instance_id, str) and raw_parent_scope_instance_id:
        parent_scope_instance_id = raw_parent_scope_instance_id
    else:
        raise RunOutputResolutionError(
            "trace descriptor payload field 'parent_scope_instance_id' must be present for retained parent"
        )
    return RetainedOutputIdentity(
        scope_name=_required_payload_str(payload, "scope_name"),
        scope_ref=_required_payload_str(payload, "scope_ref"),
        scope_instance_id=_required_payload_str(payload, "scope_instance_id"),
        parent_ref=parent_ref,
        parent_scope_name=_required_payload_str(payload, "parent_scope_name"),
        parent_scope_instance_id=parent_scope_instance_id,
        binding=citation.binding,
        output_world_oid=citation.output_world_oid,
        handoff_ref=citation.custody_ref,
        parent_basis_world_oid=citation.parent_basis_world_oid,
        store_id=citation.store_id,
        resource_id=citation.resource_id,
        candidate_id=_required_payload_str(payload, "candidate_id"),
        candidate_ref=_required_payload_str(payload, "candidate_ref"),
        candidate_head=_required_payload_str(payload, "candidate_head"),
    )


def _retained_row_matches(citation: RunOutputCitationRef, row: Any) -> bool:
    return RunRetainedCustody.from_output_citation(citation).matches_retained_output(row)


def _run_output_ref_from_authorities(
    citation: RunOutputCitationRef,
    retained: Any,
    descriptor_record: ProjectedRunOutputDescriptor,
) -> RunOutputRef:
    required = _required_retained_fields(retained)
    identity = run_output_identity_for(
        output_name=citation.output_name,
        binding=required.binding,
        parent_scope_name=required.parent_scope_name,
        parent_ref=required.parent_ref,
        parent_scope_instance_id=required.parent_scope_instance_id,
        scope_name=required.scope_name,
        scope_ref=required.scope_ref,
        scope_instance_id=required.scope_instance_id,
        candidate_id=required.candidate_id,
        candidate_head=required.candidate_head,
        output_world_oid=required.output_world_oid,
        handoff_ref=required.handoff_ref,
    )
    if identity.output_id != citation.output_id:
        raise RunOutputResolutionError("run output citation output_id disagrees with retained custody tuple")

    payload = dict(descriptor_record.citation_payload)
    descriptor = _descriptor_from_trace_payload(payload)
    changed_paths = _changed_paths(getattr(retained, "changed_paths", ()), "retained changed_paths")
    _validate_trace_descriptor_payload(payload, citation=citation, retained=required, changed_paths=changed_paths)

    owner = RunOutputOwner(
        kind="run",
        run_id=citation.trace_ref.run_id,
        execution_id=citation.trace_ref.execution_id,
        frontier_id=citation.trace_ref.frontier_id,
    )
    state = getattr(retained, "state", None)
    if not isinstance(state, str) or not state:
        raise RunOutputResolutionError("retained-output row is missing state")
    return RunOutputRef(
        identity=identity,
        owner=owner,
        descriptor=descriptor,
        state=state,
        parent_basis_world_oid=required.parent_basis_world_oid,
        candidate_ref=required.candidate_ref,
        store_id=required.store_id,
        resource_id=required.resource_id,
        changed_paths=changed_paths,
        settlement_ref=getattr(retained, "settlement_ref", None),
        invalid_reason=getattr(retained, "invalid_reason", None),
        descriptor_locator=descriptor_record.locator,
    )


def _required_retained_fields(retained: Any) -> _RetainedOutputFields:
    field_names = (
        "scope_name",
        "scope_ref",
        "scope_instance_id",
        "parent_ref",
        "parent_scope_name",
        "binding",
        "output_world_oid",
        "handoff_ref",
        "parent_basis_world_oid",
        "store_id",
        "resource_id",
        "candidate_id",
        "candidate_ref",
        "candidate_head",
    )
    required: dict[str, str] = {}
    missing: list[str] = []
    for field_name in field_names:
        value = getattr(retained, field_name, None)
        if not isinstance(value, str) or not value:
            missing.append(field_name)
            continue
        required[field_name] = value
    if missing:
        raise RunOutputResolutionError(f"retained-output row is missing custody fields: {', '.join(missing)}")
    raw_parent_instance_id = getattr(retained, "parent_scope_instance_id", None)
    if required["parent_ref"] == _GROUND_REF:
        parent_scope_instance_id = None
    elif isinstance(raw_parent_instance_id, str) and raw_parent_instance_id:
        parent_scope_instance_id = raw_parent_instance_id
    else:
        raise RunOutputResolutionError("retained-output row is missing custody fields: parent_scope_instance_id")
    return _RetainedOutputFields(
        scope_name=required["scope_name"],
        scope_ref=required["scope_ref"],
        scope_instance_id=required["scope_instance_id"],
        parent_ref=required["parent_ref"],
        parent_scope_name=required["parent_scope_name"],
        parent_scope_instance_id=parent_scope_instance_id,
        binding=required["binding"],
        output_world_oid=required["output_world_oid"],
        handoff_ref=required["handoff_ref"],
        parent_basis_world_oid=required["parent_basis_world_oid"],
        store_id=required["store_id"],
        resource_id=required["resource_id"],
        candidate_id=required["candidate_id"],
        candidate_ref=required["candidate_ref"],
        candidate_head=required["candidate_head"],
    )


def _descriptor_from_trace_payload(payload: dict[str, Any]) -> RunOutputDescriptor:
    if payload.get("schema") != RUN_OUTPUT_SCHEMA:
        raise RunOutputResolutionError("trace descriptor payload is not a RunOutput payload")
    materialization_kind = _required_payload_member(payload, "materialization_kind", {"tree", "external"})
    return RunOutputDescriptor(
        output_name=_required_payload_str(payload, "output_name"),
        world_binding=_required_payload_str(payload, "binding"),
        store_id=_required_payload_str(payload, "store_id"),
        resource_id=_required_payload_str(payload, "resource_id"),
        materialization_kind=cast("RunOutputMaterializationKind", materialization_kind),
    )


def _validate_trace_descriptor_payload(
    payload: dict[str, Any],
    *,
    citation: RunOutputCitationRef,
    retained: _RetainedOutputFields,
    changed_paths: tuple[str, ...],
) -> None:
    expected = {
        "output_name": citation.output_name,
        "binding": citation.binding,
        "parent_scope_name": retained.parent_scope_name,
        "parent_ref": retained.parent_ref,
        "scope_name": retained.scope_name,
        "scope_ref": retained.scope_ref,
        "scope_instance_id": retained.scope_instance_id,
        "output_world_oid": retained.output_world_oid,
        "handoff_ref": retained.handoff_ref,
        "candidate_id": retained.candidate_id,
        "candidate_head": retained.candidate_head,
        "candidate_ref": retained.candidate_ref,
        "parent_basis_world_oid": retained.parent_basis_world_oid,
        "store_id": retained.store_id,
        "resource_id": retained.resource_id,
        "materialization_kind": citation.materialization_kind,
        "trace_run_id": citation.trace_ref.run_id,
        "trace_execution_id": citation.trace_ref.execution_id,
        "trace_frontier_id": citation.trace_ref.frontier_id,
    }
    if retained.parent_scope_instance_id is None:
        if payload.get("parent_scope_instance_id") is not None:
            raise RunOutputResolutionError(
                "trace descriptor payload field 'parent_scope_instance_id' must be absent for ground parent"
            )
    else:
        expected["parent_scope_instance_id"] = retained.parent_scope_instance_id
    for field_name, expected_value in expected.items():
        _require_payload_equal(payload, field_name, expected_value)
    if _changed_paths(payload.get("changed_paths"), "trace descriptor changed_paths") != changed_paths:
        raise RunOutputResolutionError("trace descriptor changed_paths disagree with retained custody")


def _require_payload_equal(payload: dict[str, Any], field_name: str, expected: str) -> None:
    actual = _required_payload_str(payload, field_name)
    if actual != expected:
        raise RunOutputResolutionError(
            f"trace descriptor payload field {field_name!r} disagrees with run citation or retained custody"
        )


def _required_payload_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RunOutputResolutionError(f"trace descriptor payload field {field_name!r} must be a non-empty string")
    return value


def _required_payload_member(
    payload: dict[str, Any],
    field_name: str,
    allowed: set[str],
) -> str:
    value = _required_payload_str(payload, field_name)
    if value not in allowed:
        raise RunOutputResolutionError(f"trace descriptor payload field {field_name!r} is unsupported: {value!r}")
    return value


def _changed_paths(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise RunOutputResolutionError(f"{field_name} must be a list or tuple")
    paths: list[str] = []
    for path in value:
        if not isinstance(path, str) or not path:
            raise RunOutputResolutionError(f"{field_name} must contain non-empty strings")
        paths.append(path)
    return tuple(paths)


__all__ = [
    "RUN_OUTPUT_SCHEMA",
    "DescriptorResolver",
    "RunOutputPublicationDraft",
    "RunOutputResolutionError",
    "RunOutputResolver",
    "TraceDescriptorNotResolvedError",
    "run_output_publication_from_retained_row",
    "run_output_publication_from_seal_handoff",
]
