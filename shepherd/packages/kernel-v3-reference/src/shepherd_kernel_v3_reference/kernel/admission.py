"""Production AdmittedObservation bundle + validator for the `-lite` profile.

Per `260521-0600-kernel.md` §"Validation Responsibilities" → "Admission
Validation" and 2026-05-24 §"Post-#72 design pass" item F.

The validator is additive on top of `resume_kernel_replay(...)`: it adds
source/admission-basis coherence, frontier-prefix, restart-artifact
agreement, and observation-schema validation, then the caller delegates
to `resume_kernel_replay(...)` for execution.

The bundle's `request` field bridges the identity gap between
`request.source_key` (content-addressed; keyed in `state.open_requests`)
and `source.source_ref` (trace-local: `declaration:0`, `resumption:N`).

The reference spike at `shepherd_kernel_v3_reference.spikes.admission_validation`
covers steps 1-6 of the cheapest-first check order as a 9-case standing
regression; this production module covers all 7 steps including
observation-schema validation (step 7), addressing the taxonomic
inconsistency surfaced by `260524-observation-stream-spike.md` §"What
Was Surprising" #1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shepherd_kernel_v3_reference.kernel.replay import (
    ContinuationReplayArtifact,
    ExternalEffectRequest,
    HostCompleted,
    KernelReplayState,
)
from shepherd_kernel_v3_reference.semantic import (
    AdmissionBasis,
    ContinuationSource,
    ObservedFrontier,
)


class AdmittedObservationError(ValueError):
    """Raised when an AdmittedObservation does not validate against state.

    Carries a `rejection_class` naming the failing validator step so the
    `KernelRejection(kind="observation-admission", rejection_class=...)`
    construction in #76 can populate the wire-shape field without parsing
    the diagnostic string.
    """

    def __init__(self, message: str, *, rejection_class: RejectionClass) -> None:
        super().__init__(message)
        self.rejection_class: RejectionClass = rejection_class


RejectionClass = Literal[
    "state-level",
    "source-open",
    "open-request-agreement",
    "source-basis-coherence",
    "frontier-prefix",
    "restart-artifact-agreement",
    "observation-schema",
]


@dataclass(frozen=True)
class AdmittedObservation:
    """Bundle for resuming a kernel run via observed host completion.

    Per `260521-0600-kernel.md` §"API Shape" and 2026-05-24
    §"Post-#72 design pass" item F. The bundle carries `request` to bridge
    `request.source_key` (content-addressed; keyed in `state.open_requests`)
    and `source.source_ref` (trace-local).
    """

    source: ContinuationSource
    restart_artifact: ContinuationReplayArtifact
    admission_basis: AdmissionBasis
    observation: HostCompleted
    request: ExternalEffectRequest

    def __post_init__(self) -> None:
        if not isinstance(self.source, ContinuationSource):
            raise TypeError("AdmittedObservation.source must be a ContinuationSource")
        if not isinstance(self.restart_artifact, ContinuationReplayArtifact):
            raise TypeError("AdmittedObservation.restart_artifact must be a ContinuationReplayArtifact")
        if not isinstance(self.admission_basis, AdmissionBasis):
            raise TypeError("AdmittedObservation.admission_basis must be an AdmissionBasis")
        if not isinstance(self.observation, HostCompleted):
            raise TypeError("AdmittedObservation.observation must be a HostCompleted")
        if not isinstance(self.request, ExternalEffectRequest):
            raise TypeError("AdmittedObservation.request must be an ExternalEffectRequest")


def _frontier_is_prefix(frontier: ObservedFrontier, state_transition_refs: tuple[str, ...]) -> bool:
    """Sequence-prefix relation: `observed.record_refs ⊑ state.transition_refs`.

    Per 2026-05-24 §"Post-#72 design pass" item F: sequence-prefix, NOT set
    inclusion. Empty frontier is admitted (initial run case).
    """

    if not frontier.record_refs:
        return True
    if len(frontier.record_refs) > len(state_transition_refs):
        return False
    return tuple(frontier.record_refs) == tuple(state_transition_refs[: len(frontier.record_refs)])


def validate_admitted_observation(
    observation: AdmittedObservation,
    state: KernelReplayState,
) -> None:
    """Validate an AdmittedObservation against the current replay state.

    Check order is cheapest-first; raises `AdmittedObservationError` on the
    first failure with a stable diagnostic and `rejection_class` naming the
    failing step.

      1. state-level                — state is not terminal, not rejected
      2. source-open                — request.source_key open AND not consumed
      3. open-request-agreement     — open_request.program_ref / declaration_ref match
      4. source-basis-coherence     — source ↔ admission_basis fields agree
      5. frontier-prefix            — basis.observed_frontier ⊑ state.transition_refs
      6. restart-artifact-agreement — restart_artifact.{program_ref, source_ref,
                                       operation_result_schema_ref} agree with source/state
      7. observation-schema         — observation.value validates against
                                       source.operation_result_schema_ref
                                       (skipped if source.operation_result_schema_ref is None)
    """

    src = observation.source
    basis = observation.admission_basis
    obs = observation.observation
    request = observation.request

    # 1. state-level
    if state.terminal:
        raise AdmittedObservationError(
            "KernelReplayState is terminal; no admission possible",
            rejection_class="state-level",
        )
    if state.rejected:
        raise AdmittedObservationError(
            "KernelReplayState is rejected; no admission possible",
            rejection_class="state-level",
        )

    # 2. source-open: one-shot + currently open
    if request.source_key in state.consumed_source_keys:
        raise AdmittedObservationError(
            f"source_key {request.source_key!r} already consumed (one-shot violation)",
            rejection_class="source-open",
        )
    open_request = state.open_requests.get(request.source_key)
    if open_request is None:
        raise AdmittedObservationError(
            f"source_key {request.source_key!r} is not open in KernelReplayState",
            rejection_class="source-open",
        )

    # 3. open-request agreement
    if open_request.program_ref != state.program_ref:
        raise AdmittedObservationError(
            "OpenReplayRequest program_ref does not match state",
            rejection_class="open-request-agreement",
        )
    if open_request.declaration_ref != src.declaration_ref:
        raise AdmittedObservationError(
            f"OpenReplayRequest.declaration_ref {open_request.declaration_ref!r} != "
            f"source.declaration_ref {src.declaration_ref!r}",
            rejection_class="open-request-agreement",
        )

    # 4. source ↔ admission_basis coherence
    if basis.source_ref != src.source_ref:
        raise AdmittedObservationError(
            f"AdmissionBasis.source_ref {basis.source_ref!r} != source.source_ref {src.source_ref!r}",
            rejection_class="source-basis-coherence",
        )
    if basis.source_kind != src.source_kind:
        raise AdmittedObservationError(
            f"AdmissionBasis.source_kind {basis.source_kind!r} != source.source_kind {src.source_kind!r}",
            rejection_class="source-basis-coherence",
        )
    if basis.source_generation != src.source_generation:
        raise AdmittedObservationError(
            "AdmissionBasis.source_generation != source.source_generation",
            rejection_class="source-basis-coherence",
        )
    if basis.one_shot_key != src.one_shot_key:
        raise AdmittedObservationError(
            "AdmissionBasis.one_shot_key != source.one_shot_key",
            rejection_class="source-basis-coherence",
        )
    if basis.program_ref != state.program_ref:
        raise AdmittedObservationError(
            f"AdmissionBasis.program_ref {basis.program_ref!r} != state.program_ref {state.program_ref!r}",
            rejection_class="source-basis-coherence",
        )

    # 5. frontier-prefix
    if not _frontier_is_prefix(basis.observed_frontier, state.transition_refs):
        raise AdmittedObservationError(
            f"AdmissionBasis.observed_frontier is not a sequence-prefix of "
            f"state.transition_refs ({list(basis.observed_frontier.record_refs)} "
            f"vs {list(state.transition_refs)})",
            rejection_class="frontier-prefix",
        )

    # 6. restart-artifact agreement
    artifact = observation.restart_artifact
    if artifact.program_ref != state.program_ref:
        raise AdmittedObservationError(
            f"restart_artifact.program_ref {artifact.program_ref!r} != state.program_ref {state.program_ref!r}",
            rejection_class="restart-artifact-agreement",
        )
    if artifact.source_ref != src.source_ref:
        raise AdmittedObservationError(
            f"restart_artifact.source_ref {artifact.source_ref!r} != source.source_ref {src.source_ref!r}",
            rejection_class="restart-artifact-agreement",
        )
    if artifact.operation_result_schema_ref != src.operation_result_schema_ref:
        raise AdmittedObservationError(
            f"restart_artifact.operation_result_schema_ref "
            f"{artifact.operation_result_schema_ref!r} != "
            f"source.operation_result_schema_ref {src.operation_result_schema_ref!r}",
            rejection_class="restart-artifact-agreement",
        )

    # 7. observation schema — resolve operation_result_schema_ref via the
    # program's schema catalog and validate observation.value against it.
    # Skipped if source carries no schema ref (legacy/non-typed effects).
    schema_ref = src.operation_result_schema_ref
    if schema_ref is not None:
        schema_defs = state.prepared_program.program.schemas
        schema_def = schema_defs.get(schema_ref)
        if schema_def is None:
            raise AdmittedObservationError(
                f"operation_result_schema_ref {schema_ref!r} is not present in the "
                f"program's schema catalog (state.prepared_program.program.schemas)",
                rejection_class="observation-schema",
            )
        err = schema_def.schema.validate(obs.value)
        if err is not None:
            raise AdmittedObservationError(
                f"observation.value does not match schema {schema_ref!r}: {err}",
                rejection_class="observation-schema",
            )


__all__ = [
    "AdmittedObservation",
    "AdmittedObservationError",
    "RejectionClass",
    "validate_admitted_observation",
]
