"""Materialization plan computation, execution, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from vcs_core._errors import VcsCoreError
from vcs_core._substrate_runtime import InternalMaterializerProvider
from vcs_core._upstream import PreflightResult
from vcs_core.types import CommitInfo, DiffSummary, FileChange, MaterializationPhase, MaterializationPlan, Status

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


PHASE_ORDER = {"auto": 0, "compensable": 1, "none": 2}
PreflightMode = Literal["pure", "recording"]


def _is_operation_boundary_commit(commit: CommitInfo) -> bool:
    return commit.metadata.get("type") in {"OperationStarted", "OperationCompleted", "OperationAborted"}


def _semantic_pending_commit_indices(pending_commits: Sequence[CommitInfo]) -> dict[str, int]:
    indices: dict[str, int] = {}
    semantic_index = 0
    for commit in pending_commits:
        if _is_operation_boundary_commit(commit):
            continue
        indices[commit.oid] = semantic_index
        semantic_index += 1
    return indices


@dataclass(frozen=True)
class MaterializationUnit:
    """Internal executable materialization work item."""

    unit_id: str
    materializer_key: str
    substrate: str
    target_id: str
    reversibility: str
    commit_index: int
    upstream_aware: bool = False
    basis_token: str | None = None
    frontier: str | None = None
    file_changes: tuple[FileChange, ...] = ()
    intents: tuple[dict[str, object], ...] = ()


class MaterializationPlanningStore(Protocol):
    """Store capability required to compute materialization work."""

    def walk_pending(self) -> Sequence[CommitInfo]: ...

    def diff(self) -> DiffSummary: ...

    def status(self) -> Status: ...


@dataclass(frozen=True)
class PlannedMaterialization:
    """Internal plan carrying both units and the public summary DTO."""

    units: tuple[MaterializationUnit, ...]
    plan: MaterializationPlan


@dataclass(frozen=True)
class MaterializationPreflightBlocker:
    """Expected non-ready preflight verdict for a planned materialization unit."""

    unit: MaterializationUnit
    result: PreflightResult

    @property
    def message(self) -> str:
        reason = f": {self.result.reason}" if self.result.reason else ""
        return (
            f"Materialization preflight failed for unit {self.unit.unit_id!r} (status={self.result.status!r}){reason}"
        )


class MaterializationPreflightError(VcsCoreError, RuntimeError):
    """Expected materialization preflight blocker distinct from planner contract bugs."""

    def __init__(self, blockers: tuple[MaterializationPreflightBlocker, ...]) -> None:
        if not blockers:
            msg = "Materialization preflight failed without blockers."
            raise ValueError(msg)
        self.blockers = blockers
        super().__init__(blockers[0].message)


@dataclass(frozen=True)
class MaterializationAssessment:
    """Materialization plan plus expected preflight blockers, without applying units."""

    planned: PlannedMaterialization
    preflight_blockers: tuple[MaterializationPreflightBlocker, ...] = ()


@dataclass(frozen=True)
class VerificationResult:
    """Internal verification verdict for a materialization unit."""

    ok: bool
    reason: str | None = None


class InternalMaterializer(Protocol):
    """Internal planner/apply seam for substrate-specific push work."""

    materializer_key: str

    def collect_units(
        self,
        *,
        pending_commits: Sequence[CommitInfo],
        diff: DiffSummary,
        status: Status,
    ) -> Sequence[MaterializationUnit]: ...

    def apply_units(self, units: Sequence[MaterializationUnit]) -> None: ...


@runtime_checkable
class InternalPreflightProvider(Protocol):
    """Preflight hook required for upstream-aware materialization units."""

    def preflight_units(
        self,
        units: Sequence[MaterializationUnit],
        *,
        mode: PreflightMode = "pure",
    ) -> Mapping[str, PreflightResult]: ...


@runtime_checkable
class InternalRunArtifactProvider(Protocol):
    """Optional hook for preparing recovery artifacts before push side effects."""

    def prepare_run_artifacts(
        self,
        units: Sequence[MaterializationUnit],
        *,
        run_directory: Path,
    ) -> Mapping[str, Mapping[str, object]]: ...


@runtime_checkable
class InternalVerificationProvider(Protocol):
    """Optional hook for verify-oriented crash recovery."""

    def verify_units(
        self,
        units: Sequence[MaterializationUnit],
        *,
        run_state: Mapping[str, Mapping[str, object]],
        run_directory: Path,
    ) -> Mapping[str, VerificationResult]: ...


class _FilesystemMaterializer:
    """Compatibility materializer for the current diff-based filesystem path."""

    materializer_key = "builtin:filesystem"

    def __init__(self, substrate: object | None = None) -> None:
        self._substrate = substrate

    def collect_units(
        self,
        *,
        pending_commits: Sequence[CommitInfo],
        diff: DiffSummary,
        status: Status,
    ) -> Sequence[MaterializationUnit]:
        del status
        if not diff.files:
            return ()

        return (
            MaterializationUnit(
                unit_id="filesystem:workspace",
                materializer_key=self.materializer_key,
                substrate="filesystem",
                target_id="workspace",
                reversibility="auto",
                commit_index=self._commit_index(pending_commits, diff=diff),
                file_changes=tuple(diff.files),
            ),
        )

    def apply_units(self, units: Sequence[MaterializationUnit]) -> None:
        del units
        if self._substrate is None:
            return
        materialize_workspace = getattr(self._substrate, "materialize_workspace", None)
        if materialize_workspace is None:
            msg = "Filesystem substrate does not provide planner-owned workspace materialization."
            raise RuntimeError(msg)
        materialize_workspace()

    @staticmethod
    def _commit_index(pending_commits: Sequence[CommitInfo], *, diff: DiffSummary) -> int:
        semantic_indices = _semantic_pending_commit_indices(pending_commits)
        for commit in pending_commits:
            if commit.metadata.get("substrate") != "filesystem":
                continue
            if commit.metadata.get("type") not in {
                "FileCreate",
                "FilePatch",
                "FileDelete",
                "WorkspaceBaselineAdopt",
            }:
                continue
            return semantic_indices[commit.oid]
        if diff.files:
            msg = "Pending workspace diff could not be ordered because no filesystem planning commit was found."
            raise RuntimeError(msg)
        return 0


DEFAULT_MATERIALIZERS: tuple[InternalMaterializer, ...] = (_FilesystemMaterializer(),)


def build_materializers(
    substrates: Sequence[object] | None = None,
) -> tuple[InternalMaterializer, ...]:
    """Build the authoritative materializer set for the active substrates."""
    materializers: list[InternalMaterializer] = []
    if substrates is not None:
        for substrate in substrates:
            if not isinstance(substrate, InternalMaterializerProvider):
                continue
            materializers.extend(substrate.materializers())

    return tuple(materializers)


def _normalize_materializers(
    materializers: Sequence[InternalMaterializer] | None = None,
) -> dict[str, InternalMaterializer]:
    materializer_map: dict[str, InternalMaterializer] = {}
    source = DEFAULT_MATERIALIZERS if materializers is None else materializers
    for materializer in source:
        key = materializer.materializer_key
        if key in materializer_map:
            msg = f"Duplicate materializer registration for key {key!r}."
            raise ValueError(msg)
        materializer_map[key] = materializer
    return materializer_map


def _required_materializer_keys(pending_commits: Sequence[CommitInfo]) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    for commit in pending_commits:
        key = commit.metadata.get("materializer_key")
        if key is None:
            continue
        if not isinstance(key, str):
            msg = f"Pending commit {commit.oid[:12]} has non-string materializer_key metadata."
            raise RuntimeError(msg)  # noqa: TRY004
        required.setdefault(key, []).append(commit.oid)
    return required


def _validate_required_materializers(
    pending_commits: Sequence[CommitInfo],
    *,
    diff: DiffSummary,
    materializers: dict[str, InternalMaterializer],
) -> None:
    if diff.files and _FilesystemMaterializer.materializer_key not in materializers:
        msg = (
            "Pending filesystem materialization requires unavailable materializer "
            f"{_FilesystemMaterializer.materializer_key!r}."
        )
        raise RuntimeError(msg)

    required = _required_materializer_keys(pending_commits)
    missing = sorted(key for key in required if key not in materializers)
    if not missing:
        return

    key = missing[0]
    sample_oids = ", ".join(oid[:8] for oid in required[key][:3])
    msg = f"Pending materialization requires unavailable materializer {key!r} (commits: {sample_oids})."
    raise RuntimeError(msg)


def _order_units(units: Sequence[MaterializationUnit]) -> tuple[MaterializationUnit, ...]:
    return tuple(
        sorted(
            units,
            key=lambda unit: (
                PHASE_ORDER.get(unit.reversibility, len(PHASE_ORDER)),
                unit.commit_index,
                unit.unit_id,
            ),
        )
    )


def _validate_upstream_basis_contract(units: Sequence[MaterializationUnit]) -> None:
    grouped_basis_tokens: dict[tuple[str, str, str], set[str]] = {}
    for unit in units:
        if not unit.upstream_aware:
            continue
        if unit.frontier is None:
            msg = f"Upstream-aware unit {unit.unit_id!r} is missing a frontier."
            raise RuntimeError(msg)
        if unit.basis_token is None:
            msg = f"Upstream-aware unit {unit.unit_id!r} is missing a basis_token."
            raise RuntimeError(msg)
        grouped_basis_tokens.setdefault(
            (unit.materializer_key, unit.target_id, unit.frontier),
            set(),
        ).add(unit.basis_token)

    for (materializer_key, target_id, frontier), basis_tokens in grouped_basis_tokens.items():
        if len(basis_tokens) == 1:
            continue
        msg = (
            "Upstream-aware materialization requires exactly one basis_token per "
            f"(materializer_key, target_id, frontier). Got {len(basis_tokens)} for "
            f"{materializer_key!r}, {target_id!r}, frontier={frontier!r}."
        )
        raise RuntimeError(msg)


def _run_preflight(
    units: Sequence[MaterializationUnit],
    *,
    materializers: Mapping[str, InternalMaterializer],
    mode: PreflightMode,
) -> tuple[MaterializationPreflightBlocker, ...]:
    _validate_upstream_basis_contract(units)

    grouped_units: dict[str, list[MaterializationUnit]] = {}
    for unit in units:
        grouped_units.setdefault(unit.materializer_key, []).append(unit)

    blockers: list[MaterializationPreflightBlocker] = []
    for materializer_key, batch in grouped_units.items():
        materializer = materializers[materializer_key]
        requires_preflight = any(unit.upstream_aware for unit in batch)
        if not isinstance(materializer, InternalPreflightProvider):
            if requires_preflight:
                msg = (
                    "Upstream-aware materialization requires a preflight-capable materializer. "
                    f"Missing preflight support for {materializer_key!r}."
                )
                raise RuntimeError(msg)
            continue
        results = materializer.preflight_units(tuple(batch), mode=mode)
        for unit in batch:
            result = results.get(unit.unit_id)
            if result is None:
                if unit.upstream_aware:
                    msg = (
                        "Materialization preflight must return a verdict for every "
                        f"upstream-aware unit. Missing result for {unit.unit_id!r}."
                    )
                    raise RuntimeError(msg)
                result = PreflightResult(status="ready")
            if result.ok:
                continue
            blockers.append(MaterializationPreflightBlocker(unit=unit, result=result))
    return tuple(blockers)


def _summarize_units(units: Sequence[MaterializationUnit], *, status: Status) -> MaterializationPlan:
    phases: list[MaterializationPhase] = []
    for reversibility in ("auto", "compensable", "none"):
        phase_units = [unit for unit in units if unit.reversibility == reversibility]
        if not phase_units:
            continue
        file_changes: list[FileChange] = []
        intents: list[dict[str, object]] = []
        for unit in phase_units:
            file_changes.extend(unit.file_changes)
            intents.extend(unit.intents)
        phases.append(
            MaterializationPhase(
                reversibility=reversibility,
                file_changes=file_changes,
                intents=intents,
            )
        )
    return MaterializationPlan(phases=phases, commits_ahead=status.commits_ahead)


def prepare_run_artifacts(
    planned: PlannedMaterialization,
    *,
    materializers: Sequence[InternalMaterializer] | None = None,
    run_directory: Path,
) -> dict[str, dict[str, object]]:
    """Prepare durable recovery artifacts before push side effects begin."""
    materializer_map = _normalize_materializers(materializers)
    grouped_units: dict[str, list[MaterializationUnit]] = {}
    for unit in planned.units:
        grouped_units.setdefault(unit.materializer_key, []).append(unit)

    unit_state: dict[str, dict[str, object]] = {}
    for materializer_key, batch in grouped_units.items():
        materializer = materializer_map[materializer_key]
        if not isinstance(materializer, InternalRunArtifactProvider):
            continue
        prepared = materializer.prepare_run_artifacts(tuple(batch), run_directory=run_directory)
        for unit_id, state in prepared.items():
            unit_state[unit_id] = dict(state)
    return unit_state


def assess_materialization(
    store: MaterializationPlanningStore,
    *,
    materializers: Sequence[InternalMaterializer] | None = None,
    skip_preflight: bool = False,
    preflight_mode: PreflightMode = "pure",
) -> MaterializationAssessment:
    """Build the internal materialization plan and collect expected preflight blockers."""
    pending_commits = store.walk_pending()
    diff = store.diff()
    status = store.status()

    materializer_map = _normalize_materializers(materializers)
    _validate_required_materializers(
        pending_commits,
        diff=diff,
        materializers=materializer_map,
    )

    units: list[MaterializationUnit] = []
    for materializer in materializer_map.values():
        units.extend(
            materializer.collect_units(
                pending_commits=pending_commits,
                diff=diff,
                status=status,
            )
        )

    ordered_units = _order_units(units)
    preflight_blockers: tuple[MaterializationPreflightBlocker, ...] = ()
    if not skip_preflight:
        preflight_blockers = _run_preflight(ordered_units, materializers=materializer_map, mode=preflight_mode)
    planned = PlannedMaterialization(
        units=ordered_units,
        plan=_summarize_units(ordered_units, status=status),
    )
    return MaterializationAssessment(planned=planned, preflight_blockers=preflight_blockers)


def plan_materialization(
    store: MaterializationPlanningStore,
    *,
    materializers: Sequence[InternalMaterializer] | None = None,
    skip_preflight: bool = False,
    preflight_mode: PreflightMode = "pure",
) -> PlannedMaterialization:
    """Build the internal materialization units and public plan summary."""
    assessment = assess_materialization(
        store,
        materializers=materializers,
        skip_preflight=skip_preflight,
        preflight_mode=preflight_mode,
    )
    if assessment.preflight_blockers:
        raise MaterializationPreflightError(assessment.preflight_blockers)
    return assessment.planned


def apply_materialization(
    planned: PlannedMaterialization,
    *,
    materializers: Sequence[InternalMaterializer] | None = None,
    on_units_completed: Callable[[Sequence[MaterializationUnit]], None] | None = None,
) -> None:
    """Apply materialization units in planner order."""
    materializer_map = _normalize_materializers(materializers)
    batch: list[MaterializationUnit] = []
    current_key: str | None = None

    def flush() -> None:
        if not batch or current_key is None:
            return
        materializer = materializer_map.get(current_key)
        if materializer is None:
            unit = batch[0]
            msg = f"Materialization unit {unit.unit_id!r} requires unavailable materializer {unit.materializer_key!r}."
            raise RuntimeError(msg)
        materializer.apply_units(tuple(batch))
        if on_units_completed is not None:
            on_units_completed(tuple(batch))

    for unit in planned.units:
        if current_key is None:
            current_key = unit.materializer_key
        if unit.materializer_key != current_key:
            flush()
            batch = []
            current_key = unit.materializer_key
        batch.append(unit)
    flush()


def verify_materialization(
    planned: PlannedMaterialization,
    *,
    materializers: Sequence[InternalMaterializer] | None = None,
    run_state: Mapping[str, Mapping[str, object]],
    run_directory: Path,
) -> None:
    """Verify that previously-started materialization completed successfully."""
    materializer_map = _normalize_materializers(materializers)
    grouped_units: dict[str, list[MaterializationUnit]] = {}
    for unit in planned.units:
        grouped_units.setdefault(unit.materializer_key, []).append(unit)

    for materializer_key, batch in grouped_units.items():
        materializer = materializer_map.get(materializer_key)
        if materializer is None:
            unit = batch[0]
            msg = f"Materialization unit {unit.unit_id!r} requires unavailable materializer {unit.materializer_key!r}."
            raise RuntimeError(msg)
        if not isinstance(materializer, InternalVerificationProvider):
            raise NotImplementedError(
                f"Materialization verification requires a verify-capable materializer for {materializer_key!r}."
            )
        results = materializer.verify_units(
            tuple(batch),
            run_state=run_state,
            run_directory=run_directory,
        )
        for unit in batch:
            result = results.get(unit.unit_id)
            if result is None:
                msg = f"Materialization verification is missing a verdict for {unit.unit_id!r}."
                raise RuntimeError(msg)
            if result.ok:
                continue
            reason = f": {result.reason}" if result.reason else ""
            raise RuntimeError(f"Materialization verification failed for unit {unit.unit_id!r}{reason}")


def compute_materialization_plan(store: MaterializationPlanningStore) -> MaterializationPlan:
    """Backward-compatible public summary over the internal planner."""
    return plan_materialization(store).plan
