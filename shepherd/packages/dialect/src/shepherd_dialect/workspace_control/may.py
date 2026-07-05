"""Slice-local ``may=`` lowering for workspace-control authority.

This module is intentionally narrower than the dialect-wide ``may=`` story.
It supports the current workspace-control filesystem/retained-output vertical
slice while older confinement and nucleus paths keep their existing lowering
rules until the full authority model converges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MayProfileName = Literal["ReadOnly", "ReadWrite", "Permissive"]
WorkspaceRepoAuthority = Literal["readonly", "readwrite"]

DEFAULT_WORKSPACE_MAY_PROFILE: MayProfileName = "ReadWrite"


class MayProfileError(ValueError):
    """Raised when a may profile cannot be used safely."""


class UnsupportedMayProfileError(MayProfileError):
    """Raised when a profile has no lowering in the workspace-control slice."""


class MayProfileWideningError(MayProfileError):
    """Raised when a caller tries to widen a task's declared authority."""


@dataclass(frozen=True)
class MayProfile:
    """Normalized workspace-control may profile.

    The rank is the current filesystem/workspace ordering. ``ReadOnly``
    is a strict subset of ``ReadWrite``; ``Permissive`` remains the legacy broad
    top profile but lowers to the same workspace GitRepo authority as
    ``ReadWrite`` in this slice.
    """

    name: MayProfileName
    rank: int
    workspace_repo_authority: WorkspaceRepoAuthority
    workspace_selection_can_mutate: bool


@dataclass(frozen=True)
class WorkspaceAuthorityDecision:
    """Resolved authority facts for one workspace-control run.

    This is the first internal decision boundary for the walking
    skeleton. It records the effective profile once, then exposes the concrete
    authority facts consumed by handle acquisition and retained-output
    selection.
    """

    task_default: MayProfile
    requested: MayProfile | None
    effective: MayProfile
    gitrepo_grant_clamp: Any | None = None

    @property
    def may_profile_name(self) -> MayProfileName:
        """Return the profile name persisted on the run record."""
        return self.effective.name

    @property
    def repo_authority(self) -> WorkspaceRepoAuthority:
        """Return the temporary skeleton GitRepo authority."""
        if self.gitrepo_grant_clamp is not None and not _gitrepo_grant_clamp_allows_mutation(self.gitrepo_grant_clamp):
            return "readonly"
        return self.effective.workspace_repo_authority

    @property
    def workspace_selection_can_mutate(self) -> bool:
        """Return whether retained-output selection may mutate workspace."""
        if self.gitrepo_grant_clamp is not None and not _gitrepo_grant_clamp_allows_mutation(self.gitrepo_grant_clamp):
            return False
        return self.effective.workspace_selection_can_mutate


_MAY_PROFILES: dict[str, MayProfile] = {
    "ReadOnly": MayProfile(
        name="ReadOnly",
        rank=10,
        workspace_repo_authority="readonly",
        workspace_selection_can_mutate=False,
    ),
    "ReadWrite": MayProfile(
        name="ReadWrite",
        rank=20,
        workspace_repo_authority="readwrite",
        workspace_selection_can_mutate=True,
    ),
    "Permissive": MayProfile(
        name="Permissive",
        rank=30,
        workspace_repo_authority="readwrite",
        workspace_selection_can_mutate=True,
    ),
}


def supported_may_profile_names() -> tuple[MayProfileName, ...]:
    """Return the supported names in authority order."""
    return ("ReadOnly", "ReadWrite", "Permissive")


def normalize_may_profile(value: str) -> MayProfile:
    """Return a canonical profile for ``value`` or fail closed."""
    if not isinstance(value, str) or not value:
        raise UnsupportedMayProfileError(f"may={value!r} is not a supported workspace-control profile")
    profile = _MAY_PROFILES.get(value)
    if profile is None:
        raise UnsupportedMayProfileError(f"may={value!r} has no workspace authority lowering")
    return profile


def canonical_may_profile_name(value: str) -> MayProfileName:
    """Return the canonical spelling for a supported profile."""
    return normalize_may_profile(value).name


def may_profile_allows(requested: MayProfile, ceiling: MayProfile) -> bool:
    """Return whether ``requested`` is no broader than ``ceiling``."""
    return requested.rank <= ceiling.rank


def resolve_run_may_profile(*, task_default: str, requested: str | None) -> MayProfile:
    """Resolve a run's effective profile without allowing call-site widening."""
    return resolve_workspace_authority_decision(task_default=task_default, requested=requested).effective


def resolve_workspace_authority_decision(
    *,
    task_default: str,
    requested: str | None,
    gitrepo_grant: Any | None = None,
) -> WorkspaceAuthorityDecision:
    """Resolve the workspace-control authority facts for a run."""
    ceiling = normalize_may_profile(task_default)
    requested_profile = normalize_may_profile(requested) if requested is not None else None
    effective = ceiling if requested_profile is None else requested_profile
    if not may_profile_allows(effective, ceiling):
        raise MayProfileWideningError(f"may={effective.name!r} exceeds task may_default={ceiling.name!r}")
    return WorkspaceAuthorityDecision(
        task_default=ceiling,
        requested=requested_profile,
        effective=effective,
        gitrepo_grant_clamp=_clamp_gitrepo_grant_to_profile(effective, gitrepo_grant),
    )


def repo_authority_for_may(value: str) -> WorkspaceRepoAuthority:
    """Lower a supported profile to the temporary GitRepo authority."""
    return normalize_may_profile(value).workspace_repo_authority


def _clamp_gitrepo_grant_to_profile(profile: MayProfile, gitrepo_grant: Any | None) -> Any | None:
    if gitrepo_grant is None:
        return None
    from shepherd_dialect.workspace_control.authority import (
        GitRepoGrantClause,
        GitRepoGrantDescriptor,
        clamp_gitrepo_grants,
    )

    profile_grant = GitRepoGrantDescriptor(
        grant_ref=f"workspace-effective-profile:{profile.name}",
        clauses=(
            GitRepoGrantClause(
                binding_ref="workspace",
                mutates=False if not profile.workspace_selection_can_mutate else None,
            ),
        ),
    )
    return clamp_gitrepo_grants(
        parent_ceiling=profile_grant,
        requested=gitrepo_grant,
        grant_ref=f"workspace-effective:{profile.name}:{gitrepo_grant.digest}",
    )


def _gitrepo_grant_clamp_allows_mutation(grant_clamp: Any) -> bool:
    effective = getattr(grant_clamp, "effective", None)
    clauses = getattr(effective, "clauses", ())
    return any(getattr(clause, "mutates", None) is not False for clause in clauses)
