"""Shared authority metadata helpers for workspace-control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shepherd_dialect.workspace_control.authority import (
    GitRepoAuthorityDecisionPolicy,
    GitRepoAuthoritySurface,
    GitRepoGrantClamp,
    GitRepoGrantClause,
    GitRepoGrantDescriptor,
    clamp_gitrepo_grants,
    gitrepo_authority_surface_for_grant,
)
from shepherd_dialect.workspace_control.schemas import RunAuthorityContext

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shepherd_dialect.workspace_control.may import MayProfile, WorkspaceAuthorityDecision

WORKSPACE_FILESYSTEM_AUTHORITY_BINDING_ROOTS = {"workspace": ""}
WORKSPACE_AUTHORITY_CONTEXT_SCHEMA = "shepherd.workspace-control.authority-context.v1"


@dataclass(frozen=True)
class ValidatedRunAuthorityContext:
    """Semantically validated executable view of a persisted run authority context."""

    raw: RunAuthorityContext
    task_default: MayProfile
    requested: MayProfile | None
    profile: MayProfile
    grant_clamp: GitRepoGrantClamp
    effective_grant: GitRepoGrantDescriptor
    surface: GitRepoAuthoritySurface
    policy: GitRepoAuthorityDecisionPolicy
    uses_signature_gitrepo_grant: bool


def workspace_gitrepo_grant_for_profile(profile: MayProfile, *, grant_ref: str) -> GitRepoGrantDescriptor:
    """Return the coarse whole-workspace GitRepo grant for one may profile."""
    # ReadOnly only authorizes non-mutating facts. ReadWrite/Permissive leave
    # mutates unconstrained in this coarse whole-workspace slice.
    mutates = False if profile.workspace_repo_authority == "readonly" else None
    return GitRepoGrantDescriptor(
        grant_ref=grant_ref,
        clauses=(GitRepoGrantClause(binding_ref="workspace", mutates=mutates),),
    )


def workspace_filesystem_authority_grant_clamp(decision: WorkspaceAuthorityDecision) -> GitRepoGrantClamp:
    """Build the effective GitRepo grant clamp for one workspace-control run."""
    if decision.gitrepo_grant_clamp is not None:
        return decision.gitrepo_grant_clamp
    parent_ceiling = workspace_gitrepo_grant_for_profile(
        decision.task_default,
        grant_ref=f"workspace-task-default:{decision.task_default.name}",
    )
    requested_profile = decision.requested or decision.task_default
    requested = workspace_gitrepo_grant_for_profile(
        requested_profile,
        grant_ref=f"workspace-requested:{requested_profile.name}",
    )
    return clamp_gitrepo_grants(
        parent_ceiling=parent_ceiling,
        requested=requested,
        grant_ref=f"workspace-effective:{decision.effective.name}",
    )


def workspace_retained_output_authority_policy_for_profile(profile: MayProfile) -> GitRepoAuthorityDecisionPolicy:
    """Return the retained-output selection classifier policy for one profile."""
    return GitRepoAuthorityDecisionPolicy(
        routes=("retained_output_selection",),
        binding_refs=("workspace",),
        allowed_classification_bases=("exact_tree_diff",),
        allow_changed_paths_fallback=True,
        allow_changed_paths_fallback_for_path_sensitive=False,
        reason_code_subject="retained_output",
        allow_reason_code=f"may_{profile.name}_retained_output_selection_match",
        outside_match_reason_code="retained_output_selection_outside_effective_match",
        denied_when_mutates_outside_match=not profile.workspace_selection_can_mutate,
        mutating_outside_match_reason_code=f"may_{profile.name}_retained_output_selection_mutates_workspace",
        invalid_view_reason_code_prefix="retained_output_match_view_invalid",
        match_evaluation_failed_reason_code_prefix="retained_output_match_evaluation_failed",
        unclassifiable_reason_code="unclassifiable_retained_output",
        default_monitor_basis="carrier_check_at_commit",
    )


def workspace_retained_output_authority_policy_for_grant(
    profile: MayProfile,
    grant_clamp: GitRepoGrantClamp,
) -> GitRepoAuthorityDecisionPolicy:
    """Return the retained-output classifier policy for an explicit GitRepo grant."""
    binding_refs = tuple(sorted({clause.binding_ref for clause in grant_clamp.effective.clauses}))
    return GitRepoAuthorityDecisionPolicy(
        routes=("retained_output_selection",),
        binding_refs=binding_refs or None,
        allowed_classification_bases=("exact_tree_diff",),
        allow_changed_paths_fallback=True,
        allow_changed_paths_fallback_for_path_sensitive=False,
        reason_code_subject="retained_output",
        allow_reason_code="gitrepo_grant_retained_output_selection_match",
        outside_match_reason_code="gitrepo_grant_retained_output_selection_outside_effective_grant",
        denied_when_mutates_outside_match=True,
        mutating_outside_match_reason_code=("gitrepo_grant_retained_output_selection_mutates_outside_effective_grant"),
        invalid_view_reason_code_prefix="retained_output_match_view_invalid",
        match_evaluation_failed_reason_code_prefix="retained_output_match_evaluation_failed",
        unclassifiable_reason_code="unclassifiable_retained_output",
        default_monitor_basis="carrier_check_at_commit",
    )


def workspace_retained_output_authority_surface_for_grant(grant: GitRepoGrantDescriptor) -> GitRepoAuthoritySurface:
    """Lower an effective GitRepo grant to the retained-output selection Match surface."""
    return gitrepo_authority_surface_for_grant(
        grant,
        label="workspace-control.retained-output.effective",
        route="retained_output_selection",
    )


def run_authority_context_for_decision(decision: WorkspaceAuthorityDecision) -> RunAuthorityContext:
    """Build the durable authority metadata persisted on workspace-control runs."""
    grant_clamp = workspace_filesystem_authority_grant_clamp(decision)
    surface = workspace_retained_output_authority_surface_for_grant(grant_clamp.effective)
    if decision.gitrepo_grant_clamp is None:
        policy = workspace_retained_output_authority_policy_for_profile(decision.effective)
    else:
        policy = workspace_retained_output_authority_policy_for_grant(decision.effective, grant_clamp)
    return RunAuthorityContext(
        task_default_may=decision.task_default.name,
        requested_may=None if decision.requested is None else decision.requested.name,
        effective_may=decision.effective.name,
        repo_authority=decision.repo_authority,
        workspace_selection_can_mutate=decision.workspace_selection_can_mutate,
        grant_clamp=grant_clamp.to_descriptor(),
        effective_grant=grant_clamp.effective.to_descriptor(),
        effective_grant_digest=grant_clamp.effective.digest,
        effective_match_digest=surface.effective_match_digest,
        authority_surface_plan_digest=surface.authority_surface_plan_digest,
        classifier_policy=policy.to_descriptor(),
    )


def validate_run_authority_context(context: RunAuthorityContext) -> ValidatedRunAuthorityContext:
    """Validate persisted run authority evidence before using it as settlement policy."""
    if not isinstance(context, RunAuthorityContext):
        raise TypeError("run authority context must be RunAuthorityContext")

    from shepherd_dialect.workspace_control.may import (
        WorkspaceAuthorityDecision,
        may_profile_allows,
        normalize_may_profile,
    )

    task_default = normalize_may_profile(context.task_default_may)
    requested = normalize_may_profile(context.requested_may) if context.requested_may is not None else None
    profile = normalize_may_profile(context.effective_may)
    expected_effective = task_default if requested is None else requested
    if profile != expected_effective:
        raise ValueError("run authority context effective may disagrees with task default/requested may")
    if not may_profile_allows(profile, task_default):
        raise ValueError("run authority context effective may exceeds task default may")

    grant_clamp = GitRepoGrantClamp.from_descriptor(context.grant_clamp)
    effective_grant = GitRepoGrantDescriptor.from_descriptor(context.effective_grant)
    if grant_clamp.effective.to_descriptor() != effective_grant.to_descriptor():
        raise ValueError("run authority context effective grant disagrees with grant clamp")
    if effective_grant.digest != context.effective_grant_digest:
        raise ValueError("run authority context effective grant digest disagrees with descriptor")

    decision = WorkspaceAuthorityDecision(
        task_default=task_default,
        requested=requested,
        effective=profile,
        gitrepo_grant_clamp=grant_clamp,
    )
    if decision.repo_authority != context.repo_authority:
        raise ValueError("run authority context repo authority disagrees with effective grant")
    if decision.workspace_selection_can_mutate != context.workspace_selection_can_mutate:
        raise ValueError("run authority context workspace-selection authority disagrees with effective grant")

    surface = workspace_retained_output_authority_surface_for_grant(effective_grant)
    if surface.effective_match_digest != context.effective_match_digest:
        raise ValueError("run authority context effective match digest disagrees with descriptor")
    if surface.authority_surface_plan_digest != context.authority_surface_plan_digest:
        raise ValueError("run authority context authority-surface plan digest disagrees with descriptor")

    uses_signature_gitrepo_grant = grant_clamp.requested.grant_ref.startswith("signature:")
    if uses_signature_gitrepo_grant:
        policy = workspace_retained_output_authority_policy_for_grant(profile, grant_clamp)
    else:
        policy = workspace_retained_output_authority_policy_for_profile(profile)
    if policy.to_descriptor() != context.classifier_policy:
        raise ValueError("run authority context classifier policy disagrees with effective authority")

    return ValidatedRunAuthorityContext(
        raw=context,
        task_default=task_default,
        requested=requested,
        profile=profile,
        grant_clamp=grant_clamp,
        effective_grant=effective_grant,
        surface=surface,
        policy=policy,
        uses_signature_gitrepo_grant=uses_signature_gitrepo_grant,
    )


def vcscore_authority_context_for_run_authority_context(
    validated: ValidatedRunAuthorityContext | RunAuthorityContext,
    *,
    transaction_kind: str,
    shepherd_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project validated run authority context into vcs-core evidence metadata."""
    if isinstance(validated, RunAuthorityContext):
        validated = validate_run_authority_context(validated)
    if not isinstance(transaction_kind, str) or not transaction_kind:
        raise ValueError("transaction_kind must be a non-empty string")
    context: dict[str, object] = {
        "schema": WORKSPACE_AUTHORITY_CONTEXT_SCHEMA,
        "source": "shepherd.workspace_control",
        "transaction_kind": transaction_kind,
        "may_profile": validated.profile.name,
        "profile": {
            "name": validated.profile.name,
            "workspace_repo_authority": validated.profile.workspace_repo_authority,
            "workspace_selection_can_mutate": validated.profile.workspace_selection_can_mutate,
        },
        "effective_match_digest": validated.surface.effective_match_digest,
        "authority_surface_plan_digest": validated.surface.authority_surface_plan_digest,
        "grant": {
            "grant_ref": validated.grant_clamp.effective.grant_ref,
            "grant_digest": validated.grant_clamp.effective.digest,
        },
        "clamp": {
            "digest": validated.grant_clamp.digest,
            "descriptor": validated.grant_clamp.to_descriptor(),
            "parent_ceiling": validated.grant_clamp.parent_ceiling.to_descriptor(),
            "requested": validated.grant_clamp.requested.to_descriptor(),
            "effective": validated.grant_clamp.effective.to_descriptor(),
        },
    }
    if shepherd_context is not None:
        context["shepherd"] = dict(shepherd_context)
    return context
