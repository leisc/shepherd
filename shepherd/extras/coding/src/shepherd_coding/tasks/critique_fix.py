"""CritiqueFix — LLM-powered @task for fix critique (Phase 4).

Function-form (DECISIONS D5 / Tranche 7): the previous class-form
``@task class CritiqueFix(BaseModel)`` is replaced with the
function-form ``@task async def critique_fix(...) -> CritiqueResult``
shape per CONTRACTS A4.

Evaluates a proposed fix for correctness, minimality, and side effects.
Validated by Spike 17: 100% accuracy with Opus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from shepherd_contexts.workspace import WorkspaceRef
from shepherd_runtime.nucleus import deliver, task
from shepherd_runtime.scope import current_binding
from shepherd_runtime.task.markers import InputMarker

_CRITIQUE_GUIDANCE = """\
You are a senior code reviewer evaluating a proposed automated fix.

Evaluate the fix on these criteria:
1. Does it actually resolve the stated issue?
2. Is the fix minimal (no unnecessary changes)?
3. Does the fix introduce any new problems (type errors, broken imports, changed behavior)?
4. Is the fix idiomatic (follows existing code style)?

A fix that resolves the issue via a workaround (e.g., adding a # noqa comment instead of
removing unused code) should be REJECTED — it masks the problem without fixing it.

Return a clear APPROVE or REJECT verdict with reasoning.
"""


@dataclass(frozen=True)
class CritiqueResult:
    """Verdict on a proposed code fix."""

    approved: bool = False
    reasoning: str = ""


@task(guidance=_CRITIQUE_GUIDANCE)
async def critique_fix(
    issue_description: Annotated[str, InputMarker(description="The issue being fixed")],
    original_code: Annotated[str, InputMarker(description="Original file content before the fix")],
    proposed_fix: Annotated[str, InputMarker(description="Proposed fixed file content")],
    fix_description: Annotated[str, InputMarker(description="What the fix claims to do")] = "",
) -> CritiqueResult:
    """Critique a proposed code fix for correctness and quality.

    The active workspace (looked up by type via
    ``current_binding(WorkspaceRef)``) provides read-only access for
    inspecting callers and related code during critique.

    Skipped for programmatic-source issues (verified by tool re-run).
    Enabled for LLM-sourced fixes when using an Opus-class model
    (per Spike 17 findings).
    """
    workspace = current_binding(WorkspaceRef)
    return await deliver(
        CritiqueResult,
        goal=(
            "Evaluate the proposed fix for correctness, minimality, "
            "and side effects; return APPROVE / REJECT verdict with "
            "reasoning."
        ),
        evidence=[
            f"workspace={workspace.value}",
            f"issue_description={issue_description}",
            f"fix_description={fix_description}",
            f"original_code={original_code}",
            f"proposed_fix={proposed_fix}",
        ],
    )


__all__ = ["CritiqueResult", "critique_fix"]
