"""ValidateIssue — LLM-powered @task for concern validation (Phase 3).

Function-form (DECISIONS D5 / Tranche 7): the previous class-form
``@task class ValidateIssue(BaseModel)`` is replaced with the
function-form ``@task async def validate_issue(...) -> ValidationResult``
shape per CONTRACTS A4.

Receives a single LLM-sourced issue and validates it against the codebase.
Returns a verdict: confirmed, dropped, or inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from shepherd_contexts.workspace import WorkspaceRef
from shepherd_runtime.nucleus import deliver, task
from shepherd_runtime.scope import current_binding
from shepherd_runtime.task.markers import InputMarker

_VALIDATION_GUIDANCE = """\
You are a senior code reviewer validating a potential issue found by an automated analyzer.

Your job is to determine whether the issue is REAL by examining the actual code.

Procedure:
1. Read the cited code at the specified file and line range.
2. Search for related code (callers, tests, type definitions) to understand context.
3. Assess whether the hypothesis holds given the full evidence.

Return one of three verdicts:
- "confirmed": The issue is real and should be fixed. Cite the specific code that proves it.
- "dropped": The issue is a false positive. Explain why (e.g., the code is actually correct, or the pattern is intentional).
- "inconclusive": Cannot determine from static analysis alone (e.g., runtime behavior, concurrency).

You MUST provide evidence for your verdict — a verdict without evidence is not actionable.
"""


@dataclass(frozen=True)
class ValidationResult:
    """Verdict from validating an LLM-sourced issue."""

    verdict: str = "inconclusive"
    explanation: str = ""
    suggested_fix_approach: str = ""


@task(guidance=_VALIDATION_GUIDANCE)
async def validate_issue(
    issue_description: Annotated[str, InputMarker(description="Description of the potential issue")],
    issue_hypothesis: Annotated[
        str,
        InputMarker(description="What the analyzer believes is wrong and why"),
    ],
    issue_file_path: Annotated[str, InputMarker(description="File containing the potential issue")],
    issue_line_range: Annotated[str, InputMarker(description="Line range (e.g., '10-25')")],
    issue_evidence: Annotated[str, InputMarker(description="Code snippet that triggered the finding")],
) -> ValidationResult:
    """Validate a single LLM-sourced issue against the codebase.

    The active workspace (looked up by type via
    ``current_binding(WorkspaceRef)``) provides read-only access for
    workspace tools (Read, Glob, Grep) used during validation.
    """
    workspace = current_binding(WorkspaceRef)
    return await deliver(
        ValidationResult,
        goal=(
            "Validate the issue against the codebase using the active "
            "workspace; produce a verdict, evidence-based explanation, "
            "and suggested fix approach."
        ),
        evidence=[
            f"workspace={workspace.value}",
            f"issue_description={issue_description}",
            f"issue_hypothesis={issue_hypothesis}",
            f"issue_file_path={issue_file_path}",
            f"issue_line_range={issue_line_range}",
            f"issue_evidence={issue_evidence}",
        ],
    )


__all__ = ["ValidationResult", "validate_issue"]
