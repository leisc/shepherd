"""GenerateFix — LLM-powered @task for code fix generation (Phase 4).

Function-form (DECISIONS D5 / Tranche 7): the previous class-form
``@task class GenerateFix(BaseModel)`` is replaced with the
function-form ``@task async def generate_fix(...) -> FixResult``
shape per CONTRACTS A4.

Receives an issue and code context, produces a minimal fix as the
complete fixed file content. Supports two strategies: minimal (snippet
only) and broader (full file + callers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from shepherd_contexts.workspace import WorkspaceRef
from shepherd_runtime.nucleus import deliver, task
from shepherd_runtime.scope import current_binding
from shepherd_runtime.task.markers import InputMarker

_FIX_GUIDANCE = """\
You are a code repair agent. Your job is to produce the SMALLEST possible fix
that resolves the stated issue without changing any unrelated code.

Rules:
- Fix ONLY the stated issue. Do not refactor, clean up, or improve surrounding code.
- The fix must be syntactically valid Python.
- Preserve existing style, naming, and formatting conventions.
- If you cannot fix the issue, return an empty fixed_content.

Return the COMPLETE fixed file content. The system will diff it against the original
to produce a patch.
"""


@dataclass(frozen=True)
class FixResult:
    """A generated fix for a code issue."""

    fixed_content: str = ""
    fix_description: str = ""


@task(guidance=_FIX_GUIDANCE)
async def generate_fix(
    issue_description: Annotated[str, InputMarker(description="Description of the issue to fix")],
    issue_file_path: Annotated[str, InputMarker(description="File containing the issue")],
    issue_category: Annotated[
        str,
        InputMarker(description="Issue category (type_error, doc_gap, etc.)"),
    ],
    code_context: Annotated[
        str,
        InputMarker(description="Code snippet or full file content for context"),
    ],
    strategy: Annotated[str, InputMarker(description="Fix strategy: minimal or broader")] = "minimal",
    suggested_fix_approach: Annotated[
        str,
        InputMarker(description="Suggested approach from validation"),
    ] = "",
) -> FixResult:
    """Generate a fix for a confirmed code quality issue.

    The active workspace (looked up by type via
    ``current_binding(WorkspaceRef)``) provides read-only access when
    the strategy expands beyond the supplied snippet.
    """
    workspace = current_binding(WorkspaceRef)
    return await deliver(
        FixResult,
        goal=(
            "Produce the smallest fix that resolves the stated issue. "
            "Return the complete fixed file content (empty if a fix "
            "cannot be generated) plus a brief fix description."
        ),
        evidence=[
            f"workspace={workspace.value}",
            f"issue_description={issue_description}",
            f"issue_file_path={issue_file_path}",
            f"issue_category={issue_category}",
            f"strategy={strategy}",
            f"suggested_fix_approach={suggested_fix_approach}",
            f"code_context={code_context}",
        ],
    )


__all__ = ["FixResult", "generate_fix"]
