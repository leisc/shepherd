"""ConfigurePRReview task — infer PR review config from codebase analysis.

Function-form (DECISIONS D5 / Tranche 7): the previous class-form
``@task class ConfigurePRReview(BaseModel)`` is replaced with the
function-form ``@task async def configure_pr_review(...) -> PRReviewConfig``
shape per CONTRACTS A4.
"""

from __future__ import annotations

from typing import Annotated

from shepherd.autoconfig import WORKSPACE_ANALYSIS_GUIDANCE
from shepherd_contexts.workspace.ref import WorkspaceRef
from shepherd_runtime.nucleus import deliver, task
from shepherd_runtime.scope import current_binding
from shepherd_runtime.task.markers import InputMarker

from ..workflows.pr_review.config import PRReviewConfig

PR_REVIEW_GUIDANCE = f"""\
{WORKSPACE_ANALYSIS_GUIDANCE}

## PR Review Configuration

You are inferring a `PRReviewConfig` for this repository. Focus on these fields:

### guidelines (string)
Synthesize review guidelines from:
- CONTRIBUTING.md or similar docs (look for PR review norms, coding standards)
- Linter configuration (what the project cares about stylistically)
- README sections about development workflow
Keep it concise — 2-4 sentences that capture the project's review philosophy.
If no guidance exists in the repo, leave empty.

### focus_areas (list[str])
Derive from repository structure and purpose:
- API-heavy projects -> "api-stability", "backwards-compatibility"
- Data pipelines -> "data-integrity", "error-handling"
- Security-sensitive -> "security", "input-validation"
- Always include "correctness" unless the repo is trivially simple.

### file_patterns_to_skip (list[str])
Extend the defaults (`*.lock`, `*.generated.*`) with patterns from:
- `.gitignore` entries for generated/vendored files
- Ruff/ESLint exclude patterns
- Common patterns: `vendor/**`, `dist/**`, `*.min.js`, `*.pb.go`

### verify (VerifyConfig | None)
Populate ONLY if CI config reveals explicit test/build commands:
- `test_command`: exact command from CI (e.g., `pytest tests/ -x`)
- `build_command`: exact build/typecheck command if present
- `setup_commands`: dependency install steps from CI
- `container_image`: match the CI image or use a sensible default
If no CI config exists, set verify to null.

### Infrastructure fields — LEAVE NULL
The following fields are populated at runtime by the pipeline infrastructure.
Do NOT attempt to discover or populate them:
- `repo` — repository in owner/repo format
- `github_token` — authentication token
- `clone_url` — clone URL for the repository
"""


@task(guidance=PR_REVIEW_GUIDANCE)
async def configure_pr_review(
    hints: Annotated[str, InputMarker(description="Optional user hints for configuration")] = "",
) -> PRReviewConfig:
    """Analyze a codebase to infer PR review configuration.

    The active workspace (looked up by type via
    ``current_binding(WorkspaceRef)``) provides read access for
    discovering CONTRIBUTING, CI config, lint config, and other
    repository signals.
    """
    workspace = current_binding(WorkspaceRef)
    return await deliver(
        PRReviewConfig,
        goal=(
            "Analyze the workspace and infer the PRReviewConfig fields "
            "(guidelines, focus_areas, file_patterns_to_skip, verify) "
            "from project signals. Leave infrastructure fields null."
        ),
        evidence=[
            f"workspace={workspace.value}",
            f"hints={hints}",
        ],
    )


__all__ = ["configure_pr_review"]
