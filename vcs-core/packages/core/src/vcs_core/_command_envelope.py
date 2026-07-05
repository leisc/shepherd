"""Framework-owned command execution controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vcs_core._execution_capability import NON_REVERSIBLE_RUN_FLAG

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class CommandEnvelopeError(ValueError):
    """Raised when framework-owned command controls are malformed."""


ExecutionSuccessDisposition = Literal["merge", "seal", "authority_merge"]
_EXECUTION_SUCCESS_DISPOSITIONS = frozenset({"merge", "seal", "authority_merge"})


@dataclass(frozen=True)
class AuthorityMergeControl:
    """Framework-owned controls for authority terminalization.

    This object is callback-bearing and intentionally not session/CLI
    transportable. It exists so a trusted in-process dialect can ask the
    execution-bound reversible wrap to settle a run scope through
    ``VcsCore.merge_with_authority(...)`` instead of ordinary ``merge(...)``.
    """

    binding_roots: Mapping[str, str]
    decide: Callable[[object], object]
    effective_match_digest: str | None = None
    authority_surface_plan_digest: str | None = None
    permission_plan_digest: str | None = None
    permission_plan_descriptor: Mapping[str, object] | None = None
    authority_context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        try:
            binding_roots = dict(self.binding_roots)
        except (TypeError, ValueError) as exc:
            raise CommandEnvelopeError("authority merge binding_roots must be a mapping.") from exc
        if not callable(self.decide):
            raise CommandEnvelopeError("authority merge decide must be callable.")
        _require_optional_digest(self.effective_match_digest, "effective_match_digest")
        _require_optional_digest(self.authority_surface_plan_digest, "authority_surface_plan_digest")

        from vcs_core._authority import normalize_authority_context
        from vcs_core._permission_plan_evidence import PermissionPlanEvidenceError, validate_permission_plan_evidence

        try:
            permission_plan_descriptor = validate_permission_plan_evidence(
                permission_plan_digest_value=self.permission_plan_digest,
                permission_plan_descriptor=self.permission_plan_descriptor,
                expected_route="carrier_diff",
                expected_effective_match_digest=self.effective_match_digest,
                expected_authority_surface_plan_digest=self.authority_surface_plan_digest,
            )
        except PermissionPlanEvidenceError as exc:
            raise CommandEnvelopeError(f"authority merge PermissionPlan evidence invalid: {exc}") from exc

        object.__setattr__(self, "binding_roots", binding_roots)
        object.__setattr__(self, "permission_plan_descriptor", permission_plan_descriptor)
        object.__setattr__(
            self,
            "authority_context",
            normalize_authority_context(dict(self.authority_context) if self.authority_context is not None else None),
        )


@dataclass(frozen=True)
class CommandExecutionOptions:
    """Framework controls attached to an exec invocation, separate from driver params."""

    non_reversible_run: bool = False
    success_disposition: ExecutionSuccessDisposition = "merge"
    seal_output_binding: str = "workspace"
    authority_merge: AuthorityMergeControl | None = None


def command_execution_options_from_mapping(options: Mapping[str, object] | None) -> CommandExecutionOptions:
    """Parse framework execution controls from a dedicated transport object."""
    if options is None:
        return CommandExecutionOptions()
    unknown = sorted(set(options) - {NON_REVERSIBLE_RUN_FLAG})
    if unknown:
        raise CommandEnvelopeError(f"Unknown command execution option(s): {', '.join(unknown)}.")
    value = options.get(NON_REVERSIBLE_RUN_FLAG, False)
    if type(value) is not bool:
        raise CommandEnvelopeError(f"Command execution option '{NON_REVERSIBLE_RUN_FLAG}' must be a bool.")
    return CommandExecutionOptions(non_reversible_run=value)


def command_execution_options_to_mapping(options: CommandExecutionOptions) -> dict[str, object]:
    """Render framework execution controls for session/raw transport."""
    validate_command_execution_options(options)
    if options.success_disposition != "merge":
        raise CommandEnvelopeError(
            f"Command success disposition {options.success_disposition!r} is not supported by command transport."
        )
    return {NON_REVERSIBLE_RUN_FLAG: options.non_reversible_run}


def validate_command_execution_options(options: CommandExecutionOptions) -> None:
    """Validate a typed options object that may have been constructed dynamically."""
    if type(options.non_reversible_run) is not bool:
        raise CommandEnvelopeError(f"Command execution option '{NON_REVERSIBLE_RUN_FLAG}' must be a bool.")
    if options.success_disposition not in _EXECUTION_SUCCESS_DISPOSITIONS:
        raise CommandEnvelopeError(
            f"Command execution option 'success_disposition' is unsupported: {options.success_disposition!r}."
        )
    if type(options.seal_output_binding) is not str or not options.seal_output_binding:
        raise CommandEnvelopeError("Command execution option 'seal_output_binding' must be a non-empty string.")
    if options.authority_merge is not None and not isinstance(options.authority_merge, AuthorityMergeControl):
        raise CommandEnvelopeError("Command execution option 'authority_merge' must be AuthorityMergeControl or None.")
    if options.success_disposition == "authority_merge":
        if options.authority_merge is None:
            raise CommandEnvelopeError("Authority merge success disposition requires 'authority_merge' controls.")
    elif options.authority_merge is not None:
        raise CommandEnvelopeError(
            "Command execution option 'authority_merge' is only valid with authority_merge success disposition."
        )
    if options.non_reversible_run and options.success_disposition in {"seal", "authority_merge"}:
        raise CommandEnvelopeError(
            f"{options.success_disposition!r} success disposition requires a reversible execution-bound run."
        )


def _require_optional_digest(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise CommandEnvelopeError(f"authority merge {field_name} must be a non-empty string or None.")
