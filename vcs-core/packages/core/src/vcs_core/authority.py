"""Runtime authority reporting for active substrates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from vcs_core._errors import VcsCoreError

AuthorityRegime = Literal["complete", "partial", "none"]
InterceptionTier = Literal["container", "python", "recording"]
AuthorityLevel = Literal["authoritative", "best-effort", "cooperative"]
_VALID_REGIMES = frozenset(("complete", "partial", "none"))
_VALID_TIERS = frozenset(("container", "python", "recording"))


@dataclass(frozen=True)
class AuthorityAspect:
    """One axis of runtime substrate authority reporting."""

    regime: AuthorityRegime
    access_gated: bool
    tier: InterceptionTier
    reason: str
    level: AuthorityLevel = field(init=False)

    def __post_init__(self) -> None:
        if self.regime not in _VALID_REGIMES:
            raise AuthorityValidationError(
                f"AuthorityAspect regime must be one of {_VALID_REGIMES}, got {self.regime!r}."
            )
        if self.access_gated.__class__ is not bool:
            actual_type = type(self.access_gated).__name__
            raise AuthorityValidationError(f"AuthorityAspect access_gated must be bool, got {actual_type}.")
        if self.tier not in _VALID_TIERS:
            raise AuthorityValidationError(f"AuthorityAspect tier must be one of {_VALID_TIERS}, got {self.tier!r}.")
        _validate_reason("AuthorityAspect", "", self.reason)
        object.__setattr__(self, "level", derive_authority_level(regime=self.regime, access_gated=self.access_gated))


@dataclass(frozen=True)
class SubstrateAuthority:
    """Runtime authority report for one active substrate.

    `containment` answers how completely the substrate can gate and
    preserve final state changes. `provenance` answers how complete the
    canonical low-level event history is.
    """

    substrate: str
    containment: AuthorityAspect
    provenance: AuthorityAspect
    reason: str


class AuthorityValidationError(VcsCoreError, ValueError):
    """Raised when a substrate reports an invalid authority payload."""


def validate_authority_report(substrate_name: str, report: object) -> SubstrateAuthority:
    """Validate one substrate authority report payload."""
    if not isinstance(report, SubstrateAuthority):
        actual_type = type(report).__name__
        raise AuthorityValidationError(
            f"Substrate '{substrate_name}' authority() must return SubstrateAuthority, got {actual_type}."
        )
    if report.substrate != substrate_name:
        raise AuthorityValidationError(
            f"Substrate '{substrate_name}' authority() must report substrate='{substrate_name}', "
            f"got {report.substrate!r}."
        )

    _validate_authority_aspect(substrate_name, "containment", report.containment)
    _validate_authority_aspect(substrate_name, "provenance", report.provenance)
    _validate_reason(substrate_name, "authority", report.reason)
    return report


def derive_authority_level(*, regime: AuthorityRegime, access_gated: bool) -> AuthorityLevel:
    """Collapse runtime capture properties into a consumer-facing trust label."""
    if regime == "complete" and access_gated:
        return "authoritative"
    if regime == "none":
        return "cooperative"
    return "best-effort"


def make_authority_aspect(
    *,
    regime: AuthorityRegime,
    access_gated: bool,
    tier: InterceptionTier,
    reason: str,
) -> AuthorityAspect:
    """Construct one authority aspect with its derived trust label."""
    return AuthorityAspect(
        regime=regime,
        access_gated=access_gated,
        tier=tier,
        reason=reason,
    )


def _validate_authority_aspect(substrate_name: str, label: str, aspect: object) -> None:
    if not isinstance(aspect, AuthorityAspect):
        actual_type = type(aspect).__name__
        raise AuthorityValidationError(
            f"Substrate '{substrate_name}' authority().{label} must be AuthorityAspect, got {actual_type}."
        )
    expected_level = derive_authority_level(regime=aspect.regime, access_gated=aspect.access_gated)
    if aspect.level != expected_level:
        raise AuthorityValidationError(
            f"Substrate '{substrate_name}' authority().{label}.level must be {expected_level!r}, got {aspect.level!r}."
        )
    _validate_reason(substrate_name, f"authority().{label}", aspect.reason)


def _validate_reason(substrate_name: str, label: str, reason: object) -> None:
    if isinstance(reason, str) and reason.strip():
        return
    prefix = f"{substrate_name} {label}".strip()
    raise AuthorityValidationError(f"{prefix} reason must be a non-empty string.")
