"""Named semantic profiles for the storage-free reference kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticProfile:
    """A named validation and source-surface boundary.

    `requires_source_admission` marks a profile whose stamp makes a promise
    about the *source* program that cannot be verified at the IR level (the
    distinguishing source constructs do not survive `elaborate()`). Such a
    profile may only be minted via the source-level `admit_and_prepare(...)`
    entry point, which runs `validate_profile_admission(...)` before stamping
    — never by stamping raw IR (2026-05-26 §"Profile admission boundary").
    Permissive profiles (`CORE_A`, `CORE0`) make no source promise and are
    stampable directly on IR.
    """

    name: str
    version: str
    validated: bool
    requires_source_admission: bool = False


CORE0 = SemanticProfile(name="core0", version="v0", validated=True)
CORE_A = SemanticProfile(name="core_a", version="v0", validated=True)
CORE_REFERENCE_V0_LITE = SemanticProfile(
    name="core_reference_v0_lite",
    version="v0",
    validated=True,
    requires_source_admission=True,
)
PUBLICATION_EXPERIMENTAL = SemanticProfile(
    name="publication_experimental",
    version="v0",
    validated=False,
)

DEFAULT_CORE_PROFILE = CORE_A

_REGISTRY: dict[str, SemanticProfile] = {
    p.name: p for p in (CORE0, CORE_A, CORE_REFERENCE_V0_LITE, PUBLICATION_EXPERIMENTAL)
}


def lookup_profile(name: str) -> SemanticProfile:
    """Return the SemanticProfile for a registered profile name.

    Raises `KeyError` if no profile with that name is registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown profile name {name!r}; known: {sorted(_REGISTRY)!r}") from None


__all__ = [
    "CORE0",
    "CORE_A",
    "CORE_REFERENCE_V0_LITE",
    "DEFAULT_CORE_PROFILE",
    "PUBLICATION_EXPERIMENTAL",
    "SemanticProfile",
    "lookup_profile",
]
