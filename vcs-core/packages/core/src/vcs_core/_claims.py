"""Internal claim registry for substrate-owned real resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vcs_core._errors import VcsCoreError

ClaimPolicy = Literal["observe", "exclusive", "authoritative_suppress_fs"]


@dataclass(frozen=True)
class ResourceClaim:
    """One substrate-owned claim over a real resource path."""

    substrate: str
    target_id: str
    path: str
    policy: ClaimPolicy


class ClaimConflictError(VcsCoreError, ValueError):
    """Two incompatible substrates attempted to claim the same real path."""

    def __init__(self, *, existing: ResourceClaim, requested: ResourceClaim) -> None:
        self.existing = existing
        self.requested = requested
        super().__init__(
            "Path "
            f"{existing.path!r} is already claimed by {existing.substrate}:{existing.target_id} "
            f"({existing.policy}); cannot register {requested.substrate}:{requested.target_id} "
            f"({requested.policy})."
        )


class ClaimRegistry:
    """Minimal path-backed claim registry."""

    def __init__(self) -> None:
        self._claims: dict[str, ResourceClaim] = {}

    def register(
        self,
        *,
        substrate: str,
        target_id: str,
        path: str | Path,
        policy: ClaimPolicy,
    ) -> ResourceClaim:
        if not substrate:
            raise ValueError("substrate must not be empty.")
        if not target_id:
            raise ValueError("target_id must not be empty.")
        normalized = self._normalize(path)
        claim = ResourceClaim(
            substrate=substrate,
            target_id=target_id,
            path=normalized,
            policy=policy,
        )
        existing = self._claims.get(normalized)
        if existing is not None:
            if existing == claim:
                return existing
            raise ClaimConflictError(existing=existing, requested=claim)
        self._claims[normalized] = claim
        return claim

    def lookup(self, path: str | Path) -> ResourceClaim | None:
        return self._claims.get(self._normalize(path))

    def clear(self) -> None:
        self._claims.clear()

    @staticmethod
    def _normalize(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve(strict=False))
