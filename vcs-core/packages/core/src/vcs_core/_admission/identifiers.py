"""Parse user-facing identifiers before they reach storage internals."""

from __future__ import annotations

import re
from typing import ClassVar

import pygit2

from vcs_core._errors import VcsCoreError


class ParseError(VcsCoreError, ValueError):
    """User input could not be parsed into a domain identifier."""


_WHITESPACE_RE = re.compile(r"\s")


class ScopeName(str):
    """Validated vcs-core scope name."""

    __slots__ = ()

    GROUND: ClassVar[str] = "ground"

    @classmethod
    def parse(cls, raw: str, *, allow_ground: bool = True) -> ScopeName:
        """Parse one user-provided scope name.

        The storage layer ultimately embeds scope names in Git refs, so the
        parser rejects names that would be invalid as a flat ref component.
        """
        if raw == cls.GROUND:
            if allow_ground:
                return cls(raw)
            raise ParseError("scope name 'ground' is reserved.")
        if raw == "":
            raise ParseError("scope name is empty.")
        if raw.startswith("."):
            raise ParseError(f"scope name {raw!r} starts with '.'.")
        if "/" in raw:
            raise ParseError(f"scope name {raw!r} contains '/'; use '-' as a separator.")
        if raw in {".", ".."} or ".." in raw:
            raise ParseError(f"scope name {raw!r} contains '..'.")
        if _WHITESPACE_RE.search(raw):
            raise ParseError(f"scope name {raw!r} contains whitespace.")
        if raw.endswith(".lock"):
            raise ParseError(f"scope name {raw!r} ends with '.lock'.")
        if not pygit2.reference_is_valid_name(f"refs/vcscore/scopes/{raw}"):
            raise ParseError(f"scope name {raw!r} is not a valid Git ref component.")
        return cls(raw)


def parse_optional_scope_name(raw: str | None, *, allow_ground: bool = True) -> str | None:
    """Parse an optional scope name and return the normalized string."""
    if raw is None:
        return None
    return str(ScopeName.parse(raw, allow_ground=allow_ground))
