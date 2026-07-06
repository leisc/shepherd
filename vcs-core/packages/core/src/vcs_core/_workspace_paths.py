"""Single source of truth for the workspace-relative path guard.

The check "reject empty, absolute, or traversal (`..`) workspace-relative paths"
was copy-pasted byte-for-byte across five carrier/substrate modules
(260704-1400-review F2). Hardening one copy left the others weak; this module
consolidates the guard so a fix lands once.

`normalize_workspace_relative_path` is the byte-equivalent of the former inline
`_normalize_relative_path` / `_workspace_path` guards: it validates and returns a
normalized *relative* `Path`. Callers keep their own wrapping (joining to a
workspace root, returning a `str`, extra control-plane checks).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def normalize_workspace_relative_path(path: str) -> Path:
    """Validate a workspace-relative path and return it as a normalized relative Path.

    Rejects empty, absolute, and traversal (`..`) paths, and any input that
    normalizes to no path components, with a ``ValueError`` naming the offending
    input. Byte-equivalent to the guard it replaces.
    """
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts:
        msg = f"Invalid workspace-relative path: {path!r}"
        raise ValueError(msg)
    normalized = Path(*pure.parts)
    if not normalized.parts:
        msg = f"Invalid workspace-relative path: {path!r}"
        raise ValueError(msg)
    return normalized
