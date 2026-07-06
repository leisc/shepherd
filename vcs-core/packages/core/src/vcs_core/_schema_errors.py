"""Shared validation error types for runtime ingress contracts."""

from __future__ import annotations

from vcs_core._errors import VcsCoreError


class SchemaValidationError(VcsCoreError, ValueError):
    """Raised when a runtime ingress payload violates its declared schema."""
