"""Internal framework-owned command-admission seam."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vcs_core._errors import VcsCoreError
from vcs_core._immutable_payload import immutable_payload_view

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vcs_core.types import ScopeInfo


class CommandAdmissionError(VcsCoreError, ValueError):
    """Named validation error for framework-routed command admission."""


@runtime_checkable
class InternalCommandAdmissionProvider(Protocol):
    """Optional internal hook for rejecting command execution before runtime effects.

    Framework-routed callers are expected to invoke this under the active
    recursion/interception guard so admission cannot accidentally trigger
    patched runtime behavior before execution begins.
    """

    def validate_command_invocation(
        self,
        command: str,
        scope: ScopeInfo,
        *,
        params: Mapping[str, Any],
    ) -> None: ...


def admit_command_invocation(
    substrate: object,
    command: str,
    scope: ScopeInfo,
    *,
    params: Mapping[str, Any],
) -> None:
    """Run optional framework-owned admission before substrate execution."""
    validator = getattr(substrate, "validate_command_invocation", None)
    if validator is None:
        return
    try:
        validator(
            command,
            scope,
            params=immutable_payload_view(params),
        )
    except CommandAdmissionError:
        raise
    except ValueError as exc:
        raise CommandAdmissionError(str(exc)) from exc
