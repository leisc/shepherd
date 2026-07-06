# under-test: vcs_core._cli_errors
from __future__ import annotations

from vcs_core._cli_errors import prefixed_error_message


def test_prefixed_error_message_adds_missing_prefix() -> None:
    assert prefixed_error_message("something failed") == "Error: something failed"


def test_prefixed_error_message_preserves_rendered_app_error() -> None:
    message = "Error: cannot push:\n  - live scope remains"

    assert prefixed_error_message(message) == message
