# under-test: vcs_core._fork_hints
"""PD2a: typed ``ForkHints`` + reject-unknown-keys at the ``branch()`` boundary."""

from __future__ import annotations

import pytest
from vcs_core import UnknownForkHintError
from vcs_core._fork_hints import (
    ACCEPTED_BRANCH_HINT_KEYS,
    ACCEPTED_FORK_HINT_KEYS,
    ForkHints,
    validate_branch_hints,
)


def test_from_value_accepts_none_mapping_and_typed() -> None:
    assert ForkHints.from_value(None) == ForkHints()
    assert ForkHints.from_value({"isolated": True}) == ForkHints(isolated=True)
    assert ForkHints.from_value({"isolated": False}) == ForkHints()
    typed = ForkHints(isolated=True, restore=True)
    assert ForkHints.from_value(typed) is typed


def test_from_value_rejects_misspelled_key_and_names_accepted_keys() -> None:
    with pytest.raises(UnknownForkHintError) as excinfo:
        ForkHints.from_value({"isoalted": True})
    message = str(excinfo.value)
    assert "isoalted" in message
    for key in sorted(ACCEPTED_FORK_HINT_KEYS):
        assert key in message


def test_restore_dunder_does_not_pass_through_the_typed_layer() -> None:
    with pytest.raises(UnknownForkHintError):
        ForkHints.from_value({"__restore__": True})
    with pytest.raises(UnknownForkHintError):
        ForkHints.from_value({"isolated": True, "__restore__": True})


def test_to_branch_hints_lowering() -> None:
    assert ForkHints().to_branch_hints() is None
    assert ForkHints(isolated=True).to_branch_hints() == {"isolated": True}
    assert ForkHints(isolated=True, restore=True).to_branch_hints() == {
        "isolated": True,
        "__restore__": True,
    }


def test_lowered_hints_always_pass_the_branch_boundary() -> None:
    for hints in (ForkHints(), ForkHints(isolated=True), ForkHints(isolated=True, restore=True)):
        validate_branch_hints(hints.to_branch_hints())


def test_validate_branch_hints_rejects_unknown_key_and_names_accepted_keys() -> None:
    with pytest.raises(UnknownForkHintError) as excinfo:
        validate_branch_hints({"isolated": True, "mount": "/tmp"})
    message = str(excinfo.value)
    assert "mount" in message
    for key in sorted(ACCEPTED_BRANCH_HINT_KEYS):
        assert key in message


def test_validate_branch_hints_accepts_known_keys_and_empty() -> None:
    validate_branch_hints(None)
    validate_branch_hints({})
    validate_branch_hints({"isolated": True})
    validate_branch_hints({"isolated": True, "__restore__": True})
