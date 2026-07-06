# under-test: vcs_core._authority
from __future__ import annotations

import math

import pytest
from vcs_core._authority import normalize_authority_context


def test_authority_context_normalizes_json_payload() -> None:
    payload = {
        "schema": "shepherd.workspace-control.filesystem-authority-context.v1",
        "shepherd": {"run_ref": "run-1", "attempts": 1, "tags": ("a", "b")},
        "complete": True,
        "score": 1.25,
        "missing": None,
    }

    assert normalize_authority_context(payload) == {
        "shepherd": {"attempts": 1, "run_ref": "run-1", "tags": ["a", "b"]},
        "complete": True,
        "missing": None,
        "schema": "shepherd.workspace-control.filesystem-authority-context.v1",
        "score": 1.25,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {1: "not a string key"},
        {"": "empty key"},
        {"bad": object()},
        {"bad": b"bytes need an explicit schema"},
        {"bad": math.nan},
        {"bad": "contains\0nul"},
        {"bad\0key": "value"},
    ],
)
def test_authority_context_rejects_non_durable_values(payload: dict[object, object]) -> None:
    with pytest.raises((TypeError, ValueError), match="authority context"):
        normalize_authority_context(payload)
