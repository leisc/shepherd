# under-test: vcs_core._query_inventory
"""Conformance for the `Health` discriminated union.

Each variant serializes to the flat shape `health_to_json` produces, and the
union is closed to its variants (exhaustive under `assert_never`). Phase 2 split
the absent state into `Expected` (benign, primary_issue="none") and `Missing`
(problematic, primary_issue="missing") — the honest values the severity table
derives from.
"""

from __future__ import annotations

from typing import assert_never

from vcs_core._query_inventory import (
    Expected,
    Health,
    Missing,
    PresentInvalid,
    PresentValid,
    expected,
    health_to_json,
    missing,
    present_invalid,
    present_valid,
    severity_for,
)


def test_present_valid_flat_shape() -> None:
    h = present_valid(lifecycle="active", authority_role="authoritative")
    assert health_to_json(h) == {
        "presence": "present",
        "validity": "valid",
        "primary_issue": "none",
        "issue_codes": [],
        "lifecycle": "active",
        "authority_role": "authoritative",
        "status": "present_valid",
    }


def test_present_valid_terminal_status_derived() -> None:
    assert present_valid(lifecycle="terminal").status == "present_terminal"


def test_present_invalid_flat_shape_and_derived_status() -> None:
    payload = health_to_json(present_invalid(primary_issue="corrupt", issue_codes=("world_unreadable",)))
    assert payload["presence"] == "present"
    assert payload["validity"] == "invalid"
    assert payload["primary_issue"] == "corrupt"
    assert payload["issue_codes"] == ["world_unreadable"]
    assert payload["status"] == "present_corrupt"  # _invalid_status fallback


def test_present_invalid_status_override_preserved() -> None:
    h = present_invalid(primary_issue="schema_mismatch", issue_codes=(), status="custom_status")
    assert health_to_json(h)["status"] == "custom_status"


def test_expected_flat_shape_is_benign() -> None:
    # Benign absence (pre-first-publish ground/authority, or a queried miss):
    # primary_issue stays "none".
    payload = health_to_json(expected(issue_codes=("authority_ref_missing",)))
    assert payload["presence"] == "absent"
    assert payload["validity"] == "unknown"
    assert payload["primary_issue"] == "none"
    assert payload["issue_codes"] == ["authority_ref_missing"]
    assert payload["status"] == "absent"


def test_missing_flat_shape_is_honest() -> None:
    # Phase 2: a problematic absence reports primary_issue="missing" (Phase 1 lossily
    # reported "none"); this is the honest value the severity table derives from.
    payload = health_to_json(missing(issue_codes=("operation_authority_missing",)))
    assert payload["presence"] == "absent"
    assert payload["validity"] == "unknown"
    assert payload["primary_issue"] == "missing"
    assert payload["status"] == "missing"


def test_health_union_is_exhaustive() -> None:
    def classify(h: Health) -> str:
        match h:
            case PresentValid():
                return "present-valid"
            case PresentInvalid():
                return "present-invalid"
            case Expected():
                return "expected"
            case Missing():
                return "missing"
        assert_never(h)

    assert classify(present_valid()) == "present-valid"
    assert classify(present_invalid(primary_issue="corrupt", issue_codes=())) == "present-invalid"
    assert classify(expected()) == "expected"
    assert classify(missing()) == "missing"


def test_severity_for_derives_from_health() -> None:
    # Ratified Decision 2: severity is a total function of the Health verdict.
    assert severity_for(present_valid()) == "info"
    assert severity_for(expected()) == "info"  # benign absence (the issue-02 fix)
    assert severity_for(missing()) == "error"  # should-exist-and-gone
    assert severity_for(missing(lifecycle="recoverable")) == "warning"  # recoverable-pending


def test_severity_for_present_invalid_maps_by_primary_issue() -> None:
    assert severity_for(present_invalid(primary_issue="corrupt", issue_codes=())) == "error"
    assert severity_for(present_invalid(primary_issue="schema_mismatch", issue_codes=())) == "error"
    assert severity_for(present_invalid(primary_issue="dangling_dependency", issue_codes=())) == "error"
    assert severity_for(present_invalid(primary_issue="unknown", issue_codes=())) == "warning"


def test_item_to_json_derives_issue_severity_from_health() -> None:
    # The item boundary injects severity_for(health) into each issue; an Expected
    # item's issue is info (was hand-set "error" pre-Phase-3 — the issue-02 fix).
    from vcs_core._query_inventory import InventoryIssue, InventoryItem

    item = InventoryItem(
        id="authority_ref:x",
        domain="authority_ref",
        kind="world_authority_ref",
        locator="refs/vcscore/ground",
        source_kind="git_ref",
        source_store="coordinator",
        health=expected(issue_codes=("authority_ref_missing",), authority_role="authoritative"),
        issues=(
            InventoryIssue(
                id="i",
                code="authority_ref_missing",
                message="world authority ref is missing",
                subject_id="authority_ref:x",
            ),
        ),
    )
    payload = item.to_json()
    assert payload["issues"][0]["severity"] == "info"


def test_severity_invariant_no_error_while_primary_issue_none() -> None:
    # Read-model load-bearing invariant (DESIGN-derived-fact-authorities.md): severity is
    # *derived* from Health, so nothing can be error-severity while Health reports no
    # problem (primary_issue="none"). Pins the structural guarantee as a property so a
    # future change to severity_for or the variants cannot reintroduce issue-02 (a benign
    # workspace flagged as an error).
    benign: list[Health] = [
        present_valid(),
        present_valid(lifecycle="terminal"),
        expected(),
        expected(issue_codes=("authority_ref_missing",), authority_role="authoritative"),
    ]
    for h in benign:
        assert h.primary_issue == "none"
        assert severity_for(h) != "error"
