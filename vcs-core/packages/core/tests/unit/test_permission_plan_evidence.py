# under-test: vcs_core._permission_plan_evidence
from __future__ import annotations

import pytest
from vcs_core._permission_plan_evidence import (
    PermissionPlanEvidenceError,
    permission_plan_digest,
    validate_permission_plan_evidence,
)


def _descriptor(*, route: str = "retained_output_selection") -> dict[str, object]:
    return {
        "schema": "shepherd.permission-plan.v1",
        "fallback": "enforce",
        "assignments": [
            {
                "monitor": "carrier_check_at_commit",
                "timing": "commit",
                "route": route,
                "completeness_basis": "exact carrier evidence",
                "tamper_basis": "coordinator-owned settlement",
                "confinement": None,
                "evidence": {
                    "effective_match_digest": "effective-match",
                    "authority_surface_plan_digest": "surface-plan",
                },
            }
        ],
    }


def test_permission_plan_evidence_validates_carrier_descriptor() -> None:
    descriptor = _descriptor()

    assert (
        validate_permission_plan_evidence(
            permission_plan_digest_value=permission_plan_digest(descriptor),
            permission_plan_descriptor=descriptor,
            expected_route="retained_output_selection",
            expected_effective_match_digest="effective-match",
            expected_authority_surface_plan_digest="surface-plan",
        )
        == descriptor
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda descriptor: descriptor.update(schema="wrong"), "schema"),
        (lambda descriptor: descriptor.update(fallback="maybe"), "fallback"),
        (lambda descriptor: descriptor["assignments"].append(dict(descriptor["assignments"][0])), "more than once"),
        (lambda descriptor: descriptor["assignments"][0].update(route="carrier_diff"), "route"),
        (lambda descriptor: descriptor["assignments"][0].update(monitor="other"), "carrier_check_at_commit"),
        (lambda descriptor: descriptor["assignments"][0].update(timing="pre_action"), "commit"),
        (lambda descriptor: descriptor["assignments"][0].pop("evidence"), "evidence"),
        (
            lambda descriptor: descriptor["assignments"][0]["evidence"].update(effective_match_digest="other"),
            "effective_match_digest",
        ),
        (
            lambda descriptor: descriptor["assignments"][0]["evidence"].update(authority_surface_plan_digest="other"),
            "authority_surface_plan_digest",
        ),
    ],
)
def test_permission_plan_evidence_rejects_mismatched_carrier_descriptor(mutate, match: str) -> None:
    descriptor = _descriptor()
    mutate(descriptor)

    with pytest.raises(PermissionPlanEvidenceError, match=match):
        validate_permission_plan_evidence(
            permission_plan_digest_value=permission_plan_digest(descriptor),
            permission_plan_descriptor=descriptor,
            expected_route="retained_output_selection",
            expected_effective_match_digest="effective-match",
            expected_authority_surface_plan_digest="surface-plan",
        )


def test_permission_plan_evidence_rejects_digest_descriptor_mismatch() -> None:
    descriptor = _descriptor()

    with pytest.raises(PermissionPlanEvidenceError, match="digest"):
        validate_permission_plan_evidence(
            permission_plan_digest_value="not-the-real-digest",
            permission_plan_descriptor=descriptor,
            expected_route="retained_output_selection",
            expected_effective_match_digest="effective-match",
            expected_authority_surface_plan_digest="surface-plan",
        )
