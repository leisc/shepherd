"""Schema-neutral validation for Shepherd PermissionPlan evidence.

The dialect owns the monitor-assignment compiler. vcs-core only persists and
checks its JSON evidence: schema shape, digest, route, and the Match-surface
digests that the carrier monitor claims it is enforcing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

from vcs_core._errors import VcsCoreError

PERMISSION_PLAN_SCHEMA = "shepherd.permission-plan.v1"
CARRIER_MONITOR = "carrier_check_at_commit"
_FALLBACKS = frozenset({"enforce", "refuse"})
_TIMINGS = frozenset({"pre_action", "commit"})
_ASSIGNMENT_FIELDS = frozenset(
    {
        "monitor",
        "timing",
        "completeness_basis",
        "tamper_basis",
        "confinement",
        "route",
        "evidence",
    }
)


class PermissionPlanEvidenceError(VcsCoreError, ValueError):
    """Raised when persisted PermissionPlan evidence is malformed or mismatched."""


def permission_plan_digest(descriptor: Mapping[str, object]) -> str:
    """Return the canonical digest for a normalized PermissionPlan descriptor."""
    normalized = normalize_permission_plan_descriptor(descriptor)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_permission_plan_evidence(
    *,
    permission_plan_digest_value: str | None,
    permission_plan_descriptor: Mapping[str, object] | None,
    expected_route: str,
    expected_effective_match_digest: str | None,
    expected_authority_surface_plan_digest: str | None,
) -> dict[str, object]:
    """Validate carrier-only PermissionPlan evidence and return a normalized descriptor."""
    _require_non_empty_str(permission_plan_digest_value, "permission_plan_digest")
    _require_non_empty_str(expected_route, "expected_route")
    _require_non_empty_str(expected_effective_match_digest, "expected_effective_match_digest")
    _require_non_empty_str(expected_authority_surface_plan_digest, "expected_authority_surface_plan_digest")
    if permission_plan_descriptor is None:
        raise PermissionPlanEvidenceError("PermissionPlan descriptor is required")
    descriptor = normalize_permission_plan_descriptor(permission_plan_descriptor)
    actual_digest = permission_plan_digest(descriptor)
    if actual_digest != permission_plan_digest_value:
        raise PermissionPlanEvidenceError("PermissionPlan digest disagrees with descriptor")
    assignments = cast("list[dict[str, object]]", descriptor["assignments"])
    if len(assignments) != 1:
        raise PermissionPlanEvidenceError("carrier PermissionPlan must contain exactly one assignment")
    (assignment,) = assignments
    if assignment["monitor"] != CARRIER_MONITOR:
        raise PermissionPlanEvidenceError("carrier PermissionPlan must assign carrier_check_at_commit")
    if assignment["timing"] != "commit":
        raise PermissionPlanEvidenceError("carrier PermissionPlan assignment must run at commit")
    if assignment.get("confinement") is not None:
        raise PermissionPlanEvidenceError("carrier PermissionPlan assignment must not carry confinement")
    if assignment.get("route") != expected_route:
        raise PermissionPlanEvidenceError("PermissionPlan route disagrees with authority route")
    evidence = assignment.get("evidence")
    if not isinstance(evidence, dict):
        raise PermissionPlanEvidenceError("carrier PermissionPlan assignment must carry evidence")
    if evidence.get("effective_match_digest") != expected_effective_match_digest:
        raise PermissionPlanEvidenceError("PermissionPlan effective_match_digest disagrees with authority surface")
    if evidence.get("authority_surface_plan_digest") != expected_authority_surface_plan_digest:
        raise PermissionPlanEvidenceError(
            "PermissionPlan authority_surface_plan_digest disagrees with authority surface"
        )
    return descriptor


def normalize_permission_plan_descriptor(descriptor: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize a PermissionPlan descriptor as durable JSON."""
    if not isinstance(descriptor, Mapping):
        raise PermissionPlanEvidenceError("PermissionPlan descriptor must be an object")
    unknown = sorted(set(descriptor) - {"schema", "fallback", "assignments"})
    if unknown:
        raise PermissionPlanEvidenceError(f"PermissionPlan descriptor has unknown fields: {unknown!r}")
    if descriptor.get("schema") != PERMISSION_PLAN_SCHEMA:
        raise PermissionPlanEvidenceError("PermissionPlan descriptor has unsupported schema")
    fallback = descriptor.get("fallback")
    if fallback not in _FALLBACKS:
        raise PermissionPlanEvidenceError(f"PermissionPlan fallback is unsupported: {fallback!r}")
    raw_assignments = descriptor.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise PermissionPlanEvidenceError("PermissionPlan assignments must be a non-empty list")
    assignments: list[dict[str, object]] = []
    monitors: set[str] = set()
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = _normalize_assignment(raw_assignment, index=index)
        monitor = cast("str", assignment["monitor"])
        if monitor in monitors:
            raise PermissionPlanEvidenceError(f"PermissionPlan assigns monitor {monitor!r} more than once")
        monitors.add(monitor)
        assignments.append(assignment)
    return {
        "schema": PERMISSION_PLAN_SCHEMA,
        "fallback": fallback,
        "assignments": assignments,
    }


def _normalize_assignment(raw_assignment: object, *, index: int) -> dict[str, object]:
    if not isinstance(raw_assignment, Mapping):
        raise PermissionPlanEvidenceError(f"PermissionPlan assignment {index} must be an object")
    unknown = sorted(set(raw_assignment) - _ASSIGNMENT_FIELDS)
    if unknown:
        raise PermissionPlanEvidenceError(f"PermissionPlan assignment {index} has unknown fields: {unknown!r}")
    monitor = _required_str(raw_assignment, "monitor", index=index)
    timing = _required_str(raw_assignment, "timing", index=index)
    if timing not in _TIMINGS:
        raise PermissionPlanEvidenceError(f"PermissionPlan assignment {index} timing is unsupported: {timing!r}")
    normalized: dict[str, object] = {
        "monitor": monitor,
        "timing": timing,
        "completeness_basis": _required_str(raw_assignment, "completeness_basis", index=index),
        "tamper_basis": _required_str(raw_assignment, "tamper_basis", index=index),
        "confinement": _json_value(raw_assignment.get("confinement"), path=f"assignments[{index}].confinement"),
    }
    route = raw_assignment.get("route")
    if route is not None:
        normalized["route"] = _non_empty_str(route, f"assignments[{index}].route")
    evidence = raw_assignment.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, Mapping) or not evidence:
            raise PermissionPlanEvidenceError(f"PermissionPlan assignment {index} evidence must be an object")
        normalized_evidence: dict[str, object] = {}
        for key, value in sorted(evidence.items(), key=lambda item: str(item[0])):
            field = _non_empty_str(key, f"assignments[{index}].evidence key")
            if field in normalized_evidence:
                raise PermissionPlanEvidenceError(f"PermissionPlan assignment {index} repeats evidence key {field!r}")
            normalized_evidence[field] = _non_empty_str(value, f"assignments[{index}].evidence[{field}]")
        normalized["evidence"] = normalized_evidence
    return normalized


def _required_str(mapping: Mapping[str, object], field: str, *, index: int) -> str:
    return _non_empty_str(mapping.get(field), f"assignments[{index}].{field}")


def _require_non_empty_str(value: object, field_name: str) -> None:
    _non_empty_str(value, field_name)


def _non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PermissionPlanEvidenceError(f"{field_name} must be a non-empty string")
    if "\0" in value:
        raise PermissionPlanEvidenceError(f"{field_name} must not contain NUL")
    return value


def _json_value(value: object, *, path: str) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            if not isinstance(key, str) or not key:
                raise PermissionPlanEvidenceError(f"{path} contains a non-string or empty key")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, list):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, bytes):
        raise PermissionPlanEvidenceError(f"{path} must not be bytes")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PermissionPlanEvidenceError(f"{path} must be finite")
        return value
    if isinstance(value, str):
        if "\0" in value:
            raise PermissionPlanEvidenceError(f"{path} must not contain NUL")
        return value
    if isinstance(value, Sequence):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise PermissionPlanEvidenceError(f"{path} must be JSON-compatible")
