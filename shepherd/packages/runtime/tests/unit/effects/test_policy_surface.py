"""Path-A ``Match`` / ``Plan`` conformance core."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from shepherd_runtime.effects import (
    Ask,
    EffectNotPermitted,
    EffectSurfaceEmpty,
    EffectSurfaceTooWide,
    Match,
    OverbroadHandler,
    Plan,
    PlanNotExtractable,
    Subset,
    Tell,
)


@dataclass(frozen=True)
class Pick(Ask[str], kind="policy_surface.pick"):
    options: tuple[str, ...]
    severity: int = 0


@dataclass(frozen=True)
class Audit(Tell, kind="policy_surface.audit"):
    message: str
    severity: int = 0


class SpecialAudit(Audit, kind="policy_surface.audit.special"):
    pass


def test_match_kind_constructors_and_class_subtree_matching() -> None:
    exact = Match.exact(Audit)
    subtree = Match.subtree(Audit)
    descendants = Match.descendants(Audit)

    assert exact.matches(Audit(message="x"))
    assert not exact.matches(SpecialAudit(message="x"))
    assert subtree.matches(SpecialAudit(message="x"))
    assert descendants.matches(SpecialAudit(message="x"))
    assert not descendants.matches(Audit(message="x"))

    assert Match.exact(Audit).subset_of(Match.subtree(Audit)) is Subset.Yes
    assert Match.subtree(Audit).subset_of(Match.exact(Audit)) is Subset.Unknown
    assert Match.subtree(Audit).subset_of(Match.descendants(Audit)) is Subset.No


def test_category_root_subset_reasoning_preserves_class_evidence() -> None:
    assert Match.exact(Audit).subset_of(Match.subtree(Tell)) is Subset.Yes
    assert Match.subtree(Audit).subset_of(Match.subtree(Tell)) is Subset.Yes
    assert Plan().allow_only(Match.exact(Audit)).subset_of(Plan().allow_only(Match.subtree(Tell))) is Subset.Yes

    # A bare string carries no Python category evidence unless a later slice wires
    # a public-kind registry lookup into structural reasoning.
    assert Match.exact("policy_surface.audit").subset_of(Match.subtree(Tell)) is Subset.Unknown


def test_match_algebra_normalizes_core_identities() -> None:
    audit = Match.subtree(Audit)
    pick = Match.subtree(Pick)

    assert audit | Match.nothing() == audit
    assert audit & Match.all() == audit
    assert audit & ~audit == Match.nothing()
    assert audit | ~audit == Match.all()
    assert Match.exact(Audit) & Match.exact(Pick) == Match.nothing()
    assert Match.exact(Audit) & Match.exact("policy_surface.pick") == Match.nothing()
    assert audit | (audit & pick) == audit
    assert audit & (audit | pick) == audit
    assert audit & pick == Match.nothing()
    assert (Match.exact(Audit) - audit).is_empty() is Subset.Yes
    assert ~(audit | pick) == (~audit & ~pick)
    assert (audit | pick).subset_of(Match.all()) is Subset.Yes
    assert Match.nothing().subset_of(audit) is Subset.Yes
    assert Match.predicate(lambda event: False).is_empty() is Subset.Unknown


def test_match_wildcard_sugars_and_public_kind_roots() -> None:
    subtree = Match.of("policy_surface.audit.**")
    descendants = Match.of("policy_surface.audit.*")

    assert subtree.matches(Audit(message="root"))
    assert subtree.matches(SpecialAudit(message="child"))
    assert not descendants.matches(Audit(message="root"))
    assert descendants.matches(SpecialAudit(message="child"))
    assert Match.of("policy_surface.audit").equivalent_to(Match.exact(Audit)) is Subset.Yes


def test_match_field_predicates_match_and_support_basic_subset() -> None:
    high = Match.subtree(Audit).where(severity__gte=5)
    exact = Match.subtree(Audit).where(severity=7)
    selected = Match.field("severity", "in", {7, 8})

    assert high.matches(Audit(message="risk", severity=7))
    assert not high.matches(Audit(message="low", severity=1))
    assert exact.subset_of(selected) is Subset.Yes
    assert Match.field("message", "contains", "risk").subset_of(Match.all()) is Subset.Yes
    assert (
        Match.field("message", "contains", "risk").subset_of(Match.field("message", "contains", "risk")) is Subset.Yes
    )
    assert (
        Match.field("message", "contains", "risk").subset_of(Match.field("message", "contains", "other"))
        is Subset.Unknown
    )
    assert Audit.where(severity__gte=5) == high
    assert Audit.where_not(severity=1) == Match.subtree(Audit).where_not(severity=1)


def test_plan_permission_stacking_and_composition_are_immutable() -> None:
    base = Plan().allow_only(Match.subtree(Tell), ref="allow-tells")
    narrowed = base.deny_kind(Match.exact(Audit), ref="deny-audit")
    composed = base | Plan().deny_kind(Match.exact(Audit), ref="deny-audit")

    assert base.installations() != narrowed.installations()
    assert narrowed == composed
    assert narrowed.installation("deny-audit").matcher == Match.exact(Audit)
    assert narrowed.effective_surface() == Match.subtree(Tell) - Match.exact(Audit)
    assert narrowed.subset_of(base) is Subset.Yes


def test_plan_as_may_requires_allow_only() -> None:
    with pytest.raises(PlanNotExtractable):
        Plan().deny_kind(Match.exact(Audit)).extract_may()

    assert Plan().allow_only(Match.subtree(Tell)).extract_may() == Match.subtree(Tell)


def test_deny_tool_is_not_static_may_extraction() -> None:
    allow = Match.subtree("tool")

    with pytest.raises(PlanNotExtractable):
        Plan().deny_tool("read_file").extract_may()

    assert Plan().allow_only(allow).deny_tool("read_file").extract_may() == allow


def test_overbroad_authoritative_handler_is_rejected() -> None:
    with pytest.raises(OverbroadHandler):
        Plan().handle(Match.all(), lambda event: event)

    plan = Plan().observe(Match.all(), lambda event: None)
    assert plan.installations()[0].kind == "observe"


def test_authoritative_handler_field_fragment_is_conservative() -> None:
    plan = Plan().handle(Match.field("severity", "eq", 2), lambda event: event)
    assert plan.installations()[0].matcher == Match.field("severity", "eq", 2)

    for matcher in (
        Match.field("severity", "gte", 3),
        Match.field("severity", "not_in", {0}),
        Match.field("message", "contains", "risk"),
    ):
        with pytest.raises(OverbroadHandler):
            Plan().handle(matcher, lambda event: event)


def test_in_cut_error_classes_carry_structured_attributes() -> None:
    declared = Match.subtree(Tell)
    attempted = Pick(options=("a",))
    error = EffectNotPermitted(declared=declared, effective=declared, attempted=attempted)

    assert error.declared == declared
    assert error.effective == declared
    assert error.attempted is attempted
    assert error.attempted_kind == "policy_surface.pick"

    too_wide = EffectSurfaceTooWide(caller_may=Match.subtree(Tell), callee_may=Match.all())
    assert too_wide.excess == Match.all() - Match.subtree(Tell)

    empty = EffectSurfaceEmpty(declared=Match.nothing(), reason="empty at task entry")
    assert empty.reason == "empty at task entry"
