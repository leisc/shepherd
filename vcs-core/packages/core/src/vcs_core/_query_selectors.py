"""Small pure selector evaluator for inventory snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vcs_core._errors import VcsCoreError

if TYPE_CHECKING:
    from collections.abc import Callable

    from vcs_core._query_inventory import InventoryItem, InventorySnapshot


class InventorySelectorError(VcsCoreError, ValueError):
    """Raised when an inventory selector cannot be parsed or evaluated."""


@dataclass(frozen=True)
class InventorySelector:
    """Pure predicate over an inventory item."""

    predicate: Callable[[InventoryItem], bool]
    expression: str

    def matches(self, item: InventoryItem) -> bool:
        return self.predicate(item)

    def __and__(self, other: InventorySelector) -> InventorySelector:
        return InventorySelector(
            predicate=lambda item: self.matches(item) and other.matches(item),
            expression=f"({self.expression} {other.expression})",
        )

    def __or__(self, other: InventorySelector) -> InventorySelector:
        return InventorySelector(
            predicate=lambda item: self.matches(item) or other.matches(item),
            expression=f"({self.expression} | {other.expression})",
        )

    def __sub__(self, other: InventorySelector) -> InventorySelector:
        return InventorySelector(
            predicate=lambda item: self.matches(item) and not other.matches(item),
            expression=f"({self.expression} - {other.expression})",
        )


def select_inventory_items(snapshot: InventorySnapshot, selector: str | InventorySelector) -> tuple[InventoryItem, ...]:
    """Select items from a snapshot without mutating it or consulting storage."""
    parsed = parse_selector(selector) if isinstance(selector, str) else selector
    return tuple(item for item in snapshot.items if parsed.matches(item))


def parse_selector(expression: str) -> InventorySelector:
    text = expression.strip()
    if not text:
        raise InventorySelectorError("inventory selector must not be empty")
    union_parts = [part.strip() for part in text.split("|")]
    if any(not part for part in union_parts):
        raise InventorySelectorError(f"invalid inventory selector union: {expression!r}")
    selector = _parse_intersection(union_parts[0])
    for part in union_parts[1:]:
        selector = selector | _parse_intersection(part)
    return selector


def domain(value: str) -> InventorySelector:
    return InventorySelector(lambda item: item.domain == value, f"domain={value}")


def kind(value: str) -> InventorySelector:
    return InventorySelector(lambda item: item.kind == value, f"kind={value}")


def role(value: str) -> InventorySelector:
    return InventorySelector(lambda item: value in item.role, f"role={value}")


def health_status(value: str) -> InventorySelector:
    return InventorySelector(lambda item: item.health.status == value, f"health.status={value}")


def health_validity(value: str) -> InventorySelector:
    return InventorySelector(lambda item: item.health.validity == value, f"health.validity={value}")


def authority_role(value: str) -> InventorySelector:
    return InventorySelector(lambda item: item.health.authority_role == value, f"authority={value}")


def issue(value: str) -> InventorySelector:
    return InventorySelector(lambda item: any(current.code == value for current in item.issues), f"issue={value}")


def field_equals(name: str, value: str) -> InventorySelector:
    if not name:
        raise InventorySelectorError("field selector requires a field name")
    return InventorySelector(lambda item: str(item.fields.get(name)) == value, f"field.{name}={value}")


def _parse_intersection(expression: str) -> InventorySelector:
    tokens = expression.split()
    if not tokens:
        raise InventorySelectorError("inventory selector clause must not be empty")
    positive: InventorySelector | None = None
    negative: list[InventorySelector] = []
    for token in tokens:
        if token.startswith("-"):
            if len(token) == 1:
                raise InventorySelectorError("inventory selector '-' must prefix a selector token")
            negative.append(_parse_token(token[1:]))
            continue
        parsed = _parse_token(token)
        positive = parsed if positive is None else positive & parsed
    selector = positive or InventorySelector(lambda _item: True, "*")
    for excluded in negative:
        selector = selector - excluded
    return selector


def _parse_token(token: str) -> InventorySelector:
    key, value = _split_token(token)
    if key == "domain":
        return domain(value)
    if key == "kind":
        return kind(value)
    if key == "role":
        return role(value)
    if key == "health.status":
        return health_status(value)
    if key == "health.validity":
        return health_validity(value)
    if key == "authority":
        return authority_role(value)
    if key == "issue":
        return issue(value)
    if key.startswith("field."):
        return field_equals(key.removeprefix("field."), value)
    raise InventorySelectorError(f"unknown inventory selector token: {token!r}")


def _split_token(token: str) -> tuple[str, str]:
    if "=" in token:
        key, value = token.split("=", 1)
    elif ":" in token:
        key, value = token.split(":", 1)
    else:
        raise InventorySelectorError(f"inventory selector token must use '=' or ':': {token!r}")
    if not key or not value:
        raise InventorySelectorError(f"inventory selector token has empty key or value: {token!r}")
    return key, value
