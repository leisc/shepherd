"""Effect-kind naming and canonical effect identity helpers.

CONTRACTS B5 plus DECISIONS D3, D9.

Public kinds are hierarchical dot-separated names. Each segment matches
``[a-z][a-z0-9_]*`` (Python identifier rules, lower case). Some specialized
helpers such as ``tool_kind`` and ``model_kind`` still expose a single-dot
namespace/name contract.

D9 reserves ``model.embed`` and ``model.stream``; v1 allocates only
``model.call``. The validator rejects reserved verbs at the
construction layer; owner plans (Plan 01) MAY further restrict to
``call`` at registration time.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

__all__ = [
    "ConflictingKind",
    "EffectIdentity",
    "effect_key_for_class",
    "effect_key_for_event",
    "is_explicit_effect_kind",
    "model_kind",
    "parse_matcher_kind_sugar",
    "register_effect_class",
    "split_effect_kind",
    "tool_kind",
    "validate_public_effect_kind",
]


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_MODEL_VERBS = frozenset({"embed", "stream"})
_EXPLICIT_KIND_OWNERS: dict[str, tuple[str, str, str]] = {}


class ConflictingKind(TypeError):  # noqa: N818
    """An effect class declared an invalid or already-claimed public kind."""


@dataclass(frozen=True)
class EffectIdentity:
    """Canonical identity attached to an ``Ask`` / ``Tell`` class."""

    key: str
    explicit: bool
    category: str | None
    owner: tuple[str, str, int | None]


def split_effect_kind(kind: str) -> tuple[str, str]:
    """Split ``"namespace.name"`` into ``(namespace, name)``.

    Raises:
        ValueError: if the form is wrong, has multiple dots, or the
            name does not match ``[a-z][a-z0-9_]*``.
    """
    if kind.count(".") != 1:
        raise ValueError(f"effect_kind must be 'namespace.name'; got {kind!r}")
    namespace, name = kind.split(".", 1)
    if not _NAME_RE.match(namespace):
        raise ValueError(f"effect_kind namespace {namespace!r} must match [a-z][a-z0-9_]*")
    if not _NAME_RE.match(name):
        raise ValueError(f"effect_kind name {name!r} must match [a-z][a-z0-9_]*")
    return namespace, name


def validate_public_effect_kind(kind: str) -> str:
    """Validate a hierarchical public effect kind.

    Public effect kinds are authority names such as ``filesystem`` or
    ``review.verdict``. The reserved ``local`` root is kept for derived-local
    runtime keys and cannot be claimed by user-authored ``kind=`` declarations.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"effect kind must be a non-empty string; got {kind!r}")
    parts = kind.split(".")
    for part in parts:
        if not _NAME_RE.match(part):
            raise ValueError(f"effect kind segment {part!r} must match [a-z][a-z0-9_]* in {kind!r}")
    if parts[0] == "local":
        raise ValueError("explicit effect kinds cannot use the reserved 'local' root")
    return kind


def parse_matcher_kind_sugar(value: str) -> tuple[str, str]:
    """Parse a kind string or wildcard matcher sugar.

    Returns ``(mode, root_kind)`` where mode is ``"exact"``, ``"descendants"``,
    or ``"subtree"``. Wildcard strings are matcher syntax, not concrete effect
    kinds.
    """
    if value.endswith(".**"):
        return "subtree", validate_public_effect_kind(value[:-3])
    if value.endswith(".*"):
        return "descendants", validate_public_effect_kind(value[:-2])
    return "exact", validate_public_effect_kind(value)


def register_effect_class(cls: type, *, explicit_kind: str | None, category: str | None) -> EffectIdentity:
    """Attach and return canonical identity metadata for an effect class."""
    owner = _owner_for_class(cls)
    if explicit_kind is not None:
        try:
            key = validate_public_effect_kind(explicit_kind)
        except ValueError as exc:
            raise ConflictingKind(f"{cls.__qualname__}: invalid kind={explicit_kind!r}: {exc}") from exc
        _validate_public_parent_hierarchy(cls, key)
        _claim_explicit_kind(cls, key)
        identity = EffectIdentity(key=key, explicit=True, category=category, owner=owner)
    else:
        parents = _public_kind_ancestors(cls)
        if parents:
            parent_list = ", ".join(repr(parent.key) for parent in parents)
            raise ConflictingKind(
                f"{cls.__qualname__}: subclasses of public kind-bearing parents must declare "
                f"an explicit descendant kind=; inherited parents: {parent_list}"
            )
        identity = EffectIdentity(
            key=_derive_local_key(cls, owner),
            explicit=False,
            category=category,
            owner=owner,
        )
    if identity.explicit:
        cls.kind = identity.key  # type: ignore[attr-defined]
    cls.__shepherd_effect_identity__ = identity  # type: ignore[attr-defined]
    return identity


def effect_key_for_class(cls: type) -> str:
    """Return the canonical effect key for a class."""
    identity = getattr(cls, "__shepherd_effect_identity__", None)
    if isinstance(identity, EffectIdentity):
        return identity.key
    return _derive_local_key(cls, _owner_for_class(cls))


def effect_key_for_event(event: object) -> str:
    """Return the canonical effect key for an event instance."""
    return effect_key_for_class(type(event))


def is_explicit_effect_kind(cls: type) -> bool:
    """Return true when ``cls`` declared a stable public ``kind=``."""
    identity = getattr(cls, "__shepherd_effect_identity__", None)
    return isinstance(identity, EffectIdentity) and identity.explicit


def tool_kind(name: str) -> str:
    """Construct ``"tool.<name>"`` after validating ``name`` per D3."""
    split_effect_kind(f"tool.{name}")
    return f"tool.{name}"


def model_kind(verb: str) -> str:
    """Construct ``"model.<verb>"`` after validating ``verb`` per D9.

    First and only verb in v1 is ``model.call``. Reserved but not
    allocated: ``model.embed``, ``model.stream``. Plans must not use
    reserved verbs without updating D9.
    """
    if verb in _RESERVED_MODEL_VERBS:
        raise ValueError(f"model verb {verb!r} is reserved; not allocated in v1")
    if not _NAME_RE.match(verb):
        raise ValueError(f"model verb {verb!r} must match [a-z][a-z0-9_]*")
    return f"model.{verb}"


def _claim_explicit_kind(cls: type, key: str) -> None:
    owner = (cls.__module__, cls.__qualname__, key)
    existing = _EXPLICIT_KIND_OWNERS.get(key)
    if existing is not None and existing != owner:
        raise ConflictingKind(f"{cls.__qualname__}: kind {key!r} is already claimed by {existing[0]}.{existing[1]}")
    _EXPLICIT_KIND_OWNERS[key] = owner


def _validate_public_parent_hierarchy(cls: type, key: str) -> None:
    parents = _public_kind_ancestors(cls)
    if not parents:
        return
    parent_keys = tuple(parent.key for parent in parents)
    if not _single_kind_branch(parent_keys):
        parent_list = ", ".join(repr(parent_key) for parent_key in parent_keys)
        raise ConflictingKind(
            f"{cls.__qualname__}: multiple public kind-bearing parents are ambiguous in this cut: {parent_list}"
        )
    for parent in parents:
        if not _strict_kind_descendant(key, parent.key):
            raise ConflictingKind(
                f"{cls.__qualname__}: kind {key!r} must be a descendant of public parent kind {parent.key!r}"
            )


def _public_kind_ancestors(cls: type) -> tuple[EffectIdentity, ...]:
    parents: list[EffectIdentity] = []
    seen: set[str] = set()
    for base in cls.__mro__[1:]:
        identity = getattr(base, "__shepherd_effect_identity__", None)
        if not isinstance(identity, EffectIdentity) or not identity.explicit:
            continue
        if identity.key in seen:
            continue
        seen.add(identity.key)
        parents.append(identity)
    return tuple(parents)


def _single_kind_branch(kinds: tuple[str, ...]) -> bool:
    for left in kinds:
        for right in kinds:
            if left == right:
                continue
            if not (_kind_contains(left, right) or _kind_contains(right, left)):
                return False
    return True


def _strict_kind_descendant(child: str, parent: str) -> bool:
    return child.startswith(f"{parent}.")


def _kind_contains(parent: str, child: str) -> bool:
    return child == parent or child.startswith(f"{parent}.")


def _owner_for_class(cls: type) -> tuple[str, str, int | None]:
    try:
        _, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        line = None
    return cls.__module__, cls.__qualname__, line


def _derive_local_key(cls: type, owner: tuple[str, str, int | None]) -> str:
    module, qualname, line = owner
    parts = ("local", *_split_owner(module), *_split_owner(qualname))
    suffix = f"l{line}" if line is not None else f"dynamic_{id(cls):x}"
    return ".".join((*parts, suffix))


def _split_owner(value: str) -> tuple[str, ...]:
    parts = re.split(r"[.<>\s:/\\-]+", value)
    return tuple(_sanitize_segment(part) for part in parts if part)


def _sanitize_segment(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not value:
        return "anonymous"
    if value[0].isdigit():
        value = f"n{value}"
    return value
