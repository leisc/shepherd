"""Ask / Tell base classes with class-creation `on_unhandled` validation.

CONTRACTS B1, B2 plus DECISIONS D7.

The metaclass enforces:

- ``Ask`` subclasses reject ``on_unhandled="ignore"`` at class
  creation (R has no sensible default).
- ``Tell`` subclasses accept ``"raise"``, ``"suspend"``, or
  ``"ignore"``; default is ``"ignore"``.
- Both forms of declaration are accepted: keyword
  (``class X(Ask[str], on_unhandled="raise")``) and class-attribute
  (``class X(Ask[str]): on_unhandled = "raise"``).

Subclasses compose with ``@dataclass(frozen=True)``; the metaclass
must not interfere with field collection or hashability.
"""

from __future__ import annotations

from typing import ClassVar, Generic, Literal, TypeVar

from shepherd_runtime.effects.effect_kind import ConflictingKind, register_effect_class

__all__ = ["Ask", "Tell", "_EffectMeta"]


R_co = TypeVar("R_co", covariant=True)


class _EffectMeta(type):
    """Metaclass for ``Ask`` and ``Tell`` base classes.

    Validates ``on_unhandled`` at class creation per DECISIONS D7.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict,
        on_unhandled: str | None = None,
        kind: str | None = None,
        **kwargs,
    ) -> _EffectMeta:
        attr_kind = namespace.get("kind")
        if kind is not None and isinstance(attr_kind, str) and attr_kind != kind:
            raise ConflictingKind(f"{name}: conflicting kind declarations {kind!r} and {attr_kind!r}")
        explicit_kind = kind if kind is not None else attr_kind if isinstance(attr_kind, str) else None
        if kind is not None:
            namespace["kind"] = kind
        if on_unhandled is not None:
            namespace["on_unhandled"] = on_unhandled
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        category = _effect_category(cls)
        ou = getattr(cls, "on_unhandled", None)
        if ou is not None:
            is_ask_subclass = any(getattr(b, "__name__", None) == "Ask" for b in cls.__mro__)
            if is_ask_subclass and ou == "ignore":
                raise TypeError(
                    f"{name}: Ask subclass cannot use on_unhandled='ignore' "
                    f"(R has no sensible default; use 'raise' or 'suspend')"
                )
            if ou not in ("raise", "suspend", "ignore"):
                raise TypeError(f"{name}: invalid on_unhandled={ou!r}; must be 'raise', 'suspend', or 'ignore'")
        register_effect_class(cls, explicit_kind=explicit_kind, category=category)
        return cls


def _effect_category(cls: type) -> str | None:
    if cls.__name__ == "Ask":
        return "ask"
    if cls.__name__ == "Tell":
        return "tell"
    categories = {
        getattr(getattr(base, "__shepherd_effect_identity__", None), "category", None) for base in cls.__mro__[1:]
    }
    if "ask" in categories:
        return "ask"
    if "tell" in categories:
        return "tell"
    return None


class Ask(Generic[R_co], metaclass=_EffectMeta):
    """Base class for typed ask effects (CONTRACTS B1).

    Subclasses parameterize the response type ``R``. Default
    ``on_unhandled`` is ``None`` (resolves at perform-time per
    DECISIONS D7). ``Ask`` subclasses cannot use
    ``on_unhandled="ignore"`` because ``R`` has no sensible default.
    """

    on_unhandled: ClassVar[Literal["raise", "suspend"] | None] = None

    @classmethod
    def where(cls, **constraints: object) -> object:
        """Return ``Match.subtree(cls).where(...)`` without importing policy at module load."""
        from shepherd_runtime.effects.policy import Match

        return Match.subtree(cls).where(**constraints)

    @classmethod
    def where_not(cls, **constraints: object) -> object:
        """Return ``Match.subtree(cls).where_not(...)`` without importing policy at module load."""
        from shepherd_runtime.effects.policy import Match

        return Match.subtree(cls).where_not(**constraints)


class Tell(metaclass=_EffectMeta):
    """Base class for typed tell effects (CONTRACTS B2).

    Default ``on_unhandled`` is ``"ignore"``. Subclasses may override
    to ``"raise"`` or ``"suspend"``.
    """

    on_unhandled: ClassVar[Literal["raise", "suspend", "ignore"]] = "ignore"

    @classmethod
    def where(cls, **constraints: object) -> object:
        """Return ``Match.subtree(cls).where(...)`` without importing policy at module load."""
        from shepherd_runtime.effects.policy import Match

        return Match.subtree(cls).where(**constraints)

    @classmethod
    def where_not(cls, **constraints: object) -> object:
        """Return ``Match.subtree(cls).where_not(...)`` without importing policy at module load."""
        from shepherd_runtime.effects.policy import Match

        return Match.subtree(cls).where_not(**constraints)
