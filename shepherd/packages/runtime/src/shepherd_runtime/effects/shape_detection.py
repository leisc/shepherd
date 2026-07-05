"""Detect handler-body shape from a callable signature.

CONTRACTS C2 + DECISIONS D6, D14.

Rules:

- one positional parameter -> ``"pure_response"``
- two positional parameters AND second annotated ``Resumption[...]``
  (parameterized or bare) -> ``"supervisor"``
- otherwise -> ``HandlerSignatureError``

D14: supervisor handlers must be ``async def``. Sync second-
parameter-``Resumption`` shapes are rejected at registration. Pure-
response handlers may be sync or async (per D6). Sync/async dispatch
is orthogonal to shape dispatch.

Annotation-keyed dispatch only. Name-only dispatch on ``resume`` is
rejected per D6 to avoid silent breakage on rename.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal, get_origin, get_type_hints

from shepherd_runtime.effects.resumption import Resumption

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["HandlerShape", "HandlerSignatureError", "detect_handler_shape"]


HandlerShape = Literal["pure_response", "supervisor"]


class HandlerSignatureError(TypeError):
    """The handler callable's signature does not match a supported shape."""


def detect_handler_shape(fn: Callable) -> HandlerShape:
    """Inspect ``fn`` and return its handler-body shape.

    Raises ``HandlerSignatureError`` for any other signature.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise HandlerSignatureError(f"{getattr(fn, '__qualname__', fn)!r}: cannot inspect signature: {exc}") from exc

    positional = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required_keyword_only = [
        p for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    ]
    if required_keyword_only:
        names = ", ".join(p.name for p in required_keyword_only)
        raise HandlerSignatureError(
            f"{fn.__qualname__}: required keyword-only parameters are not "
            f"supported by Phase 1 handler dispatch: {names}"
        )

    if len(positional) == 1:
        return "pure_response"

    if len(positional) == 2:
        # Prefer get_type_hints (resolves PEP 563 string annotations and
        # forward references), fall back to the raw signature annotation
        # if hint resolution fails (annotation visible to the caller but
        # not to get_type_hints, e.g. locally-imported Resumption inside
        # a closure).
        ann = None
        param = positional[1]
        try:
            hints = get_type_hints(fn, include_extras=True)
            ann = hints.get(param.name)
        except Exception:  # noqa: BLE001  # pragma: no cover - resolution best-effort
            ann = None
        if ann is None:
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise HandlerSignatureError(
                    f"{fn.__qualname__}: second parameter must be annotated "
                    f"Resumption[...] (D6); name-only dispatch is rejected"
                )
        if _annotation_is_resumption(ann):
            if not inspect.iscoroutinefunction(fn):
                raise HandlerSignatureError(
                    f"{fn.__qualname__}: supervisor handlers must be `async def` (DECISIONS D14); got sync function"
                )
            return "supervisor"
        raise HandlerSignatureError(f"{fn.__qualname__}: second parameter annotation must be Resumption (got {ann!r})")

    raise HandlerSignatureError(
        f"{fn.__qualname__}: invalid signature; expected 1 or 2 positional parameters (got {len(positional)})"
    )


def _annotation_is_resumption(ann: object) -> bool:
    """Detect ``Resumption`` annotations across resolution forms.

    Handles three cases:

    - resolved class object (``Resumption`` itself)
    - generic alias (``Resumption[T_in, T_out]``) via ``get_origin``
    - PEP 563 string annotation (``"Resumption"`` or
      ``"Resumption[int, str]"``) when ``get_type_hints`` can't resolve
      because the annotation references a closure-local import
    """
    if ann is Resumption:
        return True
    origin = get_origin(ann)
    if origin is Resumption:
        return True
    if isinstance(ann, str):
        return ann == "Resumption" or ann.startswith("Resumption[")
    return False
