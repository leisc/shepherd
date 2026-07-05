"""``handle()`` context manager for Plan 04 pure-response effects.

CONTRACTS C1 + DECISIONS D6, D11.

Three call shapes:

- single-class:    ``handle(AmbiguousDesign, lambda e: ...)``
- dict-of-classes: ``handle({E1: fn1, E2: fn2})``
- string-kind:     ``handle("tool.read_file", fake_read_file)``

Sync ``with`` and async ``async with`` both work; the body's
call shape is determined by ``inspect.iscoroutinefunction(fn)`` per
D6.

Per D11 (inside vs outside task lookup): this first behavioral tranche
uses a contextvar-backed dynamic stack. That keeps handlers scoped to
the lexical/async extent of the context manager without exposing the
older handler registry as a public compatibility layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from shepherd_runtime.effects._handler_stack import (
    HandlerBinding,
    make_bindings,
    pop_handlers,
    push_handlers,
)
from shepherd_runtime.effects.shape_detection import HandlerSignatureError

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextvars import Token

__all__ = ["handle"]


@overload
def handle(effect: type, fn: Callable[..., Any], /) -> _HandleContext: ...


@overload
def handle(effects: dict[type, Callable[..., Any]], /) -> _HandleContext: ...


@overload
def handle(effect_kind: str, fn: Callable[..., Any], /) -> _HandleContext: ...


def handle(*args, **kwargs) -> _HandleContext:
    """Install a handler in the active binding env for the lexical block.

    See module docstring for accepted call shapes. Signatures and keys
    are validated eagerly; bindings are active only for the dynamic
    extent of the returned sync or async context manager.
    """
    if kwargs:
        raise HandlerSignatureError("handle(...) does not accept keyword arguments")
    if len(args) == 1 and isinstance(args[0], dict):
        bindings = args[0]
        return _HandleContext(bindings=make_bindings(tuple(bindings.items())))
    if len(args) == 2:
        effect_or_kind, fn = args
        return _HandleContext(bindings=make_bindings(((effect_or_kind, fn),)))
    raise HandlerSignatureError("handle(...) accepts (effect, fn) or (effect_kind, fn) or (dict)")


class _HandleContext:
    """Scoped handler installer for sync and async ``with`` blocks."""

    def __init__(self, *, bindings: tuple[HandlerBinding, ...]) -> None:
        self._bindings = bindings
        self._token: Token[tuple[HandlerBinding, ...]] | None = None

    # Sync context manager protocol
    def __enter__(self) -> None:
        self._enter()

    def __exit__(self, *exc: object) -> None:
        del exc
        self._exit()

    # Async context manager protocol
    async def __aenter__(self) -> None:
        self._enter()

    async def __aexit__(self, *exc: object) -> None:
        del exc
        self._exit()

    def _enter(self) -> None:
        if self._token is not None:
            raise RuntimeError("handle(...) context manager cannot be re-entered")
        self._token = push_handlers(self._bindings)

    def _exit(self) -> None:
        if self._token is None:
            return
        pop_handlers(self._token)
        self._token = None
