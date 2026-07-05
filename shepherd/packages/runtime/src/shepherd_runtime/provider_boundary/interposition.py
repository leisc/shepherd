"""D1 ``ProviderInterposition`` Protocol + D4 ``BypassInterposition``.

One adapter implements ``ProviderInterposition`` per provider
(Claude, OpenAI, LiteLLM, OpenCode, ``BypassInterposition``).
Adapters translate provider-native message streams into the
kernel-v3 sextet via the recorder; they do not emit kernel records
directly.

``BypassInterposition`` is the substitution mechanism: when
registered as ``handle(model.call, fn)``, calls ``fn(request)``
instead of invoking the provider SDK. The Phase 1 callable spine records
runtime-normalized ``model.call`` evidence around this adapter; full
provider sextet synthesis remains deferred to the provider-interposition
implementation track.

Pinned by `docs/design/proposed/260505-plans/CONTRACTS.md` D1, D4.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from shepherd_runtime.provider_boundary.payloads import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from shepherd_runtime.provider_boundary.runtime import ProviderRuntime
    from shepherd_runtime.provider_boundary.tools import ToolHandler

__all__ = ["BypassInterposition", "ProviderInterposition", "ResponderFn"]


ResponderFn = Callable[
    [ModelRequest],
    "ModelResponse | Awaitable[ModelResponse]",
]


class ProviderInterposition(Protocol):
    """Translate one provider call into kernel-v3 sextet records.

    One ``perform_model_call`` invocation corresponds to one
    ``EffectDeclaration("model.call")`` and one ``EffectCapture``
    (or ``SelectionClosed`` if aborted). Tool calls inside the turn
    each emit their own ``EffectDeclaration("tool.<name>")`` sextet.

    Adapters do not retain SDK-side state across calls; session
    identity is carried through ``ModelResponse.session_id``.
    """

    async def perform_model_call(
        self,
        request: ModelRequest,
        runtime: ProviderRuntime,
        tool_handler: ToolHandler,
    ) -> ModelResponse: ...


class BypassInterposition:
    """``ProviderInterposition`` adapter that bypasses the SDK.

    When registered as ``handle(model.call, fn)``, calls ``fn(request)``
    instead of invoking the provider SDK. Used for record-replay
    fixtures, mocking, and the integration gate.

    The Phase 1 runtime records ``model.call`` request/selection/completion
    around this adapter in ``nucleus.delivery``. The adapter intentionally does
    not emit the full provider sextet itself; ``ContinuationResume`` /
    ``ResumeReturn`` and cancellation closure records remain deferred
    provider-boundary work. It honors the ``ProviderInterposition`` Protocol so
    consumer adapters can test against it.
    """

    def __init__(self, fn: ResponderFn, *, handler_id: str = "bypass.v1") -> None:
        self._fn = fn
        self.handler_id = handler_id

    async def perform_model_call(
        self,
        request: ModelRequest,
        runtime: ProviderRuntime,
        tool_handler: ToolHandler,
    ) -> ModelResponse:
        """Invoke the responder and return the synthesized response.

        Bypass semantics: no provider-boundary recorder calls. The delivery
        layer records the Phase 1 runtime evidence around this call; full
        provider sextet emission remains deferred.
        """
        result = self._fn(request)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ModelResponse):
            raise TypeError(f"BypassInterposition responder must return a ModelResponse; got {type(result).__name__}")
        return result
