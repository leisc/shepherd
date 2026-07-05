"""Owner-path pipeline API for legacy class-form workflow composition.

The top-level ``shepherd`` facade no longer exports ``Pipeline``. Import from
``shepherd.pipeline`` only when working on advanced workflow migration or
class-form task orchestration. First-run examples should use function-form
``@task`` and ``deliver(...)`` from the callable spine.

This module provides the legacy Level 3 entry point for task composition:

    from shepherd.pipeline import Pipeline
    from shepherd_runtime.device import Device

    flow = (
        Pipeline(WriteCode)
        .retry(max_attempts=3)
        .gate(lambda r: "TODO" not in r.code_written)
    )

    with Device("container"):
        result = flow.run(feature="auth", filename="auth.py")

    if result.rejected:
        print(f"Gate rejected: {result.reason}")
    else:
        print(result.code_written)

Pipeline wraps class-form @task classes and provides a fluent interface for adding
combinators like retry(), gate(), and timeout(). The run() method executes
the composed pipeline synchronously, while arun() provides async execution.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from shepherd_runtime.combinators import Rejected
from shepherd_runtime.combinators import gate as _gate
from shepherd_runtime.combinators import recover as _recover
from shepherd_runtime.combinators import retry as _retry
from shepherd_runtime.combinators import timeout as _timeout
from shepherd_runtime.sync import run_sync

from .adapters import task_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from shepherd_core.scope.stream import Stream
    from shepherd_runtime.scope import Scope

T = TypeVar("T")


@dataclass
class PipelineResult(Generic[T]):
    """Result container for Pipeline execution.

    Wraps the task result with metadata about the execution:
    - value: The task instance (with Output fields populated)
    - effects: The effect stream from this execution
    - rejected: Whether a gate rejected the result
    - reason: Rejection reason (if rejected)

    Attributes delegate to value for convenience:
        result.code_written  # Delegates to result.value.code_written

    Example:
        result = Pipeline(WriteCode).gate(valid).run(feature="auth")

        if result.rejected:
            print(f"Rejected: {result.reason}")
        else:
            print(result.code_written)  # Delegated access
            print(result.value.code_written)  # Explicit access
    """

    value: T | None
    effects: Stream
    rejected: bool = False
    reason: str | None = None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped value.

        This enables `result.field` instead of `result.value.field`.

        Raises:
            AttributeError: If value is None (rejected without value) or
                           if the attribute doesn't exist on value
        """
        if name.startswith("_"):
            raise AttributeError(name)

        if self.value is None:
            raise AttributeError(
                f"Cannot access '{name}' on rejected PipelineResult with no value. "
                f"Check result.rejected before accessing attributes."
            )
        return getattr(self.value, name)

    def __dir__(self) -> list[str]:
        """Include value's attributes for tab completion."""
        base = ["value", "effects", "rejected", "reason"]
        if self.value is not None:
            base.extend(a for a in dir(self.value) if not a.startswith("_"))
        return base

    def __repr__(self) -> str:
        if self.rejected:
            return f"PipelineResult(rejected=True, reason={self.reason!r})"
        return f"PipelineResult({self.value!r})"


class Pipeline(Generic[T]):
    """Fluent wrapper for composing tasks with combinators.

    This is the owner-path Level 3 entry point for legacy class-form task
    composition. It wraps a @task class and allows chaining combinators like
    retry(), gate(), and timeout().

    Example:
        from shepherd.pipeline import Pipeline
        from shepherd_runtime.device import Device

        # Build a pipeline with retry and quality gate
        flow = (
            Pipeline(WriteCode)
            .retry(max_attempts=3)
            .gate(lambda r: "TODO" not in r.code_written)
        )

        # Execute with device context
        with Device("container"):
            result = flow.run(feature="auth", filename="auth.py")

        # Handle result
        if result.rejected:
            print(f"Gate rejected: {result.reason}")
        else:
            print(result.code_written)

    Attributes:
        _task_class: The underlying @task class
        _combinators: List of combinator transformations to apply
    """

    def __init__(self, task_class: type[T]) -> None:
        """Initialize a Pipeline for a task class.

        Args:
            task_class: A @task-decorated class
        """
        self._task_class = task_class
        self._combinators: list[Callable] = []

    def retry(
        self,
        max_attempts: int = 3,
        delay_seconds: float = 1.0,
        backoff: float = 1.0,
    ) -> Pipeline[T]:
        """Add retry behavior for transient failures.

        Args:
            max_attempts: Maximum number of attempts (default 3)
            delay_seconds: Initial delay between retries (default 1.0)
            backoff: Multiplier for delay after each attempt (default 1.0)

        Returns:
            self (for chaining)

        Example:
            Pipeline(FlakyTask).retry(max_attempts=5, delay_seconds=2, backoff=2)
        """
        self._combinators.append(
            lambda t, ma=max_attempts, ds=delay_seconds, bo=backoff: _retry(
                t, max_attempts=ma, delay_seconds=ds, backoff=bo
            )
        )
        return self

    def gate(
        self,
        predicate: Callable[[T], bool] | Callable[[T, Stream], bool],
    ) -> Pipeline[T]:
        """Add a quality gate that can reject results.

        The gate runs the task, then evaluates the predicate. If the
        predicate returns False, the result is wrapped in Rejected and
        effects are not committed to the parent scope.

        Args:
            predicate: Either a single-arg (result) -> bool or
                      two-arg (result, effects) -> bool function.
                      Return True to accept, False to reject.

        Returns:
            self (for chaining)

        Example:
            # Single-arg predicate (common case)
            Pipeline(WriteCode).gate(lambda r: "TODO" not in r.code_written)

            # Two-arg predicate (access effects)
            Pipeline(WriteCode).gate(lambda r, e: len(list(e.query(FilePatch))) < 10)
        """
        # Detect signature and wrap single-arg predicates
        # Use default argument to capture predicate by value, not reference
        sig = inspect.signature(predicate)
        if len(sig.parameters) == 1:

            def wrapped(r: Any, e: Any, _orig: Any = predicate) -> Any:
                return _orig(r)
        else:
            wrapped = predicate

        self._combinators.append(lambda t, p=wrapped: _gate(t, p))
        return self

    def timeout(self, seconds: float) -> Pipeline[T]:
        """Add a timeout that raises TaskTimeoutError if exceeded.

        Args:
            seconds: Maximum execution time in seconds

        Returns:
            self (for chaining)

        Example:
            Pipeline(SlowTask).timeout(30)  # 30 second timeout
        """
        self._combinators.append(lambda t, s=seconds: _timeout(t, seconds=s))
        return self

    def recover(
        self,
        handler: Callable[[Exception], T],
    ) -> Pipeline[T]:
        """Provide a fallback value on error.

        Unlike retry, this provides a default value on failure instead of
        retrying. The recovery handler receives the exception and returns
        an appropriate fallback value.

        Args:
            handler: Function (exception) -> fallback_value. Can be sync
                    or async. May also re-raise a different exception.

        Returns:
            self (for chaining)

        Example:
            # Return empty result on failure
            Pipeline(SearchTask).recover(
                lambda e: SearchResults(items=[], error=str(e))
            )

            # Log and return default
            def handle_error(e):
                logging.error(f"Task failed: {e}")
                return default_value

            Pipeline(RiskyTask).recover(handle_error)

            # Compose with retry - try 3 times, then fallback
            Pipeline(FlakyTask).retry(3).recover(lambda e: fallback)
        """
        self._combinators.append(lambda t, h=handler: _recover(t, h))
        return self

    def build(self) -> Any:
        """Return the underlying combinator-compatible callable.

        This is the Level 4 escape hatch for advanced composition.
        The returned callable can be passed directly to other combinators.

        Returns:
            A callable with signature (inputs: dict, scope: Scope) -> Awaitable[T]

        Example:
            # Build and use with additional combinators
            base = Pipeline(WriteCode).retry(3).build()
            gated = gate(base, lambda r, e: r.is_valid)
            result = await gated({"feature": "auth"}, scope)
        """
        base = task_fn(self._task_class)
        result = base
        for combinator in self._combinators:
            result = combinator(result)
        return result

    async def arun(
        self,
        scope: Scope | None = None,
        **kwargs: Any,
    ) -> PipelineResult[T]:
        """Execute the pipeline asynchronously.

        Args:
            scope: Optional scope to execute in. If None, uses current scope.
            **kwargs: Input values for the task

        Returns:
            PipelineResult wrapping the task result and effects

        Raises:
            ScopeNotConfiguredError: If no scope is available

        Example:
            result = await Pipeline(WriteCode).retry(3).arun(feature="auth")
        """
        from shepherd_core.errors import ScopeNotConfiguredError
        from shepherd_runtime.scope import current_scope

        if scope is None:
            scope = current_scope()
        if scope is None:
            raise ScopeNotConfiguredError(
                "No scope available. Pass scope=... or run inside shepherd_runtime.scope.Scope."
            )

        task_callable = self.build()
        result = await task_callable(kwargs, scope)

        return self._wrap_result(result, scope)

    def run(
        self,
        scope: Scope | None = None,
        **kwargs: Any,
    ) -> PipelineResult[T]:
        """Execute the pipeline synchronously.

        This owner-path helper bridges to async execution via run_sync(),
        preserving ContextVars (including Device context) across the thread
        boundary for legacy workflow callers.

        Args:
            scope: Optional scope to execute in. If None, uses current scope.
            **kwargs: Input values for the task

        Returns:
            PipelineResult wrapping the task result and effects

        Raises:
            ScopeNotConfiguredError: If no scope is available

        Example:
            with Device("container"):
                result = Pipeline(WriteCode).retry(3).run(feature="auth")
        """
        return run_sync(self.arun(scope=scope, **kwargs))

    def _wrap_result(
        self,
        result: T | Rejected,
        scope: Scope,
    ) -> PipelineResult[T]:
        """Wrap result in PipelineResult, handling Rejected case."""
        if isinstance(result, Rejected):
            return PipelineResult(
                value=result.value,
                effects=result.effects if result.effects is not None else scope.effects,
                rejected=True,
                reason=result.reason,
            )
        # Access effects from task instance or scope
        task_scope = getattr(result, "_task_scope", None)
        effects = task_scope.effects if task_scope is not None else scope.effects
        return PipelineResult(
            value=result,
            effects=effects,
            rejected=False,
            reason=None,
        )

    def __repr__(self) -> str:
        """Return readable representation of the pipeline."""
        task_name = getattr(self._task_class, "__name__", str(self._task_class))
        if not self._combinators:
            return f"Pipeline({task_name})"

        # Try to extract combinator names
        parts = []
        for c in self._combinators:
            # Lambda functions don't have useful names, so we look at what they wrap
            name = getattr(c, "__name__", "combinator")
            parts.append(name)

        return f"Pipeline({task_name}).{'.'.join(parts)}"


__all__ = [
    "Pipeline",
    "PipelineResult",
]
