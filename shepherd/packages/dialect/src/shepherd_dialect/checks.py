"""Check markers — declarative input/output validation (authoring re-pin W1).

The legacy contract (`shepherd_runtime.task.markers`), re-pinned function-form
(tranche D1): a ``Check`` is a frozen predicate + message template attached via
``Annotated`` metadata on task parameters (preconditions) or the return
annotation (postconditions). Preconditions refuse **before the reversible
fork** (S1 seam 1 — nucleus-side, zero carrier cost; trace terminal
``refused``); postconditions raise inside the body, so the wrap discards
(trace terminal ``discarded``). Either way the violation is durable trace
evidence: a namespaced ``check.violation`` event (S1 seam 2).
"""

from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Check",
    "CheckFailed",
    "FileExists",
    "InRange",
    "Matches",
    "MaxLength",
    "NonEmpty",
    "extract_checks",
    "run_checks",
]


@dataclass(frozen=True)
class Check:
    """Marker for runtime verification of inputs/outputs."""

    predicate: Callable[..., bool]
    message: str = ""

    def __call__(self, value: Any) -> bool:
        return self.predicate(value)

    def format_message(self, value: Any, field_name: str = "field") -> str:
        if not self.message:
            return f"Check failed for {field_name}: {value!r}"
        try:
            return self.message.format(value=value, field=field_name)
        except (KeyError, IndexError):
            return self.message


class CheckFailed(Exception):  # noqa: N818 — the legacy CheckFailedError observable, dialect-spelled
    """A declarative Check on a task input or output failed.

    Attributes mirror the legacy ``CheckFailedError``: ``task_name``,
    ``field_name``, ``value``, ``check``, ``phase`` ("precondition" |
    "postcondition").
    """

    def __init__(self, task_name: str, field_name: str, value: Any, check: Check, phase: str) -> None:
        self.task_name = task_name
        self.field_name = field_name
        self.value = value
        self.check = check
        self.phase = phase
        super().__init__(
            f"[{phase}] check failed for {field_name!r}: {check.format_message(value, field_name)} (task {task_name})"
        )


def FileExists(message: str = "") -> Check:
    """Check that a file or directory exists at the given path."""

    def _check(v: Any) -> bool:
        return Path(v).exists()

    return Check(predicate=_check, message=message or "File does not exist: {value}")


def NonEmpty(message: str = "") -> Check:
    """Check that a value is not empty."""

    def _check(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        if isinstance(v, (list, dict, set, tuple)):
            return len(v) > 0
        return True

    return Check(predicate=_check, message=message or "Value must not be empty: {value!r}")


def InRange(min_val: Any = None, max_val: Any = None, message: str = "") -> Check:
    """Check that a numeric value is within inclusive bounds."""

    def _check(v: Any) -> bool:
        if min_val is not None and v < min_val:
            return False
        return not (max_val is not None and v > max_val)

    if not message:
        if min_val is not None and max_val is not None:
            message = f"Value {{value}} not in range [{min_val}, {max_val}]"
        elif min_val is not None:
            message = f"Value {{value}} must be >= {min_val}"
        else:
            message = f"Value {{value}} must be <= {max_val}"
    return Check(predicate=_check, message=message)


def Matches(pattern: str, message: str = "") -> Check:
    """Check that a string value matches a regex pattern."""
    compiled = re.compile(pattern)
    safe_pattern = pattern.replace("{", "{{").replace("}", "}}")

    def _check(v: Any) -> bool:
        return compiled.search(str(v)) is not None

    return Check(
        predicate=_check,
        message=message or f"Value {{value!r}} does not match pattern '{safe_pattern}'",
    )


def MaxLength(length: int, message: str = "") -> Check:
    """Check that len(value) <= length."""

    def _check(v: Any) -> bool:
        return len(v) <= length

    return Check(predicate=_check, message=message or f"Length exceeds maximum of {length}")


def _checks_in(annotation: Any) -> tuple[Check, ...]:
    if typing.get_origin(annotation) is not Annotated:
        return ()
    return tuple(m for m in typing.get_args(annotation)[1:] if isinstance(m, Check))


def extract_checks(fn: Any) -> tuple[dict[str, tuple[Check, ...]], tuple[Check, ...]]:
    """Function-form extraction of Check markers (tranche D1).

    Parameter ``Annotated`` metadata → input checks (fields with no Check get
    no entry; instance identity preserved); return annotation → output checks.
    """
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except NameError:
        # Stringified annotations (PEP 563) referencing closure locals cannot be
        # resolved from the function object; fall back to whatever is live.
        hints = {k: v for k, v in getattr(fn, "__annotations__", {}).items() if not isinstance(v, str)}
    input_checks = {name: checks for name, ann in hints.items() if name != "return" and (checks := _checks_in(ann))}
    return input_checks, _checks_in(hints.get("return"))


def run_checks(
    task_name: str,
    fn: Any,
    args: tuple,
    kwargs: dict,
    input_checks: dict[str, tuple[Check, ...]],
) -> None:
    """Run preconditions over the bound call; raise on the FIRST failing check."""
    if not input_checks:
        return
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    for field, checks in input_checks.items():
        if field not in bound.arguments:
            continue
        value = bound.arguments[field]
        for chk in checks:
            if not chk(value):
                raise CheckFailed(task_name, field, value, chk, "precondition")
