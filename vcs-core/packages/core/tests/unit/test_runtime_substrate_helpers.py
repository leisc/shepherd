# under-test: vcs_core._runtime_substrate_helpers
"""The runtime-substrate seam helpers that stay vcs-core-side after PD5.

Tier-A resolution and the in-process provider are generic pieces of the seam;
the run driver that composes them lives in shepherd-dialect (see its
``test_run_driver.py`` for the coordinator round-trip).
"""

from __future__ import annotations

from typing import Any

import pytest
from vcs_core._runtime_substrate_helpers import (
    HandlerStack,
    InProcessExecutionProvider,
    TaskIdResolutionError,
    UnhandledAsk,
    resolve_task_id,
)


def demo_task(stack: HandlerStack, *, marker: str = "hello") -> dict[str, Any]:
    del stack
    return {"marker": marker}


def test_resolve_task_id_colon_and_dotted_forms() -> None:
    assert resolve_task_id(f"{__name__}:demo_task") is demo_task
    assert resolve_task_id(f"{__name__}.demo_task") is demo_task


def test_resolve_task_id_failures_are_loud() -> None:
    with pytest.raises(TaskIdResolutionError, match="cannot import"):
        resolve_task_id("no.such.module:fn")
    with pytest.raises(TaskIdResolutionError, match="has no attribute"):
        resolve_task_id(f"{__name__}:nope")
    with pytest.raises(TaskIdResolutionError, match="non-callable"):
        resolve_task_id(f"{__name__}:__doc__")
    with pytest.raises(TaskIdResolutionError, match="fully-qualified"):
        resolve_task_id("bare-name")


def test_in_process_provider_runs_body_with_args() -> None:
    outcome = InProcessExecutionProvider().execute(demo_task, HandlerStack(), None, {"marker": "unit"})
    assert outcome == {"status": "ok", "provider": "in-process", "result": {"marker": "unit"}}


def test_in_process_provider_requires_a_body() -> None:
    with pytest.raises(TaskIdResolutionError, match="resolved task body"):
        InProcessExecutionProvider().execute(None, HandlerStack(), None, {})


def test_handler_stack_lifo_dispatch_and_unhandled() -> None:
    stack = HandlerStack()
    stack.push({str: lambda e: f"outer:{e}"})
    stack.push({str: lambda e: f"inner:{e}"})
    assert stack.dispatch("x") == "inner:x"
    with pytest.raises(UnhandledAsk):
        stack.dispatch(42)
