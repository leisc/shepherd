"""Tests for the public shepherd facade routing."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import warnings
from dataclasses import dataclass

from shepherd_runtime.context import BindableContext
from shepherd_runtime.effects import (
    Ask,
    EffectNotPermitted,
    EffectSurfaceEmpty,
    EffectSurfaceTooWide,
    Match,
    OverbroadHandler,
    Plan,
    PlanNotExtractable,
    Resumption,
    Subset,
    Tell,
)
from shepherd_runtime.effects import ask as runtime_ask
from shepherd_runtime.effects import handle as runtime_handle
from shepherd_runtime.effects import sync_ask as runtime_sync_ask
from shepherd_runtime.effects import sync_tell as runtime_sync_tell
from shepherd_runtime.effects import tell as runtime_tell
from shepherd_runtime.lifecycle import execute
from shepherd_runtime.nucleus import (
    Artifact,
    DeliveryException,
    DeliveryFailed,
    Run,
    RunInProgress,
    RunRef,
    Workspace,
    deliver,
    emit_artifact,
    task,
    workspace,
)
from shepherd_runtime.scope import Scope, current_binding
from shepherd_runtime.task.markers import InputMarker


@dataclass(frozen=True)
class _Pick(Ask[str]):
    options: tuple[str, ...]


@dataclass(frozen=True)
class _Audit(Tell):
    message: str


def test_shepherd_import_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.reload(importlib.import_module("shepherd"))

    assert module.Workspace is Workspace
    assert caught == []


def test_shepherd_facade_wiring() -> None:
    module = importlib.reload(importlib.import_module("shepherd"))

    # The exact manifest is pinned in test_sp_facade.py. Here we assert the
    # spine and the consolidated handle surface are both present (WS-A)...
    for name in ("workspace", "Workspace", "task", "Match", "Plan", "Subset", "Run", "RunRef", "handle"):
        assert name in module.__all__
    for name in ("GitRepo", "May", "RunOutput", "ShepherdWorkspace", "open", "Flow"):
        assert name in module.__all__
        assert getattr(module, name) is not None
    # ...and (below) that each re-export is the real owner-package object.
    assert module.workspace is workspace
    assert module.Workspace is Workspace
    assert module.task is task
    assert module.deliver is deliver
    assert module.Match is Match
    assert module.Plan is Plan
    assert module.Subset is Subset
    assert module.Run is Run
    assert module.RunRef is RunRef
    assert module.RunInProgress is RunInProgress
    assert module.DeliveryException is DeliveryException
    assert module.DeliveryFailed is DeliveryFailed
    assert module.emit_artifact is emit_artifact
    assert module.Artifact is Artifact
    assert module.handle is runtime_handle
    assert callable(module.ask)
    assert callable(module.tell)
    assert module.ask is not runtime_ask
    assert module.tell is not runtime_tell
    assert module.EffectNotPermitted is EffectNotPermitted
    assert module.EffectSurfaceEmpty is EffectSurfaceEmpty
    assert module.EffectSurfaceTooWide is EffectSurfaceTooWide
    assert module.OverbroadHandler is OverbroadHandler
    assert module.PlanNotExtractable is PlanNotExtractable
    assert module.current_binding is current_binding


def test_shepherd_top_level_ask_tell_are_sync_facade() -> None:
    module = importlib.reload(importlib.import_module("shepherd"))
    seen: list[str] = []

    with (
        module.handle(_Pick, lambda effect: effect.options[0]),
        module.handle(_Audit, lambda effect: seen.append(effect.message)),
    ):
        assert module.ask(_Pick(options=("sync", "fallback"))) == "sync"
        assert module.tell(_Audit(message="recorded")) is None

    assert seen == ["recorded"]


def test_shepherd_top_level_sync_facade_preserves_context_inside_event_loop() -> None:
    module = importlib.reload(importlib.import_module("shepherd"))

    async def run() -> str:
        with module.handle(_Pick, lambda effect: effect.options[0]):
            await asyncio.sleep(0)
            return module.ask(_Pick(options=("thread-hop", "fallback")))

    assert asyncio.run(run()) == "thread-hop"


def test_shepherd_top_level_facade_works_inside_sync_task(tmp_path) -> None:
    module = importlib.reload(importlib.import_module("shepherd"))
    seen: list[str] = []

    @module.task
    def choose(topic: str) -> str:
        value = module.ask(_Pick(options=(topic, "fallback")))
        module.tell(_Audit(message=value))
        return value

    with (
        module.workspace(model=object(), root=tmp_path),
        module.handle(_Pick, lambda effect: effect.options[0]),
        module.handle(_Audit, lambda effect: seen.append(effect.message)),
    ):
        run = choose.detailed("sync-task")

    assert run.unwrap() == "sync-task"
    assert seen == ["sync-task"]


def test_shepherd_top_level_facade_works_inside_async_task(tmp_path) -> None:
    module = importlib.reload(importlib.import_module("shepherd"))
    seen: list[str] = []

    @module.task
    async def choose(topic: str) -> str:
        value = await module.ask(_Pick(options=(topic, "fallback")))
        await module.tell(_Audit(message=value))
        return value

    async def run_task() -> Run[str]:
        with (
            module.workspace(model=object(), root=tmp_path),
            module.handle(_Pick, lambda effect: effect.options[0]),
            module.handle(_Audit, lambda effect: seen.append(effect.message)),
        ):
            return await choose.detailed("async-task")

    run = asyncio.run(run_task())

    assert run.unwrap() == "async-task"
    assert seen == ["async-task"]


def test_owner_path_ask_tell_remain_async() -> None:
    assert inspect.iscoroutinefunction(runtime_ask)
    assert inspect.iscoroutinefunction(runtime_tell)
    assert not inspect.iscoroutinefunction(runtime_sync_ask)
    assert not inspect.iscoroutinefunction(runtime_sync_tell)


def test_advanced_runtime_apis_stay_on_owner_paths() -> None:
    module = importlib.reload(importlib.import_module("shepherd"))

    assert not hasattr(module, "Scope")
    assert not hasattr(module, "BindableContext")
    assert not hasattr(module, "execute")
    assert not hasattr(module, "mock_steps")
    assert not hasattr(module, "TaskFailed")

    assert Scope.__module__.startswith("shepherd_runtime.")
    assert BindableContext.__module__.startswith("shepherd_runtime.")
    assert execute.__module__.startswith("shepherd_runtime.")


def test_effects_submodule_is_explicit_narrow_public_namespace() -> None:
    effects = importlib.import_module("shepherd.effects")

    assert effects.__all__ == [
        "Ask",
        "Tell",
        "Resumption",
        "ResumptionAborted",
        "ResumptionConsumed",
    ]
    assert effects.Ask is Ask
    assert effects.Tell is Tell
    assert effects.Resumption is Resumption


def test_markers_submodule_is_explicit_narrow_public_namespace() -> None:
    markers = importlib.import_module("shepherd.markers")

    assert markers.__all__ == ["InputMarker"]
    assert markers.InputMarker is InputMarker
    assert not hasattr(markers, "Input")
    assert not hasattr(markers, "Output")
