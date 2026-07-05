"""Phase 1 behavioral-spine acceptance gate.

The gate is intentionally stricter than the Appendix C nucleus smoke test:
it does not use MockProvider, class-form task machinery, name-keyed binding,
or live provider credentials.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from shepherd_core.schema import SINGLE_OUTPUT_KEY
from shepherd_kernel_v3_reference.trace.validate import validate_runtime_trace
from shepherd_runtime.nucleus.workspace import reset_workspace_for_tests
from shepherd_runtime.provider_boundary import ModelRequest, ModelResponse

from shepherd import ask, deliver, handle, task, tell, workspace
from shepherd.effects import Ask, Tell


def test_phase1_behavioral_spine_without_mock_provider(tmp_path) -> None:
    """Run the offline callable syntax spine with runtime trace evidence."""
    reset_workspace_for_tests()

    @dataclass(frozen=True)
    class PickOne(Ask[str], kind="phase1_async.pick_one"):
        options: tuple[str, ...]

    @dataclass(frozen=True)
    class Audit(Tell, kind="phase1_async.audit"):
        message: str

    @dataclass(frozen=True)
    class GateModel:
        name: str = "phase1-gate"

    fake_model_calls: list[ModelRequest] = []

    async def fake_model(request: ModelRequest) -> ModelResponse:
        fake_model_calls.append(request)
        return ModelResponse(
            structured_output={SINGLE_OUTPUT_KEY: "alpha"},
            session_id=None,
            finish_reason="end_turn",
        )

    @task
    async def choose(topic: str) -> str:
        value = await ask(PickOne(options=("alpha", "beta")))
        await tell(Audit(message=value))
        return await deliver(str, goal=f"return the selected value for {topic}", evidence=[value])

    async def run_gate() -> object:
        with (
            workspace(model=GateModel(), root=tmp_path),
            handle("model.call", fake_model),
            handle(PickOne, lambda effect: effect.options[0]),
        ):
            return await choose.detailed("topic")

    try:
        run = asyncio.run(run_gate())
        assert run.unwrap() == "alpha"
        assert fake_model_calls
        assert fake_model_calls[0].settings.model == "phase1-gate"
        assert fake_model_calls[0].messages[0].role == "user"
        assert run.trace is not None
        assert run.trace.kernel
        assert run.trace.surface
        validate_runtime_trace(run.trace.kernel)

        surface = run.trace.surface
        assert all(record.claim_level == "phase1-runtime" for record in surface)
        assert all(record.proof_profile == "runtime_only" for record in surface)
        assert any(
            record.kind == "effect_requested" and record.effect_key == "phase1_async.pick_one" for record in surface
        )
        assert any(
            record.kind == "handler_selected" and record.effect_key == "phase1_async.pick_one" for record in surface
        )
        assert any(
            record.kind == "handler_returned" and record.effect_key == "phase1_async.pick_one" for record in surface
        )
        assert any(
            record.kind == "effect_requested" and record.effect_key == "phase1_async.audit" for record in surface
        )
        assert any(
            record.kind == "handler_selected"
            and record.effect_key == "phase1_async.audit"
            and record.handler_key == "runtime.default_ignore.v1"
            and record.status == "default_ignored"
            for record in surface
        )
        assert any(record.kind == "provider_call_requested" and record.effect_key == "model.call" for record in surface)
        assert any(record.kind == "handler_selected" and record.effect_key == "model.call" for record in surface)
        assert any(record.kind == "provider_call_completed" and record.effect_key == "model.call" for record in surface)
        assert any(record.kind == "delivery_completed" and record.status == "completed" for record in surface)
        assert type(run.trace).from_json(run.trace.to_json()) == run.trace
    finally:
        reset_workspace_for_tests()


def test_phase1_sync_task_spine_uses_top_level_ask_tell(tmp_path) -> None:
    """Run the sync callable syntax spine through the top-level effect facade."""
    reset_workspace_for_tests()

    @dataclass(frozen=True)
    class PickOne(Ask[str], kind="phase1_sync.pick_one"):
        options: tuple[str, ...]

    @dataclass(frozen=True)
    class Audit(Tell, kind="phase1_sync.audit"):
        message: str

    @dataclass(frozen=True)
    class GateModel:
        name: str = "phase1-sync-gate"

    fake_model_calls: list[ModelRequest] = []

    async def fake_model(request: ModelRequest) -> ModelResponse:
        fake_model_calls.append(request)
        return ModelResponse(
            structured_output={SINGLE_OUTPUT_KEY: "alpha"},
            session_id=None,
            finish_reason="end_turn",
        )

    @task
    def choose(topic: str) -> str:
        value = ask(PickOne(options=("alpha", "beta")))
        tell(Audit(message=value))
        return deliver(str, goal=f"return the selected value for {topic}", evidence=[value])

    try:
        with (
            workspace(model=GateModel(), root=tmp_path),
            handle("model.call", fake_model),
            handle(PickOne, lambda effect: effect.options[0]),
        ):
            run = choose.detailed("topic")

        assert run.unwrap() == "alpha"
        assert fake_model_calls
        assert fake_model_calls[0].settings.model == "phase1-sync-gate"
        assert run.trace is not None
        assert run.trace.kernel
        assert run.trace.surface
        validate_runtime_trace(run.trace.kernel)

        surface = run.trace.surface
        assert any(
            record.kind == "effect_requested" and record.effect_key == "phase1_sync.pick_one" for record in surface
        )
        assert any(
            record.kind == "handler_returned" and record.effect_key == "phase1_sync.pick_one" for record in surface
        )
        assert any(
            record.kind == "handler_selected"
            and record.effect_key == "phase1_sync.audit"
            and record.handler_key == "runtime.default_ignore.v1"
            and record.status == "default_ignored"
            for record in surface
        )
        assert any(record.kind == "delivery_completed" and record.status == "completed" for record in surface)
        assert type(run.trace).from_json(run.trace.to_json()) == run.trace
    finally:
        reset_workspace_for_tests()
