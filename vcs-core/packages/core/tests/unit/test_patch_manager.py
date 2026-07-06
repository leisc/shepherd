# under-test: vcs_core._patch_manager
"""Unit tests for PatchManager install/uninstall mechanics.

Focus is the contract, not effect recording:
- install wraps the target, uninstall restores the original
- extra_patches() coexists with activate-time install_substrates() via
  ref-counted _shared_patches cleanup
- exceptions in the extra_patches body still uninstall
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest
from vcs_core import UnresolvedPatchPathError
from vcs_core._patch_manager import PatchManager
from vcs_core._patch_paths import PatchPathCandidate
from vcs_core._substrate_runtime import PerformedEventSpec, PythonPatch


def _install_synthetic_module(name: str, fn_name: str = "fn") -> tuple[types.ModuleType, object]:
    """Create and register a throwaway module with a single function, return (module, original_fn)."""
    module = types.ModuleType(name)

    def original(*_args, **_kwargs):
        return "ORIGINAL"

    module.__dict__[fn_name] = original
    sys.modules[name] = module
    return module, original


class _FakePipeline:
    """Minimal RecordingPipeline stand-in for PatchManager mechanics tests."""

    context = types.SimpleNamespace(world=None)
    execution_context = None


class _FakeSubstrate:
    """Minimal performed-event patch provider for install mechanics tests."""

    def __init__(self, patches, name: str = "test-sub") -> None:
        self._patches = tuple(patches)
        self.name = name

    def python_patches(self):
        return self._patches

    def performed_event_specs(self):
        return {"event": PerformedEventSpec()}

    def performed_effects(self, event, scope, *, params):
        del event, scope, params
        return ()


class _PatchProviderWithoutPerformedEvents:
    """Patch provider that deliberately lacks PerformedEventProvider."""

    def __init__(self, patches, name: str = "after-only") -> None:
        self._patches = tuple(patches)
        self.name = name

    def python_patches(self):
        return self._patches


@pytest.fixture
def synthetic_module():
    name = f"_patch_manager_test_{uuid.uuid4().hex[:8]}"
    module, original = _install_synthetic_module(name)
    try:
        yield name, module, original
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def manager(tmp_path: Path) -> PatchManager:
    return PatchManager(tmp_path, _FakePipeline())


def _translator(*_args, _result=None, **_kwargs):
    # Returning None is fine for these tests — no scope is set so the
    # after-translator code path would short-circuit before reaching it anyway.
    return None


# ---------------------------------------------------------------------------
# Basic install / uninstall cycle
# ---------------------------------------------------------------------------


def test_extra_patches_installs_and_uninstalls(synthetic_module, manager: PatchManager) -> None:
    mod_name, module, original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _FakeSubstrate([patch])

    assert module.fn is original

    with manager.extra_patches(sub, [patch]):
        assert module.fn is not original, "wrapper should be installed inside the with-block"
        # Wrapper still calls through to the original
        assert module.fn("x") == "ORIGINAL"

    assert module.fn is original, "original should be restored after the with-block exits"


def test_extra_patches_uninstalls_on_exception(synthetic_module, manager: PatchManager) -> None:
    mod_name, module, original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _FakeSubstrate([patch])

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom), manager.extra_patches(sub, [patch]):
        assert module.fn is not original
        raise _Boom("inner body blew up")

    assert module.fn is original, "uninstall must run even when the body raises"


def test_external_write_after_patch_authorizes_before_original(synthetic_module, manager: PatchManager) -> None:
    mod_name, module, _original = synthetic_module
    calls = 0

    def original(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "ORIGINAL"

    module.fn = original

    class _Blocked(RuntimeError):
        pass

    def block(operation: str) -> None:
        raise _Blocked(operation)

    patch = PythonPatch(
        target=f"{mod_name}.fn",
        after_translator=_translator,
        mutation_intent="external_write",
    )
    sub = _FakeSubstrate([patch])
    manager.set_external_write_authorizer(block)

    with pytest.raises(_Blocked, match=f"external write via {mod_name}.fn"), manager.extra_patches(sub, [patch]):
        module.fn("x")

    assert calls == 0


def test_external_write_wrap_patch_authorizes_before_original(synthetic_module, manager: PatchManager) -> None:
    mod_name, module, _original = synthetic_module
    calls = 0

    def original(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "ORIGINAL"

    def wrap_handler(*_args, **_kwargs):
        pytest.fail("wrap handler should not run when external write authority is denied")

    module.fn = original

    class _Blocked(RuntimeError):
        pass

    patch = PythonPatch(
        target=f"{mod_name}.fn",
        wrap_handler=wrap_handler,
        path_candidates=lambda *_args, **_kwargs: (manager._workspace / "target.txt",),
        mutation_intent="external_write",
    )
    sub = _FakeSubstrate([patch])
    manager.set_external_write_authorizer(lambda operation: (_ for _ in ()).throw(_Blocked(operation)))

    with pytest.raises(_Blocked, match=f"external write via {mod_name}.fn"), manager.extra_patches(sub, [patch]):
        module.fn("x")

    assert calls == 0


def test_install_substrates_rejects_after_translator_provider_without_performed_events(
    synthetic_module,
    manager: PatchManager,
) -> None:
    mod_name, module, original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _PatchProviderWithoutPerformedEvents([patch])

    with pytest.raises(TypeError, match=r"after_translator.*PerformedEventProvider"):
        manager.install_substrates([sub])

    assert module.fn is original


def test_extra_patches_rejects_after_translator_provider_without_performed_events(
    synthetic_module,
    manager: PatchManager,
) -> None:
    mod_name, module, original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _PatchProviderWithoutPerformedEvents([])

    with (
        pytest.raises(TypeError, match=r"after_translator.*PerformedEventProvider"),
        manager.extra_patches(sub, [patch]),  # type: ignore[arg-type]
    ):
        pass

    assert module.fn is original


def test_wrap_patch_provider_without_capture_can_install(synthetic_module, manager: PatchManager) -> None:
    mod_name, module, original = synthetic_module

    def wrap_handler(original_fn, patch_manager, substrate, *args, **kwargs):
        del patch_manager, substrate
        return f"WRAPPED:{original_fn(*args, **kwargs)}"

    patch = PythonPatch(
        target=f"{mod_name}.fn",
        wrap_handler=wrap_handler,
        path_candidates=lambda *_args, **_kwargs: (manager._workspace / "target.txt",),
    )
    sub = _PatchProviderWithoutPerformedEvents([patch], name="wrap-only")

    manager.install_substrates([sub])
    try:
        assert module.fn("x") == "WRAPPED:ORIGINAL"
    finally:
        manager.uninstall_all()

    assert module.fn is original


# ---------------------------------------------------------------------------
# Ref-counted coexistence with activate-time install_substrates()
# ---------------------------------------------------------------------------


def test_extra_patches_ref_counts_against_install_substrates(synthetic_module, manager: PatchManager) -> None:
    """install_substrates → extra_patches → exit extra → wrapper stays → uninstall_all → original."""
    mod_name, module, original = synthetic_module

    activate_patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    run_patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    activate_sub = _FakeSubstrate([activate_patch], name="activate-sub")
    run_sub = _FakeSubstrate([run_patch], name="run-sub")

    manager.install_substrates([activate_sub])
    assert module.fn is not original, "activate-time install wraps the target"
    outer_wrapper = module.fn

    with manager.extra_patches(run_sub, [run_patch]):
        # Wrapper identity does not change — same _SharedPatch entry, new
        # registration appended. The wrapper dispatches to all registrations.
        assert module.fn is outer_wrapper, (
            "extra_patches should not replace the existing shared wrapper when one already exists"
        )

    # Outer wrapper survives extra_patches exit because activate-time
    # registration is still present.
    assert module.fn is outer_wrapper, "wrapper must survive inner extra_patches cleanup"

    manager.uninstall_all()
    assert module.fn is original, "uninstall_all restores the original once all registrations are gone"


def test_extra_patches_alone_installs_and_removes_shared_entry(synthetic_module, manager: PatchManager) -> None:
    """When no activate-time registration exists, extra_patches owns the full lifecycle."""
    from vcs_core._patch_manager import PatchManager as _PM

    mod_name, module, original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _FakeSubstrate([patch])
    key = (module, "fn")

    assert key not in _PM._shared_patches

    with manager.extra_patches(sub, [patch]):
        assert key in _PM._shared_patches
        assert len(_PM._shared_patches[key].registrations) == 1

    assert key not in _PM._shared_patches, "shared entry must be removed when last registration goes"
    assert module.fn is original


# ---------------------------------------------------------------------------
# Wrapper dispatch still respects workspace matching from inside extra_patches
# ---------------------------------------------------------------------------


def test_extra_patches_wrapper_passes_through_untouched(synthetic_module, manager: PatchManager) -> None:
    """A patch with no path_candidates and no scope set should pass args through."""
    mod_name, module, _original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _FakeSubstrate([patch])

    with manager.extra_patches(sub, [patch]):
        # pipeline.context.world is None → after-translator path short-circuits before
        # calling the translator, but original() still runs and returns its value.
        assert module.fn("arg1", key="v") == "ORIGINAL"


def test_wrapper_tolerates_caller_kwargs_with_internal_names(synthetic_module, manager: PatchManager) -> None:
    """A caller whose own kwargs happen to be named `key`, `target`, `original`,
    or `bindings` must not collide with dispatch-chain positional params.

    Regression test: before positional-only `/` was added to the dispatch chain,
    e.g. `patched_fn(key="v")` raised ``TypeError: got multiple values for
    argument 'key'`` from inside ``_dispatch_shared_wrapper``.
    """
    mod_name, module, _original = synthetic_module
    patch = PythonPatch(target=f"{mod_name}.fn", after_translator=_translator)
    sub = _FakeSubstrate([patch])

    with manager.extra_patches(sub, [patch]):
        assert module.fn(key="v") == "ORIGINAL"
        assert module.fn(target="x") == "ORIGINAL"
        assert module.fn(original="o") == "ORIGINAL"
        assert module.fn(bindings=()) == "ORIGINAL"
        assert module.fn(key=1, target=2, original=3, bindings=4) == "ORIGINAL"


def test_mutating_patch_fails_closed_for_unresolved_fd_relative_path(
    synthetic_module,
    manager: PatchManager,
) -> None:
    mod_name, module, _original = synthetic_module
    patch = PythonPatch(
        target=f"{mod_name}.fn",
        after_translator=_translator,
        path_candidates=lambda *_args, **_kwargs: (PatchPathCandidate("victim.txt", dir_fd=-1),),
        requires_scope=True,
    )
    sub = _FakeSubstrate([patch])

    with manager.extra_patches(sub, [patch]), pytest.raises(UnresolvedPatchPathError, match="cannot resolve safely"):
        module.fn("victim.txt", dir_fd=-1)
