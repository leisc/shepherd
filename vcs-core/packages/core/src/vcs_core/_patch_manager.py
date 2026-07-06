"""Python interception infrastructure for experimental substrate patches."""

from __future__ import annotations

import functools
import importlib
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from vcs_core._errors import UnresolvedPatchPathError, UnscopedMutationError
from vcs_core._patch_paths import (
    PatchPathCandidateLike,
    resolve_patch_path,
    resolve_patch_path_status,
    workspace_relative,
)
from vcs_core._performed_event_admission import admit_performed_event
from vcs_core._substrate_runtime import (
    PatchMutationIntent,
    PerformedEventProvider,
    PythonPatch,
    PythonPatchProvider,
)
from vcs_core.types import EffectRecord

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from vcs_core._runtime_types import ExecutionContext
    from vcs_core.recording import RecordingPipeline
    from vcs_core.types import ScopeInfo


class NamedSubstrate(Protocol):
    """Internal named effect provider used for direct effect recording."""

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class _PatchBinding:
    substrate: PerformedEventProvider | PythonPatchProvider
    patch: PythonPatch


@dataclass(frozen=True)
class _InstalledPatch:
    owner: object
    attr_name: str


@dataclass(frozen=True)
class _TargetRegistration:
    manager: PatchManager
    bindings: tuple[_PatchBinding, ...]


@dataclass
class _SharedPatch:
    owner: object
    attr_name: str
    original: Callable[..., Any]
    registrations: list[_TargetRegistration]


PatchCallable = Callable[..., Any]
ExternalWriteAuthorizer = Callable[[str], None]


class PatchManager:
    """Install and remove Python-level interception for active substrates."""

    _tls = threading.local()
    _registry_lock = threading.Lock()
    _shared_patches: ClassVar[dict[tuple[object, str], _SharedPatch]] = {}

    def __init__(self, workspace: Path, pipeline: RecordingPipeline) -> None:
        self._workspace = workspace.resolve()
        self._pipeline = pipeline
        self._installed: list[_InstalledPatch] = []
        self._runtime_activity_opener: Any | None = None
        self._external_write_authorizer: ExternalWriteAuthorizer | None = None

    @property
    def scope(self) -> ScopeInfo | None:
        return self._pipeline.context.world

    @property
    def execution_context(self) -> ExecutionContext | None:
        return self._pipeline.execution_context

    def set_runtime_activity_opener(self, opener: Any) -> None:
        self._runtime_activity_opener = opener

    def set_external_write_authorizer(self, authorizer: ExternalWriteAuthorizer) -> None:
        self._external_write_authorizer = authorizer

    @contextmanager
    def guard(self) -> Iterator[None]:
        depth = getattr(self._tls, "depth", 0)
        self._tls.depth = depth + 1
        try:
            yield
        finally:
            self._tls.depth -= 1

    @contextmanager
    def activity(
        self,
        *,
        operation_label: str,
        operation_kind: str,
        operation_metadata: dict[str, object] | None = None,
        operation_id: str | None = None,
        boundary_policy: str = "forced_child",
        scope: ScopeInfo | None = None,
    ) -> Iterator[None]:
        activity_depth = getattr(self._tls, "activity_depth", 0)
        self._tls.activity_depth = activity_depth + 1
        effective_scope = scope or self.scope
        opener = self._runtime_activity_opener
        try:
            if opener is None or effective_scope is None:
                yield
                return
            with opener(
                scope=effective_scope,
                operation_label=operation_label,
                operation_kind=operation_kind,
                boundary_policy=boundary_policy,
                operation_id=operation_id,
                operation_metadata=operation_metadata,
            ):
                yield
        finally:
            self._tls.activity_depth -= 1

    def _install_grouped(
        self,
        grouped: dict[str, list[_PatchBinding]],
    ) -> list[tuple[tuple[object, str], _TargetRegistration]]:
        """Install grouped bindings into the shared-patch registry.

        Caller holds ``self._registry_lock``. Returns the (key, registration)
        pairs added, so callers can track them for scoped cleanup
        (``extra_patches``) or long-term tracking (``install_substrates``).
        """
        added: list[tuple[tuple[object, str], _TargetRegistration]] = []
        for target, bindings in grouped.items():
            self._validate_bindings(bindings)
            owner, attr_name, original = self._resolve_target(target)
            key = (owner, attr_name)
            shared = self._shared_patches.get(key)
            if shared is None:
                wrapper = self._make_shared_wrapper(key, target, original)
                setattr(owner, attr_name, wrapper)
                shared = _SharedPatch(
                    owner=owner,
                    attr_name=attr_name,
                    original=original,
                    registrations=[],
                )
                self._shared_patches[key] = shared
            registration = _TargetRegistration(self, tuple(bindings))
            shared.registrations.append(registration)
            added.append((key, registration))
        return added

    @contextmanager
    def extra_patches(
        self,
        substrate: PerformedEventProvider,
        patches: Sequence[PythonPatch],
    ) -> Iterator[None]:
        """Install additional patches for one substrate, then uninstall.

        Scoped install for callers like ``vcs-core run`` that need Python
        interception for the duration of a user script even when the
        substrate's default ``python_patches()`` would return empty (e.g.,
        because an overlay backend is present).
        """
        grouped: dict[str, list[_PatchBinding]] = {}
        for patch in patches:
            grouped.setdefault(patch.target, []).append(_PatchBinding(substrate=substrate, patch=patch))
        added: list[tuple[tuple[object, str], _TargetRegistration]] = []
        try:
            with self._registry_lock:
                added = self._install_grouped(grouped)
            yield
        finally:
            with self._registry_lock:
                for key, registration in added:
                    shared = self._shared_patches.get(key)
                    if shared is None:
                        continue
                    shared.registrations = [r for r in shared.registrations if r is not registration]
                    if not shared.registrations:
                        setattr(shared.owner, shared.attr_name, shared.original)
                        del self._shared_patches[key]

    def install_substrates(self, substrates: Sequence[object]) -> None:
        grouped: dict[str, list[_PatchBinding]] = {}
        for substrate in substrates:
            if not isinstance(substrate, PythonPatchProvider):
                continue
            for patch in substrate.python_patches():
                grouped.setdefault(patch.target, []).append(_PatchBinding(substrate=substrate, patch=patch))

        try:
            with self._registry_lock:
                added = self._install_grouped(grouped)
                for (owner, attr_name), _registration in added:
                    self._installed.append(_InstalledPatch(owner=owner, attr_name=attr_name))
        except Exception:
            self.uninstall_all()
            raise

    def uninstall_all(self) -> None:
        with self._registry_lock:
            for installed in reversed(self._installed):
                key = (installed.owner, installed.attr_name)
                shared = self._shared_patches.get(key)
                if shared is None:
                    continue
                shared.registrations = [
                    registration for registration in shared.registrations if registration.manager is not self
                ]
                if shared.registrations:
                    continue
                setattr(shared.owner, shared.attr_name, shared.original)
                del self._shared_patches[key]
            self._installed.clear()

    def record_effects(
        self,
        substrate: NamedSubstrate,
        effects: Sequence[EffectRecord],
        *,
        scope: ScopeInfo | None = None,
        boundary_policy: str | None = None,
    ) -> list[str]:
        effective_scope = scope or self.scope
        if effective_scope is None or not effects:
            return []
        effective_boundary_policy = self._resolve_boundary_policy(boundary_policy)
        with self.guard():
            return self._pipeline.record_runtime_effects(
                list(effects),
                substrate=substrate.name,
                scope=effective_scope,
                boundary_policy=effective_boundary_policy,
                operation_kind=f"{substrate.name}.runtime",
                operation_label=f"{substrate.name}-runtime",
            )

    def require_scope_for_mutation(self, operation: str, *candidates: str | Path | object) -> None:
        if self.scope is not None:
            return
        path = next((self.workspace_relative(candidate) for candidate in candidates), None)
        raise UnscopedMutationError(operation, path=path)

    def require_external_write_allowed(self, operation: str) -> None:
        authorizer = self._external_write_authorizer
        if authorizer is None:
            return
        with self.guard():
            authorizer(operation)

    def record_performed_event(
        self,
        substrate: PerformedEventProvider,
        event: str,
        params: dict[str, Any],
        *,
        scope: ScopeInfo | None = None,
        boundary_policy: str | None = None,
    ) -> list[str]:
        effective_scope = scope or self.scope
        if effective_scope is None:
            return []
        effective_boundary_policy = self._resolve_boundary_policy(boundary_policy)
        with self.guard():
            normalized = admit_performed_event(substrate, event, effective_scope, params=params)
            effects = _require_performed_effects(
                substrate.name,
                event,
                substrate.performed_effects(event, effective_scope, params=normalized.params),
                allowed_effect_types=normalized.effect_types,
            )
            if not effects:
                return []
            return self._pipeline.record_runtime_effects(
                list(effects),
                substrate=substrate.name,
                scope=effective_scope,
                boundary_policy=effective_boundary_policy,
                operation_kind=f"{substrate.name}.{event}",
                operation_label=f"{substrate.name}-{event}",
            )

    def path_in_workspace(self, candidate: PatchPathCandidateLike) -> bool:
        return self.workspace_relative(candidate) is not None

    def workspace_relative(self, candidate: PatchPathCandidateLike) -> str | None:
        return workspace_relative(candidate, self._workspace)

    def resolve_path(self, candidate: PatchPathCandidateLike) -> Path | None:
        return resolve_patch_path(candidate)

    def _validate_bindings(self, bindings: Sequence[_PatchBinding]) -> None:
        wrap_bindings = [binding for binding in bindings if binding.patch.wrap_handler is not None]
        if len(wrap_bindings) > 1:
            msg = "Only one wrap_handler may claim a given Python patch target."
            raise ValueError(msg)
        for binding in bindings:
            if binding.patch.after_translator is None:
                continue
            if isinstance(binding.substrate, PerformedEventProvider):
                continue
            substrate_name = getattr(binding.substrate, "name", type(binding.substrate).__name__)
            msg = (
                f"Python patch target {binding.patch.target!r} uses after_translator for substrate "
                f"{substrate_name!r}, but the substrate does not implement PerformedEventProvider."
            )
            raise TypeError(msg)

    @classmethod
    def _make_shared_wrapper(
        cls,
        key: tuple[object, str],
        target: str,
        original: PatchCallable,
    ) -> Any:
        # Dispatch-chain params are positional-only so caller kwargs named
        # `key`, `target`, `original`, `bindings`, etc. cannot collide with
        # the wrapper's own positionals.
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return cls._dispatch_shared_wrapper(key, target, original, *args, **kwargs)

        return wrapped

    @classmethod
    def _dispatch_shared_wrapper(
        cls,
        key: tuple[object, str],
        target: str,
        original: PatchCallable,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with cls._registry_lock:
            shared = cls._shared_patches.get(key)
            registrations = list(shared.registrations) if shared is not None else []

        registration = cls._select_registration(registrations, *args, **kwargs)
        if registration is None:
            return original(*args, **kwargs)
        return registration.manager._invoke_registration(target, original, registration.bindings, *args, **kwargs)

    @classmethod
    def _select_registration(
        cls,
        registrations: Sequence[_TargetRegistration],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _TargetRegistration | None:
        applicable = [
            registration
            for registration in registrations
            if any(
                registration.manager._patch_applies(binding.patch, *args, **kwargs) for binding in registration.bindings
            )
        ]
        if not applicable:
            return None
        applicable.sort(key=lambda registration: len(registration.manager._workspace.parts), reverse=True)
        return applicable[0]

    def _invoke_registration(
        self,
        target: str,
        original: PatchCallable,
        bindings: Sequence[_PatchBinding],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        wrap_binding = next((binding for binding in bindings if binding.patch.wrap_handler is not None), None)
        if wrap_binding is not None:
            return self._invoke_wrap_binding(target, original, wrap_binding, *args, **kwargs)
        return self._invoke_after_bindings(original, bindings, *args, **kwargs)

    def _invoke_wrap_binding(
        self,
        target: str,
        original: PatchCallable,
        binding: _PatchBinding,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        wrap_handler = binding.patch.wrap_handler
        if wrap_handler is None:
            raise RuntimeError("wrap_handler missing for wrap binding")
        if self._guard_active():
            return original(*args, **kwargs)
        candidates = self._candidate_paths(binding.patch, *args, **kwargs)
        if binding.patch.requires_scope and self._has_unknown_path(candidates):
            raise UnresolvedPatchPathError(target)
        if not self._matches_workspace_candidates(candidates):
            return original(*args, **kwargs)
        if binding.patch.requires_scope and self.scope is None:
            self.require_scope_for_mutation(target, *candidates)
        if self._patch_mutation_intent(binding.patch, *args, **kwargs) == "external_write":
            self.require_external_write_allowed(f"external write via {target}")
        with self.guard():
            return wrap_handler(original, self, binding.substrate, *args, **kwargs)

    def _invoke_after_bindings(
        self,
        original: PatchCallable,
        bindings: Sequence[_PatchBinding],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._guard_active():
            return original(*args, **kwargs)

        applicable = [binding for binding in bindings if self._patch_applies(binding.patch, *args, **kwargs)]
        if not applicable:
            return original(*args, **kwargs)
        unknown = next(
            (binding for binding in applicable if self._patch_has_unknown_mutation(binding.patch, *args, **kwargs)),
            None,
        )
        if unknown is not None:
            raise UnresolvedPatchPathError(unknown.patch.target)
        if self.scope is None:
            required = next((binding for binding in applicable if binding.patch.requires_scope), None)
            if required is not None:
                self.require_scope_for_mutation(
                    required.patch.target,
                    *self._candidate_paths(required.patch, *args, **kwargs),
                )
            self._require_external_write_allowed_for_applicable(applicable, *args, **kwargs)
            return original(*args, **kwargs)

        self._require_external_write_allowed_for_applicable(applicable, *args, **kwargs)
        result = original(*args, **kwargs)
        for binding in applicable:
            translator = binding.patch.after_translator
            if translator is None:
                continue
            translated = translator(*args, _result=result, **kwargs)
            if translated is None:
                continue
            event, params = translated
            # Reached only when after_translator is set, and install_substrates rejects any such
            # binding whose substrate is not also a PerformedEventProvider — so this cast is sound.
            self.record_performed_event(
                cast("PerformedEventProvider", binding.substrate), event, params, boundary_policy="forced_child"
            )
            break
        return result

    def _candidate_paths(self, patch: PythonPatch, *args: Any, **kwargs: Any) -> Sequence[PatchPathCandidateLike]:
        if patch.path_candidates is None:
            return ()
        try:
            return patch.path_candidates(*args, **kwargs)
        except (AttributeError, OSError, TypeError, ValueError):
            return ()

    def _matches_workspace(self, patch: PythonPatch, *args: Any, **kwargs: Any) -> bool:
        if patch.path_candidates is None:
            return True
        candidates = self._candidate_paths(patch, *args, **kwargs)
        return self._matches_workspace_candidates(candidates)

    def _patch_applies(self, patch: PythonPatch, *args: Any, **kwargs: Any) -> bool:
        if patch.path_candidates is None:
            return True
        candidates = self._candidate_paths(patch, *args, **kwargs)
        return self._matches_workspace_candidates(candidates) or (
            patch.requires_scope and self._has_unknown_path(candidates)
        )

    def _patch_has_unknown_mutation(self, patch: PythonPatch, *args: Any, **kwargs: Any) -> bool:
        return patch.requires_scope and self._has_unknown_path(self._candidate_paths(patch, *args, **kwargs))

    def _patch_mutation_intent(self, patch: PythonPatch, *args: Any, **kwargs: Any) -> PatchMutationIntent:
        intent = patch.mutation_intent
        if callable(intent):
            return intent(*args, **kwargs)
        return intent

    def _require_external_write_allowed_for_applicable(
        self,
        bindings: Sequence[_PatchBinding],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        external_write = next(
            (
                binding
                for binding in bindings
                if self._patch_mutation_intent(binding.patch, *args, **kwargs) == "external_write"
            ),
            None,
        )
        if external_write is not None:
            self.require_external_write_allowed(f"external write via {external_write.patch.target}")

    def _matches_workspace_candidates(self, candidates: Sequence[PatchPathCandidateLike]) -> bool:
        return any(self.path_in_workspace(candidate) for candidate in candidates)

    def _has_unknown_path(self, candidates: Sequence[PatchPathCandidateLike]) -> bool:
        return any(resolve_patch_path_status(candidate).unknown for candidate in candidates)

    def _guard_active(self) -> bool:
        return getattr(self._tls, "depth", 0) > 0

    def _resolve_boundary_policy(self, boundary_policy: str | None) -> str:
        if boundary_policy is not None:
            return boundary_policy
        if getattr(self._tls, "activity_depth", 0):
            return "append_or_root"
        if self._pipeline.current_operation() is not None:
            return "append_or_root"
        return "forced_child"

    def _resolve_target(self, target: str) -> tuple[object, str, PatchCallable]:
        parts = target.split(".")
        module: object | None = None
        remainder: list[str] = []
        for index in range(len(parts), 0, -1):
            module_name = ".".join(parts[:index])
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            remainder = parts[index:]
            break

        if module is None or not remainder:
            msg = f"Unable to resolve Python patch target: {target!r}"
            raise ValueError(msg)

        owner = module
        for attr in remainder[:-1]:
            owner = getattr(owner, attr)
        attr_name = remainder[-1]
        original = getattr(owner, attr_name)
        if not callable(original):
            msg = f"Python patch target {target!r} is not callable."
            raise TypeError(msg)
        return owner, attr_name, original


def _require_performed_effects(
    substrate_name: str,
    event: str,
    result: object,
    *,
    allowed_effect_types: frozenset[str] = frozenset(),
) -> tuple[EffectRecord, ...]:
    if isinstance(result, (str, bytes, bytearray)):
        actual_type = type(result).__name__
        raise TypeError(
            f"Substrate '{substrate_name}' performed event '{event}' must return EffectRecord sequence, "
            f"got {actual_type}."
        )
    try:
        effects: tuple[EffectRecord, ...] = tuple(result)  # type: ignore[arg-type]
    except TypeError as exc:
        actual_type = type(result).__name__
        raise TypeError(
            f"Substrate '{substrate_name}' performed event '{event}' must return EffectRecord sequence, "
            f"got {actual_type}."
        ) from exc
    invalid = next((effect for effect in effects if not isinstance(effect, EffectRecord)), None)
    if invalid is not None:
        actual_type = type(invalid).__name__
        raise TypeError(
            f"Substrate '{substrate_name}' performed event '{event}' returned non-EffectRecord item: {actual_type}."
        )
    if allowed_effect_types:
        invalid_effect = next((effect for effect in effects if effect.effect_type not in allowed_effect_types), None)
        if invalid_effect is not None:
            allowed = ", ".join(sorted(allowed_effect_types))
            raise TypeError(
                f"Substrate '{substrate_name}' performed event '{event}' returned undeclared effect type "
                f"{invalid_effect.effect_type!r}; allowed: {allowed}."
            )
    return effects
