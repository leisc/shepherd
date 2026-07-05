"""Runtime effect registry composition and decode helpers.

Pre-Plan-04 effect-class plumbing: composes kernel-default + cache-
provided + discovered effect contributor classes into one
``EffectTypeRegistry``. Used by Plan 00 (`compose_effect_registry`),
the persistence layer, the export layer, and provider effect
collectors.

Lives under ``shepherd_runtime.effects.registry`` so the new typed-
effect surface (``Ask``, ``Tell``, ``handle``) at
``shepherd_runtime.effects`` can re-export it without colliding on the
``shepherd_runtime.effects`` import path. The old top-level module
``shepherd_runtime/effects.py`` was deleted in PR 18; consumers still
import ``compose_effect_registry`` and ``decode_effect`` from
``shepherd_runtime.effects``.

Tranche 7+ deletion target per DECISIONS D5: this module retires when
``shepherd_core.effects.effects`` retires (Plan 02 wave 2).
"""

from __future__ import annotations

from typing import Any

from shepherd_core.effects import (
    KERNEL_EFFECT_REGISTRY,
    Effect,
    EffectTypeRegistry,
    effect_from_dict,
)
from shepherd_core.effects.contributors import (
    EFFECTS_GROUP,
    EffectContributorConflictError,
    EffectContributorNameConflictError,
    EffectContributorValidationError,
    discover_effects,
    discover_effects_with_owners,
)

from shepherd_runtime.cache import get_effect_types as get_cache_effect_types


def discover_effect_types() -> dict[str, type[Effect]]:
    """Discover runtime effect contributors and normalize them to concrete effect types."""
    return discover_effects()


def compose_effect_registry() -> EffectTypeRegistry:
    """Compose the runtime effect registry from kernel defaults plus discovered contributors."""
    runtime_effect_types = dict(get_cache_effect_types())
    discovered, contributor_by_type = discover_effects_with_owners()

    for effect_type in sorted(runtime_effect_types):
        if effect_type in KERNEL_EFFECT_REGISTRY:
            raise EffectContributorConflictError(effect_type, "kernel", "runtime")

    for effect_type in sorted(discovered):
        if effect_type in KERNEL_EFFECT_REGISTRY:
            raise EffectContributorConflictError(effect_type, "kernel", contributor_by_type[effect_type])
        if effect_type in runtime_effect_types:
            raise EffectContributorConflictError(effect_type, "runtime", contributor_by_type[effect_type])

    return KERNEL_EFFECT_REGISTRY.extend(runtime_effect_types).extend(discovered)


def decode_effect(data: dict[str, Any], *, registry: EffectTypeRegistry | None = None) -> Effect:
    """Decode an effect using an explicit or composed runtime registry."""
    decode_registry = registry or compose_effect_registry()
    return effect_from_dict(data, registry=decode_registry)


__all__ = [
    "EFFECTS_GROUP",
    "EffectContributorConflictError",
    "EffectContributorNameConflictError",
    "EffectContributorValidationError",
    "compose_effect_registry",
    "decode_effect",
    "discover_effect_types",
]
