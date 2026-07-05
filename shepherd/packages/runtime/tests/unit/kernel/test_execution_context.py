"""Contract-import test for E0 ``ExecutionContext`` re-export.

Satisfies CONTRACTS Maintenance Rule 3 ("the empirical definition of
`stub-ready` is that the consumer plan's spike directory contains a
test that imports the contract from the stub and exercises the consumer
behavior"). After this test passes, E0 is empirically stub-ready.
"""

from __future__ import annotations

import dataclasses
from dataclasses import is_dataclass

import pytest


def test_execution_context_imports_from_runtime_kernel() -> None:
    """E0: import path is ``shepherd_runtime.kernel.ExecutionContext``."""
    from shepherd_runtime.kernel import ExecutionContext

    assert ExecutionContext.__name__ == "ExecutionContext"


def test_execution_context_is_frozen_dataclass_with_three_refs() -> None:
    """CONTRACTS E0: three Ref fields, frozen dataclass."""
    from shepherd_runtime.kernel import ExecutionContext

    assert is_dataclass(ExecutionContext)
    ctx = ExecutionContext()
    field_names = {f.name for f in ctx.__dataclass_fields__.values()}
    assert field_names == {"binding_env_ref", "region_ref", "authority_ref"}


def test_execution_context_defaults_match_root_context() -> None:
    """CONTRACTS E0 stub: defaults are ``env:root`` / ``region:root`` / ``authority:root``."""
    from shepherd_runtime.kernel import ExecutionContext

    ctx = ExecutionContext()
    assert ctx.binding_env_ref == "env:root"
    assert ctx.region_ref == "region:root"
    assert ctx.authority_ref == "authority:root"


def test_execution_context_is_immutable() -> None:
    """Frozen-dataclass discipline (DECISIONS D17)."""
    from shepherd_runtime.kernel import ExecutionContext

    ctx = ExecutionContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.binding_env_ref = "env:other"  # type: ignore[misc]


def test_execution_context_equal_triples_are_equal() -> None:
    """Structural equality on the triple, per CONTRACTS E0."""
    from shepherd_runtime.kernel import ExecutionContext

    a = ExecutionContext(binding_env_ref="env:a", region_ref="region:a", authority_ref="auth:a")
    b = ExecutionContext(binding_env_ref="env:a", region_ref="region:a", authority_ref="auth:a")
    assert a == b
    assert hash(a) == hash(b)
