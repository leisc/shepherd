"""VcsCoreError root guard (260704-1410-plan.md V1.3).

The frozen inventory (`vcs_core_exception_inventory.json`) enumerates every vcs_core
exception class. These tests assert: (a) the inventory is still exhaustive — a new
exception class that forgets the root is a red diff, not a silent gap; (b) every
inventoried class is a `VcsCoreError` so consumers can catch at the boundary; and
(c) classes that also subclass a stdlib exception keep that base (standing rule 4),
so existing `except ValueError` / `except RuntimeError` call sites are unaffected.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
from pathlib import Path

import pytest
import vcs_core
from vcs_core import VcsCoreError

SRC = Path(vcs_core.__file__).resolve().parent
INVENTORY_PATH = Path(__file__).with_name("vcs_core_exception_inventory.json")

_STD_EXC = {
    "Exception",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "OSError",
    "NotImplementedError",
    "LookupError",
}


def _discover_exceptions() -> dict[str, dict]:
    """Re-derive the exception inventory from the current source tree."""
    found: dict[str, dict] = {}
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases]
            is_exc = (
                any(b in _STD_EXC for b in bases)
                or node.name.endswith(("Error", "Denied", "Refused"))
                or any(str(b).endswith(("Error", "Denied", "Refused")) for b in bases)
            )
            if is_exc:
                mod = str(path.relative_to(SRC)).replace("/", ".")[:-3]
                found[node.name] = {"module": f"vcs_core.{mod}", "bases": bases}
    return found


def _load_inventory() -> dict[str, dict]:
    return json.loads(INVENTORY_PATH.read_text())


def test_frozen_inventory_is_exhaustive() -> None:
    """A newly added exception class must be added to the frozen inventory.

    Regenerating the inventory from source must produce no diff. If this fails,
    a new exception was added (or one removed/renamed); update the fixture with the
    same reasoning that reparents it under VcsCoreError.
    """
    current = _discover_exceptions()
    frozen = _load_inventory()
    missing = sorted(set(current) - set(frozen))
    extra = sorted(set(frozen) - set(current))
    assert not missing, f"exception classes missing from the frozen inventory: {missing}"
    assert not extra, f"frozen inventory names classes no longer in source: {extra}"


@pytest.mark.parametrize("name", sorted(_load_inventory()))
def test_every_exception_is_a_vcs_core_error(name: str) -> None:
    entry = _load_inventory()[name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, name)
    assert issubclass(cls, VcsCoreError), f"{name} is not a VcsCoreError subclass"


@pytest.mark.parametrize("name", sorted(_load_inventory()))
def test_stdlib_bases_preserved(name: str) -> None:
    """A class whose frozen bases include a stdlib exception still subclasses it."""
    entry = _load_inventory()[name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, name)
    for base in entry["bases"]:
        if base in {"ValueError", "RuntimeError", "OSError", "TypeError", "KeyError", "LookupError"}:
            assert issubclass(cls, getattr(builtins, base)), f"{name} lost its stdlib base {base}"


def test_root_is_public() -> None:
    assert vcs_core.VcsCoreError is VcsCoreError
    assert "VcsCoreError" in vcs_core.__all__
