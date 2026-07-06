"""Guard the vcs_core.testing seam boundary (260704-1410-plan.md V1.1).

The test-support seam is one-directional: it may reach into vcs_core internals,
but (a) it must not import shepherd code, and (b) no production vcs_core module
may import from it. A violation of (b) would drag test-only machinery into the
product import graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

import vcs_core.testing as seam

SRC = Path(vcs_core_src := __import__("vcs_core").__file__).parent


def test_seam_exports_are_importable():
    for name in seam.__all__:
        assert hasattr(seam, name), name


def test_seam_imports_no_shepherd():
    tree = ast.parse((SRC / "testing" / "__init__.py").read_text())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        for m in mods:
            assert not m.startswith(("shepherd", "shepherd2")), m


def test_no_production_module_imports_from_testing():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "testing" in path.relative_to(SRC).parts:
            continue  # the seam itself and any nested testing kits are exempt
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module)
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names)
            if any(t == "vcs_core.testing" or t.startswith("vcs_core.testing.") for t in targets):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], f"production modules import vcs_core.testing: {offenders}"
