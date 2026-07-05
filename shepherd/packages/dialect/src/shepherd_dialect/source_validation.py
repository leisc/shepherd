"""Source validation — the advisory pre-filter (authoring re-pin W3a).

The D2 guarantee (block ``os``/``subprocess``/``eval``/dunders in task source),
re-pinned **dependency-free** — the legacy ``shepherd_core``'s ``ast`` walk
ported verbatim, with **no RestrictedPython**.

Its role, stated honestly against the containment layers
([`confinement-compiler.md`](../../../../docs/engineering/convergence/confinement-compiler.md)
`source-validation-is-advisory-the-jail-enforces`): this is the **framework
tier** — a fast, in-process, best-effort filter that fails closed with a good
error *before* a run spins up a jail. It is **not** the enforcement boundary.
The complete, tamperproof monitor for "reconstructed source attempts a
dangerous effect" is the jail at syscall altitude (the confined body): an
``import os; os.system(...)`` that slips past this filter is denied at the
syscall regardless. The legacy RestrictedPython ``secure_reconstruct_task_class``
— an in-process sandbox that *executes* reconstructed code under guards — is
the advisory-dressed-as-enforcement shape the architecture supersedes; it does
not port (see the doc sidebar).
"""

from __future__ import annotations

import ast

__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_IMPORTS",
    "FORBIDDEN_NAMES",
    "SourceValidationError",
    "check_task_source",
    "validate_task_source",
]


class SourceValidationError(Exception):
    """Raised by ``check_task_source`` when source fails the advisory filter."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Source validation failed: {violations}")


FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "ctypes",
        "importlib",
        "builtins",
        "socket",
        "multiprocessing",
        "threading",
        "signal",
        "shutil",
        "tempfile",
        "pathlib",
        "io",
    }
)

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "memoryview",
    }
)

FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
    }
)


def validate_task_source(source: str, *, strict: bool = True) -> list[str]:
    """Return advisory violations for dangerous patterns in task source (empty = clean)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    forbidden = FORBIDDEN_IMPORTS if strict else FORBIDDEN_IMPORTS - {"pathlib"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    violations.append(f"Line {node.lineno}: Forbidden import '{alias.name}'")
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in forbidden:
            violations.append(f"Line {node.lineno}: Forbidden import from '{node.module}'")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            violations.append(f"Line {node.lineno}: Forbidden call '{node.func.id}()'")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            violations.append(f"Line {node.lineno}: Forbidden attribute '{node.attr}'")
    return violations


def check_task_source(source: str, *, strict: bool = True) -> None:
    """Advisory gate: raise ``SourceValidationError`` if the source is non-clean.

    Fail-fast convenience over ``validate_task_source`` — the framework-tier
    refusal. Enforcement remains the jail (the docstring's note).
    """
    violations = validate_task_source(source, strict=strict)
    if violations:
        raise SourceValidationError(violations)
