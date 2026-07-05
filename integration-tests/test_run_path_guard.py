"""The run-path executor guard — layer (b) of "real ⇒ jailed" (PD7).

The dialect run-composition layer invokes an executor ONLY via
``launch_confined``; the only raw spawn lives in the containment backends
(which ARE ``launch_confined``'s implementation). "No raw subprocess" is
deliberately NOT the invariant — the framework spawns ``sandbox-exec`` by
design and ``ruff`` waives S603 — so this is an AST *call* scan (styled on
``test_d2_boundary.py``'s full-tree import scan), graduated move-not-build
from ``spikes/260609-run-path-guard`` (11/11) per the execplan's PD7 row.

The guard runs live against the production dialect package run path (landed at
PD5). It currently holds — the in-process driver spawns nothing —
so this is a plain green invariant rather than the signposted xfail the plan
anticipated for a not-yet-landed run path; B3c-1's ``launch_confined``
composition lands *inside* the sanctioned verb and stays green by
construction.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = REPO_ROOT / "shepherd" / "packages" / "dialect" / "src" / "shepherd_dialect"

#: The containment backends — the jail IS the sanctioned spawn; raw subprocess
#: is its mechanism. Everything else scanned is run-path.
IMPL_FILES = frozenset(
    {
        "_containment.py",
        "_seatbelt_containment.py",
        "_landlock_containment.py",
    }
)

# Executor-spawning APIs. A run-path module must reach these only *through*
# launch_confined.
_BANNED: dict[str, frozenset[str]] = {
    "subprocess": frozenset({"run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput"}),
    "os": frozenset(
        {
            "system",
            "popen",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "posix_spawn",
            "posix_spawnp",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
        }
    ),
    "pty": frozenset({"spawn", "fork"}),
    "asyncio": frozenset({"create_subprocess_exec", "create_subprocess_shell"}),
    "multiprocessing": frozenset({"Process"}),
}

#: The one sanctioned verb. A call to it is never a violation, on any receiver.
_SANCTIONED = "launch_confined"


@dataclass(frozen=True)
class Violation:
    """One banned executor call: where, and which API."""

    filename: str
    lineno: int
    api: str


def _import_maps(tree: ast.AST) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Full-tree (incl. function-local) import resolution.

    Deliberately scope-flat — a conservative over-approximation appropriate
    for a deny-guard.
    """
    amap: dict[str, str] = {}
    fmap: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                amap[bound] = (a.asname and a.name) or a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                fmap[a.asname or a.name] = (node.module, a.name)
    return amap, fmap


def _resolve(call: ast.Call, amap: dict[str, str], fmap: dict[str, tuple[str, str]]) -> str | None:
    """Return the banned 'module.attr' a call resolves to, or None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr == _SANCTIONED:
            return None
        base = func.value
        if isinstance(base, ast.Name):
            mod = amap.get(base.id)
            if mod and func.attr in _BANNED.get(mod, frozenset()):
                return f"{mod}.{func.attr}"
    elif isinstance(func, ast.Name):
        if func.id == _SANCTIONED:
            return None
        if func.id in fmap:
            mod, orig = fmap[func.id]
            if orig in _BANNED.get(mod, frozenset()):
                return f"{mod}.{orig}"
    return None


def find_violations(source: str, filename: str, *, role: str) -> list[Violation]:
    """Scan one module; containment_impl is the sanctioned-spawn role."""
    if role == "containment_impl":
        return []  # the jail IS the sanctioned spawn; raw subprocess is its mechanism.
    tree = ast.parse(source, filename=filename)
    amap, fmap = _import_maps(tree)
    return [
        Violation(filename, node.lineno, api)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (api := _resolve(node, amap, fmap)) is not None
    ]


def scan_paths(paths: list[Path], *, impl_files: frozenset[str]) -> list[Violation]:
    """Scan trees: every .py is run_path unless its basename is in impl_files."""
    out: list[Violation] = []
    for p in paths:
        for f in sorted(p.rglob("*.py")) if p.is_dir() else [p]:
            role = "containment_impl" if f.name in impl_files else "run_path"
            out.extend(find_violations(f.read_text(encoding="utf-8"), f.name, role=role))
    return out


# --- The live invariant -------------------------------------------------------


def test_run_path_invokes_no_executor_outside_launch_confined() -> None:
    """The live invariant against the production dialect run path."""
    assert RUN_PATH.is_dir(), f"run path missing: {RUN_PATH}"
    violations = scan_paths([RUN_PATH], impl_files=IMPL_FILES)
    assert violations == [], (
        f"Executor call(s) outside launch_confined in the dialect run path (real ⇒ jailed, layer b): {violations!r}"
    )


# --- The guard guards itself (the spike's self-test corpus, carried over) ------


def test_guard_accepts_the_sanctioned_verb() -> None:
    """launch_confined is never a violation, on any receiver."""
    clean = (
        "def prepare_bound(ctx, req, execution):\n"
        "    spec = map_may_to_spec(req.may)\n"
        "    return execution.launch_confined([req.entrypoint, *req.args], spec)\n"
    )
    assert find_violations(clean, "clean.py", role="run_path") == []


def test_guard_catches_executor_bypasses() -> None:
    """Every spawn family the spike pinned still trips the guard."""
    violating = {
        "raw-subprocess-run": "import subprocess\ndef f(cmd):\n    return subprocess.run(cmd, check=True)\n",
        "aliased-Popen": "from subprocess import Popen as P\ndef f(cmd):\n    return P(cmd)\n",
        "os-system": "import os\ndef f(cmd):\n    os.system(cmd)\n",
        "os-execv": "import os\ndef f(p, a):\n    os.execv(p, a)\n",
        "asyncio-exec": "import asyncio\nasync def f(cmd):\n    return await asyncio.create_subprocess_exec(*cmd)\n",
        "local-import": "def f(cmd):\n    import subprocess\n    return subprocess.Popen(cmd)\n",
    }
    for name, source in violating.items():
        assert find_violations(source, f"{name}.py", role="run_path"), f"guard missed: {name}"


def test_containment_backends_are_impl_not_run_path() -> None:
    """The jail IS the sanctioned spawn; its raw subprocess is allowed."""
    raw = "import subprocess\ndef launch(profile, root, cmd):\n    return subprocess.run(cmd)\n"
    assert find_violations(raw, "_seatbelt_containment.py", role="containment_impl") == []
