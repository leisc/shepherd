from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "vcs_core"
TEST_ROOT = ROOT / "tests"
THIS_FILE = Path(__file__).resolve()


def _python_files() -> tuple[Path, ...]:
    roots = (SRC_ROOT, TEST_ROOT)
    return tuple(path for root in roots for path in root.rglob("*.py") if path.resolve() != THIS_FILE)


def test_retired_scalar_command_capture_symbols_are_absent_from_live_python() -> None:
    retired_needles = (
        "ScalarCommandSubstrate",
        "execute_and_record",
        "require_command_outcome",
        "_legacy_driver_schema.py",
        "HookCommand",
        "_command_coercion",
        "coerce_command_mapping",
        "compile_command_contract_from_specs",
        "_legacy_tool_projection",
        "_legacy_validate_projected_surface_input",
        "split_command_execution_envelope",
        "merge_command_execution_envelope_params",
        "RESERVED_COMMAND_ENVELOPE_PARAM_NAMES",
    )
    retired_exact_needles = (
        "class CommandProjection:",
        "class ProjectedParam:",
        "def project_command(",
        "def anthropic_tool_schema(",
    )
    violations: list[str] = []
    command_outcome = re.compile(r"\bCommandOutcome\b")
    performed_true = re.compile(r"\bperformed\s*=\s*True\b")

    for path in _python_files():
        text = path.read_text()
        for needle in retired_needles:
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)} contains {needle!r}")
        for needle in retired_exact_needles:
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)} contains retired compatibility symbol {needle!r}")
        if command_outcome.search(text):
            violations.append(f"{path.relative_to(ROOT)} contains retired bare CommandOutcome")
        if path == SRC_ROOT / "_patch_manager.py" and performed_true.search(text):
            violations.append(f"{path.relative_to(ROOT)} routes patch capture through performed=True")

    assert violations == []


def test_performed_is_not_a_command_dispatch_option_in_runtime_modules() -> None:
    checked_paths = (
        SRC_ROOT / "_vcscore_runtime.py",
        SRC_ROOT / "vcscore.py",
        SRC_ROOT / "_command_admission.py",
        SRC_ROOT / "sqlite_substrate.py",
        SRC_ROOT / "_app.py",
    )
    violations: list[str] = []
    performed_signature = re.compile(r"\bperformed\s*:")
    performed_kwarg = re.compile(r"\bperformed\s*=")

    for path in checked_paths:
        text = path.read_text()
        if performed_signature.search(text) or performed_kwarg.search(text):
            violations.append(f"{path.relative_to(ROOT)} still exposes performed as command dispatch state")

    assert violations == []


def test_prelaunch_compatibility_shims_are_absent_from_live_source() -> None:
    checked_paths = tuple(SRC_ROOT.rglob("*.py"))
    retired_exact_needles = ("pending_path_candidates",)
    retired_patterns = (
        (
            re.compile(r"\b(?:from\s+vcs_core\._overlay\s+import|import\s+vcs_core\._overlay(?:\s|$))"),
            "retired _overlay import path",
        ),
        (
            re.compile(r"\bdef\s+_safe_component\s*\(|\b_safe_component\s*\("),
            "retired workspace-authority safe-component helper",
        ),
        (re.compile(r"\bOverlayBackend\b"), "retired OverlayBackend alias"),
        (re.compile(r"\bdef\s+substrates\s*\("), "retired VcsCore.substrates property"),
        (re.compile(r"\bself\._substrates\b"), "retired VcsCore._substrates alias"),
        (re.compile(r"\b(?:self\._pipeline|pipeline)\.scope\b"), "retired RecordingPipeline.scope view"),
        (
            re.compile(r"DriverContext\([^)]*\bingress_kind\s*=", re.DOTALL),
            "retired DriverContext.ingress_kind constructor field",
        ),
        (
            re.compile(r"diagnostics:\s*tuple\[Diagnostic\s*\|\s*Mapping"),
            "retired mapping-shaped diagnostic tolerance",
        ),
        (
            re.compile(r"DriverIngressResult\([^)]*diagnostics\s*=\s*\(\s*\{", re.DOTALL),
            "mapping-shaped DriverIngressResult diagnostics",
        ),
    )
    violations: list[str] = []

    for path in checked_paths:
        text = path.read_text()
        for needle in retired_exact_needles:
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)} contains {needle!r}")
        for pattern, label in retired_patterns:
            if pattern.search(text):
                violations.append(f"{path.relative_to(ROOT)} contains {label}")

    assert violations == []
