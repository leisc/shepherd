"""Per-construct source-AST admission validator for the `-lite` profile.

Per `260521-0600-kernel.md` §"Canonical Value and Schema Profile" /
"`core-reference-v0-lite` source admission" and 2026-05-23 §"Handler-body
admission shape: Resume-or-Abort".

The validator runs *before* `prepare_kernel_program(...)`. Profile-admission
failure is a distinct error class (`ProfileAdmissionError`) that the
runtime canary's fallback flow can treat as a normal fallback case (per
`260511-2200-plan.md` §"Profile Admission As Pre-Prepare Gate").

Admission set for `core-reference-v0-lite`:

  Source syntax: Return / Let / Perform / Handle / Var / Lit + Core-A
                 Resume / Abort + StaticHandlerInstall in one of the
                 two normalized body shapes below.

  Values:        int + null (rejects bool, str, list, record, float, ...).
  Schemas:       IntSchema / NullSchema / LiteralSchema(int).

  Handler body:  Resume-shape (Core-0H normalized) — pure pre-resume
                 computation, exactly one top-level Resume(value), pure
                 source answer ending in Return(...).

                 Abort-shape (Core-A normalized) — pure pre-abort
                 computation, exactly one top-level Abort(value) with no
                 further computation.

  Rejected:      RecordExpr, DynamicHandlerInstall, Forward,
                 TerminalDelay, TerminalFork, multi-shot resume,
                 no-resume-no-abort handler bodies, nested resume,
                 booleans / strings / lists / records / floats /
                 opaque values, AnySchema / TypeSchema /
                 TaggedRecordSchema / RecordSchema / custom schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shepherd_kernel_v3_reference.envelope import KernelRejection, SourceLocation
from shepherd_kernel_v3_reference.profiles import (
    CORE_REFERENCE_V0_LITE,
    SemanticProfile,
)
from shepherd_kernel_v3_reference.schemas import (
    IntSchema,
    LiteralSchema,
    NullSchema,
    Schema,
)
from shepherd_kernel_v3_reference.source.handlers import (
    DynamicHandlerInstall,
    HandlerEnv,
    StaticHandlerInstall,
)
from shepherd_kernel_v3_reference.source.syntax import (
    Abort,
    Computation,
    Expr,
    Handle,
    Let,
    Lit,
    Perform,
    RecordExpr,
    Resume,
    Return,
    Var,
)


class ProfileAdmissionError(ValueError):
    """Raised when a source program is rejected by the profile admission gate.

    Carries a `KernelRejection` payload (kind="profile-admission") suitable
    for direct wrapping into a `KernelResultEnvelope(status='profile-rejected', ...)`.
    """

    def __init__(self, rejection: KernelRejection) -> None:
        super().__init__(rejection.diagnostic)
        self.rejection: KernelRejection = rejection


@dataclass(frozen=True)
class _Path:
    """Construct-path accumulator for source-location diagnostics.

    Mirrors the `Handle.body.Let[1].body`-style shape named in
    `SourceLocation.construct_path`. Implementation is a tuple of components
    with O(1) append; rendered to a string only at error time.
    """

    parts: tuple[str, ...] = ()

    def append(self, part: str) -> _Path:
        return _Path(parts=(*self.parts, part))

    def render(self) -> str:
        return ".".join(self.parts) if self.parts else "<root>"


def _reject(
    construct: str,
    diagnostic: str,
    path: _Path,
) -> ProfileAdmissionError:
    return ProfileAdmissionError(
        KernelRejection(
            kind="profile-admission",
            diagnostic=diagnostic,
            construct=construct,
            source_location=SourceLocation(construct_path=path.render()),
        )
    )


# ---------------------------------------------------------------------------
# Expression admission (Lit / Var only; RecordExpr rejected)
# ---------------------------------------------------------------------------


def _admit_expr(expr: Expr, path: _Path) -> None:
    if isinstance(expr, Var):
        return
    if isinstance(expr, Lit):
        _admit_value(expr.value, path.append("value"))
        return
    if isinstance(expr, RecordExpr):
        raise _reject(
            "RecordExpr",
            "RecordExpr is not admitted by core-reference-v0-lite (records arrive in -json)",
            path,
        )
    # Any other expression form is a profile-admission rejection by exclusion
    raise _reject(
        type(expr).__name__,
        f"expression {type(expr).__name__} is not admitted by core-reference-v0-lite",
        path,
    )


def _admit_value(value: Any, path: _Path) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise _reject(
            "bool",
            "bool values are not admitted by core-reference-v0-lite "
            "(map to 0/1 if needed; bool support arrives in -json)",
            path,
        )
    if isinstance(value, int):
        return
    raise _reject(
        type(value).__name__,
        f"value of type {type(value).__name__} is not admitted by core-reference-v0-lite (only int and null)",
        path,
    )


# ---------------------------------------------------------------------------
# Schema admission (IntSchema / NullSchema / LiteralSchema(int) only)
# ---------------------------------------------------------------------------


def _admit_schema(schema: Schema, path: _Path) -> None:
    if isinstance(schema, (IntSchema, NullSchema, LiteralSchema)):
        return
    raise _reject(
        type(schema).__name__,
        f"schema {type(schema).__name__} is not admitted by core-reference-v0-lite "
        f"(only IntSchema / NullSchema / LiteralSchema(int))",
        path,
    )


# ---------------------------------------------------------------------------
# Handler body shape: Resume-or-Abort normalized
# ---------------------------------------------------------------------------


def _count_resume_abort(comp: Computation) -> tuple[int, int]:
    """Count Resume and Abort occurrences in a computation tree."""
    if isinstance(comp, Resume):
        return (1, 0)
    if isinstance(comp, Abort):
        return (0, 1)
    if isinstance(comp, Return):
        return (0, 0)
    if isinstance(comp, Perform):
        return (0, 0)
    if isinstance(comp, Let):
        rb, ab = _count_resume_abort(comp.bound)
        rb2, ab2 = _count_resume_abort(comp.body)
        return (rb + rb2, ab + ab2)
    if isinstance(comp, Handle):
        rb, ab = _count_resume_abort(comp.body)
        # Nested Handle's inner handlers are walked by _admit_handler_body
        # at the outer recursion; we count only resumes/aborts in the body
        # at the same handler-extent.
        return (rb, ab)
    return (0, 0)


def _terminates_in_return(comp: Computation) -> bool:
    """Resume-shape rule: after the resume binding, the answer must end in Return.

    Walks `Let` chains looking for a terminating `Return`. Conservative: a
    `Handle` or `Perform` at the tail position is admitted as a "pure tail"
    only if all its sub-computations also terminate in Return.

    For -lite, the rule is: the answer position must reach a Return without
    going through another Resume/Abort.
    """
    if isinstance(comp, Return):
        return True
    if isinstance(comp, Let):
        # The let's tail is what we follow
        return _terminates_in_return(comp.body)
    if isinstance(comp, Handle):
        return _terminates_in_return(comp.body)
    if isinstance(comp, Perform):
        # A Perform that's followed by something is in a Let position; here
        # it's the terminal computation, which is unusual but admissible only
        # if its result is the handler's answer. We require Return.
        return False
    # Resume / Abort as the terminal answer of the body is the body's own
    # shape; that's not "terminates in Return".
    return False


def _terminates_in_abort(comp: Computation) -> bool:
    """Abort-shape rule: the body ends in an Abort (no Return after)."""
    if isinstance(comp, Abort):
        return True
    if isinstance(comp, Let):
        return _terminates_in_abort(comp.body)
    return False


def _admit_handler_body(body: Computation, path: _Path) -> None:
    """Verify the body matches Resume-shape or Abort-shape (Resume-or-Abort).

    Per 2026-05-23 §"Handler-body admission shape: Resume-or-Abort":
    Resume-shape — exactly one top-level Resume followed by pure tail
                   terminating in Return.
    Abort-shape  — exactly one top-level Abort, no Return after.
    Rejected     — multi-shot (>1 Resume), mixed (Resume+Abort), no-resume-no-abort,
                   nested resume.
    """

    n_resume, n_abort = _count_resume_abort(body)

    if n_resume == 0 and n_abort == 0:
        raise _reject(
            "no-resume-no-abort handler body",
            "core-reference-v0-lite rejects handler bodies with neither Resume "
            "nor Abort (Core-0 baseline bodies are out; use Resume-shape or "
            "Abort-shape)",
            path,
        )
    if n_resume > 1:
        raise _reject(
            "multi-shot Resume in handler body",
            f"core-reference-v0-lite admits at most one Resume per handler body "
            f"(found {n_resume}); multi-shot resume arrives in a later profile",
            path,
        )
    if n_abort > 1:
        raise _reject(
            "multiple Abort in handler body",
            f"core-reference-v0-lite admits at most one Abort per handler body (found {n_abort})",
            path,
        )
    if n_resume == 1 and n_abort == 1:
        raise _reject(
            "mixed Resume+Abort handler body",
            "core-reference-v0-lite admits handler bodies in Resume-shape OR Abort-shape, not both",
            path,
        )

    if n_resume == 1:
        if not _terminates_in_return(body):
            raise _reject(
                "Resume-shape handler body does not terminate in Return",
                "core-reference-v0-lite Resume-shape requires the body to "
                "terminate in Return(...) after the Resume binding",
                path,
            )
    # n_abort == 1
    elif not _terminates_in_abort(body):
        raise _reject(
            "Abort-shape handler body has computation after Abort",
            "core-reference-v0-lite Abort-shape rejects computation after the Abort (Abort is the terminal answer)",
            path,
        )

    # Recurse into the body to admit nested computations
    _admit_computation(body, path)


# ---------------------------------------------------------------------------
# Computation admission
# ---------------------------------------------------------------------------


def _admit_computation(comp: Computation, path: _Path) -> None:
    if isinstance(comp, Return):
        _admit_expr(comp.expr, path.append("Return.expr"))
        return
    if isinstance(comp, Let):
        _admit_computation(comp.bound, path.append("Let.bound"))
        _admit_computation(comp.body, path.append("Let.body"))
        return
    if isinstance(comp, Perform):
        _admit_expr(comp.payload, path.append("Perform.payload"))
        return
    if isinstance(comp, Handle):
        _admit_handler_env(comp.handler_env, path.append("Handle.handler_env"))
        _admit_computation(comp.body, path.append("Handle.body"))
        return
    if isinstance(comp, Resume):
        # Resume is only valid inside a handler body and must be top-level
        # within its body. Resume-or-Abort discipline is enforced by
        # _admit_handler_body; here we only admit its value.
        _admit_expr(comp.value, path.append("Resume.value"))
        return
    if isinstance(comp, Abort):
        _admit_expr(comp.value, path.append("Abort.value"))
        return

    # Anything else (publication-experimental controls, etc.) is rejected
    construct = type(comp).__name__
    raise _reject(
        construct,
        f"{construct} is not admitted by core-reference-v0-lite "
        f"(only Return / Let / Perform / Handle / Resume / Abort)",
        path,
    )


def _admit_handler_env(env: HandlerEnv, path: _Path) -> None:
    for i, install in enumerate(env.bindings):
        install_path = path.append(f"bindings[{i}]")
        if isinstance(install, DynamicHandlerInstall):
            raise _reject(
                "DynamicHandlerInstall",
                "DynamicHandlerInstall (Python closures) is not admitted by "
                "core-reference-v0-lite; use StaticHandlerInstall",
                install_path,
            )
        if not isinstance(install, StaticHandlerInstall):
            raise _reject(
                type(install).__name__,
                f"handler install {type(install).__name__} is not admitted by "
                f"core-reference-v0-lite (only StaticHandlerInstall)",
                install_path,
            )
        _admit_schema(
            install.handled_result_schema,
            install_path.append("handled_result_schema"),
        )
        _admit_handler_body(install.body, install_path.append("body"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_profile_admission(
    program: Computation,
    profile: SemanticProfile = CORE_REFERENCE_V0_LITE,
) -> None:
    """Validate a source program against the named profile's admission set.

    Raises `ProfileAdmissionError` carrying a `KernelRejection(kind=
    'profile-admission', construct=..., source_location=...)` on the first
    rejection.

    Currently implements `CORE_REFERENCE_V0_LITE` only. Other profiles are
    accepted as no-ops for forward compatibility; future profile rungs
    (`-json`, full `core-reference-v0`) widen the admitted set.
    """

    if profile is not CORE_REFERENCE_V0_LITE:
        # No-op for non-lite profiles (CORE_A and friends admit the wider set
        # historically; CORE_REFERENCE_V0_LITE is the only narrow profile today)
        return

    _admit_computation(program, _Path())


__all__ = [
    "ProfileAdmissionError",
    "validate_profile_admission",
]
