from copy import copy
from dataclasses import replace

import pytest

from shepherd_kernel_v3_reference.kernel import elaborate_publication_experimental
from shepherd_kernel_v3_reference.schemas import AnySchema
from shepherd_kernel_v3_reference.semantic import ObservedFrontier
from shepherd_kernel_v3_reference.source.experimental import TerminalFork
from shepherd_kernel_v3_reference.source.handlers import HandlerEnv, StaticHandlerInstall
from shepherd_kernel_v3_reference.source.syntax import Handle, Let, Lit, Perform, Return, Var
from shepherd_kernel_v3_reference.spikes.branch_replay import (
    BranchReplayValidationError,
    build_single_child_branch_replay_certificate,
    replay_single_child_branch_from_image,
    validate_single_child_branch_replay_certificate,
)
from shepherd_kernel_v3_reference.trace.machine import run_trace
from shepherd_kernel_v3_reference.trace.records import ContinuationResume, ForkBranch, TerminalResumeResult

pytestmark = pytest.mark.xfail(
    reason="K1b defers executable replay from legacy ContinuationImage",
    strict=True,
)


def install(effect_kind: str, body, handler_id: str = "h.v1") -> StaticHandlerInstall:
    return StaticHandlerInstall(
        effect_kind=effect_kind,
        handler_id=handler_id,
        handled_result_schema=AnySchema(),
        body=body,
        payload_name="_payload",
    )


def test_single_child_branch_certificate_replays_exact_terminal_suffix() -> None:
    term = Handle(
        Let("y", Perform("eff.a", Lit("payload")), Return(Var("y"))),
        HandlerEnv((install("eff.a", TerminalFork((("branch:A", Lit("value-A")),)), "h.fork"),)),
    )
    program = elaborate_publication_experimental(term)
    result = run_trace(program, include_debug_evidence=True)

    cert = build_single_child_branch_replay_certificate(result, "branch:A")

    assert cert.replay_input.admission_basis().observed_frontier.record_refs == cert.parent_prefix_refs
    assert isinstance(cert.parent_prefix_records[-1], ForkBranch)
    assert isinstance(cert.child_suffix_records[0], ContinuationResume)
    assert isinstance(cert.child_suffix_records[-1], TerminalResumeResult)
    assert replay_single_child_branch_from_image(program, cert) == cert.child_suffix_records


def test_single_child_branch_replay_includes_downstream_handled_work() -> None:
    term = Handle(
        Let("y", Perform("eff.a", Lit("payload")), Perform("eff.b", Var("y"))),
        HandlerEnv(
            (
                install("eff.a", TerminalFork((("branch:A", Lit("fork-value")),)), "h.fork"),
                install("eff.b", Return(Lit("handled-b")), "h.b"),
            )
        ),
    )
    program = elaborate_publication_experimental(term)
    result = run_trace(program, include_debug_evidence=True)

    cert = build_single_child_branch_replay_certificate(result, "branch:A")

    assert any(getattr(record, "effect_kind", None) == "eff.b" for record in cert.child_suffix_records)
    assert replay_single_child_branch_from_image(program, cert) == cert.child_suffix_records


def test_branch_replay_certificate_rejects_bad_frontier() -> None:
    cert = _simple_cert()
    bad_input = replace(cert.replay_input, observed_frontier=ObservedFrontier(("record:wrong",)))

    with pytest.raises(BranchReplayValidationError, match="observed frontier"):
        validate_single_child_branch_replay_certificate(replace(cert, replay_input=bad_input))


def test_branch_replay_certificate_rejects_missing_restart_image() -> None:
    cert = _simple_cert()
    images = tuple(
        image for image in cert.continuation_images if image.ref != cert.replay_input.restart_continuation_ref
    )

    with pytest.raises(BranchReplayValidationError, match="restart continuation image is missing"):
        validate_single_child_branch_replay_certificate(replace(cert, continuation_images=images))


def test_branch_replay_certificate_rejects_missing_terminal_continuation_ref() -> None:
    cert = _simple_cert()
    fork_branch = cert.parent_prefix_records[-1]
    assert isinstance(fork_branch, ForkBranch)
    bad_prefix = cert.parent_prefix_records[:-1] + (replace(fork_branch, terminal_continuation_ref=None),)

    with pytest.raises(BranchReplayValidationError, match="restart continuation ref mismatch"):
        validate_single_child_branch_replay_certificate(replace(cert, parent_prefix_records=bad_prefix))


def test_branch_replay_certificate_rejects_wrong_replay_input_value() -> None:
    cert = _simple_cert()
    bad_input = replace(cert.replay_input, input_value="wrong")

    with pytest.raises(BranchReplayValidationError, match="input value"):
        validate_single_child_branch_replay_certificate(replace(cert, replay_input=bad_input))


def test_branch_replay_certificate_rejects_wrong_branch_scope_ref() -> None:
    cert = _simple_cert()
    bad_input = replace(cert.replay_input, branch_scope_ref="resume:stale")

    with pytest.raises(BranchReplayValidationError, match="branch scope"):
        validate_single_child_branch_replay_certificate(replace(cert, replay_input=bad_input))


def test_branch_replay_certificate_rejects_prefix_without_fork_branch_source() -> None:
    cert = _simple_cert()

    with pytest.raises(BranchReplayValidationError, match="parent prefix"):
        validate_single_child_branch_replay_certificate(
            replace(cert, parent_prefix_records=cert.parent_prefix_records[:-1])
        )


def test_branch_replay_certificate_rejects_multi_branch_fork_for_first_and_last_child() -> None:
    result = _two_branch_result()

    for branch_ref in ("branch:A", "branch:B"):
        with pytest.raises(BranchReplayValidationError, match="single-child"):
            build_single_child_branch_replay_certificate(result, branch_ref)


def test_branch_replay_certificate_rejects_stale_certificate_boundary_fields() -> None:
    cert = _simple_cert()

    with pytest.raises(BranchReplayValidationError, match="fork index"):
        validate_single_child_branch_replay_certificate(replace(cert, fork_index=999))
    with pytest.raises(BranchReplayValidationError, match="parent branch"):
        validate_single_child_branch_replay_certificate(replace(cert, parent_branch_ref="branch:wrong"))


def test_branch_replay_certificate_rejects_stale_replay_input_fields() -> None:
    cert = _simple_cert()

    bad_inputs = (
        (replace(cert.replay_input, source_ref="fork-branch:wrong"), "source ref"),
        (replace(cert.replay_input, source_kind="ContinuationPending"), "ForkBranch replay inputs"),
        (replace(cert.replay_input, source_path_ref="path:wrong"), "source path"),
        (replace(cert.replay_input, continuation_ref="continuation-image:wrong"), "continuation"),
        (replace(cert.replay_input, handler_continuation_ref="continuation-image:wrong"), "handler continuation"),
        (replace(cert.replay_input, handler_dynamic_tail_ref="continuation-image:wrong"), "dynamic tail"),
        (replace(cert.replay_input, worker_context_ref="ctx:wrong"), "worker context"),
        (replace(cert.replay_input, handler_context_ref="ctx:wrong"), "handler context"),
        (replace(cert.replay_input, operation_result_schema_ref="schema:wrong"), "operation-result schema"),
    )
    for bad_input, message in bad_inputs:
        with pytest.raises(BranchReplayValidationError, match=message):
            validate_single_child_branch_replay_certificate(replace(cert, replay_input=bad_input))


def test_replay_input_admission_basis_rejects_wrong_source_kind() -> None:
    cert = _simple_cert()
    bad_input = replace(cert.replay_input, source_kind="ContinuationPending")

    with pytest.raises(BranchReplayValidationError, match="ForkBranch replay inputs"):
        bad_input.admission_basis()


def test_branch_replay_certificate_rejects_bad_restart_image_fields() -> None:
    cert = _simple_cert()
    restart_ref = cert.replay_input.restart_continuation_ref
    image = next(image for image in cert.continuation_images if image.ref == restart_ref)

    replacements = (
        (_tampered_image(image, branch_ref="branch:wrong"), "branch_ref"),
        (_tampered_image(image, branch_scope_ref="resume:wrong"), "branch scope"),
        (_tampered_image(image, continuation_kind="captured-worker"), "continuation kind"),
    )
    for bad_image, message in replacements:
        images = tuple(
            bad_image if candidate.ref == restart_ref else candidate for candidate in cert.continuation_images
        )
        with pytest.raises(BranchReplayValidationError, match=message):
            validate_single_child_branch_replay_certificate(replace(cert, continuation_images=images))


def _simple_cert():
    term = Handle(
        Let("y", Perform("eff.a", Lit("payload")), Return(Var("y"))),
        HandlerEnv((install("eff.a", TerminalFork((("branch:A", Lit("value-A")),)), "h.fork"),)),
    )
    result = run_trace(elaborate_publication_experimental(term), include_debug_evidence=True)
    return build_single_child_branch_replay_certificate(result, "branch:A")


def _two_branch_result():
    term = Handle(
        Let("y", Perform("eff.a", Lit("payload")), Return(Var("y"))),
        HandlerEnv(
            (
                install(
                    "eff.a",
                    TerminalFork((("branch:A", Lit("value-A")), ("branch:B", Lit("value-B")))),
                    "h.fork",
                ),
            )
        ),
    )
    return run_trace(elaborate_publication_experimental(term), include_debug_evidence=True)


def _tampered_image(image, **changes):
    tampered = copy(image)
    for name, value in changes.items():
        object.__setattr__(tampered, name, value)
    return tampered
