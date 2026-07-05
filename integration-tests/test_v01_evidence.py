"""Regression coverage for the v0.1 release evidence packet."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts" / "run_v01_evidence.py"
READINESS_DOC = REPO / "docs" / "engineering" / "convergence" / "v01-release-readiness.md"
CONVERGENCE_INDEX = REPO / "docs" / "engineering" / "convergence" / "README.md"
MAKEFILE = REPO / "Makefile"

sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("run_v01_evidence", RUNNER_PATH)
assert spec is not None
run_v01_evidence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_v01_evidence
spec.loader.exec_module(run_v01_evidence)


def test_v01_evidence_commands_pin_current_release_floor() -> None:
    """The normal evidence packet covers the current v0.1 release floor."""
    names = [command.name for command in run_v01_evidence.COMMANDS]
    pending = [item["name"] for item in run_v01_evidence.PENDING_EVIDENCE]
    commands = {command.name: command for command in run_v01_evidence.COMMANDS}

    assert names == [
        "claim_state_guard",
        "public_handle_floor",
        "authority_and_output_read_models",
        "lower_path_jailed_enforcement",
        "explicit_readonly_retained_runs",
        "facade_confined_enforcement",
        "best_of_n_example",
        "retry_until_acceptable_example",
    ]
    assert pending == []
    assert "test_workspace_control_public_surface.py" in " ".join(commands["public_handle_floor"].argv)
    assert commands["public_handle_floor"].min_executed_tests == 10
    assert commands["authority_and_output_read_models"].min_executed_tests == 54
    assert "test_jailed_run.py" in " ".join(commands["lower_path_jailed_enforcement"].argv)
    assert commands["lower_path_jailed_enforcement"].min_executed_tests == 3
    assert "test_workspace_control_core_loop.py" in " ".join(commands["explicit_readonly_retained_runs"].argv)
    assert commands["explicit_readonly_retained_runs"].min_executed_tests == 10
    assert "test_workspace_control_workstream3.py" in " ".join(commands["facade_confined_enforcement"].argv)
    assert commands["facade_confined_enforcement"].min_executed_tests == 12


def test_v01_evidence_claim_rows_prove_current_release_claim() -> None:
    """Claim rows prove the release claim once every named command is green."""
    commands = [{"name": command.name, "status": "passed"} for command in run_v01_evidence.COMMANDS]

    claims = {claim["name"]: claim for claim in run_v01_evidence._evaluate_claims(commands)}

    assert claims["workstream_2_public_handle_floor"]["state"] == "proven"
    assert claims["lower_path_jailed_enforcement"]["state"] == "proven"
    assert claims["explicit_readonly_retained_runs"]["state"] == "proven"
    assert claims["facade_confined_enforcement"]["state"] == "proven"
    assert claims["v01_release_claim"]["state"] == "proven"
    assert claims["v01_release_claim"]["pending_evidence"] == []


def test_v01_packet_status_passes_for_proven_claims_or_incomplete_for_skipped_proof() -> None:
    """A fully green packet passes, but zero-exit skipped proof is not enough."""
    commands = [{"name": command.name, "status": "passed"} for command in run_v01_evidence.COMMANDS]
    claims = run_v01_evidence._evaluate_claims(commands)

    assert (
        run_v01_evidence._packet_status(
            commands=commands,
            claims=claims,
            workspace_status_failed=False,
            command_failed=False,
        )
        == "passed"
    )

    skipped_commands = [
        {"name": command.name, "status": "skipped" if command.name == "lower_path_jailed_enforcement" else "passed"}
        for command in run_v01_evidence.COMMANDS
    ]
    skipped_claims = run_v01_evidence._evaluate_claims(skipped_commands)

    assert (
        run_v01_evidence._packet_status(
            commands=skipped_commands,
            claims=skipped_claims,
            workspace_status_failed=False,
            command_failed=False,
        )
        == "incomplete"
    )
    assert {claim["name"]: claim["state"] for claim in skipped_claims}["lower_path_jailed_enforcement"] == "incomplete"


def test_v01_public_floor_claim_needs_executed_proof_tests() -> None:
    """Skipped public-floor pytest evidence cannot prove Workstream 2."""
    skipped_commands = [
        {"name": command.name, "status": "skipped" if command.name == "public_handle_floor" else "passed"}
        for command in run_v01_evidence.COMMANDS
    ]

    claims = {claim["name"]: claim for claim in run_v01_evidence._evaluate_claims(skipped_commands)}

    assert claims["workstream_2_public_handle_floor"]["state"] == "incomplete"
    assert (
        run_v01_evidence._packet_status(
            commands=skipped_commands,
            claims=list(claims.values()),
            workspace_status_failed=False,
            command_failed=False,
        )
        == "incomplete"
    )


def test_v01_claim_pending_evidence_is_not_only_metadata() -> None:
    """Claim-local pending evidence must still prevent promotion."""
    commands = [{"name": command.name, "status": "passed"} for command in run_v01_evidence.COMMANDS]
    original_claims = run_v01_evidence.CLAIMS
    try:
        run_v01_evidence.CLAIMS = tuple(
            replace(claim, pending_evidence=("future_proof",)) if claim.name == "v01_release_claim" else claim
            for claim in original_claims
        )
        claims = {claim["name"]: claim for claim in run_v01_evidence._evaluate_claims(commands)}
    finally:
        run_v01_evidence.CLAIMS = original_claims

    assert claims["v01_release_claim"]["state"] == "pending"
    assert claims["v01_release_claim"]["pending_evidence"] == ["future_proof"]


def test_v01_evidence_dry_run_lists_commands_without_writing(tmp_path: Path) -> None:
    """Dry-run gives reviewers the command packet without running the suite."""
    out_dir = tmp_path / "packet"
    proc = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--dry-run", "--out", str(out_dir)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "workspace_snapshot_mode: ignore-working-copy" in proc.stdout
    assert "public_handle_floor:" in proc.stdout
    assert "lower_path_jailed_enforcement:" in proc.stdout
    assert "explicit_readonly_retained_runs:" in proc.stdout
    assert "facade_confined_enforcement:" in proc.stdout
    assert "-k 'test_public_workspace_run_explicit_readonly_uses_confined_process_for_raw_write or" in proc.stdout
    assert not out_dir.exists()


def test_make_target_uses_strict_snapshot_bookends() -> None:
    """The official release packet snapshots jj state instead of using sandbox-safe metadata."""
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert "uv run python scripts/run_v01_evidence.py --snapshot-working-copy" in makefile


def test_v01_release_readiness_doc_is_indexed_and_claim_bounded() -> None:
    """The claim-lock task list is discoverable and keeps v0.1 nonclaims explicit."""
    readiness = READINESS_DOC.read_text(encoding="utf-8")
    index = CONVERGENCE_INDEX.read_text(encoding="utf-8")

    assert "v01-release-readiness.md" in index
    assert "make v01-evidence" in readiness
    assert "facade_confined_enforcement" in readiness
    assert "explicit_readonly_retained_runs" in readiness
    assert "Workstream 1 has landed" in readiness
    assert "Workstream 3 has landed" in readiness
    assert "`incomplete` means the packet generated successfully" in readiness
    assert "generic `workspace.apply(...)`" in readiness
    assert "framework `best_of_n`, `gather`, or speculative combinator APIs" in readiness
