"""Initialization and config-oriented CLI integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pygit2
from click.testing import CliRunner
from vcs_core import canonical_bytes
from vcs_core._lifecycle_run import LifecycleRun, LifecycleScopeState, read_lifecycle_run, write_lifecycle_run
from vcs_core._projection_store import SCOPE_REGISTRY_CURRENT_REF
from vcs_core._workspace_authority import WorkspaceAuthorityPending, write_pending_workspace_authority
from vcs_core._world_operation_journal import OPERATION_JOURNAL_PATH, OPERATION_JOURNAL_SCHEMA
from vcs_core._world_storage_installation import open_existing_default_world_storage, open_or_init_default_world_storage
from vcs_core.cli import main
from vcs_core.git_store import create_commit_with_recovery, insert_tree_entry
from vcs_core.store import Store
from vcs_core.testing import DEFAULT_GROUND_REF, WorldStorageManager, operation_journal_ref, write_dirty_flag

from ...support.cli import init_repo as _init


def test_init_creates_vcscore_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".vcscore").exists()
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert SCOPE_REGISTRY_CURRENT_REF in store._repo.references
    assert "Initialized" in result.output
    assert f"Managed workspace: {tmp_path.resolve()}" in result.output
    assert "Host environment outside this workspace is untracked." in result.output


def test_init_idempotent(tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(main, ["init", str(tmp_path)])
    assert first.exit_code == 0, first.output
    result = runner.invoke(main, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Already initialized .vcscore/ repository" in result.output
    assert "Initialized .vcscore/ repository" not in result.output


def test_init_auto_detects_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    runner = CliRunner()
    result = runner.invoke(main, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert "auto-detected" in result.output or "git" in result.output.lower()


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True, text=True)


def _commit_git_baseline(workspace: Path) -> None:
    _run_git(workspace, "init", "-q")
    _run_git(workspace, "config", "user.name", "Meta Git Test")
    _run_git(workspace, "config", "user.email", "vcs-core-test@example.invalid")
    (workspace / "README.md").write_text("hello\n")
    executable = workspace / "bin" / "tool"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\necho tool\n")
    executable.chmod(0o755)
    _run_git(workspace, "add", "README.md", "bin/tool")
    _run_git(workspace, "commit", "-qm", "initial")


def test_init_adopt_git_head_records_materialized_baseline(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Adopted 2 filesystem change(s) from git-head" in result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "README.md") == b"hello\n"
    assert store.workspace_file_mode(store.GROUND_REF, "bin/tool") == pygit2.GIT_FILEMODE_BLOB_EXECUTABLE
    adoption_effects = store.filter_effects(effect_type="WorkspaceBaselineAdopt", substrate="filesystem")
    assert len(adoption_effects) == 1
    assert adoption_effects[0].metadata["path_count"] == 2
    assert adoption_effects[0].metadata["created_count"] == 2
    assert store.status().local_changes == 0
    assert store.status().commits_ahead == 0
    manager = open_existing_default_world_storage(tmp_path / ".vcscore")
    selected_world = manager.read_world(store.GROUND_REF)
    selected_head = selected_world.snapshot.head_for("workspace").head
    provenance = manager.store("store_workspace").validate_prepared_candidate(
        selected_head,
        evidence_resolver=manager.world_store.resolve_evidence_ref,
    )
    assert provenance.transition.semantic_op == "workspace-adoption"


def test_init_adopt_rejects_already_adopted_baseline(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    runner = CliRunner()
    first = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])
    assert first.exit_code == 0, first.output

    result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])

    assert result.exit_code == 1
    assert "already has an adopted filesystem baseline" in result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert len(store.filter_effects(effect_type="WorkspaceBaselineAdopt", substrate="filesystem")) == 1


def test_init_adopt_empty_baseline_records_marker_and_blocks_rebaseline(tmp_path: Path) -> None:
    runner = CliRunner()

    first = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])
    assert first.exit_code == 0, first.output
    assert "Adopted 0 filesystem change(s) from worktree" in first.output

    store = Store.open_existing(str(tmp_path / ".vcscore"))
    adoption_effects = store.filter_effects(effect_type="WorkspaceBaselineAdopt", substrate="filesystem")
    assert len(adoption_effects) == 1
    assert adoption_effects[0].metadata["path_count"] == 0
    assert adoption_effects[0].metadata["created_count"] == 0
    assert store.status().local_changes == 0
    assert store.status().commits_ahead == 0

    (tmp_path / "later.txt").write_text("later\n")
    second = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert second.exit_code == 1
    assert "already has an adopted filesystem baseline" in second.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "later.txt") is None
    assert len(store.filter_effects(effect_type="WorkspaceBaselineAdopt", substrate="filesystem")) == 1


def test_init_adopt_after_metadata_only_init_remains_allowed(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("later baseline\n")
    runner = CliRunner()
    initial = runner.invoke(main, ["init", str(tmp_path)])
    assert initial.exit_code == 0, initial.output

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Already initialized .vcscore/ repository" in result.output
    assert "Adopted 1 filesystem change(s) from worktree" in result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "notes.txt") == b"later baseline\n"


def test_init_adopt_git_head_from_git_subdirectory_uses_workspace_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    selected = workspace / "subws"
    selected.mkdir(parents=True)
    _run_git(workspace, "init", "-q")
    _run_git(workspace, "config", "user.name", "Meta Git Test")
    _run_git(workspace, "config", "user.email", "vcs-core-test@example.invalid")
    (workspace / "root.txt").write_text("root\n")
    (selected / "inside.txt").write_text("inside\n")
    _run_git(workspace, "add", "root.txt", "subws/inside.txt")
    _run_git(workspace, "commit", "-qm", "initial")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(selected)])

    assert result.exit_code == 0, result.output
    store = Store.open_existing(str(selected / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "inside.txt") == b"inside\n"
    assert store.read_workspace_file(store.GROUND_REF, "root.txt") is None
    assert store.read_workspace_file(store.GROUND_REF, "subws/inside.txt") is None


def test_init_adopt_git_head_rejects_dirty_tracked_file(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])

    assert result.exit_code == 1
    assert "Cannot adopt Git HEAD" in result.output
    assert "README.md" in result.output


def test_init_adopt_git_head_rejects_untracked_nonignored_file(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / "notes.txt").write_text("untracked\n")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "git-head", "--all", str(tmp_path)])

    assert result.exit_code == 1
    assert "Cannot adopt Git HEAD" in result.output
    assert "notes.txt" in result.output


def test_init_adopt_worktree_records_non_git_baseline(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("worktree\n")
    executable = tmp_path / "script.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 0, result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "notes.txt") == b"worktree\n"
    assert store.workspace_file_mode(store.GROUND_REF, "script.sh") == pygit2.GIT_FILEMODE_BLOB_EXECUTABLE
    assert store.status().local_changes == 0


def test_init_adopt_worktree_excludes_gitignored_files(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / ".gitignore").write_text("debug.log\n")
    _run_git(tmp_path, "add", ".gitignore")
    _run_git(tmp_path, "commit", "-qm", "ignore debug")
    (tmp_path / "debug.log").write_text("ignored\n")
    (tmp_path / "notes.txt").write_text("untracked\n")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 0, result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "README.md") == b"hello\n"
    assert store.read_workspace_file(store.GROUND_REF, "notes.txt") == b"untracked\n"
    assert store.read_workspace_file(store.GROUND_REF, "debug.log") is None


def test_init_adopt_worktree_ignores_gitignored_symlink(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored-link\n")
    _run_git(tmp_path, "add", ".gitignore")
    _run_git(tmp_path, "commit", "-qm", "ignore symlink")
    (tmp_path / "ignored-link").symlink_to("README.md")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 0, result.output
    store = Store.open_existing(str(tmp_path / ".vcscore"))
    assert store.read_workspace_file(store.GROUND_REF, "README.md") == b"hello\n"
    assert store.read_workspace_file(store.GROUND_REF, "ignored-link") is None


def test_init_adopt_worktree_rejects_nonignored_git_symlink(tmp_path: Path) -> None:
    _commit_git_baseline(tmp_path)
    (tmp_path / "link.txt").symlink_to("README.md")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 1
    assert "Cannot adopt symbolic link: link.txt" in result.output


def test_init_adopt_rejects_without_all(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", str(tmp_path)])

    assert result.exit_code == 2
    assert "baseline adoption currently requires `--all`" in result.output


def test_init_adopt_worktree_rejects_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        return
    (tmp_path / "target.txt").write_text("target\n")
    (tmp_path / "link.txt").symlink_to("target.txt")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--adopt", "worktree", "--all", str(tmp_path)])

    assert result.exit_code == 1
    assert "Cannot adopt symbolic link" in result.output


def test_activate_validates_repo_and_reports_active_substrates(tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)

    result = runner.invoke(main, ["activate", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Repository validated." in result.output
    assert "filesystem" in result.output
    assert "marker" in result.output


def test_activate_requires_existing_repo(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["activate", str(tmp_path)])

    assert result.exit_code != 0
    assert "not a vcs-core repository" in result.output


def test_common_commands_fail_closed_without_repo(tmp_path: Path) -> None:
    runner = CliRunner()
    commands = (["status"], ["push"], ["operations"], ["log"], ["diff"], ["inspect"], ["checkout", "ground"])

    with runner.isolated_filesystem(temp_dir=tmp_path):
        for command in commands:
            result = runner.invoke(main, list(command))
            assert result.exit_code != 0, (command, result.output)
            assert "not a vcs-core repository" in result.output
        assert not Path(".vcscore").exists()


def test_inspect_json_reports_invalid_workspace_authority(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    pending = tmp_path / ".vcscore" / "workspace-authority" / "pending"
    pending.mkdir(parents=True)
    (pending / "broken.json").write_text("not json")

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["inspect", "--domain", "workspace_authority", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "vcscore/inspect-result/experimental-v1"
    assert payload["items"][0]["health"]["status"] == "present_corrupt"
    assert payload["issues"][0]["code"] == "workspace_authority_payload_corrupt"


def test_inspect_json_filters_with_selector(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    pending = tmp_path / ".vcscore" / "workspace-authority" / "pending"
    pending.mkdir(parents=True)
    (pending / "broken.json").write_text("not json")

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(
            main,
            [
                "inspect",
                "--domain",
                "workspace_authority",
                "--selector",
                "issue=workspace_authority_payload_corrupt",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query"]["selector"] == "issue=workspace_authority_payload_corrupt"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["health"]["status"] == "present_corrupt"


def test_inspect_json_reports_scope_and_world_domains(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["inspect", "--domain", "scope", "--domain", "authority_ref", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query"]["domains"] == ["scope", "authority_ref"]
    assert [item["domain"] for item in payload["items"]] == ["scope", "authority_ref"]
    assert payload["items"][0]["health"]["status"] == "present_valid"
    assert payload["items"][1]["health"]["status"] == "absent"


def test_readiness_json_reports_shepherd_status(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["readiness", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "vcscore/shepherd-query-readiness/v1"
    assert payload["readiness"]["command"] == "shepherd.status"
    assert payload["readiness"]["allowed"] is True


def test_readiness_json_reports_shepherd_run_as_best_effort_blocked(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["readiness", "--command", "shepherd.run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["readiness"]["command"] == "shepherd.run"
    assert payload["readiness"]["allowed"] is False
    assert payload["readiness"]["state"] == "blocked"
    assert payload["blockers"]


def test_readiness_json_unknown_command_emits_structured_error(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["readiness", "--json", "--command", "bogus.nonsense"])

    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    payload = json.loads(result.output)
    assert "unknown readiness command" in payload["error"]
    assert "bogus.nonsense" in payload["error"]
    assert isinstance(payload["valid_commands"], list)
    assert "shepherd.status" in payload["valid_commands"]


def test_readiness_unknown_command_without_json_is_clean_usage_error(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["readiness", "--command", "bogus.nonsense"])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "unknown readiness command" in result.output
    assert "shepherd.status" in result.output


def test_inspect_json_reports_invalid_operation_journal(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    manager = open_or_init_default_world_storage(tmp_path / ".vcscore")
    _write_manual_journal_commit(
        manager,
        payload_bytes=canonical_bytes(
            {
                "schema": OPERATION_JOURNAL_SCHEMA,
                "operation_id": "op-payload",
                "operation_kind": "shepherd.task",
                "status": "opened",
                "seq": 0,
                "target_ref": DEFAULT_GROUND_REF,
                "input_world_oid": None,
                "candidate_refs": [],
                "candidate_outcomes": [],
                "selected": {},
                "created_at_unix_ns": 1,
                "updated_at_unix_ns": 1,
            }
        ),
        ref=operation_journal_ref("open", "op-locator"),
    )

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["inspect", "--domain", "operation_journal", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "vcscore/inspect-result/experimental-v1"
    assert payload["items"][0]["health"]["status"] == "identity_mismatch"
    assert payload["issues"][0]["code"] == "operation_journal_identity_mismatch"


def test_inspect_json_reports_recovery_domain(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    write_dirty_flag(str(tmp_path / ".vcscore"), "crashed-session")

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["inspect", "--domain", "recovery", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "vcscore/inspect-result/experimental-v1"
    assert payload["query"]["domains"] == ["recovery"]
    assert any(item["domain"] == "recovery" and item["kind"] == "dirty_push" for item in payload["items"])


def test_inspect_json_reports_partial_domain_failure(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    open_or_init_default_world_storage(tmp_path / ".vcscore")
    (tmp_path / ".vcscore" / "world-vectors" / "world-stores.json").write_text("{not-json")
    pending = tmp_path / ".vcscore" / "workspace-authority" / "pending"
    pending.mkdir(parents=True)
    (pending / "broken.json").write_text("not json")

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["inspect", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {issue["code"] for issue in payload["issues"]} == {
        "authority_ref_unreadable",
        "query_domain_unreadable",
        "world_unreadable",
        "workspace_authority_payload_corrupt",
    }
    assert any(item["domain"] == "workspace_authority" for item in payload["items"])


def test_status_reports_workspace_authority_recovery(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    write_pending_workspace_authority(
        tmp_path / ".vcscore",
        WorkspaceAuthorityPending(
            operation_id="wv_status_pending",
            source_operation_id="op_status_pending",
            driver_command="scan",
            scope_name="ground",
            scope_ref=Store.GROUND_REF,
            scope_instance_id="ground",
            scope_world_id=None,
            expected_input_world_oid=None,
            scalar_source_commit=None,
        ).with_update(phase="scalar_committed"),
    )

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "Pending workspace authority" in result.output
    assert "wv_status_pending" in result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        recovery = runner.invoke(main, ["recovery"])

    assert recovery.exit_code == 0, recovery.output
    assert "Pending workspace authority" in recovery.output
    assert "wv_status_pending" in recovery.output


def _write_manual_journal_commit(
    manager: WorldStorageManager,
    *,
    payload_bytes: bytes,
    ref: str,
) -> str:
    repo = manager.world_store.repo
    meta_builder = repo.TreeBuilder()
    insert_tree_entry(
        repo,
        meta_builder,
        OPERATION_JOURNAL_PATH.split("/")[-1],
        repo.create_blob(payload_bytes),
        pygit2.GIT_FILEMODE_BLOB,
    )
    root_builder = repo.TreeBuilder()
    insert_tree_entry(repo, root_builder, "meta", meta_builder.write(), pygit2.GIT_FILEMODE_TREE)
    signature = pygit2.Signature("vcs-core operation journal", "vcs-core@example.invalid")
    oid = create_commit_with_recovery(
        repo,
        None,
        signature,
        signature,
        "manual journal",
        root_builder.write(),
        [],
    )
    repo.references.create(ref, oid, force=True)
    return str(oid)


def test_activate_recover_lifecycle_resume(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["init", str(tmp_path)])
    repo_path = str(tmp_path / ".vcscore")
    write_lifecycle_run(
        repo_path,
        LifecycleRun(
            session_id="crashed-session",
            operation="merge",
            phase="merge_store",
            scope=LifecycleScopeState(
                name="task-cli-resume",
                ref="refs/vcscore/scopes/task-cli-resume",
                instance_id="task-cli-resume",
                creation_oid="root",
                world_id="world-task-cli-resume",
            ),
            parent=LifecycleScopeState(
                name="ground",
                ref="refs/vcscore/ground",
                instance_id="ground",
                creation_oid="",
            ),
        ),
    )

    result = runner.invoke(main, ["activate", "--recover-lifecycle", "resume", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Repository validated." in result.output
    assert read_lifecycle_run(repo_path) is None


def test_recover_workspace_authority_noops_without_pending_state(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    with runner.isolated_filesystem():
        os.chdir(tmp_path)
        result = runner.invoke(main, ["recover-workspace-authority"])

    assert result.exit_code == 0, result.output
    assert "No pending workspace authority." in result.output


def test_coverage_shows_substrates() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        init_result = runner.invoke(main, ["init", "."])
        assert init_result.exit_code == 0, init_result.output

        result = runner.invoke(main, ["coverage"])
        assert result.exit_code == 0
        assert "filesystem" in result.output
        assert "marker" in result.output
        assert "Contain" in result.output
        assert "Gated" in result.output


def test_coverage_does_not_create_vcscore_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["coverage"])

        assert result.exit_code != 0, result.output
        assert "not a vcs-core repository" in result.output
        assert not (Path(".vcscore")).exists()


def test_substrate_list() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["substrate", "list"])
    assert result.exit_code == 0
    assert "filesystem" in result.output
    assert "git" in result.output
    assert "http" in result.output
    assert "[planned]" in result.output


def test_substrate_list_available() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["substrate", "list", "--available"])
    assert result.exit_code == 0
    assert "filesystem" in result.output
    assert "git" in result.output
    assert "marker" in result.output


def test_substrate_show() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["substrate", "show", "filesystem"])
    assert result.exit_code == 0
    assert "always" in result.output
    assert "Commands:" in result.output
    assert "Effects:" not in result.output
    assert "write:" in result.output


def test_substrate_show_unknown_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["substrate", "show", "does-not-exist"])
    assert result.exit_code != 0
    assert "unknown substrate" in result.output


def test_substrate_show_missing_secret_errors_cleanly(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    (tmp_path / "vcscore.toml").write_text(
        '[bindings.filesystem]\ntype = "filesystem"\ndsn = { env = "MISSING_SECRET" }\n'
    )

    result = runner.invoke(main, ["substrate", "show", "filesystem"])

    assert result.exit_code == 0
    assert "Commands:" in result.output


def _register_cli_show_driver(monkeypatch, *, driver_id: str = "test.cli_show_driver") -> str:  # type: ignore[no-untyped-def]
    from vcs_core import discovery
    from vcs_core.manifest import SubstrateManifest

    from ...support.drivers import PlainCommandDriver

    module_name = "_test_cli_show_driver"
    driver_module = types.ModuleType(module_name)

    class CliShowDriver(PlainCommandDriver):
        pass

    CliShowDriver.driver_id = driver_id
    driver_module.CliShowDriver = CliShowDriver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, driver_module)
    real_discover = discovery.discover_plugin_registrations

    def patched_discover(*, strict: bool = True):  # type: ignore[no-untyped-def]
        available = dict(real_discover(strict=strict))
        available[driver_id] = discovery.DiscoveredSubstrate(
            name=driver_id,
            module_name=module_name,
            class_name="CliShowDriver",
            source="plugin",
            manifest=SubstrateManifest(name=driver_id, description="Driver for CLI show tests."),
            implementation_kind="driver",
        )
        return available

    monkeypatch.setattr(discovery, "discover_plugin_registrations", patched_discover)
    return driver_id


def test_substrate_show_displays_driver_kind_schema(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    driver_id = _register_cli_show_driver(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(main, ["substrate", "show", driver_id])

    assert result.exit_code == 0, result.output
    assert f"{driver_id} -- Driver for CLI show tests." in result.output
    assert "Commands:" in result.output
    assert "echo:" in result.output
    assert "message: str (required)" in result.output
    assert "Effects:" not in result.output


def test_binding_list_shows_alias_and_type(tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    (tmp_path / "vcscore.toml").write_text('[bindings.analytics]\ntype = "marker"\n')

    result = runner.invoke(main, ["binding", "list"])

    assert result.exit_code == 0, result.output
    assert "analytics" in result.output
    assert "marker" in result.output


def test_binding_add_writes_binding_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "tomli_w",
        types.SimpleNamespace(dumps=lambda data: '[bindings.analytics]\ntype = "marker"\n'),
    )

    result = runner.invoke(main, ["binding", "add", "analytics", "--type", "marker"])

    assert result.exit_code == 0, result.output
    assert "Added [bindings.analytics]" in result.output
    config_text = (tmp_path / "vcscore.toml").read_text()
    assert "[bindings.analytics]" in config_text
    assert 'type = "marker"' in config_text


def test_binding_show_displays_alias_and_type(tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    (tmp_path / "vcscore.toml").write_text('[bindings.analytics]\ntype = "marker"\n')

    result = runner.invoke(main, ["binding", "show", "analytics"])

    assert result.exit_code == 0, result.output
    assert "analytics -- marker" in result.output
    assert "Source:" in result.output
    assert "configured" in result.output


def test_binding_show_displays_driver_kind_schema(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    driver_id = _register_cli_show_driver(monkeypatch)
    runner = CliRunner()
    _init(runner, tmp_path)
    (tmp_path / "vcscore.toml").write_text(f'[bindings.runtime]\ntype = "{driver_id}"\n')

    result = runner.invoke(main, ["binding", "show", "runtime"])

    assert result.exit_code == 0, result.output
    assert f"runtime -- {driver_id}" in result.output
    assert "Source:" in result.output
    assert "configured" in result.output
    assert "Commands:" in result.output
    assert "echo:" in result.output
    assert "Effects:" not in result.output


def test_binding_show_missing_secret_errors_cleanly(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    _init(runner, tmp_path)
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    (tmp_path / "vcscore.toml").write_text(
        '[bindings.filesystem]\ntype = "filesystem"\ndsn = { env = "MISSING_SECRET" }\n'
    )

    result = runner.invoke(main, ["binding", "show", "filesystem"])

    assert result.exit_code != 0
    assert "unable to resolve bindings" in result.output
    assert "MISSING_SECRET" in result.output


def test_binding_remove_deletes_binding_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    _init(runner, tmp_path)
    (tmp_path / "vcscore.toml").write_text('[bindings.analytics]\ntype = "marker"\n')
    monkeypatch.setitem(sys.modules, "tomli_w", types.SimpleNamespace(dumps=lambda data: ""))

    result = runner.invoke(main, ["binding", "remove", "analytics"])

    assert result.exit_code == 0, result.output
    assert "Removed [bindings.analytics]" in result.output
    assert (tmp_path / "vcscore.toml").read_text() == ""


def test_substrate_check_reports_valid_configured_substrate(tmp_path: Path) -> None:
    runner = CliRunner()
    _init(runner, tmp_path)
    (tmp_path / "vcscore.toml").write_text('[bindings.git]\ntype = "git"\n')

    result = runner.invoke(main, ["substrate", "check", "git"])

    assert result.exit_code == 0, result.output
    assert "Checking substrate 'git'" in result.output
    assert "config" in result.output


def test_config_set_and_list_round_trip(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def dump_table(data: dict, prefix: tuple[str, ...] = ()) -> str:
        lines: list[str] = []
        scalars: list[str] = []
        children: list[tuple[str, dict]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                children.append((key, value))
            else:
                scalars.append(f'{key} = "{value}"')
        if prefix:
            lines.append(f"[{'.'.join(prefix)}]")
        lines.extend(scalars)
        if scalars and children:
            lines.append("")
        for index, (child_key, child_value) in enumerate(children):
            lines.extend(dump_table(child_value, (*prefix, child_key)).splitlines())
            if index != len(children) - 1:
                lines.append("")
        return "\n".join(lines)

    runner = CliRunner()
    _init(runner, tmp_path)
    fake_tomli_w = types.SimpleNamespace(dumps=lambda data: dump_table(data))
    monkeypatch.setitem(sys.modules, "tomli_w", fake_tomli_w)

    set_result = runner.invoke(main, ["config", "set", "--project", "bindings.git.enabled", "true"])
    list_result = runner.invoke(main, ["config", "list", "bindings.git"])

    assert set_result.exit_code == 0, set_result.output
    assert "Set bindings.git.enabled" in set_result.output
    assert list_result.exit_code == 0, list_result.output
    assert "bindings.git.enabled" in list_result.output


def test_config_path_user() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["config", "path", "--user"])
    assert result.exit_code == 0
    assert "config.toml" in result.output


def test_config_path_repo() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["config", "path", "--repo"])
    assert result.exit_code == 0
    assert ".vcscore" in result.output


def test_config_path_project() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["config", "path", "--project"])
    assert result.exit_code == 0
    assert "vcscore.toml" in result.output
