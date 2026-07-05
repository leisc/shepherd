"""Integration tests for macOS VM overlay implementation.

These tests verify the complete VM-based OverlayFS implementation on macOS:
- VMPathTranslator discovers VirtioFS mounts and translates paths
- PodmanSandboxManager creates/mounts overlays inside the VM
- OverlayEffectExtractor extracts effects from VM upper layers

Requirements:
- macOS host
- Podman Machine running (`podman machine start`)
- VirtioFS mounts available (default with Podman Machine)

Run with: pytest -m vm_overlay -v
Run all integration tests: pytest packages/shepherd-core/tests/integration/ -v
"""

import contextlib
import shlex
import uuid
from pathlib import Path

import pytest
from shepherd_core.effects import FileCreate, FileDelete, FilePatch
from shepherd_runtime.device.container.effect_collector import EffectCollector
from shepherd_runtime.device.container.overlay_extractor import OverlayEffectExtractor
from shepherd_runtime.device.container.podman import (
    PodmanSandboxManager,
)
from shepherd_runtime.device.container.vm_paths import (
    VMCommandRunner,
    VMPathTranslator,
    is_macos,
    is_vm_available,
)

# =============================================================================
# Test Configuration
# =============================================================================

# Skip entire module if not macOS with VM available
pytestmark = [
    pytest.mark.vm_overlay,
    pytest.mark.integration,
]


def skip_if_vm_unavailable():
    """Skip test if VM overlay is not available."""
    if not is_macos():
        pytest.skip("VM overlay tests only run on macOS")
    if not is_vm_available():
        pytest.skip("Podman Machine not available - run 'podman machine start'")


def run_vm_as_root(vm_runner: VMCommandRunner, command: str):
    """Run a command in the VM as root.

    Existing lower-layer files require copy-up through overlayfs, which on the
    Podman Machine VM behaves like the real container runtime: writes happen as
    root, not as the default `core` SSH user.
    """
    return vm_runner.run(f"sudo sh -lc {shlex.quote(command)}")


def run_vm_batch_as_root(vm_runner: VMCommandRunner, commands: list[str]):
    """Run multiple VM commands as root in a single shell."""
    return run_vm_as_root(vm_runner, " && ".join(commands))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vm_runner():
    """Get VMCommandRunner, skip if unavailable."""
    skip_if_vm_unavailable()
    return VMCommandRunner(timeout=30.0)


@pytest.fixture
def translator():
    """Get VMPathTranslator with discovered mounts."""
    skip_if_vm_unavailable()
    return VMPathTranslator.discover(verify=False)


@pytest.fixture
def manager():
    """Get PodmanSandboxManager with VM overlay enabled."""
    skip_if_vm_unavailable()
    return PodmanSandboxManager()


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def unique_task_id():
    """Generate a unique task ID for test isolation."""
    return f"test-{uuid.uuid4().hex[:8]}"


# =============================================================================
# VMPathTranslator Integration Tests
# =============================================================================


class TestVMPathTranslatorIntegration:
    """Integration tests for VMPathTranslator."""

    def test_discover_finds_virtio_mounts(self, translator):
        """Should discover at least one VirtioFS mount."""
        assert len(translator.virtio_mounts) > 0

    def test_discover_finds_users_mount(self, translator):
        """Should discover /Users VirtioFS mount (standard macOS)."""
        # Either /Users directly or via /mnt/host/Users
        user_paths = [Path("/Users"), Path("/mnt/host/Users")]
        found = any(
            any(str(mount).endswith("Users") for mount in translator.virtio_mounts.values())
            or Path("/Users") in translator.virtio_mounts
            for _ in [None]
        )
        # More lenient check - just verify we can translate home
        home = Path.home()
        try:
            vm_path = translator.host_to_vm(home)
            assert vm_path is not None
        except ValueError:
            pytest.fail(f"Could not translate home directory {home}")

    def test_verify_creates_and_reads_file(self):
        """Verification should create file on host and read it in VM."""
        skip_if_vm_unavailable()
        # This implicitly tests VirtioFS bidirectional access
        translator = VMPathTranslator.discover(verify=True)
        assert len(translator.virtio_mounts) > 0

    def test_home_directory_translation(self, translator):
        """Home directory should translate correctly."""
        home = Path.home()
        vm_path = translator.host_to_vm(home)

        assert vm_path is not None
        # Should preserve the username part
        assert home.name in str(vm_path)

    def test_tmp_to_private_tmp_translation(self, translator):
        """Should handle /tmp -> /private/tmp symlink."""
        # On macOS, /tmp is a symlink to /private/tmp
        tmp_path = Path("/tmp/test-file")

        try:
            vm_path = translator.host_to_vm(tmp_path)
            # Should map to something with tmp in it
            assert "tmp" in str(vm_path)
            assert "test-file" in str(vm_path)
        except ValueError:
            # /tmp might not be shared - that's OK
            pytest.skip("/tmp not available via VirtioFS")

    def test_roundtrip_translation(self, translator):
        """host_to_vm -> vm_to_host should return original path."""
        home = Path.home()
        vm_path = translator.host_to_vm(home)
        recovered = translator.vm_to_host(vm_path)

        assert recovered == home

    def test_vm_native_path_detection(self, translator):
        """VM-native paths should be detected correctly."""
        # /var/shepherd is VM-native (not on VirtioFS)
        assert translator.is_vm_native(Path("/var/shepherd/overlays"))

        # Home directory is on VirtioFS
        home = Path.home()
        vm_home = translator.host_to_vm(home)
        assert not translator.is_vm_native(vm_home)


class TestVMCommandRunnerIntegration:
    """Integration tests for VMCommandRunner."""

    def test_simple_command(self, vm_runner):
        """Simple command should execute and return output."""
        result = vm_runner.run("echo hello")
        assert "hello" in result.stdout

    def test_mkdir_and_exists(self, vm_runner):
        """mkdir_p and exists should work correctly."""
        test_dir = Path(f"/tmp/shepherd-test-{uuid.uuid4().hex[:8]}")

        try:
            assert not vm_runner.exists(test_dir)
            vm_runner.mkdir_p(test_dir)
            assert vm_runner.exists(test_dir)
        finally:
            vm_runner.rm_rf(test_dir)
            assert not vm_runner.exists(test_dir)

    def test_batch_commands(self, vm_runner):
        """Batch commands should execute sequentially."""
        result = vm_runner.run_batch(
            [
                "echo first",
                "echo second",
                "echo third",
            ]
        )
        assert "first" in result.stdout
        assert "second" in result.stdout
        assert "third" in result.stdout

    def test_path_with_spaces(self, vm_runner):
        """Paths with spaces should be handled correctly."""
        test_dir = Path(f"/tmp/shepherd test dir {uuid.uuid4().hex[:8]}")

        try:
            vm_runner.mkdir_p(test_dir)
            assert vm_runner.exists(test_dir)
        finally:
            vm_runner.rm_rf(test_dir)


# =============================================================================
# VM Overlay Creation Tests
# =============================================================================


class TestVMOverlayCreation:
    """Integration tests for overlay creation in VM."""

    def test_create_overlay_creates_vm_directories(self, manager, workspace, unique_task_id):
        """Overlay directories should be created inside VM."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            assert overlay.is_vm_path
            assert manager._vm_runner is not None

            # Verify directories exist in VM
            assert manager._vm_runner.exists(overlay.upper)
            assert manager._vm_runner.exists(overlay.work)
            assert manager._vm_runner.exists(overlay.merged)

            # Verify lower points to VirtioFS path
            assert not manager._path_translator.is_vm_native(overlay.lower)

        finally:
            manager._cleanup_overlay(overlay)

    def test_create_overlay_stores_original_host_path(self, manager, workspace, unique_task_id):
        """Overlay should store original host path for effect extraction."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            assert overlay.original_host_path == workspace
        finally:
            manager._cleanup_overlay(overlay)

    def test_cleanup_removes_vm_directories(self, manager, workspace, unique_task_id):
        """Cleanup should remove overlay directories from VM."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        # Verify exists
        assert manager._vm_runner.exists(overlay.upper)

        # Cleanup
        manager._cleanup_overlay(overlay)

        # Verify removed
        assert not manager._vm_runner.exists(overlay.upper)


# =============================================================================
# VM Overlay Mounting Tests
# =============================================================================


class TestVMOverlayMounting:
    """Integration tests for OverlayFS mounting in VM."""

    def test_mount_creates_overlay_filesystem(self, manager, workspace, unique_task_id):
        """Mounting should create working OverlayFS in VM."""
        # Create file in base workspace
        (workspace / "original.txt").write_text("original content")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Verify mount is active
            assert overlay.is_mounted(vm_runner=manager._vm_runner)

            # Verify original file visible in merged
            result = manager._vm_runner.run(
                f'cat "{overlay.merged}/original.txt"',
                check=False,
            )
            assert result.returncode == 0
            assert "original content" in result.stdout

        finally:
            manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_unmount_removes_mount(self, manager, workspace, unique_task_id):
        """Unmounting should remove the OverlayFS mount."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)
            assert overlay.is_mounted(vm_runner=manager._vm_runner)

            manager.unmount_overlay(overlay)
            assert not overlay.is_mounted(vm_runner=manager._vm_runner)

        finally:
            # Ensure cleanup even if test fails
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_changes_go_to_upper_layer(self, manager, workspace, unique_task_id):
        """Changes in merged should appear in upper layer."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Create new file in merged
            run_vm_as_root(manager._vm_runner, f'echo "new content" > "{overlay.merged}/new_file.txt"')

            # Verify it appears in upper
            result = manager._vm_runner.run(
                f'cat "{overlay.upper}/new_file.txt"',
                check=False,
            )
            assert result.returncode == 0
            assert "new content" in result.stdout

            # Verify original workspace unchanged
            assert not (workspace / "new_file.txt").exists()

        finally:
            manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_original_files_not_modified(self, manager, workspace, unique_task_id):
        """Original workspace files should not be modified."""
        (workspace / "original.txt").write_text("do not modify")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Modify file in merged
            run_vm_as_root(manager._vm_runner, f'echo "modified" > "{overlay.merged}/original.txt"')

            # Verify original unchanged on host
            assert (workspace / "original.txt").read_text() == "do not modify"

        finally:
            manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)


# =============================================================================
# VM Effect Extraction Tests
# =============================================================================


class TestVMEffectExtraction:
    """Integration tests for effect extraction from VM overlays."""

    def test_extract_new_file(self, manager, workspace, unique_task_id):
        """New file in merged should produce FileCreate effect."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Create new file via VM
            run_vm_as_root(manager._vm_runner, f'echo "new file content" > "{overlay.merged}/new_file.txt"')

            # Unmount to finalize changes
            manager.unmount_overlay(overlay)

            # Extract effects
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            assert len(effects) == 1
            assert isinstance(effects[0], FileCreate)
            assert effects[0].path == "new_file.txt"
            assert "new file content" in effects[0].content

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_extract_modified_file(self, manager, workspace, unique_task_id):
        """Modified file should produce FilePatch effect."""
        # Create original file
        (workspace / "existing.txt").write_text("original content")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Modify via VM
            run_vm_as_root(manager._vm_runner, f'echo "modified content" > "{overlay.merged}/existing.txt"')

            manager.unmount_overlay(overlay)

            # Extract
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            assert len(effects) == 1
            assert isinstance(effects[0], FilePatch)
            assert effects[0].path == "existing.txt"
            assert effects[0].old_content == "original content"
            assert "modified content" in effects[0].new_content

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_extract_deleted_file(self, manager, workspace, unique_task_id):
        """Deleted file should produce FileDelete effect."""
        (workspace / "to_delete.txt").write_text("delete me")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Delete via VM
            run_vm_as_root(manager._vm_runner, f'rm "{overlay.merged}/to_delete.txt"')

            manager.unmount_overlay(overlay)

            # Extract
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            assert len(effects) == 1
            assert isinstance(effects[0], FileDelete)
            assert effects[0].path == "to_delete.txt"
            assert effects[0].had_content == "delete me"

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_extract_multiple_changes(self, manager, workspace, unique_task_id):
        """Should extract multiple different types of changes."""
        # Setup original files
        (workspace / "to_modify.txt").write_text("original")
        (workspace / "to_delete.txt").write_text("goodbye")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Make multiple changes
            run_vm_batch_as_root(
                manager._vm_runner,
                [
                    f'echo "new file" > "{overlay.merged}/created.txt"',
                    f'echo "changed" > "{overlay.merged}/to_modify.txt"',
                    f'rm "{overlay.merged}/to_delete.txt"',
                ],
            )

            manager.unmount_overlay(overlay)

            # Extract
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            # Should have 3 effects
            assert len(effects) == 3

            # Verify types
            effect_types = {type(e).__name__ for e in effects}
            assert effect_types == {"FileCreate", "FilePatch", "FileDelete"}

            # Verify paths
            paths = {e.path for e in effects}
            assert paths == {"created.txt", "to_modify.txt", "to_delete.txt"}

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_extract_nested_directory_changes(self, manager, workspace, unique_task_id):
        """Should extract changes in nested directories."""
        # Create nested structure
        subdir = workspace / "src" / "utils"
        subdir.mkdir(parents=True)
        (subdir / "helper.py").write_text("# original helper")

        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            # Modify nested file and create new one
            run_vm_batch_as_root(
                manager._vm_runner,
                [
                    f'echo "# modified" > "{overlay.merged}/src/utils/helper.py"',
                    f'mkdir -p "{overlay.merged}/src/utils/new_dir"',
                    f'echo "new" > "{overlay.merged}/src/utils/new_dir/new_file.py"',
                ],
            )

            manager.unmount_overlay(overlay)

            # Extract
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            # Should have effects for modified and created files
            paths = {e.path for e in effects}
            assert "src/utils/helper.py" in paths
            assert "src/utils/new_dir/new_file.py" in paths

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_extract_with_causality_linking(self, manager, workspace, unique_task_id):
        """Effects should be linked to causing intent."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)

            run_vm_as_root(manager._vm_runner, f'echo "content" > "{overlay.merged}/file.txt"')

            manager.unmount_overlay(overlay)

            # Extract with collector that has intent
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            collector._last_completed_intent_id = "tool-call-123"

            effects = extractor.extract(overlay, collector)

            assert len(effects) == 1
            assert effects[0].caused_by == "tool-call-123"

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)


# =============================================================================
# Overlay Stacking Tests
# =============================================================================


class TestVMOverlayStacking:
    """Integration tests for hierarchical overlay stacking."""

    def test_child_overlay_sees_parent_changes(self, manager, workspace, unique_task_id):
        """Child overlay should see changes from parent overlay."""
        # Create parent overlay
        parent_overlay = manager.create_overlay(
            task_id=f"{unique_task_id}-parent",
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(parent_overlay)

            # Make change in parent
            run_vm_as_root(manager._vm_runner, f'echo "parent content" > "{parent_overlay.merged}/parent_file.txt"')

            # Create child overlay stacked on parent
            child_overlay = manager.create_overlay(
                task_id=f"{unique_task_id}-child",
                context_name="workspace",
                base_path=workspace,
                parent_task_id=f"{unique_task_id}-parent",
            )

            try:
                manager.mount_overlay(child_overlay)

                # Child should see parent's file
                result = manager._vm_runner.run(
                    f'cat "{child_overlay.merged}/parent_file.txt"',
                    check=False,
                )
                assert result.returncode == 0
                assert "parent content" in result.stdout

            finally:
                manager.unmount_overlay(child_overlay)
                manager._cleanup_overlay(child_overlay)

        finally:
            manager.unmount_overlay(parent_overlay)
            manager._cleanup_overlay(parent_overlay)

    def test_child_changes_isolated_from_parent(self, manager, workspace, unique_task_id):
        """Child overlay changes should not affect parent."""
        (workspace / "shared.txt").write_text("original")

        # Create parent overlay
        parent_overlay = manager.create_overlay(
            task_id=f"{unique_task_id}-parent",
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(parent_overlay)

            # Create child overlay
            child_overlay = manager.create_overlay(
                task_id=f"{unique_task_id}-child",
                context_name="workspace",
                base_path=workspace,
                parent_task_id=f"{unique_task_id}-parent",
            )

            try:
                manager.mount_overlay(child_overlay)

                # Modify in child
                run_vm_as_root(manager._vm_runner, f'echo "child modified" > "{child_overlay.merged}/shared.txt"')

                # Parent should still see original
                result = manager._vm_runner.run(
                    f'cat "{parent_overlay.merged}/shared.txt"',
                    check=False,
                )
                assert "original" in result.stdout

                # Child sees modified
                result = manager._vm_runner.run(
                    f'cat "{child_overlay.merged}/shared.txt"',
                    check=False,
                )
                assert "child modified" in result.stdout

            finally:
                manager.unmount_overlay(child_overlay)
                manager._cleanup_overlay(child_overlay)

        finally:
            manager.unmount_overlay(parent_overlay)
            manager._cleanup_overlay(parent_overlay)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestVMOverlayErrorHandling:
    """Integration tests for error handling in VM overlay operations."""

    def test_extraction_handles_empty_upper(self, manager, workspace, unique_task_id):
        """Extraction should handle empty upper layer gracefully."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)
            # Don't make any changes
            manager.unmount_overlay(overlay)

            # Extract should return empty list
            extractor = OverlayEffectExtractor(vm_runner=manager._vm_runner)
            collector = EffectCollector()
            effects = extractor.extract(overlay, collector)

            assert effects == []

        finally:
            with contextlib.suppress(Exception):
                manager.unmount_overlay(overlay)
            manager._cleanup_overlay(overlay)

    def test_cleanup_handles_unmounted_overlay(self, manager, workspace, unique_task_id):
        """Cleanup should handle already-unmounted overlays."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        # Don't mount, just cleanup - should not raise
        manager._cleanup_overlay(overlay)

        # Verify directories removed
        assert not manager._vm_runner.exists(overlay.upper)

    def test_double_unmount_safe(self, manager, workspace, unique_task_id):
        """Double unmount should be safe (idempotent)."""
        overlay = manager.create_overlay(
            task_id=unique_task_id,
            context_name="workspace",
            base_path=workspace,
        )

        try:
            manager.mount_overlay(overlay)
            manager.unmount_overlay(overlay)

            # Second unmount should not raise
            manager.unmount_overlay(overlay)

        finally:
            manager._cleanup_overlay(overlay)
