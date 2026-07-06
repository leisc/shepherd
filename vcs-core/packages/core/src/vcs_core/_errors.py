"""Exception hierarchy for vcs-core."""

from __future__ import annotations


class VcsCoreError(Exception):
    """Root of every vcs-core exception.

    Consumers (Shepherd, the child-runtime driver) can catch `VcsCoreError` at the
    package boundary instead of a bare `except Exception`. Domain exceptions that
    also subclass a stdlib type (`ValueError`, `RuntimeError`) keep that base via
    multiple inheritance, so existing `except ValueError` / `except RuntimeError`
    call sites are unaffected.
    """


class ActivationError(VcsCoreError):
    """Base class for errors that prevent VcsCore.activate()."""


class InvalidIdentityError(ActivationError):
    """Persistent repo identity state is invalid or unsupported."""


class InvalidRepositoryStateError(VcsCoreError, RuntimeError):
    """Repository history violates an invariant required for safe queries."""


class DirtyPushError(ActivationError):
    """Dirty flag file exists -- a prior push crashed mid-operation.

    Pass recover='repair' to activate() to auto-recover, or call
    recover_dirty_push() manually before activating.
    """

    def __init__(
        self,
        session_id: str = "",
        dirty_since: float = 0.0,
        *,
        corrupt: bool = False,
        detail: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.dirty_since = dirty_since
        self.corrupt = corrupt
        interrupted = (
            "A prior push left corrupt dirty metadata"
            if corrupt
            else f"A prior push (session {session_id}) was interrupted"
        )
        detail_line = f"\nDirty metadata error: {detail}" if detail else ""
        self.recovery_hint = (
            f"{interrupted}. Options:\n"
            f"  mg.activate(recover='repair')  # advance materialized, assume push succeeded\n"
            f"  mg.activate(recover='force')   # rewind ground, discard unpushed work\n"
            f"  vcs-core activate --recover repair\n"
            f"  vcs-core recover-materialization --mode repair"
            f"{detail_line}"
        )
        super().__init__(self.recovery_hint)


class InterruptedLifecycleError(ActivationError):
    """Lifecycle ledger exists -- a prior merge/discard was interrupted."""

    def __init__(self, *, operation: str, scope_name: str, phase: str) -> None:
        self.operation = operation
        self.scope_name = scope_name
        self.phase = phase
        self.recovery_hint = (
            f"A prior {operation} for scope {scope_name!r} was interrupted at phase {phase!r}. Options:\n"
            "  mg.activate(recover_lifecycle='resume')  # resume the interrupted lifecycle\n"
            "  mg.recover_lifecycle(mode='resume')      # resume in an active session\n"
            "  vcs-core activate --recover-lifecycle resume  # CLI equivalent"
        )
        super().__init__(self.recovery_hint)


class OpenScopeError(VcsCoreError):
    """push() was requested while a child scope is still live."""


class OrphanedOperationsError(VcsCoreError):
    """A mutating operation was attempted while orphaned operation refs exist."""

    def __init__(self, *, attempted: str, operations: list[str]) -> None:
        self.attempted = attempted
        self.operations = operations
        count = len(operations)
        sample = ", ".join(operations[:5])
        remainder = "" if count <= 5 else ", ..."
        super().__init__(
            f"Cannot {attempted} while {count} orphaned operation ref(s) from a prior session remain: "
            f"{sample}{remainder}. Run archive_orphaned_operations() first."
        )


class SiblingGroupRecoveryRequiredError(VcsCoreError):
    """A mutating operation was attempted while sibling-group recovery is pending."""

    def __init__(self, *, attempted: str, groups: list[str]) -> None:
        self.attempted = attempted
        self.groups = groups
        count = len(groups)
        sample = ", ".join(groups[:5])
        remainder = "" if count <= 5 else ", ..."
        super().__init__(
            f"Cannot {attempted} while {count} sibling-group recovery blocker(s) remain: "
            f"{sample}{remainder}. Resume, cancel, archive, or complete the sibling group first."
        )


class WorkspaceAuthorityRecoveryRequiredError(VcsCoreError):
    """A mutating operation was attempted while v2 workspace authority is pending."""

    def __init__(self, *, attempted: str, operations: list[str]) -> None:
        self.attempted = attempted
        self.operations = operations
        count = len(operations)
        sample = ", ".join(operations[:5])
        remainder = "" if count <= 5 else ", ..."
        super().__init__(
            f"Cannot {attempted} while {count} workspace authority operation(s) require recovery: "
            f"{sample}{remainder}. Run recover_workspace_authority() first."
        )


class WorldQuiescenceError(InvalidRepositoryStateError):
    """A parent mutation was attempted while a merge-bound child operation is open."""


class ScopeAdmissionError(VcsCoreError, ValueError):
    """A new child scope would violate the live-scope admission policy."""


class UnknownForkHintError(VcsCoreError, ValueError):
    """A fork/branch hint key outside the accepted set was supplied.

    Raised at ``ForkHints`` construction (the coordinator's typed layer) and
    at the built-in substrates' public ``branch(hints)`` boundary
    (reject-unknown-keys). The message names the accepted keys.
    """


class LifecycleRecoveryRequiredError(VcsCoreError):
    """A mutating operation was attempted while lifecycle recovery is pending."""

    def __init__(self, *, attempted: str, operation: str, scope_name: str, phase: str) -> None:
        self.attempted = attempted
        self.operation = operation
        self.scope_name = scope_name
        self.phase = phase
        super().__init__(
            f"Cannot {attempted} while interrupted {operation} for scope {scope_name!r} "
            f"is pending at phase {phase!r}. Run recover_lifecycle(mode='resume') first."
        )


class UnscopedMutationError(VcsCoreError):
    """A mutating workspace operation was attempted without an active scope."""

    def __init__(self, operation: str, path: str | None = None) -> None:
        self.operation = operation
        self.path = path
        detail = f" for {path}" if path is not None else ""
        super().__init__(
            f"Workspace mutation {operation!r}{detail} requires an active scope. "
            "Wrap edits in fork()/merge() before mutating the workspace."
        )


class UnresolvedPatchPathError(VcsCoreError):
    """A mutating operation used a path vcs-core cannot safely classify."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"Workspace mutation {operation!r} uses an fd-relative path that vcs-core cannot resolve safely. "
            "Retry with an absolute path or a resolvable directory fd."
        )


class OverlayDirtyError(VcsCoreError):
    """Uncommitted overlay changes exist outside any scope."""

    def __init__(self, message: str, paths: list[str] | None = None) -> None:
        self.paths = paths or []
        self.recovery_hint = (
            f"Uncommitted overlay changes: {', '.join(self.paths[:5])}. Wrap edits in fork()/merge() before pushing."
        )
        super().__init__(message)


class UnsupportedOverlayEntryError(VcsCoreError):
    """Overlay diff encountered a filesystem entry vcs-core cannot represent safely."""

    def __init__(self, *, path: str, kind: str) -> None:
        self.path = path
        self.kind = kind
        super().__init__(
            f"Unsupported overlay filesystem entry at {path!r}: {kind}. "
            "Only regular files, directories, and overlay whiteouts are supported."
        )


class ReadOnlyCarrierError(VcsCoreError):
    """A write was attempted against a read-only carrier (the EROFS tier).

    A carrier mounted ``read_only`` (lowerdir-only, no writable upper) refuses
    writes at the syscall (EROFS) for out-of-band processes; this is the
    in-process twin — framework writes refuse symmetrically, so there is no
    honor-system asymmetry. The read-only carrier is the strongest ``may=``
    enforcement tier (pessimistic check-before): used when a write must be
    refused outright, not captured-then-discarded. See
    ``docs/engineering/convergence/read-only-carrier-mode.md``.
    """


class StaleScopeError(VcsCoreError):
    """Scope ref is missing -- the scope may have been archived."""


class MergePreconditionError(VcsCoreError):
    """Parent ref moved past the scope's fork point."""


class ParentWorkingTreeDivergedError(MergePreconditionError):
    """The parent's effective carrier layer changed after a child fork."""


class VerifyFailedError(VcsCoreError):
    """Real filesystem does not match ground tree after crash recovery."""


class SubstrateNotBoundError(VcsCoreError):
    """A required substrate binding (e.g., env var) could not be resolved."""


class RefResolutionError(VcsCoreError, ValueError):
    """A commitish could not be resolved to a valid commit."""


class SubstrateCommandError(VcsCoreError):
    """Expected failure while executing a substrate command."""

    def __init__(self, *, substrate: str, command: str, message: str) -> None:
        self.substrate = substrate
        self.command = command
        self.message = message
        super().__init__(message)
