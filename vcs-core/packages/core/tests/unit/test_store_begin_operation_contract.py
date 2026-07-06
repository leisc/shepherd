"""Contract test for direct Store.begin_operation call sites."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "vcs_core"

ALLOWED_STORE_BEGIN_OPERATION_CALLS: dict[tuple[str, str, str], str] = {
    (
        "vcs_core/_operation_start_authority.py",
        "begin_executable_operation",
        "owner.store",
    ): "coordinator-authorized executable/session/shell operation start",
    (
        "vcs_core/_operation_start_authority.py",
        "_begin_allowlisted_operation",
        "owner.store",
    ): "private reason-specific diagnostic/reduction operation start",
    (
        "vcs_core/_workspace_adoption.py",
        "adopt_workspace_baseline",
        "store",
    ): "initial workspace adoption",
    (
        "vcs_core/recording.py",
        "RecordingPipeline.begin_operation",
        "self._store",
    ): "low-level recording adapter",
}

ALLOWED_ALLOWLISTED_OPERATION_START_CALLS: dict[tuple[str, str, str], str] = {
    (
        "vcs_core/_managed_exec_service.py",
        "ManagedExecutionService.record_shell_command_not_admitted",
        "begin_not_admitted_shell_command_operation",
    ): "shell-command admission diagnostic",
    (
        "vcs_core/vcscore.py",
        "VcsCore._record_capture_diagnostic",
        "begin_capture_diagnostic_operation",
    ): "capture diagnostic terminalization",
    (
        "vcs_core/vcscore.py",
        "VcsCore._reduce_capture_for_command_operation",
        "begin_capture_reduction_operation",
    ): "capture reduction terminalization",
    (
        "vcs_core/_operation_start_authority.py",
        "begin_not_admitted_shell_command_operation",
        "_begin_allowlisted_operation",
    ): "single private implementation call for shell admission diagnostics",
    (
        "vcs_core/_operation_start_authority.py",
        "begin_capture_diagnostic_operation",
        "_begin_allowlisted_operation",
    ): "single private implementation call for capture diagnostics",
    (
        "vcs_core/_operation_start_authority.py",
        "begin_capture_reduction_operation",
        "_begin_allowlisted_operation",
    ): "single private implementation call for capture reductions",
}

ALLOWED_UNCHECKED_WORLD_PUBLISH_CALLS: dict[tuple[str, str, str], str] = {
    (
        "vcs_core/_world_storage_manager.py",
        "WorldStorageManager.fork_world_ref",
        "self._world_store",
    ): "manager-mediated fork publication after new-world closure validation and receipt recording",
    (
        # V2.2c: advance_publication moved to the publication/retention controller.
        "vcs_core/_publication_retention_controller.py",
        "PublicationRetentionController.advance_publication",
        "self._world_store",
    ): "single controller-mediated publication CAS stage after new-world closure validation and receipt recording",
}

TRACKED_ALLOWLISTED_OPERATION_STARTS = frozenset(
    {
        "_begin_allowlisted_operation",
        "begin_allowlisted_operation",
        "begin_not_admitted_shell_command_operation",
        "begin_capture_diagnostic_operation",
        "begin_capture_reduction_operation",
    }
)


@dataclass(frozen=True)
class BeginOperationCallSite:
    relative_path: str
    qualname: str
    receiver: str
    line: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.relative_path, self.qualname, self.receiver)

    def display(self) -> str:
        return f"{self.relative_path}:{self.line} in {self.qualname}: {self.receiver}.begin_operation(...)"


@dataclass(frozen=True)
class NamedCallSite:
    relative_path: str
    qualname: str
    function_name: str
    line: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.relative_path, self.qualname, self.function_name)

    def display(self) -> str:
        return f"{self.relative_path}:{self.line} in {self.qualname}: {self.function_name}(...)"


@dataclass(frozen=True)
class UncheckedWorldPublishCallSite:
    relative_path: str
    qualname: str
    receiver: str
    line: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.relative_path, self.qualname, self.receiver)

    def display(self) -> str:
        return f"{self.relative_path}:{self.line} in {self.qualname}: {self.receiver}._publish_ref_unchecked(...)"


class _BeginOperationVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self._stack: list[str] = []
        self.calls: list[BeginOperationCallSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "begin_operation":
            receiver = ast.unparse(node.func.value)
            if _is_store_receiver(receiver):
                self.calls.append(
                    BeginOperationCallSite(
                        relative_path=self._relative_path,
                        qualname=".".join(self._stack) or "<module>",
                        receiver=receiver,
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)


class _NamedCallVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self._stack: list[str] = []
        self.calls: list[NamedCallSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        function_name: str | None = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name in TRACKED_ALLOWLISTED_OPERATION_STARTS:
            self.calls.append(
                NamedCallSite(
                    relative_path=self._relative_path,
                    qualname=".".join(self._stack) or "<module>",
                    function_name=function_name,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


class _UncheckedWorldPublishVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self._stack: list[str] = []
        self.calls: list[UncheckedWorldPublishCallSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_publish_ref_unchecked":
            self.calls.append(
                UncheckedWorldPublishCallSite(
                    relative_path=self._relative_path,
                    qualname=".".join(self._stack) or "<module>",
                    receiver=ast.unparse(node.func.value),
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


def _is_store_receiver(receiver: str) -> bool:
    return receiver == "store" or receiver.endswith((".store", "._store"))


def _store_begin_operation_calls() -> list[BeginOperationCallSite]:
    calls: list[BeginOperationCallSite] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_path = path.relative_to(SOURCE_ROOT.parent).as_posix()
        visitor = _BeginOperationVisitor(relative_path)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return calls


def _allowlisted_operation_start_calls() -> list[NamedCallSite]:
    calls: list[NamedCallSite] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_path = path.relative_to(SOURCE_ROOT.parent).as_posix()
        visitor = _NamedCallVisitor(relative_path)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return calls


def _unchecked_world_publish_calls() -> list[UncheckedWorldPublishCallSite]:
    calls: list[UncheckedWorldPublishCallSite] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_path = path.relative_to(SOURCE_ROOT.parent).as_posix()
        visitor = _UncheckedWorldPublishVisitor(relative_path)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return calls


def test_direct_store_begin_operation_call_sites_are_classified() -> None:
    calls = _store_begin_operation_calls()
    observed = {call.key for call in calls}
    allowed = set(ALLOWED_STORE_BEGIN_OPERATION_CALLS)

    unclassified = [call for call in calls if call.key not in allowed]
    missing = sorted(allowed - observed)

    assert not unclassified, "Unclassified Store.begin_operation call site(s):\n" + "\n".join(
        call.display() for call in unclassified
    )
    assert not missing, "Classified Store.begin_operation call site(s) no longer observed:\n" + "\n".join(
        f"{path} in {qualname}: {receiver}.begin_operation(...)" for path, qualname, receiver in missing
    )


def test_allowlisted_operation_start_call_sites_are_classified() -> None:
    calls = _allowlisted_operation_start_calls()
    observed = {call.key for call in calls}
    allowed = set(ALLOWED_ALLOWLISTED_OPERATION_START_CALLS)

    unclassified = [call for call in calls if call.key not in allowed]
    missing = sorted(allowed - observed)

    assert not unclassified, "Unclassified allowlisted operation-start call site(s):\n" + "\n".join(
        call.display() for call in unclassified
    )
    assert not missing, "Classified allowlisted operation-start call site(s) no longer observed:\n" + "\n".join(
        f"{path} in {qualname}: {function_name}(...)" for path, qualname, function_name in missing
    )


def test_unchecked_world_publish_call_sites_are_classified() -> None:
    calls = _unchecked_world_publish_calls()
    observed = {call.key for call in calls}
    allowed = set(ALLOWED_UNCHECKED_WORLD_PUBLISH_CALLS)

    unclassified = [call for call in calls if call.key not in allowed]
    missing = sorted(allowed - observed)

    assert not unclassified, "Unclassified unchecked world publish call site(s):\n" + "\n".join(
        call.display() for call in unclassified
    )
    assert not missing, "Classified unchecked world publish call site(s) no longer observed:\n" + "\n".join(
        f"{path} in {qualname}: {receiver}._publish_ref_unchecked(...)" for path, qualname, receiver in missing
    )
