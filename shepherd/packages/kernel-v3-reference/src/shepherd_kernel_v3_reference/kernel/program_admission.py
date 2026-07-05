"""Private admission and indexing for kernel programs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from shepherd_kernel_v3_reference.kernel.elaborate import KernelProgram, elaborate

if TYPE_CHECKING:
    from shepherd_kernel_v3_reference.kernel.program_identity import ProgramIdentity
    from shepherd_kernel_v3_reference.source.effects import EffectRegistry
    from shepherd_kernel_v3_reference.source.syntax import Computation
from shepherd_kernel_v3_reference.kernel.ir import (
    HandlerInstallDef,
    KAbort,
    KBind,
    KComputation,
    KForward,
    KHandle,
    KPerform,
    KPure,
    KResumeWith,
    KTerminalDelay,
    KTerminalFork,
    Ref,
)
from shepherd_kernel_v3_reference.profiles import (
    CORE_A,
    CORE_REFERENCE_V0_LITE,
    PUBLICATION_EXPERIMENTAL,
    SemanticProfile,
)


class KernelProgramValidationError(RuntimeError):
    """Raised when a `KernelProgram` is structurally malformed."""


NodeId: TypeAlias = tuple[Any, ...]
InstallNodeId: TypeAlias = tuple[Literal["install"], Ref, int]
_MAX_DIAGNOSTIC_PATH_TAIL = 8


@dataclass(frozen=True)
class ProgramIndex:
    root_node: NodeId
    control_nodes: MappingProxyType[NodeId, KComputation]
    identity_edges: MappingProxyType[NodeId, tuple[NodeId, ...]]
    identity_postorder: tuple[NodeId, ...]
    control_postorder: tuple[NodeId, ...]


class PreparedKernelProgram:
    """Opaque admitted-program artifact.

    Construct prepared programs with `prepare_kernel_program(...)`. The runtime
    trusts the index carried by this artifact, so direct construction is kept
    unavailable to callers.

    Carries the admission `profile` per 2026-05-22 §"Profile attachment on
    PreparedKernelProgram" and 2026-05-24 §"Post-#72 design pass" item A.
    Downstream APIs read profile from the prepared artifact rather than
    taking it as an argument.
    """

    __slots__ = ("_identity_cache", "_index", "_profile", "_program", "_seal")

    # Class-level annotations for the slots so mypy can type-check the
    # read-only property getters (slots are populated via
    # `object.__setattr__` because the class overrides `__setattr__` to
    # raise; without these annotations mypy reports the getters as
    # accessing an unknown attribute and returning Any).
    _program: KernelProgram
    _index: ProgramIndex
    _profile: SemanticProfile
    _identity_cache: ProgramIdentity | None
    _seal: object

    def __init__(
        self,
        program: KernelProgram | None = None,
        index: ProgramIndex | None = None,
        *,
        profile: SemanticProfile = CORE_A,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PREPARED_KERNEL_PROGRAM_SEAL or program is None or index is None:
            raise TypeError("PreparedKernelProgram cannot be constructed directly; use prepare_kernel_program(...)")
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_identity_cache", None)
        object.__setattr__(self, "_seal", _seal)

    @property
    def program(self) -> KernelProgram:
        return self._program

    @property
    def index(self) -> ProgramIndex:
        return self._index

    @property
    def profile(self) -> SemanticProfile:
        return self._profile

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PreparedKernelProgram is read-only")

    def __repr__(self) -> str:
        return f"PreparedKernelProgram(program={self.program!r}, index={self.index!r}, profile={self.profile.name!r})"


_PREPARED_KERNEL_PROGRAM_SEAL = object()


KernelProgramInput: TypeAlias = KernelProgram | PreparedKernelProgram


def ensure_prepared_kernel_program(
    program: KernelProgramInput,
    *,
    profile: SemanticProfile = CORE_A,
) -> PreparedKernelProgram:
    """Return an admitted program artifact, preserving already prepared inputs.

    The shim defaults to `CORE_A` per 2026-05-23 §"Profile-attachment migration"
    so the operational corpus (which uses Core-A surfaces) continues to admit
    cleanly. A profile that `requires_source_admission` (e.g.
    `CORE_REFERENCE_V0_LITE`) cannot be minted through this IR-level shim; use
    the source-level `admit_and_prepare(source, profile=...)` entry point
    instead (2026-05-26 §"Profile admission boundary").
    """

    if isinstance(program, PreparedKernelProgram):
        if getattr(program, "_seal", None) is not _PREPARED_KERNEL_PROGRAM_SEAL:
            raise KernelProgramValidationError("PreparedKernelProgram was not produced by prepare_kernel_program(...)")
        return program
    return prepare_kernel_program(program, profile=profile)


@dataclass(frozen=True)
class _ControlPath:
    root: str
    depth: int = 0
    tail: tuple[str, ...] = ()

    def child(self, label: str) -> _ControlPath:
        return _ControlPath(
            root=self.root,
            depth=self.depth + 1,
            tail=(*self.tail, label)[-_MAX_DIAGNOSTIC_PATH_TAIL:],
        )

    def __str__(self) -> str:
        if self.depth == 0:
            return self.root
        if self.depth == len(self.tail):
            return " ".join((self.root, *self.tail))
        return f"{self.root} ... {' '.join(self.tail)} (depth={self.depth})"


def _prepare_unchecked(
    program: KernelProgram,
    *,
    profile: SemanticProfile,
) -> PreparedKernelProgram:
    """Snapshot, validate, index, and stamp — without the profile guard.

    The single stamping path shared by IR-level `prepare_kernel_program(...)`
    (permissive profiles only) and source-level `admit_and_prepare(...)`
    (which has already run `validate_profile_admission(...)`).
    """

    program_snapshot = replace(
        program,
        binders=MappingProxyType(dict(program.binders)),
        handler_envs=MappingProxyType(dict(program.handler_envs)),
        schemas=MappingProxyType(dict(program.schemas)),
    )
    index = _AdmissionBuilder(program_snapshot).build()
    return PreparedKernelProgram(
        program_snapshot,
        index,
        profile=profile,
        _seal=_PREPARED_KERNEL_PROGRAM_SEAL,
    )


def prepare_kernel_program(
    program: KernelProgram,
    *,
    profile: SemanticProfile = CORE_A,
) -> PreparedKernelProgram:
    """Snapshot, validate, and index an IR `KernelProgram`, stamping `profile`.

    IR-level entry point. The default `profile` is `CORE_A` — the permissive
    operational profile that makes no source-level promise. A profile that
    `requires_source_admission` (e.g. `CORE_REFERENCE_V0_LITE`) CANNOT be
    stamped here: its admission contract is over the *source* AST, and the
    distinguishing constructs do not survive `elaborate()`, so an IR-level
    stamp could not honor it. Mint such a profile via the source-level
    `admit_and_prepare(source, profile=...)` instead (2026-05-26 §"Profile
    admission boundary"; the (A) resolution makes the stamp trustworthy by
    unifying enforcement with attachment).

    IR-level publication-experimental rejection (KForward / KTerminalDelay /
    KTerminalFork) is still performed by the structural admission walker
    against `program.profile` and is unchanged.
    """

    if profile.requires_source_admission:
        raise KernelProgramValidationError(
            f"profile {profile.name!r} requires source-level admission and "
            f"cannot be stamped on raw IR; mint it via "
            f"admit_and_prepare(source, profile={profile.name!r}) "
            "(2026-05-26 §'Profile admission boundary')"
        )
    return _prepare_unchecked(program, profile=profile)


def admit_and_prepare(
    source: Computation,
    *,
    profile: SemanticProfile = CORE_REFERENCE_V0_LITE,
    registry: EffectRegistry | None = None,
) -> PreparedKernelProgram:
    """Source-level admit + elaborate + prepare: the only minter of a
    profile that `requires_source_admission`.

    Runs `validate_profile_admission(source, profile)` (rejecting non-profile
    source constructs while they still exist in the AST), elaborates to IR,
    then stamps the validated profile. This is the composition the prior plan
    diagram implied but no function owned; making it the sole minter of a
    non-permissive-profile prepared program is what lets the projection trust
    the profile stamp without re-deriving it (2026-05-26 §"Profile admission
    boundary", resolution (A)).

    `profile` defaults to `CORE_REFERENCE_V0_LITE`, the strict contract
    profile. A permissive profile is accepted too (admission is a no-op for
    it), so this is also a convenient source-to-prepared one-shot.
    """

    from shepherd_kernel_v3_reference.profile_admission import validate_profile_admission

    validate_profile_admission(source, profile=profile)
    ir = elaborate(source, registry=registry)
    return _prepare_unchecked(ir, profile=profile)


class _AdmissionBuilder:
    def __init__(self, program: KernelProgram) -> None:
        self.program = program
        self.control_nodes: dict[NodeId, KComputation] = {}
        self.identity_edges: dict[NodeId, list[NodeId]] = {}
        self.node_order: list[NodeId] = []
        self._seen_nodes: set[NodeId] = set()
        self._scheduled_controls: set[NodeId] = set()

    def build(self) -> ProgramIndex:
        root_node = self._control_node(self.program.root)
        work: list[tuple[KComputation, _ControlPath]] = [(self.program.root, _ControlPath("root"))]

        for binder_ref, binder in self.program.binders.items():
            self._require(
                binder.binder_id == binder_ref,
                f"binder map key {binder_ref!r} disagrees with BinderDef.binder_id {binder.binder_id!r}",
            )
            binder_node = self._binder_node(binder_ref)
            body_node = self._control_node(binder.body)
            self._add_edges(binder_node, (body_node,))
            work.append((binder.body, _ControlPath(f"binder {binder_ref!r} body")))

        for handler_env_ref, handler_env in self.program.handler_envs.items():
            self._require(
                handler_env.handler_env_ref == handler_env_ref,
                "handler-env map key "
                f"{handler_env_ref!r} disagrees with HandlerEnvDef.handler_env_ref "
                f"{handler_env.handler_env_ref!r}",
            )
            install_nodes: list[NodeId] = []
            for idx, install in enumerate(handler_env.bindings):
                install_node = self._install_node(handler_env_ref, idx)
                install_nodes.append(install_node)
                schema_edges = self._schema_edges((install.handled_result_schema_ref,))
                body_node = self._control_node(install.body)
                self._add_edges(install_node, (body_node, *schema_edges))
                self._require_schema(
                    install.handled_result_schema_ref,
                    context=f"handler install {install.install_ref!r} handled result schema",
                )
                work.append(
                    (
                        install.body,
                        _ControlPath(f"handler-env {handler_env_ref!r} install {idx} {install.install_ref!r} body"),
                    )
                )
            self._add_edges(self._handler_env_node(handler_env_ref), tuple(install_nodes))

        for schema_ref, schema in self.program.schemas.items():
            self._require(
                schema.schema_ref == schema_ref,
                f"schema map key {schema_ref!r} disagrees with SchemaDef.schema_ref {schema.schema_ref!r}",
            )
            self._add_node(self._schema_node(schema_ref))

        self._validate_controls(work)
        identity_edges = {node: tuple(edges) for node, edges in self.identity_edges.items()}
        identity_postorder = self._identity_postorder(identity_edges)
        control_postorder = tuple(node for node in identity_postorder if node[:1] == ("control",))
        return ProgramIndex(
            root_node=root_node,
            control_nodes=MappingProxyType(dict(self.control_nodes)),
            identity_edges=MappingProxyType(identity_edges),
            identity_postorder=identity_postorder,
            control_postorder=control_postorder,
        )

    def _validate_controls(self, work: list[tuple[KComputation, _ControlPath]]) -> None:
        while work:
            control, context = work.pop()
            node = self._control_node(control)
            if node in self._scheduled_controls:
                continue
            self._scheduled_controls.add(node)

            if isinstance(control, KPure):
                continue

            if isinstance(control, KBind):
                binder = self.program.binders.get(control.binder_id)
                if binder is None:
                    raise KernelProgramValidationError(f"{context}: KBind cites missing binder {control.binder_id!r}")
                self._require(
                    binder.binder_env_ref == control.binder_env_ref,
                    f"{context}: KBind binder_env_ref {control.binder_env_ref!r} "
                    f"disagrees with BinderDef.binder_env_ref {binder.binder_env_ref!r}",
                )
                bound_node = self._control_node(control.bound)
                self._add_edges(node, (bound_node, self._binder_node(control.binder_id)))
                work.append((control.bound, context.child("bound")))
                continue

            if isinstance(control, KPerform):
                self._require_schema(
                    control.payload_schema_ref,
                    context=f"{context}: KPerform payload schema",
                )
                self._require_schema(
                    control.operation_result_schema_ref,
                    context=f"{context}: KPerform operation-result schema",
                )
                self._add_edges(
                    node,
                    self._schema_edges((control.payload_schema_ref, control.operation_result_schema_ref)),
                )
                continue

            if isinstance(control, KHandle):
                self._require(
                    control.handler_env_ref in self.program.handler_envs,
                    f"{context}: KHandle cites missing handler env {control.handler_env_ref!r}",
                )
                body_node = self._control_node(control.body)
                self._add_edges(node, (body_node, self._handler_env_node(control.handler_env_ref)))
                work.append((control.body, context.child("handled body")))
                continue

            if isinstance(control, KResumeWith):
                continue

            if isinstance(control, KAbort):
                continue

            if isinstance(control, KForward):
                self._require_publication_profile(context)
                continue

            if isinstance(control, KTerminalDelay):
                self._require_publication_profile(context)
                continue

            if isinstance(control, KTerminalFork):
                self._require_publication_profile(context)
                branch_refs = [branch_ref for branch_ref, _ in control.branches]
                self._require(
                    len(branch_refs) == len(set(branch_refs)),
                    f"{context}: KTerminalFork branch refs must be unique",
                )
                continue

            raise KernelProgramValidationError(f"{context}: unknown kernel control {control!r}")

    def _identity_postorder(self, edges: dict[NodeId, tuple[NodeId, ...]]) -> tuple[NodeId, ...]:
        states: dict[NodeId, int] = {}
        active: dict[NodeId, int] = {}
        postorder: list[NodeId] = []

        for start in self.node_order:
            if states.get(start) == 2:
                continue
            stack: list[tuple[NodeId, int]] = [(start, 0)]
            while stack:
                node, child_index = stack[-1]
                if node not in states:
                    states[node] = 1
                    active[node] = len(stack) - 1

                children = edges.get(node, ())
                if child_index < len(children):
                    child = children[child_index]
                    stack[-1] = (node, child_index + 1)
                    child_state = states.get(child)
                    if child_state == 1:
                        cycle = [item for item, _ in stack[active[child] :]] + [child]
                        raise KernelProgramValidationError(
                            "kernel program identity dependency cycle: " + self._format_cycle(cycle)
                        )
                    if child_state is None:
                        stack.append((child, 0))
                    continue

                states[node] = 2
                active.pop(node, None)
                postorder.append(node)
                stack.pop()

        return tuple(postorder)

    def _schema_edges(self, schema_refs: tuple[Ref | None, ...]) -> tuple[NodeId, ...]:
        return tuple(self._schema_node(schema_ref) for schema_ref in schema_refs if schema_ref is not None)

    def _require_schema(self, schema_ref: Ref | None, *, context: str) -> None:
        if schema_ref is None:
            return
        self._require(
            schema_ref in self.program.schemas,
            f"{context} cites missing schema {schema_ref!r}",
        )
        self._add_node(self._schema_node(schema_ref))

    def _add_edges(self, node: NodeId, edges: tuple[NodeId, ...]) -> None:
        self._add_node(node)
        node_edges = self.identity_edges.setdefault(node, [])
        node_edges.extend(edges)
        for edge in edges:
            self._add_node(edge)

    def _add_node(self, node: NodeId) -> None:
        if node in self._seen_nodes:
            return
        self._seen_nodes.add(node)
        self.node_order.append(node)
        self.identity_edges.setdefault(node, [])

    def _control_node(self, control: KComputation) -> NodeId:
        node: NodeId = ("control", id(control))
        self.control_nodes.setdefault(node, control)
        self._add_node(node)
        return node

    def _binder_node(self, binder_ref: Ref) -> NodeId:
        return ("binder", binder_ref)

    def _handler_env_node(self, handler_env_ref: Ref) -> NodeId:
        return ("handler-env", handler_env_ref)

    def _install_node(self, handler_env_ref: Ref, binding_index: int) -> InstallNodeId:
        return ("install", handler_env_ref, binding_index)

    def _schema_node(self, schema_ref: Ref) -> NodeId:
        return ("schema", schema_ref)

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise KernelProgramValidationError(message)

    def _require_publication_profile(self, context: _ControlPath) -> None:
        self._require(
            self.program.profile == PUBLICATION_EXPERIMENTAL,
            f"{context}: publication control requires publication experimental profile",
        )

    def _format_cycle(self, cycle: list[NodeId]) -> str:
        labels = [self._format_node(node) for node in cycle[:8]]
        if len(cycle) > 8:
            labels.append("...")
        return " -> ".join(labels)

    def _format_node(self, node: NodeId) -> str:
        kind = node[0]
        if kind == "control":
            control = self.control_nodes.get(node)
            if control is None:
                return "control<?>"
            return f"control<{type(control).__name__}>"
        if kind == "install":
            handler_env_ref, index = node[1], node[2]
            install = self._install_at(handler_env_ref, index)
            return f"install<{handler_env_ref!r}[{index}] {install.install_ref!r}>"
        return f"{kind}<{node[1]!r}>"

    def _install_at(self, handler_env_ref: object, index: object) -> HandlerInstallDef:
        if not isinstance(handler_env_ref, str) or not isinstance(index, int):
            raise KernelProgramValidationError("internal install node is malformed")
        return self.program.handler_envs[handler_env_ref].bindings[index]
