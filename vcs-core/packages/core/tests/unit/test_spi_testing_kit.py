"""Tests for the substrate conformance kit (vcs_core.spi.testing).

Two halves: the kit passes on real drivers (the built-ins + a bare
Protocol-only driver), and it *fails loudly* on deliberately broken fakes —
the aspirational-accepts, evidence-kind-drift, and unauthorized-execution
shapes the kit exists to catch.
"""

from __future__ import annotations

import sys

import pytest
from vcs_core.runtime_api import CommandRequest, DriverContext, DriverIngressResult
from vcs_core.spi import (
    CapabilitySet,
    CommandSpec,
    DriverSchema,
    IngressRequest,
    ParamSpec,
    ReduceRequest,
    RevisionStorageProfile,
    TransitionDraft,
    UnsupportedRequestError,
)
from vcs_core.spi.testing import (
    ConformanceCase,
    assert_execution_driver_conformant,
    assert_match_dispatch_exhaustive,
    assert_substrate_driver_conformant,
    build_probe_context,
    conformance_cases,
)

from tests.contract.conftest import DRIVERS_UNDER_TEST

# ---------------------------------------------------------------------------
# The kit passes on real drivers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver_cls", DRIVERS_UNDER_TEST, ids=lambda c: c.__name__)
def test_kit_passes_on_every_built_in_driver(driver_cls: type) -> None:
    assert_substrate_driver_conformant(driver_cls())


class _MemoryDriver:
    """A bare Protocol-only driver — no BaseSubstrateDriver identity fields.

    Exercises build_probe_context's getattr fallbacks (no binding/role/store_id).
    """

    driver_id = "test.memory"
    driver_version = "v1"
    capabilities = CapabilitySet(accepts=frozenset({CommandRequest}), selectable=True)

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
        )

    def prepare(self, context: DriverContext, request: IngressRequest) -> DriverIngressResult:
        del context
        if isinstance(request, CommandRequest):
            return DriverIngressResult(
                transitions=(
                    TransitionDraft(
                        transition_id="primary",
                        semantic_op=request.command,
                        payload={"schema": "test/memory"},
                        observation_ids=(),
                    ),
                )
            )
        raise UnsupportedRequestError(driver_id=self.driver_id, request_type=type(request))

    def capture_adapters(self, context: DriverContext) -> tuple[()]:
        del context
        return ()

    def validate_result(self, request: IngressRequest, result: DriverIngressResult) -> None:
        del request, result


def test_kit_passes_on_protocol_only_driver_via_probe_context_fallbacks() -> None:
    assert_substrate_driver_conformant(_MemoryDriver())


def test_build_probe_context_falls_back_when_identity_attrs_absent() -> None:
    context = build_probe_context(_MemoryDriver())
    # No binding/role/store_id on the driver → neutral probe defaults.
    assert context.binding == "probe"
    assert context.store_identity.store_id == "store_probe"


def test_conformance_case_ids_are_stable_and_named() -> None:
    cases = conformance_cases(_MemoryDriver())
    ids = [case.id for case in cases]
    assert ids == [
        "structural",
        "identity",
        "describe_coherence",
        "describe_context_invariance",
        "schema_validity",
        "projectability",
        "schema_anti_inference",
        "dispatch:CommandRequest",
        "evidence_kinds",
    ]
    assert all(isinstance(case, ConformanceCase) for case in cases)


# ---------------------------------------------------------------------------
# The kit fails loudly on broken fakes
# ---------------------------------------------------------------------------


class _AspirationalAcceptsDriver(_MemoryDriver):
    """Advertises ReduceRequest in accepts but has no handler for it."""

    driver_id = "test.aspirational"
    capabilities = CapabilitySet(
        accepts=frozenset({CommandRequest, ReduceRequest}),
        selectable=True,
    )
    # prepare (inherited) raises UnsupportedRequestError for ReduceRequest.


def test_kit_catches_aspirational_accepts() -> None:
    with pytest.raises(AssertionError, match="UnsupportedRequestError"):
        assert_substrate_driver_conformant(_AspirationalAcceptsDriver())


class _IncoherentDescribeDriver(_MemoryDriver):
    """describe() reports a different driver_id than the driver carries."""

    driver_id = "test.incoherent"

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id="test.lies",
            driver_version=self.driver_version,
            capabilities=self.capabilities,
        )


def test_kit_catches_describe_incoherence() -> None:
    with pytest.raises(AssertionError, match=r"disagrees with driver\.driver_id"):
        assert_substrate_driver_conformant(_IncoherentDescribeDriver())


class _ContextVariantDescribeDriver(_MemoryDriver):
    """describe() changes across calls, violating v0.1 context invariance."""

    driver_id = "test.context_variant"

    def __init__(self) -> None:
        self._describe_count = 0

    def describe(self) -> DriverSchema:
        self._describe_count += 1
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            commands={"probe": CommandSpec(description=f"Probe {self._describe_count}")},
        )


def test_kit_catches_context_variant_describe() -> None:
    with pytest.raises(AssertionError, match="context-invariant"):
        assert_substrate_driver_conformant(_ContextVariantDescribeDriver())


class _BadProjectableSchemaDriver(_MemoryDriver):
    """Forgets to mark a Python-only parameter non-projectable."""

    driver_id = "test.bad_projectable_schema"
    capabilities = CapabilitySet(accepts=frozenset({CommandRequest}), selectable=False)

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            commands={
                "probe": CommandSpec(
                    description="Probe",
                    params={"provider": ParamSpec(type="ExecutionProvider", required=False)},
                )
            },
        )


def test_kit_catches_projectable_python_only_type() -> None:
    with pytest.raises(AssertionError, match="unsupported type 'ExecutionProvider'"):
        assert_substrate_driver_conformant(_BadProjectableSchemaDriver())


class _UnboundedSnapshotDriver(_MemoryDriver):
    driver_id = "test.unbounded_snapshot"

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            storage_profile=RevisionStorageProfile(
                shape="json-snapshot",
                authority_role="authority",
                growth_bound="unbounded",
            ),
        )


def test_kit_catches_unbounded_json_snapshot_storage() -> None:
    with pytest.raises(AssertionError, match="unbounded json-snapshot"):
        assert_substrate_driver_conformant(_UnboundedSnapshotDriver())


class _BadAcceleratorStorageDriver(_MemoryDriver):
    driver_id = "test.bad_accelerator_storage"

    def describe(self) -> DriverSchema:
        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            storage_profile=RevisionStorageProfile(
                shape="derived-index",
                authority_role="accelerator",
                growth_bound="unbounded",
                read_safety="superset",
                crash_lag="atomic",
            ),
        )


def test_kit_catches_incoherent_accelerator_storage_contract() -> None:
    with pytest.raises(AssertionError, match="incoherent read_safety/crash_lag"):
        assert_substrate_driver_conformant(_BadAcceleratorStorageDriver())


class _BadEvidenceKindAdapter:
    """A capture adapter whose evidence_kind violates the <mechanism>:<kind> rule."""

    adapter_id = "probe.bad"
    adapter_version = "v1"
    mechanism = "probe"
    evidence_kinds = ("write",)  # missing the "probe:" prefix

    def parse(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError


class _BadEvidenceKindDriver(_MemoryDriver):
    driver_id = "test.bad_evidence"

    def capture_adapters(self, context: DriverContext) -> tuple[_BadEvidenceKindAdapter, ...]:
        del context
        return (_BadEvidenceKindAdapter(),)


def test_kit_catches_evidence_kind_drift() -> None:
    # describe() advertises no adapter row for probe.bad, so reconciliation
    # fails on the missing schema row (and the kind also violates the regex).
    with pytest.raises(AssertionError):
        assert_substrate_driver_conformant(_BadEvidenceKindDriver())


class _UnauthorizedExecutionDriver(_MemoryDriver):
    """An execution-bound driver that runs the body without authority — the
    silent in-process fallback the negotiation rule forbids."""

    driver_id = "test.unauthorized_exec"

    @property
    def execution_commands(self) -> frozenset[str]:
        return frozenset({"run"})

    def describe(self) -> DriverSchema:
        from vcs_core.spi import CommandSpec

        return DriverSchema(
            driver_id=self.driver_id,
            driver_version=self.driver_version,
            capabilities=self.capabilities,
            commands={"run": CommandSpec(description="run")},
        )

    def prepare(self, context: DriverContext, request: IngressRequest) -> DriverIngressResult:
        # Does NOT refuse for the execution command — just runs it like any
        # other command (the conformance violation).
        return super().prepare(context, request)

    def prepare_bound(
        self,
        context: DriverContext,
        request: IngressRequest,
        execution: object,
    ) -> DriverIngressResult:  # pragma: no cover - not reached by the probe
        return self.prepare(context, request)


def test_kit_catches_unauthorized_execution() -> None:
    driver = _UnauthorizedExecutionDriver()
    with pytest.raises(AssertionError):
        assert_execution_driver_conformant(driver)
    # And the aggregate catches it too, via the execution_negotiation case.
    with pytest.raises(AssertionError):
        assert_substrate_driver_conformant(driver)


# ---------------------------------------------------------------------------
# Opt-in exhaustiveness check + pytest-free guarantee
# ---------------------------------------------------------------------------


def test_match_dispatch_exhaustiveness_check_passes_on_match_driver() -> None:
    from vcs_core.runtime_substrate import TaskTraceSubstrateDriver

    assert_match_dispatch_exhaustive(TaskTraceSubstrateDriver)


def test_match_dispatch_check_rejects_non_match_driver() -> None:
    # _MemoryDriver dispatches with if/isinstance, not match — the opt-in check
    # is not applicable and says so rather than passing vacuously.
    with pytest.raises(AssertionError, match="no `match request:` block"):
        assert_match_dispatch_exhaustive(_MemoryDriver)


def test_kit_module_never_imports_pytest() -> None:
    """The kit ships inside the runtime wheel; it must not import pytest.

    Checked structurally (AST), not by substring — the kit's docstring shows a
    usage example that legitimately contains ``import pytest``.
    """
    import ast
    import pathlib

    import vcs_core.spi.testing as kit

    # No pytest in the kit's runtime namespace.
    assert "pytest" not in vars(kit)
    # No `import pytest` / `from pytest import ...` statement anywhere in source.
    tree = ast.parse(pathlib.Path(kit.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "pytest" not in imported_modules
    # Sanity: this test file itself uses pytest (so the check above is meaningful).
    assert "pytest" in sys.modules
