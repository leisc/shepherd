# under-test: vcs_core._query_locators
from __future__ import annotations

from vcs_core._query_locators import classify_locator_component
from vcs_core._world_refs import encode_ref_component


def test_classify_locator_component_decodes_reversible_b64u() -> None:
    component = encode_ref_component("op_123")

    classified = classify_locator_component(component)

    assert classified.encoding == "b64u"
    assert classified.decoded_value == "op_123"
    assert classified.reversible
    assert classified.to_fields("operation")["operation_decoded"] == "op_123"


def test_classify_locator_component_keeps_sha256_opaque() -> None:
    classified = classify_locator_component(f"sha256_{'a' * 64}")

    assert classified.encoding == "sha256"
    assert classified.decoded_value is None
    assert not classified.reversible


def test_classify_locator_component_reports_malformed_components() -> None:
    classified = classify_locator_component("b64u_not valid")

    assert classified.encoding == "malformed"
    assert classified.issue == "malformed_b64u_component"


def test_classify_locator_component_rejects_empty_or_non_canonical_b64u() -> None:
    assert classify_locator_component("b64u_").encoding == "malformed"
    assert classify_locator_component("b64u_@@@@").encoding == "malformed"
