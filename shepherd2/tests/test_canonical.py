from __future__ import annotations

import pytest
from shepherd2.canonical import (
    CANONICAL_VERSION,
    ROOT_WITNESS_REF,
    ROOT_WITNESS_SCHEMA_REF,
    WITNESS_SCHEMA_REF,
    canonical_json_bytes,
    canonical_record_input,
    record_digest,
    root_witness_body,
    root_witness_body_digest,
    root_witness_record_id,
    validate_witness_body,
    witness_body_digest,
)


def _witness_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "active_binding_refs": ["binding:task"],
        "actor_ref": "runtime:test",
        "authority_refs": ["trusted:internal"],
        "containment": "contained",
        "provenance_policy_refs": [],
        "semantic_environment_refs": ["schema-set:test"],
        "substrate_ref": "sqlite.local.v1",
        "visibility_policy_refs": ["visibility:payload"],
    }
    body.update(overrides)
    return body


def _witness_record_id(body: dict[str, object]) -> str:
    return record_digest(
        schema_ref=WITNESS_SCHEMA_REF,
        mode="capture",
        body=body,
        witness=root_witness_record_id(),
    )


def test_canonical_json_is_stable_and_utf8() -> None:
    left = canonical_json_bytes({"z": 1, "a": {"snowman": "snowman-☃"}})
    right = canonical_json_bytes({"a": {"snowman": "snowman-☃"}, "z": 1})

    assert left == right
    assert left == b'{"a":{"snowman":"snowman-\xe2\x98\x83"},"z":1}'


def test_record_digest_uses_kernel_v2_prefix() -> None:
    witness = _witness_record_id(_witness_body())
    digest = record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={"value": 1},
        witness=witness,
    )

    assert CANONICAL_VERSION == "shepherd.kernel.canonical.v2"
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_record_digest_is_independent_of_json_key_order() -> None:
    witness = _witness_record_id(_witness_body())

    assert record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={"z": 1, "a": 2},
        witness=witness,
    ) == record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={"a": 2, "z": 1},
        witness=witness,
    )


def test_record_digest_changes_when_mode_changes() -> None:
    witness = _witness_record_id(_witness_body())
    capture = record_digest(schema_ref="example.step.v1", mode="capture", body={"value": 1}, witness=witness)
    declaration = record_digest(
        schema_ref="example.step.v1",
        mode="declaration",
        body={"value": 1},
        witness=witness,
    )

    assert capture != declaration


def test_record_digest_preserves_parent_order() -> None:
    witness = _witness_record_id(_witness_body())

    assert record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={"value": 1},
        caused_by=("sha256:a", "sha256:b"),
        witness=witness,
    ) != record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={"value": 1},
        caused_by=("sha256:b", "sha256:a"),
        witness=witness,
    )


def test_record_digest_changes_when_retained_witness_record_id_changes() -> None:
    first = _witness_record_id(_witness_body(substrate_ref="sqlite.local.v1"))
    second = _witness_record_id(_witness_body(substrate_ref="fs.local.v1"))

    assert first != second
    assert record_digest(schema_ref="example.step.v1", mode="capture", body={}, witness=first) != record_digest(
        schema_ref="example.step.v1",
        mode="capture",
        body={},
        witness=second,
    )


def test_witness_body_digest_changes_with_substrate_and_containment() -> None:
    base = witness_body_digest(schema_ref=WITNESS_SCHEMA_REF, body=_witness_body())
    different_substrate = witness_body_digest(
        schema_ref=WITNESS_SCHEMA_REF, body=_witness_body(substrate_ref="fs.local.v1")
    )
    different_containment = witness_body_digest(schema_ref=WITNESS_SCHEMA_REF, body=_witness_body(containment="full"))

    assert base != different_substrate
    assert base != different_containment


def test_witness_body_digest_is_not_the_retained_witness_record_id() -> None:
    body = _witness_body()

    assert witness_body_digest(schema_ref=WITNESS_SCHEMA_REF, body=body) != _witness_record_id(body)


def test_root_witness_body_digest_is_deterministic() -> None:
    assert root_witness_body()["substrate_ref"] == "kernel"
    assert root_witness_body()["containment"] == "full"
    assert root_witness_body_digest() == witness_body_digest(
        schema_ref=ROOT_WITNESS_SCHEMA_REF, body=root_witness_body()
    )


def test_record_input_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        canonical_record_input(
            schema_ref="example.step.v1",
            mode="captured",
            body={},
            witness=_witness_record_id(_witness_body()),
        )  # type: ignore[arg-type]


def test_root_witness_record_input_accepts_empty_witness_sentinel() -> None:
    assert canonical_record_input(
        schema_ref=ROOT_WITNESS_SCHEMA_REF,
        mode="capture",
        body=root_witness_body(),
        witness=ROOT_WITNESS_REF,
    ) == {
        "body": root_witness_body(),
        "caused_by": [],
        "kind": "record",
        "mode": "capture",
        "schema_ref": ROOT_WITNESS_SCHEMA_REF,
        "witness": ROOT_WITNESS_REF,
    }


def test_non_root_record_input_rejects_empty_witness_sentinel() -> None:
    with pytest.raises(ValueError, match="root witness"):
        canonical_record_input(
            schema_ref="example.step.v1",
            mode="capture",
            body={},
            witness=ROOT_WITNESS_REF,
        )


def test_ordinary_witness_record_input_rejects_empty_witness_sentinel() -> None:
    with pytest.raises(ValueError, match="root witness"):
        canonical_record_input(
            schema_ref=WITNESS_SCHEMA_REF,
            mode="capture",
            body=_witness_body(),
            witness=ROOT_WITNESS_REF,
        )


def test_root_witness_record_input_rejects_non_empty_witness() -> None:
    with pytest.raises(ValueError, match="empty witness sentinel"):
        canonical_record_input(
            schema_ref=ROOT_WITNESS_SCHEMA_REF,
            mode="capture",
            body=root_witness_body(),
            witness=root_witness_body_digest(),
        )


def test_root_witness_record_input_rejects_wrong_shape_with_empty_witness() -> None:
    with pytest.raises(ValueError, match="mode"):
        canonical_record_input(
            schema_ref=ROOT_WITNESS_SCHEMA_REF,
            mode="declaration",
            body=root_witness_body(),
            witness=ROOT_WITNESS_REF,
        )

    with pytest.raises(ValueError, match="causal parents"):
        canonical_record_input(
            schema_ref=ROOT_WITNESS_SCHEMA_REF,
            mode="capture",
            body=root_witness_body(),
            caused_by=("sha256:parent",),
            witness=ROOT_WITNESS_REF,
        )

    wrong_body = root_witness_body()
    wrong_body["actor_ref"] = "kernel:not-root"
    with pytest.raises(ValueError, match="root_witness_body"):
        canonical_record_input(
            schema_ref=ROOT_WITNESS_SCHEMA_REF,
            mode="capture",
            body=wrong_body,
            witness=ROOT_WITNESS_REF,
        )


def test_witness_body_requires_substrate_and_containment() -> None:
    body = _witness_body()
    del body["substrate_ref"]

    with pytest.raises(ValueError, match="substrate_ref"):
        validate_witness_body(schema_ref=WITNESS_SCHEMA_REF, body=body)

    with pytest.raises(ValueError, match="substrate_ref"):
        witness_body_digest(schema_ref=WITNESS_SCHEMA_REF, body=body)
