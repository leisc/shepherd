# under-test: vcs_core._payload_authority
"""Unit tests for coordinator-owned payload descriptor authority."""

from __future__ import annotations

import pytest
from vcs_core import InvalidRepositoryStateError, canonical_digest
from vcs_core._payload_authority import validate_payload_descriptor_claim
from vcs_core._transition_kernel_records import ValidatedPayloadDescriptor
from vcs_core.spi import PayloadDescriptorClaim


def test_payload_authority_accepts_json_claim() -> None:
    payload = {"schema": "example/workspace", "label": "candidate"}

    descriptor = validate_payload_descriptor_claim(
        PayloadDescriptorClaim.for_json_payload(payload),
        payload=payload,
    )

    assert descriptor == ValidatedPayloadDescriptor.for_json_payload(payload)


@pytest.mark.parametrize(
    ("claim", "match"),
    [
        (
            PayloadDescriptorClaim(
                codec_id="vcscore.json",
                codec_version="v1",
                authority_mode="coordinator-native",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 2}),
                canonical_manifest={"payload_format": "canonical-json-v1"},
            ),
            "disagrees with payload",
        ),
        (
            PayloadDescriptorClaim(
                codec_id="test.unregistered-json",
                codec_version="v1",
                authority_mode="coordinator-native",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 1}),
                canonical_manifest={"payload_format": "canonical-json-v1"},
            ),
            "codec is not registered",
        ),
        (
            PayloadDescriptorClaim(
                codec_id="vcscore.json",
                codec_version="v2",
                authority_mode="coordinator-native",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 1}),
                canonical_manifest={"payload_format": "canonical-json-v1"},
            ),
            "codec version is not registered",
        ),
        (
            PayloadDescriptorClaim(
                codec_id="vcscore.json",
                codec_version="v1",
                authority_mode="registered-driver-codec",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 1}),
                canonical_manifest={"payload_format": "canonical-json-v1"},
            ),
            "authority mode is invalid",
        ),
        (
            PayloadDescriptorClaim(
                codec_id="vcscore.json",
                codec_version="v1",
                authority_mode="coordinator-native",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 1}),
                canonical_manifest={"payload_format": "other-json-v1"},
            ),
            "manifest is invalid",
        ),
        (
            PayloadDescriptorClaim(
                codec_id="vcscore.json",
                codec_version="v1",
                authority_mode="coordinator-native",
                payload_digest=canonical_digest({"schema": "example/workspace", "n": 1}),
                canonical_manifest={"payload_format": "canonical-json-v1"},
                payload_ref="refs/vcscore/payloads/test",
            ),
            "must not carry payload_ref",
        ),
    ],
)
def test_payload_authority_rejects_invalid_json_claims(claim: PayloadDescriptorClaim, match: str) -> None:
    with pytest.raises(InvalidRepositoryStateError, match=match):
        validate_payload_descriptor_claim(claim, payload={"schema": "example/workspace", "n": 1})
