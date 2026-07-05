import pytest

from shepherd_kernel_v3_reference.schemas import (
    AnySchema,
    RecordSchema,
    TaggedRecordSchema,
    TypeSchema,
    ValidationError,
    check,
    schema_fingerprint,
)


def test_any_accepts_anything() -> None:
    s = AnySchema()
    assert s.validate(1) is None
    assert s.validate("x") is None
    assert s.validate(None) is None


def test_type_schema_rejects_wrong_type() -> None:
    s = TypeSchema(int)
    assert s.validate(1) is None
    assert s.validate("1") is not None


def test_tagged_record_schema_checks_kind_field() -> None:
    s = TaggedRecordSchema("Draft")
    assert s.validate({"kind": "Draft", "text": "..."}) is None
    assert s.validate({"kind": "Prompt"}) is not None
    assert s.validate("Draft") is not None


def test_record_schema_checks_fields() -> None:
    s = RecordSchema({"x": TypeSchema(int), "y": TypeSchema(str)})
    assert s.validate({"x": 1, "y": "ok"}) is None
    assert s.validate({"x": 1}) is not None
    assert s.validate({"x": "no", "y": "ok"}) is not None


def test_check_raises_with_context() -> None:
    with pytest.raises(ValidationError, match="payload: expected int"):
        check(TypeSchema(int), "1", context="payload")


def test_check_passes_silently_on_match() -> None:
    check(TypeSchema(int), 1)


def test_schema_fingerprint_is_structural() -> None:
    assert schema_fingerprint(AnySchema()) == {"schema": "any"}
    assert schema_fingerprint(TypeSchema(int)) != schema_fingerprint(TypeSchema(str))
    assert schema_fingerprint(TaggedRecordSchema("Draft")) != schema_fingerprint(TaggedRecordSchema("Prompt"))
    assert schema_fingerprint(
        RecordSchema({"x": TypeSchema(int), "y": TaggedRecordSchema("Draft")})
    ) == schema_fingerprint(RecordSchema({"y": TaggedRecordSchema("Draft"), "x": TypeSchema(int)}))


def test_schema_fingerprint_rejects_unstable_custom_schema() -> None:
    class CustomSchema:
        def validate(self, value):
            return None

    with pytest.raises(TypeError, match="stable fingerprint"):
        schema_fingerprint(CustomSchema())


def test_schema_fingerprint_accepts_explicit_custom_identity() -> None:
    class CustomSchema:
        def validate(self, value):
            return None

        def fingerprint(self):
            return {"name": "custom.v1"}

    assert schema_fingerprint(CustomSchema()) == {
        "schema": "custom",
        "fingerprint": {"name": "custom.v1"},
    }
