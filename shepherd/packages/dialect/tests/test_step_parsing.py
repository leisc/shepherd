"""W2a re-pins — same-name port of meta/tests/unit/step/test_step_parsing.py onto shepherd_dialect.steps."""

"""Tests for step output parsing and value coercion."""

from enum import StrEnum
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from shepherd_dialect.steps import (
    StepOutputError,
    coerce_step_value,
    coerce_to_bool,
    coerce_to_enum,
    coerce_to_list,
    parse_single_output,
    parse_step_output,
    parse_tuple_output,
)


class Severity(StrEnum):
    """Test enum for step return types."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisResult(BaseModel):
    """Test Pydantic model for step return types."""

    summary: str
    score: int
    tags: list[str] = []


# =============================================================================
# Output Parsing
# =============================================================================


class TestOutputParsing:
    """Test output parsing logic."""

    def test_parse_single_output_with_result_key(self):
        """parse_single_output extracts result key correctly."""
        result = parse_single_output({"result": "hello"}, str, "test_step")
        assert result == "hello"

    def test_parse_single_output_with_different_key(self):
        """parse_single_output handles single non-result key."""
        result = parse_single_output({"output": "hello"}, str, "test_step")
        assert result == "hello"

    def test_parse_single_output_error_on_empty(self):
        """Empty result dict raises StepOutputError."""
        with pytest.raises(StepOutputError, match="Empty response"):
            parse_single_output({}, str, "test_step")

    def test_parse_single_output_error_on_multiple_keys(self):
        """Multiple keys without result key raises StepOutputError."""
        with pytest.raises(StepOutputError, match="Multiple keys"):
            parse_single_output({"a": 1, "b": 2}, str, "test_step")

    def test_parse_tuple_output(self):
        """parse_tuple_output extracts tuple values correctly."""
        result = parse_tuple_output({"output_0": "hello", "output_1": 42}, (str, int), "test_step")
        assert result == ("hello", 42)

    def test_parse_tuple_output_error_on_missing_key(self):
        """Missing tuple key raises StepOutputError."""
        with pytest.raises(StepOutputError, match="Missing required keys"):
            parse_tuple_output({"output_0": "hello"}, (str, int), "test_step")

    def test_parse_step_output_none_type(self):
        """NoneType return type returns None."""
        result = parse_step_output({}, type(None), "test_step")
        assert result is None


# =============================================================================
# Value Coercion
# =============================================================================


class TestValueCoercion:
    """Test value coercion logic."""

    def test_coerce_to_bool_true_values(self):
        """coerce_to_bool handles true-ish strings."""
        assert coerce_to_bool("true") is True
        assert coerce_to_bool("TRUE") is True
        assert coerce_to_bool("True") is True
        assert coerce_to_bool("yes") is True
        assert coerce_to_bool("YES") is True
        assert coerce_to_bool("1") is True
        assert coerce_to_bool(True) is True

    def test_coerce_to_bool_false_values(self):
        """coerce_to_bool handles false-ish strings."""
        assert coerce_to_bool("false") is False
        assert coerce_to_bool("FALSE") is False
        assert coerce_to_bool("False") is False
        assert coerce_to_bool("no") is False
        assert coerce_to_bool("NO") is False
        assert coerce_to_bool("0") is False
        assert coerce_to_bool(False) is False

    def test_coerce_to_enum_by_value(self):
        """coerce_to_enum matches by value."""
        result = coerce_to_enum("low", Severity, "test_step")
        assert result == Severity.LOW

    def test_coerce_to_enum_by_name(self):
        """coerce_to_enum matches by name if value fails."""
        result = coerce_to_enum("LOW", Severity, "test_step")
        assert result == Severity.LOW

    def test_coerce_to_enum_error_on_invalid(self):
        """coerce_to_enum raises StepOutputError on invalid value."""
        with pytest.raises(StepOutputError, match="Not a valid enum"):
            coerce_to_enum("invalid", Severity, "test_step")

    def test_coerce_to_list_from_list(self):
        """coerce_to_list passes through lists."""
        result = coerce_to_list(["a", "b"], (str,), "test_step")
        assert result == ["a", "b"]

    def test_coerce_to_list_wraps_single_value(self):
        """coerce_to_list wraps single value in list."""
        result = coerce_to_list("single", (str,), "test_step")
        assert result == ["single"]

    def test_coerce_to_list_none_returns_empty(self):
        """coerce_to_list returns empty list for None."""
        result = coerce_to_list(None, (), "test_step")
        assert result == []

    def test_coerce_step_value_literal(self):
        """coerce_step_value validates Literal values."""
        result = coerce_step_value("a", Literal["a", "b"], "test", "field")
        assert result == "a"

    def test_coerce_step_value_literal_error(self):
        """coerce_step_value raises on invalid Literal."""
        with pytest.raises(StepOutputError, match="not in allowed literals"):
            coerce_step_value("c", Literal["a", "b"], "test", "field")

    def test_coerce_step_value_primitives(self):
        """coerce_step_value handles primitive types."""
        assert coerce_step_value("hello", str, "test", "f") == "hello"
        assert coerce_step_value(42, int, "test", "f") == 42
        assert coerce_step_value(3.14, float, "test", "f") == 3.14
        assert coerce_step_value(True, bool, "test", "f") is True

    def test_coerce_step_value_none_allowed(self):
        """coerce_step_value allows None for Optional types."""
        result = coerce_step_value(None, Optional[str], "test", "field")  # noqa: UP045
        assert result is None

    def test_coerce_step_value_none_not_allowed(self):
        """coerce_step_value raises on None for non-Optional."""
        with pytest.raises(StepOutputError, match="doesn't allow None"):
            coerce_step_value(None, str, "test", "field")


# =============================================================================
# C2: Pydantic Model Coercion Error Handling
# =============================================================================


class TestPydanticCoercionErrors:
    """Tests for C2 fix: explicit errors when coercing invalid types to Pydantic models.

    Previously, coercing an int or list to a Pydantic model would silently
    return the original value (type violation). Now it raises StepOutputError.
    """

    def test_coerce_int_to_pydantic_model_raises_error(self):
        """Coercing an int to a Pydantic model should raise StepOutputError."""
        with pytest.raises(StepOutputError, match="Cannot coerce int to AnalysisResult"):
            coerce_step_value(42, AnalysisResult, "test_step", "field")

    def test_coerce_list_to_pydantic_model_raises_error(self):
        """Coercing a list to a Pydantic model should raise StepOutputError."""
        with pytest.raises(StepOutputError, match="Cannot coerce list to AnalysisResult"):
            coerce_step_value([1, 2, 3], AnalysisResult, "test_step", "field")

    def test_coerce_float_to_pydantic_model_raises_error(self):
        """Coercing a float to a Pydantic model should raise StepOutputError."""
        with pytest.raises(StepOutputError, match="Cannot coerce float to AnalysisResult"):
            coerce_step_value(3.14, AnalysisResult, "test_step", "field")

    def test_coerce_bool_to_pydantic_model_raises_error(self):
        """Coercing a bool to a Pydantic model should raise StepOutputError."""
        with pytest.raises(StepOutputError, match="Cannot coerce bool to AnalysisResult"):
            coerce_step_value(True, AnalysisResult, "test_step", "field")

    def test_coerce_dict_to_pydantic_model_succeeds(self):
        """Coercing a valid dict to a Pydantic model should succeed."""
        result = coerce_step_value(
            {"summary": "test", "score": 5, "tags": ["a"]},
            AnalysisResult,
            "test_step",
            "field",
        )
        assert isinstance(result, AnalysisResult)
        assert result.summary == "test"
        assert result.score == 5

    def test_coerce_json_string_to_pydantic_model_succeeds(self):
        """Coercing a valid JSON string to a Pydantic model should succeed."""
        json_str = '{"summary": "test", "score": 5, "tags": []}'
        result = coerce_step_value(json_str, AnalysisResult, "test_step", "field")
        assert isinstance(result, AnalysisResult)
        assert result.summary == "test"

    def test_coerce_invalid_json_string_to_pydantic_model_raises(self):
        """Coercing an invalid JSON string to a Pydantic model should raise."""
        with pytest.raises(StepOutputError, match="Cannot coerce str to AnalysisResult"):
            coerce_step_value("not valid json", AnalysisResult, "test_step", "field")

    def test_coerce_pydantic_instance_returns_same(self):
        """Coercing a Pydantic model instance to same type returns it unchanged."""
        instance = AnalysisResult(summary="test", score=5, tags=[])
        result = coerce_step_value(instance, AnalysisResult, "test_step", "field")
        assert result is instance

    def test_error_message_is_actionable(self):
        """StepOutputError message should tell user what types are expected."""
        with pytest.raises(StepOutputError) as exc_info:
            coerce_step_value(42, AnalysisResult, "my_step", "output")

        error = exc_info.value
        assert "dict" in error.reason.lower() or "json" in error.reason.lower()
        assert "AnalysisResult" in error.reason
