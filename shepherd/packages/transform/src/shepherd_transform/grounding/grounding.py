"""Behavioral grounding for task transformations.

This module provides tools for verifying that transformed tasks preserve
the behavior of original tasks:

- behavioral_grounding(): Verify transformation preserves behavior
- ground_transformation(): High-level function combining reconstruction + grounding

Example:
    >>> from shepherd_transform.grounding import behavioral_grounding, EquivalenceLevel
    >>> result = behavioral_grounding(
    ...     original_class=Calculator,
    ...     transformed_class=CalculatorWithLogging,
    ...     test_cases=[{"x": 5, "y": 3}],
    ...     equivalence=EquivalenceLevel.OUTCOME,
    ... )
    >>> if result.passed:
    ...     print("Transformation verified!")
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

from shepherd_core.schema import SINGLE_OUTPUT_KEY
from shepherd_runtime.sync import run_sync

from .equivalence import EquivalenceLevel, EquivalenceResult, compare_at_level

if TYPE_CHECKING:
    from collections.abc import Callable

    from shepherd_runtime.task.metadata import TaskMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Grounding Result
# =============================================================================


@dataclass
class Mismatch:
    """Details of a behavioral mismatch between original and transformed task.

    Attributes:
        test_input: The input that caused the mismatch
        original_output: Output from the original task
        transformed_output: Output from the transformed task
        equivalence_result: Detailed comparison result
        error: Error message if execution failed
    """

    test_input: dict[str, Any]
    original_output: dict[str, Any] | None = None
    transformed_output: dict[str, Any] | None = None
    equivalence_result: EquivalenceResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_input": self.test_input,
            "original_output": self.original_output,
            "transformed_output": self.transformed_output,
        }
        if self.equivalence_result:
            result["differences"] = self.equivalence_result.differences  # type: ignore[assignment]
        if self.error:
            result["error"] = self.error  # type: ignore[assignment]
        return result


@dataclass
class GroundingResult:
    """Result of behavioral grounding verification.

    Attributes:
        test_count: Number of test cases run
        match_count: Number of tests where behavior matched
        mismatches: List of detailed mismatch information
        goal_achieved: Whether the transformation achieved its stated goal
        confidence: Overall confidence score (0.0 to 1.0)
    """

    test_count: int
    match_count: int
    mismatches: list[Mismatch] = field(default_factory=list)
    goal_achieved: bool = True
    confidence: float = 0.0

    @property
    def match_rate(self) -> float:
        """Percentage of tests that matched (0.0 to 1.0)."""
        return self.match_count / self.test_count if self.test_count > 0 else 0.0

    @property
    def passed(self) -> bool:
        """True if match rate >= 95% and goal achieved."""
        return self.match_rate >= 0.95 and self.goal_achieved

    def summary(self) -> str:
        """Human-readable summary of grounding result."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Grounding Result: {status}",
            f"  Tests: {self.match_count}/{self.test_count} matched ({self.match_rate:.0%})",
            f"  Goal achieved: {self.goal_achieved}",
            f"  Confidence: {self.confidence:.0%}",
        ]
        if self.mismatches:
            lines.append(f"  Mismatches: {len(self.mismatches)}")
            for i, m in enumerate(self.mismatches[:3]):
                lines.append(f"    {i + 1}. Input: {m.test_input}")
                if m.error:
                    lines.append(f"       Error: {m.error}")
                elif m.equivalence_result:
                    for diff in m.equivalence_result.differences[:2]:
                        lines.append(f"       - {diff}")
            if len(self.mismatches) > 3:
                lines.append(f"    ... and {len(self.mismatches) - 3} more")
        return "\n".join(lines)


# =============================================================================
# Task Execution
# =============================================================================


def _extract_outputs_from_instance(
    instance: Any,
    meta: TaskMetadata | None = None,
) -> dict[str, Any]:
    """Extract output values from a task instance.

    Args:
        instance: The task instance
        meta: Optional TaskMetadata for output field names

    Returns:
        Dict of output field name -> value
    """
    outputs = {}

    # Try to get output field names from metadata
    if meta is not None:
        for name in meta.outputs:
            if hasattr(instance, name):
                outputs[name] = getattr(instance, name)
        return outputs

    # Fallback: try _task_meta attribute
    if hasattr(instance, "_task_meta"):
        task_meta = instance._task_meta
        if hasattr(task_meta, "outputs"):
            for name in task_meta.outputs:
                if hasattr(instance, name):
                    outputs[name] = getattr(instance, name)
            return outputs

    # Last resort: return all non-private, non-callable attributes
    for name in dir(instance):
        if name.startswith("_"):
            continue
        value = getattr(instance, name)
        if callable(value):
            continue
        outputs[name] = value

    return outputs


def _execute_task_sync(
    task: object,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Execute a task synchronously and return outputs.

    Supports function-form CallableTask objects, @task decorated classes, and
    plain Pydantic models with _task_meta attribute (for testing/spike
    compatibility).

    Args:
        task: The task object to execute
        inputs: Input values to pass to the task

    Returns:
        Tuple of (outputs dict, error message or None)
    """
    from shepherd_runtime.nucleus import CallableTask, Run
    from shepherd_runtime.task.metadata import extract_task_metadata

    if isinstance(task, CallableTask):
        valid_inputs = {
            parameter.name: inputs[parameter.name] for parameter in task.metadata.parameters if parameter.name in inputs
        }
        try:
            maybe_run = task.detailed(**valid_inputs)
            run = run_sync(maybe_run) if isawaitable(maybe_run) else maybe_run
            if not isinstance(run, Run):
                return _normalize_task_output(run), None
            return _normalize_task_output(run.unwrap()), None
        except Exception as e:  # noqa: BLE001
            logger.debug("Callable task execution failed: %s", e)
            return {}, str(e)

    if not isinstance(task, type):
        return {}, f"Unsupported task object: {type(task).__name__}"

    try:
        # Get metadata for input/output field info
        # For @task decorated classes, this extracts Input/Output markers
        # For plain models, this may return empty inputs
        try:
            meta = extract_task_metadata(task)
            has_meta = bool(meta.inputs)
        except (TypeError, AttributeError, ValueError) as e:
            logger.debug("Could not extract task metadata from %s: %s", task.__name__, e)
            meta = None
            has_meta = False

        # Determine valid inputs
        if has_meta and meta is not None:
            # Use metadata to filter inputs (for @task classes)
            valid_inputs = {k: v for k, v in inputs.items() if k in meta.inputs}
        # For plain Pydantic models, check model_fields or _task_meta
        elif hasattr(task, "model_fields"):
            # Pydantic v2: use all inputs that match model fields
            valid_inputs = {k: v for k, v in inputs.items() if k in task.model_fields}
        elif hasattr(task, "_task_meta"):
            # Use _task_meta.inputs if available
            task_meta = task._task_meta
            if hasattr(task_meta, "inputs"):
                valid_inputs = {k: v for k, v in inputs.items() if k in task_meta.inputs}
            else:
                valid_inputs = inputs
        else:
            valid_inputs = inputs

        # Create instance
        instance = task(**valid_inputs)

        # Try compute_outputs first (for spike compatibility)
        if hasattr(instance, "compute_outputs") and callable(instance.compute_outputs):
            result = instance.compute_outputs()
            if isinstance(result, dict):
                return result, None

        # Try to execute if there's an execute method
        if hasattr(instance, "execute") and callable(instance.execute):
            result = instance.execute()
            # If execute returns a dict, use that
            if isinstance(result, dict):
                return result, None

        # Extract outputs from instance
        outputs = _extract_outputs_from_instance(instance, meta)
        return outputs, None

    except Exception as e:  # noqa: BLE001
        logger.debug("Task execution failed: %s", e)
        return {}, str(e)


def _normalize_task_output(value: Any) -> dict[str, Any]:
    """Normalize task return values to the output mapping used by grounding."""
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "dict") and callable(value.dict):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {SINGLE_OUTPUT_KEY: value}


# =============================================================================
# Behavioral Grounding
# =============================================================================


def behavioral_grounding(
    original_class: object,
    transformed_class: object,
    test_cases: list[dict[str, Any]],
    equivalence: EquivalenceLevel = EquivalenceLevel.OUTCOME,
    goal_check: Callable[[object], bool] | None = None,
    important_fields: set[str] | None = None,
) -> GroundingResult:
    """Verify that a transformed task behaves the same as the original.

    Runs both original and transformed tasks on each test case, comparing
    outputs at the specified equivalence level.

    Args:
        original_class: The original task object
        transformed_class: The transformed task object
        test_cases: List of input dicts to test with
        equivalence: How strictly to compare outputs (default: OUTCOME)
        goal_check: Optional function to verify transformation goal achieved.
                   Takes the transformed class and returns True if goal met.
        important_fields: For RELAXED level, which output fields are important

    Returns:
        GroundingResult with match rate and any mismatches

    Example:
        >>> result = behavioral_grounding(
        ...     Calculator,
        ...     CalculatorWithLogging,
        ...     [{"x": 1, "y": 2}, {"x": 10, "y": 5}],
        ...     goal_check=lambda cls: hasattr(cls, "log"),
        ... )
        >>> if result.passed:
        ...     print("Transformation verified!")
    """
    if not test_cases:
        return GroundingResult(
            test_count=0,
            match_count=0,
            goal_achieved=True,
            confidence=0.0,
        )

    match_count = 0
    mismatches: list[Mismatch] = []
    confidence_scores: list[float] = []

    for inputs in test_cases:
        # Execute both tasks
        orig_outputs, orig_error = _execute_task_sync(original_class, inputs)
        trans_outputs, trans_error = _execute_task_sync(transformed_class, inputs)

        # Any execution failure is a mismatch for grounding. Treating two
        # infrastructure failures as equivalent can hide unhandled delivery or
        # setup errors in both the original and transformed task.
        if orig_error or trans_error:
            errors = []
            if orig_error:
                errors.append(f"original failed: {orig_error}")
            if trans_error:
                errors.append(f"transformed failed: {trans_error}")
            mismatches.append(
                Mismatch(
                    test_input=inputs,
                    original_output=orig_outputs if not orig_error else None,
                    transformed_output=trans_outputs if not trans_error else None,
                    error="; ".join(errors),
                )
            )
            confidence_scores.append(0.0)
            continue

        # Compare outputs at specified equivalence level
        eq_result = compare_at_level(
            orig_outputs,
            trans_outputs,
            level=equivalence,
            important_fields=important_fields,
        )

        confidence_scores.append(eq_result.confidence)

        if eq_result.equivalent:
            match_count += 1
        else:
            mismatches.append(
                Mismatch(
                    test_input=inputs,
                    original_output=orig_outputs,
                    transformed_output=trans_outputs,
                    equivalence_result=eq_result,
                )
            )

    # Check if transformation goal was achieved
    goal_achieved = True
    if goal_check is not None:
        try:
            goal_achieved = goal_check(transformed_class)
        except Exception as e:  # noqa: BLE001
            logger.debug("Goal check failed: %s", e)
            goal_achieved = False

    # Calculate overall confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    overall_confidence = avg_confidence * (0.9 if goal_achieved else 0.5)

    return GroundingResult(
        test_count=len(test_cases),
        match_count=match_count,
        mismatches=mismatches,
        goal_achieved=goal_achieved,
        confidence=overall_confidence,
    )


def ground_transformation(
    original_class: object,
    transformed_source: str,
    test_cases: list[dict[str, Any]],
    transformation_goal: str | None = None,
    equivalence: EquivalenceLevel = EquivalenceLevel.OUTCOME,
    important_fields: set[str] | None = None,
    extra_namespace: dict[str, Any] | None = None,
) -> tuple[GroundingResult, object | None]:
    """High-level function: reconstruct and ground in one step.

    Reconstructs owner-path task source after validation, then performs
    behavioral grounding.

    Args:
        original_class: The original task object
        transformed_source: Source code of the transformed task
        test_cases: List of input dicts to test with
        transformation_goal: Description of what the transformation should achieve
        equivalence: How strictly to compare outputs (default: OUTCOME)
        important_fields: For RELAXED level, which output fields are important
        extra_namespace: Additional namespace bindings for reconstruction

    Returns:
        Tuple of (GroundingResult, transformed task or None if reconstruction failed)

    Example:
        >>> result, new_class = ground_transformation(
        ...     Calculator,
        ...     transformed_source,
        ...     [{"x": 5, "y": 3}],
        ...     transformation_goal="Add logging output",
        ... )
        >>> if result.passed and new_class:
        ...     print("Transformation verified!")
    """
    from shepherd_transform.source import try_reconstruct_task, try_reconstruct_task_class

    # Try to reconstruct the transformed task.
    recon_result = try_reconstruct_task_class(transformed_source, extra_namespace=extra_namespace)
    if not recon_result.success:
        recon_result = try_reconstruct_task(
            transformed_source,
            extra_namespace=extra_namespace,
        )

    if not recon_result.success or recon_result.task is None:
        # Reconstruction failed - return failure result
        error_msg = recon_result.error or "Unknown reconstruction error"
        return GroundingResult(
            test_count=len(test_cases),
            match_count=0,
            mismatches=[
                Mismatch(
                    test_input={},
                    error=f"Reconstruction failed: {error_msg}",
                )
            ],
            goal_achieved=False,
            confidence=0.0,
        ), None

    transformed_task = recon_result.task

    # Define goal check if goal description provided
    goal_check = None
    if transformation_goal:
        # For now, goal check is just that reconstruction succeeded
        # Future: Use LLM to verify goal was achieved
        def goal_check(_task: object) -> bool:
            return True

    # Perform behavioral grounding
    result = behavioral_grounding(
        original_class=original_class,
        transformed_class=transformed_task,
        test_cases=test_cases,
        equivalence=equivalence,
        goal_check=goal_check,
        important_fields=important_fields,
    )

    return result, transformed_task


__all__ = [
    "GroundingResult",
    # Result types
    "Mismatch",
    # Grounding functions
    "behavioral_grounding",
    "ground_transformation",
]
