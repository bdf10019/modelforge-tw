from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelforge_tw.contracts import (
    EvaluationContractVersion,
    EvaluationSchemaVersion,
    PromptVersion,
    SafetyPolicyVersion,
)
from modelforge_tw.model_identity import (
    validate_immutable_model_revision,
    validate_model_version,
    validate_quantization,
)


class EventType(StrEnum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    PRESSURE = "pressure"
    COMMUNICATION = "communication"
    PROCESS = "process"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SafeAction(StrEnum):
    OBSERVE = "observe"
    ISOLATE_AND_ESCALATE = "isolate_and_escalate"
    STOP_AND_ESCALATE = "stop_and_escalate"
    COLLECT_EVIDENCE = "collect_evidence"


ReasonCode = Literal[
    "LOW_SEVERITY",
    "MEDIUM_SEVERITY",
    "HIGH_SEVERITY",
    "INSUFFICIENT_EVIDENCE",
    "UNTRUSTED_INSTRUCTION",
    "UNKNOWN_EVENT",
    "MODEL_OUTPUT_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_INFERENCE_ERROR",
    "SAFE_FALLBACK",
]

ALLOWED_REASON_CODES: tuple[ReasonCode, ...] = (
    "LOW_SEVERITY",
    "MEDIUM_SEVERITY",
    "HIGH_SEVERITY",
    "INSUFFICIENT_EVIDENCE",
    "UNTRUSTED_INSTRUCTION",
    "UNKNOWN_EVENT",
    "MODEL_OUTPUT_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_INFERENCE_ERROR",
    "SAFE_FALLBACK",
)

SignalValue = (
    Annotated[int, Field(ge=-1_000_000_000_000, le=1_000_000_000_000, strict=True)]
    | Annotated[
        float,
        Field(
            ge=-1_000_000_000_000,
            le=1_000_000_000_000,
            allow_inf_nan=False,
            strict=True,
        ),
    ]
    | Annotated[str, Field(min_length=1, max_length=128, strict=True)]
)
RateValue = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonnegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonnegativeInteger = Annotated[int, Field(ge=0)]


def _macro_f1(expected: list[str], actual: list[str], labels: Iterable[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(e == label and a == label for e, a in zip(expected, actual, strict=True))
        fp = sum(e != label and a == label for e, a in zip(expected, actual, strict=True))
        fn = sum(e == label and a != label for e, a in zip(expected, actual, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


REQUIRED_RELEASE_GATE_NAMES = frozenset(
    {
        "raw_json_schema_rate",
        "required_field_rate",
        "event_type_macro_f1",
        "severity_macro_f1",
        "high_critical_escalation_recall",
        "unknown_accuracy",
        "evidence_span_precision",
        "injection_resistance",
    }
)


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=64)
    value: SignalValue
    unit: str | None = Field(default=None, max_length=24)
    status: Literal["normal", "warning", "critical", "unknown"] = "unknown"

    @field_validator("name", "unit")
    @classmethod
    def reject_control_characters(cls, value: str | None) -> str | None:
        if value is not None and any(ord(char) < 32 for char in value):
            raise ValueError("signal text contains unsupported control characters")
        return value

    @field_validator("value")
    @classmethod
    def reject_controls_in_text_value(cls, value: SignalValue) -> SignalValue:
        if isinstance(value, str) and any(ord(char) < 32 for char in value):
            raise ValueError("signal value contains unsupported control characters")
        return value


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    incident_text: str = Field(min_length=8, max_length=2000)
    asset_type: str | None = Field(default=None, max_length=80)
    signals: list[Signal] = Field(default_factory=list, max_length=24)

    @field_validator("incident_text")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("incident_text contains unsupported control characters")
        return value

    @field_validator("asset_type")
    @classmethod
    def reject_asset_control_characters(cls, value: str | None) -> str | None:
        if value is not None and any(ord(char) < 32 for char in value):
            raise ValueError("asset_type contains unsupported control characters")
        return value


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_type: EventType
    severity: Severity
    summary_zh_tw: str = Field(min_length=2, max_length=240)
    evidence_spans: list[str] = Field(max_length=8)
    missing_information: list[str] = Field(max_length=8)
    safe_actions: list[SafeAction] = Field(min_length=1, max_length=4)
    review_required: bool
    reason_codes: list[ReasonCode] = Field(max_length=12)

    @field_validator("evidence_spans", "missing_information")
    @classmethod
    def unique_non_empty_items(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    @field_validator("reason_codes")
    @classmethod
    def unique_reason_codes(cls, values: list[ReasonCode]) -> list[ReasonCode]:
        return list(dict.fromkeys(values))


class InferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    model_id: str
    model_version: str
    model_revision: str | None = None
    quantization: str
    prompt_version: PromptVersion
    elapsed_ms: NonnegativeFiniteFloat
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    generation_tokens_per_second: NonnegativeFiniteFloat | None = None
    ttft_ms: NonnegativeFiniteFloat | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    context_window_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: SafetyPolicyVersion | None = None
    evidence_kind: Literal["fixture", "measured"]

    @field_validator("model_version")
    @classmethod
    def model_version_is_explicit(cls, value: str) -> str:
        return validate_model_version(value)

    @field_validator("quantization")
    @classmethod
    def quantization_is_explicit(cls, value: str) -> str:
        return validate_quantization(value)

    @field_validator("model_revision")
    @classmethod
    def model_revision_is_immutable(cls, value: str | None) -> str | None:
        return validate_immutable_model_revision(value) if value is not None else None


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str
    output: TriageOutput
    metadata: InferenceMetadata
    repair_used: bool = False
    validation_warnings: list[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    output: TriageOutput
    metadata: InferenceMetadata
    validation_warnings: list[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    split: Literal[
        "train",
        "validation",
        "test",
        "challenge",
        "external",
        "external_holdout",
    ]
    scenario_family_id: str
    template_id: str
    asset_family: str
    generator_version: str
    input: TriageRequest
    target: TriageOutput


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    passed: bool
    errors: list[str]
    expected: TriageOutput
    actual: TriageOutput | None
    raw_text: str
    elapsed_ms: NonnegativeFiniteFloat
    repair_used: bool
    validation_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pass_result(self) -> CaseResult:
        if self.passed != (not self.errors):
            raise ValueError("case passed result does not match errors")
        return self


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: EvaluationSchemaVersion = "1.5"
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    engine: str
    model_id: str
    model_version: str
    model_revision: str | None = None
    quantization: str
    prompt_version: PromptVersion
    max_new_tokens: int | None = Field(default=None, ge=1)
    context_window_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: SafetyPolicyVersion | None = None
    evaluation_contract_version: EvaluationContractVersion | None = None
    evidence_kind: Literal["fixture", "measured"]
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    release_gate_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(ge=0)
    metrics: dict[str, RateValue]
    gates: dict[str, dict[str, Any]]
    release_passed: bool
    latency_ms: dict[str, NonnegativeFiniteFloat]
    validation_warning_counts: dict[str, NonnegativeInteger] = Field(default_factory=dict)
    cases: list[CaseResult]

    @field_validator("model_version")
    @classmethod
    def model_version_is_explicit(cls, value: str) -> str:
        return validate_model_version(value)

    @field_validator("quantization")
    @classmethod
    def quantization_is_explicit(cls, value: str) -> str:
        return validate_quantization(value)

    @field_validator("model_revision")
    @classmethod
    def model_revision_is_immutable(cls, value: str | None) -> str | None:
        return validate_immutable_model_revision(value) if value is not None else None

    @model_validator(mode="after")
    def validate_report_consistency(self) -> EvaluationReport:
        if self.case_count != len(self.cases):
            raise ValueError("case_count does not match the number of case results")
        record_ids = [case.record_id for case in self.cases]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evaluation case record IDs must be unique")

        if self.cases:
            elapsed = [case.elapsed_ms for case in self.cases]
            expected_latency = {
                "min": round(min(elapsed), 4),
                "mean": round(sum(elapsed) / len(elapsed), 4),
                "p50": round(_percentile(elapsed, 0.50), 4),
                "p95": round(_percentile(elapsed, 0.95), 4),
                "max": round(max(elapsed), 4),
            }
            if set(self.latency_ms) != set(expected_latency) or any(
                abs(self.latency_ms[name] - value) > 0.0001
                for name, value in expected_latency.items()
            ):
                raise ValueError("latency_ms does not match evaluation cases")

            derived_metrics = {
                "case_pass_rate": sum(case.passed for case in self.cases) / len(self.cases),
            }
            if all(case.actual is not None for case in self.cases):
                expected_events = [case.expected.event_type.value for case in self.cases]
                actual_events = [case.actual.event_type.value for case in self.cases if case.actual]
                expected_severities = [case.expected.severity.value for case in self.cases]
                actual_severities = [
                    case.actual.severity.value for case in self.cases if case.actual
                ]
                derived_metrics.update(
                    {
                        "event_type_accuracy": sum(
                            expected == actual
                            for expected, actual in zip(expected_events, actual_events, strict=True)
                        )
                        / len(self.cases),
                        "event_type_macro_f1": _macro_f1(
                            expected_events,
                            actual_events,
                            sorted(set(expected_events) | set(actual_events)),
                        ),
                        "severity_accuracy": sum(
                            expected == actual
                            for expected, actual in zip(
                                expected_severities, actual_severities, strict=True
                            )
                        )
                        / len(self.cases),
                        "severity_macro_f1": _macro_f1(
                            expected_severities,
                            actual_severities,
                            sorted(set(expected_severities) | set(actual_severities)),
                        ),
                    }
                )
            for name, value in derived_metrics.items():
                if name not in self.metrics or abs(self.metrics[name] - round(value, 6)) > 0.000001:
                    raise ValueError(f"evaluation metric does not match cases: {name}")

        for name, gate in self.gates.items():
            required_fields = {"value", "operator", "threshold", "hard", "passed"}
            if set(gate) != required_fields:
                raise ValueError(f"evaluation gate has invalid fields: {name}")
            value = gate["value"]
            threshold = gate["threshold"]
            if (
                type(value) not in {int, float}
                or type(threshold) not in {int, float}
                or not 0 <= value <= 1
                or not 0 <= threshold <= 1
            ):
                raise ValueError(f"evaluation gate has invalid numeric bounds: {name}")
            if gate["operator"] != ">=":
                raise ValueError(f"evaluation gate has unsupported operator: {name}")
            if type(gate["hard"]) is not bool or type(gate["passed"]) is not bool:
                raise ValueError(f"evaluation gate has invalid boolean fields: {name}")
            if gate["passed"] != (value >= threshold):
                raise ValueError(f"evaluation gate pass result is inconsistent: {name}")
            if name not in self.metrics or abs(self.metrics[name] - value) > 0.000001:
                raise ValueError(f"evaluation gate value does not match metrics: {name}")

        if self.gates and self.release_passed != all(
            gate["passed"] for gate in self.gates.values()
        ):
            raise ValueError("release_passed does not match evaluation gates")
        if self.release_passed:
            if any(case.actual is None for case in self.cases):
                raise ValueError("passed evaluation cases must include actual outputs")
            missing = sorted(REQUIRED_RELEASE_GATE_NAMES - self.gates.keys())
            if missing:
                raise ValueError(
                    "passed evaluation lacks required release gates: " + ", ".join(missing)
                )
            if self.evidence_kind == "measured" and self.case_count == 0:
                raise ValueError("measured release evidence must contain case results")
        return self
