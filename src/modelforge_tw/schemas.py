from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


ALLOWED_REASON_CODES = (
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


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=64)
    value: float | int | str
    unit: str | None = Field(default=None, max_length=24)
    status: Literal["normal", "warning", "critical", "unknown"] = "unknown"


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


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_type: EventType
    severity: Severity
    summary_zh_tw: str = Field(min_length=2, max_length=240)
    evidence_spans: list[str] = Field(max_length=8)
    missing_information: list[str] = Field(max_length=8)
    safe_actions: list[SafeAction] = Field(min_length=1, max_length=4)
    review_required: bool
    reason_codes: list[str] = Field(max_length=12)

    @field_validator("evidence_spans", "missing_information", "reason_codes")
    @classmethod
    def unique_non_empty_items(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned


class InferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    model_id: str
    model_version: str = "unversioned"
    model_revision: str | None = None
    quantization: str = "unknown"
    prompt_version: str
    elapsed_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: str | None = None
    evidence_kind: Literal["fixture", "measured"]


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
    split: Literal["train", "validation", "test", "challenge"]
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
    elapsed_ms: float
    repair_used: bool


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.2"
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    engine: str
    model_id: str
    model_version: str
    model_revision: str | None = None
    quantization: str
    prompt_version: str
    max_new_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: str | None = None
    evaluation_contract_version: str | None = None
    evidence_kind: Literal["fixture", "measured"]
    dataset_manifest_sha256: str
    release_gate_sha256: str | None = None
    case_count: int
    metrics: dict[str, float]
    gates: dict[str, dict[str, Any]]
    release_passed: bool
    latency_ms: dict[str, float]
    cases: list[CaseResult]
