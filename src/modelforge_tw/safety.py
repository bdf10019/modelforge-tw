from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from modelforge_tw.schemas import (
    EventType,
    ReasonCode,
    SafeAction,
    Severity,
    TriageOutput,
    TriageRequest,
)

SafetyPolicyVersion = Literal["risk-precedence-v2", "risk-precedence-v3"]
SAFETY_POLICY_VERSION: SafetyPolicyVersion = "risk-precedence-v2"
EXPERIMENTAL_SAFETY_POLICY_VERSION: SafetyPolicyVersion = "risk-precedence-v3"
SUPPORTED_SAFETY_POLICY_VERSIONS = (
    SAFETY_POLICY_VERSION,
    EXPERIMENTAL_SAFETY_POLICY_VERSION,
)

INJECTION_PATTERN = re.compile(
    r"忽略(?:前述|以上|所有)?(?:規則|指令)|ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"system\s+prompt|改成正常|不要人工複核",
    re.IGNORECASE,
)

CRITICAL_RISK_PATTERN = re.compile(
    r"火花|焦味|電弧|大量.*(?:洩漏|外洩)|人員受傷|立即危害|爆炸|起火",
    re.IGNORECASE,
)
HIGH_RISK_PATTERN = re.compile(
    r"持續|連續|逐步升高|逐漸惡化|趨勢.*(?:升高|增加|未停止)|快速(?:下降|上升)|"
    r"超過(?:警戒|設定)?(?:值|上限)|越過預警|(?:9[0-9]|1[0-9]{2})°C|過熱變色|"
    r"失去連線|大量封包重送|所有.*逾時",
    re.IGNORECASE,
)
INDETERMINATE_EVENT_PATTERN = re.compile(
    r"(?:無法|無從|不能)判定事件類型|事件類型(?:無法|無從|不能)判定",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    output: TriageOutput
    repair_used: bool


def contains_untrusted_instruction(request: TriageRequest) -> bool:
    text_fields = [request.incident_text]
    if request.asset_type:
        text_fields.append(request.asset_type)
    for signal in request.signals:
        text_fields.append(signal.name)
        if isinstance(signal.value, str):
            text_fields.append(signal.value)
        if signal.unit:
            text_fields.append(signal.unit)
    return bool(INJECTION_PATTERN.search("\n".join(text_fields)))


def parse_output(raw_text: str, *, allow_bounded_repair: bool) -> ParseResult:
    text = raw_text.strip()
    try:
        return ParseResult(TriageOutput.model_validate(json.loads(text)), False)
    except (json.JSONDecodeError, ValidationError, TypeError):
        if not allow_bounded_repair:
            raise

    repaired = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if repaired.startswith("```json"):
        repaired = repaired[7:]
    elif repaired.startswith("```"):
        repaired = repaired[3:]
    if repaired.endswith("```"):
        repaired = repaired[:-3]
    first = repaired.find("{")
    last = repaired.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("model output does not contain a JSON object")
    payload = json.loads(repaired[first : last + 1])
    return ParseResult(TriageOutput.model_validate(payload), True)


def apply_safety_policy(
    request: TriageRequest,
    output: TriageOutput,
    *,
    policy_version: SafetyPolicyVersion = SAFETY_POLICY_VERSION,
) -> tuple[TriageOutput, list[str]]:
    if policy_version not in SUPPORTED_SAFETY_POLICY_VERSIONS:
        raise ValueError(f"unsupported safety policy version: {policy_version}")
    warnings: list[str] = []
    update: dict[str, object] = {}

    invalid_spans = [span for span in output.evidence_spans if span not in request.incident_text]
    grounded_spans = [span for span in output.evidence_spans if span in request.incident_text]
    if invalid_spans:
        update["evidence_spans"] = grounded_spans
        update["review_required"] = True
        warnings.append("EVIDENCE_SPAN_NOT_IN_INPUT")

    event_indeterminate = policy_version == EXPERIMENTAL_SAFETY_POLICY_VERSION and bool(
        INDETERMINATE_EVENT_PATTERN.search(request.incident_text)
    )
    if event_indeterminate and output.event_type != EventType.UNKNOWN:
        update["event_type"] = EventType.UNKNOWN
        update["review_required"] = True
        warnings.append("EVENT_TYPE_FORCED_UNKNOWN_BY_POLICY")

    must_review = output.severity in {Severity.HIGH, Severity.CRITICAL, Severity.UNKNOWN}
    must_review = must_review or output.event_type == EventType.UNKNOWN or event_indeterminate
    if must_review and not output.review_required:
        update["review_required"] = True
        warnings.append("REVIEW_FORCED_BY_POLICY")

    reason_codes = list(output.reason_codes)

    policy_severity: Severity | None = None
    policy_action: SafeAction | None = None
    if CRITICAL_RISK_PATTERN.search(request.incident_text):
        policy_severity = Severity.CRITICAL
        policy_action = SafeAction.STOP_AND_ESCALATE
    elif HIGH_RISK_PATTERN.search(request.incident_text):
        policy_severity = Severity.HIGH
        policy_action = SafeAction.ISOLATE_AND_ESCALATE
    severity_rank = {
        Severity.UNKNOWN: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }
    if (
        policy_severity is not None
        and severity_rank[output.severity] < severity_rank[policy_severity]
    ):
        update["severity"] = policy_severity
        update["safe_actions"] = [policy_action]
        update["review_required"] = True
        if "HIGH_SEVERITY" not in reason_codes:
            reason_codes.append("HIGH_SEVERITY")
        warnings.append("RISK_ESCALATED_BY_POLICY")

    if contains_untrusted_instruction(request):
        update["review_required"] = True
        if "UNTRUSTED_INSTRUCTION" not in reason_codes:
            reason_codes.append("UNTRUSTED_INSTRUCTION")
        warnings.append("UNTRUSTED_INSTRUCTION_DETECTED")

    if output.event_type == EventType.UNKNOWN or event_indeterminate or not grounded_spans:
        update["review_required"] = True
        if policy_action is None:
            update["safe_actions"] = [SafeAction.COLLECT_EVIDENCE]
        if not output.missing_information:
            update["missing_information"] = ["可驗證的設備狀態與量測證據"]
        if "INSUFFICIENT_EVIDENCE" not in reason_codes:
            reason_codes.append("INSUFFICIENT_EVIDENCE")

    priority_codes = ("HIGH_SEVERITY", "UNTRUSTED_INSTRUCTION", "INSUFFICIENT_EVIDENCE")
    if len(reason_codes) > 12:
        prioritized = [code for code in priority_codes if code in reason_codes]
        remaining = [code for code in reason_codes if code not in prioritized]
        reason_codes = (prioritized + remaining)[:12]
    if reason_codes != output.reason_codes:
        update["reason_codes"] = reason_codes

    if update:
        payload = output.model_dump(mode="json")
        payload.update(update)
        output = TriageOutput.model_validate(payload)
    return output, warnings


def conservative_fallback(reason_code: ReasonCode) -> TriageOutput:
    return TriageOutput(
        event_type=EventType.UNKNOWN,
        severity=Severity.UNKNOWN,
        summary_zh_tw="模型結果無法安全驗證，請由人員確認原始事件。",
        evidence_spans=[],
        missing_information=["可驗證的設備狀態與量測證據"],
        safe_actions=[SafeAction.COLLECT_EVIDENCE],
        review_required=True,
        reason_codes=[reason_code, "SAFE_FALLBACK"],
    )
