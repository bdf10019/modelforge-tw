from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Literal

from modelforge_tw.prompts import MINIMAL_PROMPT_VERSION, OPTIMIZED_PROMPT_VERSION
from modelforge_tw.safety import apply_safety_policy
from modelforge_tw.schemas import (
    EventType,
    InferenceMetadata,
    InferenceResult,
    SafeAction,
    Severity,
    TriageOutput,
    TriageRequest,
)

RULES: list[tuple[EventType, tuple[str, ...]]] = [
    (EventType.ELECTRICAL, ("焦味", "火花", "跳電", "漏電", "電流")),
    (EventType.THERMAL, ("溫度", "過熱", "燙", "熱點")),
    (EventType.PRESSURE, ("壓力", "洩漏", "壓降", "安全閥")),
    (EventType.MECHANICAL, ("異音", "振動", "摩擦聲", "軸承", "鬆動")),
    (EventType.COMMUNICATION, ("心跳", "失聯", "封包", "通訊", "逾時")),
    (EventType.PROCESS, ("流量", "良率", "節拍", "扭矩", "偏差")),
]


@dataclass
class FixtureEngine:
    variant: str = "adapted"

    def __post_init__(self) -> None:
        if self.variant not in {"base", "prompt", "adapted"}:
            raise ValueError("fixture variant must be base, prompt, or adapted")
        self.name = f"fixture-{self.variant}"
        self.model_id = "deterministic-fixture"
        self.model_version = "1.0.0"
        self.quantization = "none"
        self.max_new_tokens: int | None = None
        self.structured_output_contract: str = "fixture-json-v1"
        self.prompt_version = (
            MINIMAL_PROMPT_VERSION if self.variant == "base" else OPTIMIZED_PROMPT_VERSION
        )
        self.evidence_kind: Literal["fixture", "measured"] = "fixture"

    def _classify(self, text: str) -> tuple[EventType, str | None]:
        rules = RULES[:2] if self.variant == "base" else RULES
        for event_type, terms in rules:
            for term in terms:
                if term in text:
                    return event_type, term
        return EventType.UNKNOWN, None

    def _severity(self, text: str, event_type: EventType) -> Severity:
        if any(term in text for term in ("火花", "焦味", "大量洩漏", "人員受傷")):
            return Severity.CRITICAL
        if any(term in text for term in ("持續", "跳升", "超過", "連續", "失聯", "快速")):
            return Severity.HIGH
        if event_type == EventType.UNKNOWN:
            return Severity.UNKNOWN
        if any(term in text for term in ("偶爾", "輕微", "短暫")):
            return Severity.MEDIUM
        return Severity.LOW

    def generate(self, request: TriageRequest) -> InferenceResult:
        started = time.perf_counter()
        event_type, evidence = self._classify(request.incident_text)
        severity = self._severity(request.incident_text, event_type)
        review = severity in {Severity.HIGH, Severity.CRITICAL, Severity.UNKNOWN}
        if severity == Severity.CRITICAL:
            actions = [SafeAction.STOP_AND_ESCALATE]
        elif severity == Severity.HIGH:
            actions = [SafeAction.ISOLATE_AND_ESCALATE]
        elif event_type == EventType.UNKNOWN:
            actions = [SafeAction.COLLECT_EVIDENCE]
        else:
            actions = [SafeAction.OBSERVE]

        reason_codes: list[str] = []
        missing: list[str] = []
        if review:
            reason_codes.append(
                "HIGH_SEVERITY" if severity != Severity.UNKNOWN else "UNKNOWN_EVENT"
            )
        if event_type == EventType.UNKNOWN:
            missing.append("可辨識的異常現象或量測值")
        summary = (
            "事件資訊不足，需要補充可驗證證據。"
            if event_type == EventType.UNKNOWN
            else f"偵測到{event_type.value}類事件，需依風險等級處理。"
        )
        output = TriageOutput(
            event_type=event_type,
            severity=severity,
            summary_zh_tw=summary,
            evidence_spans=[evidence] if evidence else [],
            missing_information=missing,
            safe_actions=actions,
            review_required=review,
            reason_codes=reason_codes,
        )
        if self.variant == "adapted":
            output, warnings = apply_safety_policy(request, output)
        else:
            warnings = []
        raw = json.dumps(output.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(
            raw_text=raw,
            output=output,
            metadata=InferenceMetadata(
                engine=self.name,
                model_id=self.model_id,
                model_version=self.model_version,
                quantization=self.quantization,
                prompt_version=self.prompt_version,
                elapsed_ms=elapsed_ms,
                output_tokens=None,
                max_new_tokens=self.max_new_tokens,
                structured_output_contract=self.structured_output_contract,
                evidence_kind="fixture",
            ),
            validation_warnings=warnings,
        )

    def ready(self) -> bool:
        return True
