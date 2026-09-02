from typing import Literal

OptimizedPromptVersion = Literal[
    "optimized-v1",
    "optimized-v2",
    "optimized-v3",
    "optimized-v4",
    "optimized-v5",
    "optimized-v6",
]
PromptVersion = Literal["minimal-v1"] | OptimizedPromptVersion
SafetyPolicyVersion = Literal["risk-precedence-v2", "risk-precedence-v3"]
EvaluationSchemaVersion = Literal["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
EvaluationContractVersion = Literal[
    "release-metrics-v2",
    "release-metrics-v3",
    "release-metrics-v4",
    "release-metrics-v5",
]
ValidationWarning = Literal[
    "EVIDENCE_SPAN_NOT_IN_INPUT",
    "EVENT_TYPE_FORCED_UNKNOWN_BY_POLICY",
    "REVIEW_FORCED_BY_POLICY",
    "RISK_ESCALATED_BY_POLICY",
    "UNTRUSTED_INSTRUCTION_DETECTED",
    "MODEL_OUTPUT_INVALID",
    "SAFE_FALLBACK_APPLIED",
    "TIMEOUT_FALLBACK",
    "ERROR_FALLBACK",
]
